"""
Block-sparse runtime utilities for CUTE DSL kernels.

This module contains runtime execution functions for block-sparse attention kernels.
These utilities are used by CUTE DSL kernels to produce and consume block-sparse loads.
"""

from typing import Callable, Optional
from functools import partial
import math
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from quack import copy_utils
from quack import layout_utils

# Import data structures from block_sparsity
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.named_barrier import NamedBarrierBwd


# NOTE [SM100 block-sparse empty tiles: mbarrier contract]
#
# For block-sparse SM100 forward, a given (m_block, stage) Q tile can have zero active
# KV blocks (total_block_cnt == 0). In that case there is no seqlen_kv iteration, so
# the softmax warp-group has no row stats to publish.
#
# The correction warp-group seeds fully-masked-row stats and runs the usual correction
# epilogue so output/LSE have well-defined values. Both warp-groups must still perform
# the softmax<->correction mbarrier handshake so phases advance correctly across
# empty->empty and empty->non-empty tile sequences.
#
# In the no-sink case, this corresponds to the usual fully-masked-row convention:
# output is zero and LSE is -inf.
#
# Barrier contract (each is `mbar_ptr + <offset> + stage`):
#
# Producer/consumer pairs:
# - `mbar_softmax_corr_full`    : softmax arrive        -> correction wait
# - `mbar_softmax_corr_empty`   : correction arrive     -> softmax wait
# - `mbar_P_full_O_rescaled`    : softmax arrive (+ correction arrive) -> MMA wait
# - `mbar_P_full_2`             : softmax arrive        -> MMA wait
# - `mbar_corr_epi_full_/empty` : correction <-> epilogue (only when epilogue is separate)
#
# Empty tile (`total_block_cnt == 0`):
# - Softmax: skips the seqlen_kv softmax path entirely (no P stores, no `mbar_P_full_*`).
#   It only arrives `mbar_softmax_corr_full` once per stage as a synthetic "no work" signal.
#   At the `softmax_loop` level, softmax unconditionally waits `mbar_softmax_corr_empty`
#   before each tile (when block-sparse) to drain a prior correction arrival and keep
#   phases aligned across non-empty -> empty transitions.
# - Correction: waits `mbar_softmax_corr_full`, seeds stats + runs `correction_epilogue(scale=0)`,
#   and arrives `mbar_softmax_corr_empty` (and `mbar_corr_epi_full_/empty` when applicable).
# - No `mbar_P_full_*` barriers are arrived (no P, no MMA O); only the softmax<->correction
#   (and correction<->epilogue) handshakes advance phases.
#
# Non-empty tile:
# - Softmax: runs `softmax_step` (produces P) and uses `mbar_softmax_corr_full/empty` to
#   publish row_max (during seqlen_kv) and final row stats (once per tile), and to advance phases;
#   arrives `mbar_P_full_*` when P is stored.
# - Correction: waits `mbar_softmax_corr_full`, may rescale/release O, arrives `mbar_softmax_corr_empty`
#   to ack/advance, and arrives `mbar_P_full_O_rescaled` when MMA can proceed.
#
# Backward (SM100):
# - Empty KV tile: for a given `n_block`, `total_m_block_cnt == 0` means no Q tiles contribute.
# - Both the load and compute loops guard all pipeline work on `process_tile`, so empty tiles
#   skip producer/consumer operations entirely (no per-tile mbarrier phase handshake like forward).
# - In the `not dKV_postprocess` path, dK/dV for empty KV tiles are explicitly written as zeros
#   even when `process_tile == False` (see `flash_bwd_sm100.py` `should_zero_dKV`).


@cute.jit
def load_block_list(
    block_indices: cute.Tensor,
    block_count,
    first_block_preloaded: cutlass.Constexpr,
    kv_producer_state,
    load_K,
    load_V,
    pipeline_k,
    pipeline_v,
    intra_wg_overlap: cutlass.Constexpr,
):
    """Iterate over the sparse blocks and load K, V into the pipeline.
    For the intra_wg_overlap case, we overlap the loads of K and V. And this
    means we need to pipeline the last V load from the partial block case,
    with the loads for the full blocks. Set first_block_preloaded when the
    caller has already issued the first K load for the list.

    Q is loaded separately on its own mbarrier before this function is called.

    Note:
        we iterate along the block_n indices in reverse.

    Returns:
        Updated kv_producer_state after processing the block list.

    """
    if block_count > 0:
        if const_expr(not intra_wg_overlap):
            for offset in cutlass.range(block_count):
                n_block = block_indices[block_count - 1 - offset]
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block, producer_state=kv_producer_state)
                pipeline_v.producer_acquire(kv_producer_state)
                load_V(src_idx=n_block, producer_state=kv_producer_state)
                kv_producer_state.advance()
        else:
            n_block_first = block_indices[block_count - 1]
            if const_expr(not first_block_preloaded):
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block_first, producer_state=kv_producer_state)

            for idx in cutlass.range(block_count - 1, unroll=1):
                n_block_prev = block_indices[block_count - 1 - idx]
                n_block = block_indices[block_count - 2 - idx]
                kv_producer_state_prev = kv_producer_state.clone()
                kv_producer_state.advance()
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block, producer_state=kv_producer_state)
                pipeline_v.producer_acquire(kv_producer_state_prev)
                load_V(src_idx=n_block_prev, producer_state=kv_producer_state_prev)

    return kv_producer_state


@cute.jit
def finish_overlap_v_load(
    block_indices: cute.Tensor,
    block_count,
    load_V,
    pipeline_v,
    kv_producer_state,
):
    """Load the final V block after overlapped K/V loads."""
    if block_count > 0:
        n_block_last = block_indices[0]
        pipeline_v.producer_acquire(kv_producer_state)
        load_V(src_idx=n_block_last, producer_state=kv_producer_state)
        kv_producer_state.advance()

    return kv_producer_state


@cute.jit
def sparse_tensor_m_block(
    m_block,
    qhead_per_kvhead: cutlass.Constexpr[int],
    q_subtile_factor: cutlass.Constexpr[int],
):
    """Map packed m_block indices to block-sparse tensor indices."""
    block = m_block
    if const_expr(qhead_per_kvhead != 1):
        block = block // qhead_per_kvhead
    if const_expr(q_subtile_factor != 1):
        block = block // q_subtile_factor
    return block


@cute.jit
def produce_block_sparse_loads(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    kv_producer_state,
    load_K,
    load_V,
    pipeline_k,
    pipeline_v,
    intra_wg_overlap: cutlass.Constexpr,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    q_subtile_factor: cutlass.Constexpr[int] = 1,
):
    """Iterate over the mask and full block lists for a single tile.

    Q is loaded separately on its own mbarrier before this function is called.

    The masked (partial) list may leave the last V load pending when intra-warp-group
    overlap is enabled. The first full block must consume that pending V while
    issuing its own K load on the next pipeline stage.

    In the intra-wg-overlap path, the last masked block leaves its V copy in flight
    while we advance the producer state to start the next full K. Either the full list
    overlaps that pending V load, or, if no full blocks exist, we explicitly drain it.

    Args:
        qhead_per_kvhead: Pack-GQA factor. When > 1, m_block is in packed space and
            must be converted to unpacked for sparse tensor indexing.
    """

    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = blocksparse_tensors

    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    curr_mask_block_cnt = mask_block_cnt[batch_idx, head_idx, m_block_sparse]
    curr_mask_block_idx = mask_block_idx[batch_idx, head_idx, m_block_sparse, None]

    if const_expr(full_block_cnt is not None):
        curr_full_block_cnt = full_block_cnt[batch_idx, head_idx, m_block_sparse]
        curr_full_block_idx = full_block_idx[batch_idx, head_idx, m_block_sparse, None]
    else:
        curr_full_block_cnt = Int32(0)
        curr_full_block_idx = None

    mask_empty = curr_mask_block_cnt == 0
    full_empty = curr_full_block_cnt == 0

    if mask_empty:
        # No masked blocks: the full list owns the initial K load.
        kv_producer_state = load_block_list(
            curr_full_block_idx,
            curr_full_block_cnt,
            first_block_preloaded=False,
            kv_producer_state=kv_producer_state,
            load_K=load_K,
            load_V=load_V,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            intra_wg_overlap=intra_wg_overlap,
        )

        if const_expr(intra_wg_overlap) and curr_full_block_cnt > 0:
            kv_producer_state = finish_overlap_v_load(
                curr_full_block_idx,
                curr_full_block_cnt,
                load_V,
                pipeline_v,
                kv_producer_state,
            )
    else:
        # Masked blocks present. When overlap is disabled this fully drains the list.
        kv_producer_state = load_block_list(
            curr_mask_block_idx,
            curr_mask_block_cnt,
            first_block_preloaded=False,
            kv_producer_state=kv_producer_state,
            load_K=load_K,
            load_V=load_V,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            intra_wg_overlap=intra_wg_overlap,
        )

        if full_empty:
            if const_expr(intra_wg_overlap):
                kv_producer_state = finish_overlap_v_load(
                    curr_mask_block_idx,
                    curr_mask_block_cnt,
                    load_V,
                    pipeline_v,
                    kv_producer_state,
                )
        else:
            if const_expr(intra_wg_overlap):
                # Bridge the masked list to the full list by overlapping the pending masked V
                # with the first full K load.
                n_block_mask_last = curr_mask_block_idx[0]
                n_block_full_first = curr_full_block_idx[curr_full_block_cnt - 1]
                kv_producer_state_prev = kv_producer_state.clone()
                kv_producer_state.advance()
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block_full_first, producer_state=kv_producer_state)
                pipeline_v.producer_acquire(kv_producer_state_prev)
                load_V(src_idx=n_block_mask_last, producer_state=kv_producer_state_prev)

                kv_producer_state = load_block_list(
                    curr_full_block_idx,
                    curr_full_block_cnt,
                    first_block_preloaded=True,
                    kv_producer_state=kv_producer_state,
                    load_K=load_K,
                    load_V=load_V,
                    pipeline_k=pipeline_k,
                    pipeline_v=pipeline_v,
                    intra_wg_overlap=intra_wg_overlap,
                )

                kv_producer_state = finish_overlap_v_load(
                    curr_full_block_idx,
                    curr_full_block_cnt,
                    load_V,
                    pipeline_v,
                    kv_producer_state,
                )
            else:
                # Non-overlap path with both lists: run the full list normally.
                kv_producer_state = load_block_list(
                    curr_full_block_idx,
                    curr_full_block_cnt,
                    first_block_preloaded=False,
                    kv_producer_state=kv_producer_state,
                    load_K=load_K,
                    load_V=load_V,
                    pipeline_k=pipeline_k,
                    pipeline_v=pipeline_v,
                    intra_wg_overlap=intra_wg_overlap,
                )

    return kv_producer_state


@cute.jit
def consume_block_sparse_loads(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    seqlen,
    kv_consumer_state,
    mma_pv_fn,
    mma_one_n_block,
    process_first_half_block,
    process_last_half_block,
    mask_fn,
    score_mod_fn,
    O_should_accumulate,
    mask_mod,
    fastdiv_mods,
    intra_wg_overlap: cutlass.Constexpr,
    warp_scheduler_barrier_sync: Callable,
    warp_scheduler_barrier_arrive: Callable,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    q_subtile_factor: cutlass.Constexpr[int] = 1,
):
    """Consume the mask and full block lists for a single tile on the consumer side.

    Mirrors `produce_block_sparse_loads` so that the consumer pipeline uses
    the same sparse tensor indexing.

    Args:
        qhead_per_kvhead: Pack-GQA factor. When > 1, m_block is in packed space and
            must be converted to unpacked for sparse tensor indexing.
    """

    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = blocksparse_tensors

    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    curr_mask_block_cnt = mask_block_cnt[batch_idx, head_idx, m_block_sparse]
    curr_mask_block_idx = mask_block_idx[batch_idx, head_idx, m_block_sparse, None]
    curr_full_block_cnt = full_block_cnt[batch_idx, head_idx, m_block_sparse]
    curr_full_block_idx = full_block_idx[batch_idx, head_idx, m_block_sparse, None]

    processed_any = curr_mask_block_cnt + curr_full_block_cnt > 0

    if const_expr(not intra_wg_overlap):
        if curr_mask_block_cnt > 0:
            mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1]
            warp_scheduler_barrier_sync()
            kv_consumer_state = mma_one_n_block(
                kv_consumer_state,
                n_block=mask_n_block,
                mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                mask_fn=partial(
                    mask_fn,
                    mask_mod=mask_mod,
                    mask_seqlen=True,
                    fastdiv_mods=fastdiv_mods if cutlass.const_expr(mask_mod is not None) else None,
                ),
                is_first_n_block=True,
            )
            O_should_accumulate = True
            for i in cutlass.range(1, curr_mask_block_cnt):
                mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1 - i]
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=mask_n_block,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=mask_mod, mask_seqlen=False),
                    is_first_n_block=False,
                )
                O_should_accumulate = True
            if curr_full_block_cnt == 0:
                warp_scheduler_barrier_arrive()

        if curr_full_block_cnt > 0:
            full_n_block = curr_full_block_idx[curr_full_block_cnt - 1]
            if curr_mask_block_cnt == 0:
                warp_scheduler_barrier_sync()
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=full_n_block,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_seqlen=True),
                    is_first_n_block=True,
                )
                O_should_accumulate = True
                for i in cutlass.range(1, curr_full_block_cnt):
                    full_n_block = curr_full_block_idx[curr_full_block_cnt - 1 - i]
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=full_n_block,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=partial(mask_fn, mask_seqlen=False),
                        is_first_n_block=False,
                    )
                    O_should_accumulate = True
            else:
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=full_n_block,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=True),
                    is_first_n_block=False,
                )
                O_should_accumulate = True
                for i in cutlass.range(1, curr_full_block_cnt):
                    full_n_block = curr_full_block_idx[curr_full_block_cnt - 1 - i]
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=full_n_block,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=False),
                        is_first_n_block=False,
                    )
                    O_should_accumulate = True
            warp_scheduler_barrier_arrive()
    else:
        if curr_mask_block_cnt > 0:
            mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1]
            kv_consumer_state = process_first_half_block(
                n_block=mask_n_block,
                seqlen=seqlen,
                kv_consumer_state=kv_consumer_state,
                mask_fn=partial(
                    mask_fn,
                    mask_mod=mask_mod,
                    mask_seqlen=True,
                    fastdiv_mods=fastdiv_mods if cutlass.const_expr(mask_mod is not None) else None,
                ),
                score_mod_fn=score_mod_fn,
                is_first_block=True,
            )
            for i in cutlass.range(1, curr_mask_block_cnt):
                mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1 - i]
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=mask_n_block,
                    seqlen=seqlen,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=mask_mod, mask_seqlen=False),
                )
                O_should_accumulate = True

        if curr_full_block_cnt > 0:
            full_n_block = curr_full_block_idx[curr_full_block_cnt - 1]
            if curr_mask_block_cnt == 0:
                kv_consumer_state = process_first_half_block(
                    n_block=full_n_block,
                    seqlen=seqlen,
                    kv_consumer_state=kv_consumer_state,
                    mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=True),
                    score_mod_fn=score_mod_fn,
                    is_first_block=True,
                )
            else:
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=full_n_block,
                    seqlen=seqlen,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=True),
                )
                O_should_accumulate = True
            for i in cutlass.range(1, curr_full_block_cnt):
                full_n_block = curr_full_block_idx[curr_full_block_cnt - 1 - i]
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=full_n_block,
                    seqlen=seqlen,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=False),
                )
                O_should_accumulate = True

        if curr_mask_block_cnt + curr_full_block_cnt > 0:
            kv_consumer_state = process_last_half_block(
                kv_consumer_state=kv_consumer_state,
                zero_init=not O_should_accumulate,
            )
            O_should_accumulate = True

    return kv_consumer_state, O_should_accumulate, processed_any


# ============================================================================
# SM90 sparse M64N128 chunking experiment
#
# Sparse metadata remains in N64 units.  The producer gathers two list entries
# into the low/high halves of one N128 K/V stage and the consumer executes one
# N128 QK/softmax/PV tile.  Valid columns follow the same reverse order as the
# baseline helpers above.  For an odd list, the highest ordinal is transported
# first with a duplicated/masked peer; descending two-entry pairs follow.
# ============================================================================


@cute.jit
def sparse_m64n128_tail_mask(
    acc_S: cute.Tensor,
    col_limit: Int32,
    thr_mma: cute.TiledMma,
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
    n_block: Int32 = 0,
    mask_seqlen: cutlass.Constexpr[bool] = False,
):
    """Mask the duplicated high half of an odd N64 source-block pair."""
    # All regular pairs pass col_limit=N128.  Keep that steady-state path to
    # one uniform branch; only the actual odd tail touches score elements.
    if col_limit < Int32(tile_n):
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        cS = cute.make_identity_tensor((tile_m, tile_n))
        tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS))
        t0ScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cS))
        limit = col_limit - tScS_mn[0][1]
        for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
            oob = t0ScS_mn[0, c][1] >= limit
            for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                acc_S_mn[r, c] = -Float32.inf if oob else acc_S_mn[r, c]


@cute.jit
def sparse_reverse_pair(
    block_indices: cute.Tensor,
    block_count,
    pair_idx,
):
    ordinal0 = block_count - Int32(1) - pair_idx * Int32(2)
    # For an odd list, consume the unpaired highest ordinal first, then
    # continue with descending pairs.  This keeps the complete valid-column
    # stream byte-for-byte in baseline reverse-list order while placing the
    # padded transaction in the overlap prologue instead of the steady state.
    if block_count % Int32(2) == Int32(1) and pair_idx > Int32(0):
        ordinal0 = ordinal0 + Int32(1)
    ordinal1 = ordinal0 - Int32(1)
    if ordinal1 < Int32(0):
        ordinal1 = ordinal0
    return block_indices[ordinal0], block_indices[ordinal1]


@cute.jit
def load_sparse_m64n128_block_list(
    block_indices: cute.Tensor,
    block_count,
    kv_producer_state,
    load_K_pair,
    load_V_pair,
    pipeline_k,
    pipeline_v,
    intra_wg_overlap: cutlass.Constexpr,
):
    """Produce N128 stages from reverse-ordered pairs of N64 list entries."""
    if block_count > 0:
        pair_count = (block_count + Int32(1)) // Int32(2)
        block0, block1 = sparse_reverse_pair(
            block_indices, block_count, Int32(0)
        )
        pipeline_k.producer_acquire(kv_producer_state)
        load_K_pair(block0, block1, kv_producer_state)

        if const_expr(not intra_wg_overlap):
            pipeline_v.producer_acquire(kv_producer_state)
            load_V_pair(block0, block1, kv_producer_state)
            kv_producer_state.advance()
            for pair_idx in cutlass.range(1, pair_count, unroll=1):
                block0, block1 = sparse_reverse_pair(
                    block_indices, block_count, pair_idx
                )
                pipeline_k.producer_acquire(kv_producer_state)
                load_K_pair(block0, block1, kv_producer_state)
                pipeline_v.producer_acquire(kv_producer_state)
                load_V_pair(block0, block1, kv_producer_state)
                kv_producer_state.advance()
        else:
            for pair_idx in cutlass.range(1, pair_count, unroll=1):
                prev0, prev1 = sparse_reverse_pair(
                    block_indices, block_count, pair_idx - Int32(1)
                )
                block0, block1 = sparse_reverse_pair(
                    block_indices, block_count, pair_idx
                )
                kv_producer_state_prev = kv_producer_state.clone()
                kv_producer_state.advance()
                pipeline_k.producer_acquire(kv_producer_state)
                load_K_pair(block0, block1, kv_producer_state)
                pipeline_v.producer_acquire(kv_producer_state_prev)
                load_V_pair(prev0, prev1, kv_producer_state_prev)
            last0, last1 = sparse_reverse_pair(
                block_indices, block_count, pair_count - Int32(1)
            )
            pipeline_v.producer_acquire(kv_producer_state)
            load_V_pair(last0, last1, kv_producer_state)
            kv_producer_state.advance()

    return kv_producer_state


@cute.jit
def produce_block_sparse_m64n128_loads(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    kv_producer_state,
    load_K_pair,
    load_V_pair,
    pipeline_k,
    pipeline_v,
    intra_wg_overlap: cutlass.Constexpr,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    q_subtile_factor: cutlass.Constexpr[int] = 1,
):
    """Pair the existing masked list, then the existing full list."""
    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = blocksparse_tensors
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    curr_mask_block_cnt = mask_block_cnt[batch_idx, head_idx, m_block_sparse]
    curr_mask_block_idx = mask_block_idx[batch_idx, head_idx, m_block_sparse, None]
    kv_producer_state = load_sparse_m64n128_block_list(
        curr_mask_block_idx,
        curr_mask_block_cnt,
        kv_producer_state,
        load_K_pair,
        load_V_pair,
        pipeline_k,
        pipeline_v,
        intra_wg_overlap,
    )

    if const_expr(full_block_cnt is not None):
        curr_full_block_cnt = full_block_cnt[batch_idx, head_idx, m_block_sparse]
        curr_full_block_idx = full_block_idx[batch_idx, head_idx, m_block_sparse, None]
        kv_producer_state = load_sparse_m64n128_block_list(
            curr_full_block_idx,
            curr_full_block_cnt,
            kv_producer_state,
            load_K_pair,
            load_V_pair,
            pipeline_k,
            pipeline_v,
            intra_wg_overlap,
        )
    return kv_producer_state


@cute.jit
def sparse_m64n128_col_limit(
    block_count,
    pair_idx,
    tile_n: cutlass.Constexpr[int],
):
    is_odd_tail = (
        block_count % Int32(2) == Int32(1)
        and pair_idx == Int32(0)
    )
    col_limit = Int32(tile_n)
    if is_odd_tail:
        col_limit = Int32(tile_n // 2)
    return col_limit


@cute.jit
def consume_block_sparse_m64n128_loads(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    seqlen,
    kv_consumer_state,
    mma_pv_fn,
    mma_one_n_block,
    process_first_half_block,
    process_last_half_block,
    pair_tail_mask_fn,
    O_should_accumulate,
    intra_wg_overlap: cutlass.Constexpr,
    warp_scheduler_barrier_sync: Callable,
    warp_scheduler_barrier_arrive: Callable,
    tile_n: cutlass.Constexpr[int],
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    q_subtile_factor: cutlass.Constexpr[int] = 1,
):
    """Consume paired masked/full lists as genuine N128 MMA tiles."""
    mask_block_cnt, _, full_block_cnt, _ = blocksparse_tensors
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)
    curr_mask_block_cnt = mask_block_cnt[batch_idx, head_idx, m_block_sparse]
    curr_full_block_cnt = (
        full_block_cnt[batch_idx, head_idx, m_block_sparse]
        if const_expr(full_block_cnt is not None)
        else Int32(0)
    )
    mask_pair_count = (curr_mask_block_cnt + Int32(1)) // Int32(2)
    full_pair_count = (curr_full_block_cnt + Int32(1)) // Int32(2)
    processed_any = curr_mask_block_cnt + curr_full_block_cnt > Int32(0)

    if const_expr(not intra_wg_overlap):
        if curr_mask_block_cnt > 0:
            warp_scheduler_barrier_sync()
            kv_consumer_state = mma_one_n_block(
                kv_consumer_state,
                n_block=Int32(0),
                mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                mask_fn=partial(
                    pair_tail_mask_fn,
                    col_limit=sparse_m64n128_col_limit(
                        curr_mask_block_cnt, Int32(0), tile_n
                    ),
                ),
                is_first_n_block=True,
            )
            O_should_accumulate = True
            for pair_idx in cutlass.range(1, mask_pair_count, unroll=1):
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=Int32(0),
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(
                        pair_tail_mask_fn,
                        col_limit=sparse_m64n128_col_limit(
                            curr_mask_block_cnt, pair_idx, tile_n
                        ),
                    ),
                    is_first_n_block=False,
                )
                O_should_accumulate = True
        if curr_full_block_cnt > 0:
            if curr_mask_block_cnt == 0:
                warp_scheduler_barrier_sync()
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=Int32(0),
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(
                        pair_tail_mask_fn,
                        col_limit=sparse_m64n128_col_limit(
                            curr_full_block_cnt, Int32(0), tile_n
                        ),
                    ),
                    is_first_n_block=True,
                )
                O_should_accumulate = True
            else:
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=Int32(0),
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(
                        pair_tail_mask_fn,
                        col_limit=sparse_m64n128_col_limit(
                            curr_full_block_cnt, Int32(0), tile_n
                        ),
                    ),
                    is_first_n_block=False,
                )
                O_should_accumulate = True
            for pair_idx in cutlass.range(1, full_pair_count, unroll=1):
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=Int32(0),
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=None,
                    is_first_n_block=False,
                )
                O_should_accumulate = True
        if processed_any:
            warp_scheduler_barrier_arrive()
    else:
        if curr_mask_block_cnt > 0:
            kv_consumer_state = process_first_half_block(
                n_block=Int32(0),
                seqlen=seqlen,
                kv_consumer_state=kv_consumer_state,
                mask_fn=partial(
                    pair_tail_mask_fn,
                    col_limit=sparse_m64n128_col_limit(
                        curr_mask_block_cnt, Int32(0), tile_n
                    ),
                ),
                score_mod_fn=None,
                is_first_block=True,
            )
            for pair_idx in cutlass.range(1, mask_pair_count, unroll=1):
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=Int32(0),
                    seqlen=seqlen,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(
                        pair_tail_mask_fn,
                        col_limit=sparse_m64n128_col_limit(
                            curr_mask_block_cnt, pair_idx, tile_n
                        ),
                    ),
                )
                O_should_accumulate = True
        if curr_full_block_cnt > 0:
            if curr_mask_block_cnt == 0:
                kv_consumer_state = process_first_half_block(
                    n_block=Int32(0),
                    seqlen=seqlen,
                    kv_consumer_state=kv_consumer_state,
                    mask_fn=partial(
                        pair_tail_mask_fn,
                        col_limit=sparse_m64n128_col_limit(
                            curr_full_block_cnt, Int32(0), tile_n
                        ),
                    ),
                    score_mod_fn=None,
                    is_first_block=True,
                )
            else:
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=Int32(0),
                    seqlen=seqlen,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(
                        pair_tail_mask_fn,
                        col_limit=sparse_m64n128_col_limit(
                            curr_full_block_cnt, Int32(0), tile_n
                        ),
                    ),
                )
                O_should_accumulate = True
            for pair_idx in cutlass.range(1, full_pair_count, unroll=1):
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=Int32(0),
                    seqlen=seqlen,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=None,
                )
                O_should_accumulate = True
        if processed_any:
            kv_consumer_state = process_last_half_block(
                kv_consumer_state=kv_consumer_state,
                zero_init=not O_should_accumulate,
            )
            O_should_accumulate = True

    return kv_consumer_state, O_should_accumulate, processed_any


@cute.jit
def load_block_list_sm100(
    block_indices: cute.Tensor,
    block_count,
    load_q_with_first: cutlass.Constexpr,
    q_stage: cutlass.Constexpr,
    kv_producer_state,
    load_Q,
    load_K,
    load_V,
    pipeline_kv,
):
    """SM100 version of load_block_list (no intra_wg_overlap, no extra_tx_count)."""
    if block_count > 0:
        # First iteration: load Q alongside K if requested
        n_block_first = block_indices[block_count - 1]

        if const_expr(load_q_with_first):
            # SM100 loads Q0 and optionally Q1
            load_Q(block=0, stage=0)
            if const_expr(q_stage == 2):
                load_Q(block=1, stage=1)

        # SM100 doesn't use producer_acquire for pipeline_kv in load path
        # The pipeline barriers are handled inside load_KV
        load_K(block=n_block_first, producer_state=kv_producer_state, page_idx=None)
        kv_producer_state.advance()
        load_V(block=n_block_first, producer_state=kv_producer_state, page_idx=None)
        kv_producer_state.advance()

        # Remaining blocks
        for offset in cutlass.range(1, block_count):
            n_block = block_indices[block_count - 1 - offset]
            load_K(block=n_block, producer_state=kv_producer_state, page_idx=None)
            kv_producer_state.advance()
            load_V(block=n_block, producer_state=kv_producer_state, page_idx=None)
            kv_producer_state.advance()

    return kv_producer_state


# SM100-specific tile processor using SM100 helpers
@cute.jit
def produce_block_sparse_loads_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    kv_producer_state,
    load_Q,
    load_K,
    load_V,
    pipeline_kv,
    q_stage: cutlass.Constexpr,
    q_producer_phase: Int32,
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr,
):
    """SM100 entry point for sparse block iteration.

    SM100 uses PipelineTmaUmma which doesn't support extra_tx_count, so we use
    simplified block processing that just calls producer_acquire without extras.

    Args:
        m_block: which tile of m we are processing
        qhead_per_kvhead: Constexpr pack factor
    """
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = blocksparse_tensors

    curr_mask_block_cnt = mask_block_cnt[batch_idx, head_idx, m_block_sparse]
    curr_mask_block_idx = mask_block_idx[batch_idx, head_idx, m_block_sparse, None]

    if const_expr(full_block_cnt is not None):
        curr_full_block_cnt = full_block_cnt[batch_idx, head_idx, m_block_sparse]
        curr_full_block_idx = full_block_idx[batch_idx, head_idx, m_block_sparse, None]
    else:
        curr_full_block_cnt = Int32(0)
        curr_full_block_idx = None

    mask_empty = curr_mask_block_cnt == 0
    full_empty = curr_full_block_cnt == 0

    q_phase_flipped = False

    if mask_empty:
        # No masked blocks: process full list with Q loading
        kv_producer_state = load_block_list_sm100(
            curr_full_block_idx,
            curr_full_block_cnt,
            load_q_with_first=True,
            q_stage=q_stage,
            kv_producer_state=kv_producer_state,
            load_Q=load_Q,
            load_K=load_K,
            load_V=load_V,
            pipeline_kv=pipeline_kv,
        )
        q_phase_flipped = not full_empty
    else:
        # Process masked blocks with Q loading
        kv_producer_state = load_block_list_sm100(
            curr_mask_block_idx,
            curr_mask_block_cnt,
            load_q_with_first=True,
            q_stage=q_stage,
            kv_producer_state=kv_producer_state,
            load_Q=load_Q,
            load_K=load_K,
            load_V=load_V,
            pipeline_kv=pipeline_kv,
        )
        q_phase_flipped = True

        if not full_empty:
            # Process full blocks without Q loading
            kv_producer_state = load_block_list_sm100(
                curr_full_block_idx,
                curr_full_block_cnt,
                load_q_with_first=False,
                q_stage=q_stage,
                kv_producer_state=kv_producer_state,
                load_Q=load_Q,
                load_K=load_K,
                load_V=load_V,
                pipeline_kv=pipeline_kv,
            )

    if q_phase_flipped:
        q_producer_phase ^= 1

    return kv_producer_state, q_producer_phase


@cute.jit
def get_total_block_count(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr,
):
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = blocksparse_tensors
    if const_expr(full_block_cnt is not None):
        return (
            mask_block_cnt[batch_idx, head_idx, m_block_sparse]
            + full_block_cnt[batch_idx, head_idx, m_block_sparse]
        )
    else:
        return mask_block_cnt[batch_idx, head_idx, m_block_sparse]


@cute.jit
def handle_block_sparse_empty_tile_correction_sm100(
    tidx: Int32,
    q_stage: cutlass.Constexpr,
    m_block_size: cutlass.Constexpr,
    qhead_per_kvhead,
    pack_gqa: cutlass.Constexpr,
    is_split_kv: cutlass.Constexpr,
    learnable_sink,
    mLSE,
    seqlen,
    m_block: Int32,
    head_idx: Int32,
    batch_idx: Int32,
    split_idx: Int32,
    sScale: cute.Tensor,
    stats: list,
    correction_epilogue: Callable,
    thr_mma_pv: cute.core.ThrMma,
    tOtO: cute.Tensor,
    sO: cute.Tensor,
    pipeline_sm_stats: cutlass.pipeline.PipelineAsync,
    sm_stats_barrier: cutlass.pipeline.NamedBarrier,
    pipeline_o_epi: cutlass.pipeline.PipelineAsync,
    sm_stats_consumer_phase: Int32,
    o_corr_consumer_phase: Int32,
    corr_epi_producer_phase: Int32,
    softmax_scale_log2: Float32,
    max_offset: Float32,
    max_offset_scale: Float32,
    mO_cur: Optional[cute.Tensor] = None,
    gO: Optional[cute.Tensor] = None,
    gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
):
    """Handle SM100 forward block-sparse tiles with no active KV blocks.

    This path is taken when `total_block_cnt == 0`. The softmax warp-group still
    arrives `mbar_softmax_corr_full` (synthetic "no work") so the correction
    warp-group can:

    - seed fully-masked-row stats (row_sum=1; row_max=-inf when tracked) for LSE
    - run `correction_epilogue` with `scale=0` so the output tile is written as zeros
      (independent of any prior tmem contents)
    - wait on `mbar_softmax_corr_full` and arrive `mbar_softmax_corr_empty`
      (and `mbar_corr_epi_*` when applicable) so phases stay aligned across tiles

    This helper intentionally does not touch `mbar_P_full_*` since no P is produced.
    See NOTE [SM100 block-sparse empty tiles: mbarrier contract].
    """
    LOG2_E = Float32(math.log2(math.e))
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

    for stage in cutlass.range_constexpr(q_stage):
        row_sum_value = Float32(1.0)
        row_max_value = (
            -Float32.inf if const_expr(mLSE is not None or learnable_sink is not None) else None
        )
        if const_expr(learnable_sink is not None):
            sink_val = -Float32.inf
            if const_expr(not pack_gqa):
                sink_val = Float32(learnable_sink[head_idx])
            elif tidx < m_block_size:
                q_head_idx = (
                    (q_stage * m_block + stage) * m_block_size + tidx
                ) % qhead_per_kvhead + head_idx * qhead_per_kvhead
                sink_val = Float32(learnable_sink[q_head_idx])
            if sink_val != -Float32.inf and (const_expr(not is_split_kv) or split_idx == 0):
                if row_max_value == -Float32.inf:
                    row_max_value = sink_val * (LOG2_E / softmax_scale_log2)
                    row_sum_value = max_offset_scale
                else:
                    row_sum_value = row_sum_value + cute.math.exp2(
                        sink_val * LOG2_E - row_max_value * softmax_scale_log2 + max_offset,
                        fastmath=True,
                    )
        if tidx < m_block_size:
            scale_row_idx = tidx + stage * m_block_size
            sScale[scale_row_idx] = row_sum_value
            if const_expr(mLSE is not None or learnable_sink is not None):
                sScale[scale_row_idx + q_stage * m_block_size] = row_max_value
        acc_flag = row_sum_value == Float32(0.0) or row_sum_value != row_sum_value
        stats[stage] = (row_sum_value, row_max_value, acc_flag)

        # See NOTE [SM100 block-sparse empty tiles: mbarrier contract].
        # pipeline_sm_stats.consumer_wait_w_index_phase(stage, sm_stats_consumer_phase)
        sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
        pipeline_sm_stats.consumer_release_w_index(stage)

        if const_expr(gmem_tiled_copy_O is None):
            pipeline_o_epi.producer_acquire_w_index_phase(stage, corr_epi_producer_phase)

        gO_stage = gO[None, None, stage] if const_expr(gO is not None) else None
        correction_epilogue(
            thr_mma_pv,
            tOtO[None, None, None, stage],
            tidx,
            stage,
            m_block,
            seqlen.seqlen_q,
            Float32(0.0),  # zero scale ensures empty tile writes zeros into staged outputs
            sO[None, None, stage],
            mO_cur,
            gO_stage,
            gmem_tiled_copy_O,
        )
        if const_expr(gmem_tiled_copy_O is None):
            pipeline_o_epi.producer_commit_w_index(stage)

    sm_stats_consumer_phase ^= 1
    corr_epi_producer_phase ^= 1

    return (
        sm_stats_consumer_phase,
        o_corr_consumer_phase,
        corr_epi_producer_phase,
    )


@cute.jit
def softmax_block_sparse_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    softmax_step: Callable,
    mask_fn: Callable,
    mask_fn_none: Callable,
    mma_si_consumer_phase: Int32,
    si_corr_producer_phase: Int32,
    s0_s1_sequence_phase: Int32,
    pipeline_sm_stats: cutlass.pipeline.PipelineAsync,
    sm_stats_barrier: cutlass.pipeline.NamedBarrier,
    q_stage: cutlass.Constexpr,
    stage_idx: Int32,
    check_m_boundary: bool,
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr[int] = 1,
):
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = blocksparse_tensors

    curr_mask_block_cnt = mask_block_cnt[batch_idx, head_idx, m_block_sparse]
    curr_mask_block_idx = mask_block_idx[batch_idx, head_idx, m_block_sparse, None]

    if const_expr(full_block_cnt is not None):
        curr_full_block_cnt = full_block_cnt[batch_idx, head_idx, m_block_sparse]
        curr_full_block_idx = full_block_idx[batch_idx, head_idx, m_block_sparse, None]
    else:
        curr_full_block_cnt = Int32(0)
        curr_full_block_idx = None

    total_block_cnt = curr_mask_block_cnt + curr_full_block_cnt

    if total_block_cnt == 0:
        # See NOTE [SM100 block-sparse empty tiles: mbarrier contract].
        # pipeline_sm_stats.producer_commit_w_index(stage_idx)
        sm_stats_barrier.arrive_w_index(index=stage_idx * 4 + warp_idx)
    else:
        if curr_mask_block_cnt > 0:
            mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1]
            (
                mma_si_consumer_phase,
                si_corr_producer_phase,
                s0_s1_sequence_phase,
            ) = softmax_step(
                mma_si_consumer_phase,
                si_corr_producer_phase,
                s0_s1_sequence_phase,
                mask_n_block,
                is_first=True,
                mask_fn=partial(mask_fn, mask_seqlen=True, check_q_boundary=check_m_boundary),
            )
            for i in cutlass.range(1, curr_mask_block_cnt):
                mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1 - i]
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    mask_n_block,
                    mask_fn=partial(mask_fn, mask_seqlen=False, check_q_boundary=check_m_boundary),
                )

        if curr_full_block_cnt > 0:
            full_n_block = curr_full_block_idx[curr_full_block_cnt - 1]
            if curr_mask_block_cnt == 0:
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    full_n_block,
                    is_first=True,
                    mask_fn=partial(
                        mask_fn_none, mask_seqlen=True, check_q_boundary=check_m_boundary
                    ),
                )
            else:
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    full_n_block,
                    is_first=False,
                    mask_fn=partial(
                        mask_fn_none, mask_seqlen=False, check_q_boundary=check_m_boundary
                    ),
                )
            for i in cutlass.range(1, curr_full_block_cnt):
                full_n_block = curr_full_block_idx[curr_full_block_cnt - 1 - i]
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    full_n_block,
                    mask_fn=partial(
                        mask_fn_none, mask_seqlen=False, check_q_boundary=check_m_boundary
                    ),
                )

    return (
        mma_si_consumer_phase,
        si_corr_producer_phase,
        s0_s1_sequence_phase,
        total_block_cnt == 0,
    )


# =============================================================================
# Backward-specific block-sparse helpers (SM100)
# =============================================================================
#
# In backward, iteration is transposed compared to forward:
# - Forward: outer loop over m_blocks (Q tiles), inner loop over n_blocks (KV tiles)
# - Backward: outer loop over n_blocks (KV tiles), inner loop over m_blocks (Q tiles)
#
# The backward block-sparse tensors use "Q direction" indexing:
# - q_block_cnt[batch, head, n_block] → count of m_blocks to process for this KV tile
# - q_block_idx[batch, head, n_block, :] → indices of m_blocks to process
#


@cute.jit
def get_total_q_block_count_bwd(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
):
    """Count total tile iterations for given n_block (KV tile) in backward."""
    q_block_cnt, _, full_block_cnt, _ = blocksparse_tensors
    total = q_block_cnt[batch_idx, head_idx, n_block]
    if const_expr(full_block_cnt is not None):
        total = total + full_block_cnt[batch_idx, head_idx, n_block]
    return total * subtile_factor


@cute.jit
def produce_block_sparse_q_loads_bwd_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    # Pipeline states (will be returned after advancing)
    producer_state_Q_LSE,
    producer_state_dO_dPsum,
    # Pipelines
    pipeline_Q,
    pipeline_LSE,
    pipeline_dO,
    pipeline_dPsum,
    # Load functions
    load_K,
    load_V,
    load_Q,
    load_dO,
    copy_stats,
    # Global tensors for LSE/dPsum
    gLSE,
    sLSE,
    gdPsum,
    sdPsum,
    # TMA copy bytes for extra_tx_count
    tma_copy_bytes_K,
    tma_copy_bytes_V,
    # Flags for which loads to perform
    should_load_Q: cutlass.Constexpr,
    should_load_dO: cutlass.Constexpr,
    # Subtiling factor and bounds
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
):
    """SM100 backward block sparse loading with subtiling.

    Returns updated (producer_state_Q_LSE, producer_state_dO_dPsum).
    First iteration loads K/V alongside Q/dO; subsequent iterations load only Q/dO.
    """
    (
        curr_q_cnt,
        curr_q_idx,
        curr_full_cnt,
        curr_full_idx,
        loop_count,
    ) = get_block_sparse_iteration_info_bwd(
        blocksparse_tensors, batch_idx, head_idx, n_block, subtile_factor, m_block_max
    )

    for iter_idx in cutlass.range(loop_count, unroll=1):
        m_block, _ = get_m_block_from_iter_bwd(
            iter_idx,
            curr_q_cnt,
            curr_q_idx,
            curr_full_cnt,
            curr_full_idx,
            subtile_factor,
            m_block_max,
        )
        m_block_safe = m_block
        if m_block_max > 0:
            m_block_safe = cutlass.min(m_block, m_block_max - 1)

        if iter_idx == 0:
            # First block: load K/V alongside Q/dO
            if const_expr(should_load_Q):
                pipeline_Q.producer_acquire(producer_state_Q_LSE, extra_tx_count=tma_copy_bytes_K)
                load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_LSE))
                load_Q(m_block_safe, producer_state=producer_state_Q_LSE)
                pipeline_Q.producer_commit(producer_state_Q_LSE)
                pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                with cute.arch.elect_one():
                    copy_stats(
                        gLSE[None, m_block_safe],
                        sLSE[None, producer_state_Q_LSE.index],
                        mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                    )
                producer_state_Q_LSE.advance()
            if const_expr(should_load_dO):
                pipeline_dO.producer_acquire(
                    producer_state_dO_dPsum, extra_tx_count=tma_copy_bytes_V
                )
                load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_dPsum))
                load_dO(m_block_safe, producer_state=producer_state_dO_dPsum)
                pipeline_dO.producer_commit(producer_state_dO_dPsum)
                pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                with cute.arch.elect_one():
                    copy_stats(
                        gdPsum[None, m_block_safe],
                        sdPsum[None, producer_state_dO_dPsum.index],
                        mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                    )
                producer_state_dO_dPsum.advance()
        else:
            # Subsequent blocks: just load Q/dO (K/V already loaded)
            if const_expr(should_load_Q):
                pipeline_Q.producer_acquire(producer_state_Q_LSE)
                load_Q(m_block_safe, producer_state=producer_state_Q_LSE)
                pipeline_Q.producer_commit(producer_state_Q_LSE)
                pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                with cute.arch.elect_one():
                    copy_stats(
                        gLSE[None, m_block_safe],
                        sLSE[None, producer_state_Q_LSE.index],
                        mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                    )
                producer_state_Q_LSE.advance()
            if const_expr(should_load_dO):
                pipeline_dO.producer_acquire(producer_state_dO_dPsum)
                load_dO(m_block_safe, producer_state=producer_state_dO_dPsum)
                pipeline_dO.producer_commit(producer_state_dO_dPsum)
                pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                with cute.arch.elect_one():
                    copy_stats(
                        gdPsum[None, m_block_safe],
                        sdPsum[None, producer_state_dO_dPsum.index],
                        mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                    )
                producer_state_dO_dPsum.advance()

    return producer_state_Q_LSE, producer_state_dO_dPsum


@cute.jit
def get_block_sparse_iteration_info_bwd(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
):
    """Extract block-sparse iteration info for backward pass.

    Returns (curr_q_cnt, curr_q_idx, curr_full_cnt, curr_full_idx, total_count).
    """
    q_cnt, q_idx, full_cnt, full_idx = blocksparse_tensors
    curr_q_cnt = q_cnt[batch_idx, head_idx, n_block]
    curr_q_idx = q_idx[batch_idx, head_idx, n_block, None]

    if const_expr(full_cnt is not None):
        curr_full_cnt = full_cnt[batch_idx, head_idx, n_block]
        curr_full_idx = full_idx[batch_idx, head_idx, n_block, None]
    else:
        curr_full_cnt = Int32(0)
        curr_full_idx = None

    sparse_block_count = curr_q_cnt
    if const_expr(full_cnt is not None):
        sparse_block_count = sparse_block_count + curr_full_cnt
    total_count = sparse_block_count * subtile_factor

    return curr_q_cnt, curr_q_idx, curr_full_cnt, curr_full_idx, total_count


@cute.jit
def get_m_block_from_iter_bwd(
    iter_idx,
    curr_q_cnt,
    curr_q_idx: cute.Tensor,
    curr_full_cnt,
    curr_full_idx: Optional[cute.Tensor],
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
):
    """Derive m_block index and is_full_block flag from iteration index.

    Returns (m_block, is_full_block):
        - m_block: The actual Q-tile block index
        - is_full_block: True if this is a full block (no mask_mod needed)
    """
    sparse_iter_idx = iter_idx // subtile_factor
    subtile_offset = iter_idx % subtile_factor

    sparse_m_block = Int32(0)
    is_full_block = False
    if const_expr(curr_full_idx is not None):
        if sparse_iter_idx < curr_q_cnt:
            sparse_m_block = curr_q_idx[sparse_iter_idx]
        else:
            sparse_m_block = curr_full_idx[sparse_iter_idx - curr_q_cnt]
            is_full_block = True
    else:
        sparse_m_block = curr_q_idx[sparse_iter_idx]

    return sparse_m_block * subtile_factor + subtile_offset, is_full_block


@cute.jit
def _load_q_do_block_sm90(
    m_block,
    producer_state_Q,
    producer_state_dO,
    pipeline_Q,
    pipeline_dO,
    load_K,
    load_V,
    load_Q,
    load_dO,
    load_LSE,
    load_dPsum,
    tma_copy_bytes_K,
    tma_copy_bytes_V,
    Q_stage_eq_dO_stage: cutlass.Constexpr,
    load_kv: bool,
):
    """Load one Q/dO block, optionally loading K/V on first iteration."""
    if load_kv:
        pipeline_Q.producer_acquire(producer_state_Q, extra_tx_count=tma_copy_bytes_K)
        load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q))
    else:
        pipeline_Q.producer_acquire(producer_state_Q)
    load_Q(m_block, producer_state=producer_state_Q)
    load_LSE(m_block, producer_state=producer_state_Q)

    producer_state_dO_cur = (
        producer_state_dO if const_expr(not Q_stage_eq_dO_stage) else producer_state_Q
    )
    if load_kv:
        pipeline_dO.producer_acquire(producer_state_dO_cur, extra_tx_count=tma_copy_bytes_V)
        load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_cur))
    else:
        pipeline_dO.producer_acquire(producer_state_dO_cur)
    load_dO(m_block, producer_state=producer_state_dO_cur)
    load_dPsum(m_block, producer_state=producer_state_dO_cur)

    producer_state_Q.advance()
    producer_state_dO.advance()
    return producer_state_Q, producer_state_dO


@cute.jit
def produce_block_sparse_q_loads_bwd_sm90(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    producer_state_Q,
    producer_state_dO,
    pipeline_Q,
    pipeline_dO,
    load_K,
    load_V,
    load_Q,
    load_dO,
    load_LSE,
    load_dPsum,
    tma_copy_bytes_K,
    tma_copy_bytes_V,
    Q_stage_eq_dO_stage: cutlass.Constexpr,
    subtile_factor: cutlass.Constexpr,
    m_block_max: int,
):
    """SM90 backward block sparse loading with separate partial/full loops.

    K/V are loaded with the first valid block. Iterates partial blocks first,
    then full blocks, matching consumer order.

    Returns updated (producer_state_Q, producer_state_dO).
    """
    q_cnt, q_idx, full_cnt, full_idx = blocksparse_tensors
    curr_q_cnt = q_cnt[batch_idx, head_idx, n_block]
    curr_q_idx = q_idx[batch_idx, head_idx, n_block, None]

    if const_expr(full_cnt is not None):
        curr_full_cnt = full_cnt[batch_idx, head_idx, n_block]
        curr_full_idx = full_idx[batch_idx, head_idx, n_block, None]
    else:
        curr_full_cnt = Int32(0)
        curr_full_idx = None

    kv_loaded = False

    for iter_idx in cutlass.range(curr_q_cnt * subtile_factor, unroll=1):
        sparse_idx = iter_idx // subtile_factor
        subtile_offset = iter_idx % subtile_factor
        m_block = curr_q_idx[sparse_idx] * subtile_factor + subtile_offset

        if m_block < m_block_max:
            producer_state_Q, producer_state_dO = _load_q_do_block_sm90(
                m_block,
                producer_state_Q,
                producer_state_dO,
                pipeline_Q,
                pipeline_dO,
                load_K,
                load_V,
                load_Q,
                load_dO,
                load_LSE,
                load_dPsum,
                tma_copy_bytes_K,
                tma_copy_bytes_V,
                Q_stage_eq_dO_stage,
                load_kv=not kv_loaded,
            )
            kv_loaded = True

    if const_expr(full_cnt is not None):
        for iter_idx in cutlass.range(curr_full_cnt * subtile_factor, unroll=1):
            sparse_idx = iter_idx // subtile_factor
            subtile_offset = iter_idx % subtile_factor
            m_block = curr_full_idx[sparse_idx] * subtile_factor + subtile_offset

            if m_block < m_block_max:
                producer_state_Q, producer_state_dO = _load_q_do_block_sm90(
                    m_block,
                    producer_state_Q,
                    producer_state_dO,
                    pipeline_Q,
                    pipeline_dO,
                    load_K,
                    load_V,
                    load_Q,
                    load_dO,
                    load_LSE,
                    load_dPsum,
                    tma_copy_bytes_K,
                    tma_copy_bytes_V,
                    Q_stage_eq_dO_stage,
                    load_kv=not kv_loaded,
                )
                kv_loaded = True

    return producer_state_Q, producer_state_dO


@cute.jit
def consume_block_sparse_mma_bwd_sm90(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    consumer_state_Q,
    consumer_state_dO,
    mma_one_m_block_fn,
    mask,
    mask_mod,
    is_causal: cutlass.Constexpr,
    is_local: cutlass.Constexpr,
    thr_mma_SdP,
    score_mod_fn=None,
    score_mod_bwd_fn=None,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
    aux_tensors=None,
    fastdiv_mods=(None, None),
):
    """SM90 backward block sparse MMA consumption with separate partial/full loops.

    Partial blocks are processed first (with mask_mod applied), then full blocks
    (without mask_mod). This ensures mask_mod is only applied where needed.

    Returns updated (consumer_state_Q, consumer_state_dO).
    """
    q_cnt, q_idx, full_cnt, full_idx = blocksparse_tensors
    curr_q_cnt = q_cnt[batch_idx, head_idx, n_block]
    curr_q_idx = q_idx[batch_idx, head_idx, n_block, None]

    if const_expr(full_cnt is not None):
        curr_full_cnt = full_cnt[batch_idx, head_idx, n_block]
        curr_full_idx = full_idx[batch_idx, head_idx, n_block, None]
    else:
        curr_full_cnt = Int32(0)
        curr_full_idx = None

    dKV_accumulate = False

    mask_fn_partial = partial(
        mask.apply_mask,
        batch_idx=batch_idx,
        head_idx=head_idx,
        n_block=n_block,
        thr_mma=thr_mma_SdP,
        mask_seqlen=True,
        mask_causal=is_causal,
        mask_local=is_local,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        fastdiv_mods=fastdiv_mods,
    )

    mask_fn_full = partial(
        mask.apply_mask,
        batch_idx=batch_idx,
        head_idx=head_idx,
        n_block=n_block,
        thr_mma=thr_mma_SdP,
        mask_seqlen=True,
        mask_causal=is_causal,
        mask_local=is_local,
        aux_tensors=aux_tensors,
        fastdiv_mods=fastdiv_mods,
    )

    for iter_idx in cutlass.range(curr_q_cnt * subtile_factor, unroll=1):
        sparse_idx = iter_idx // subtile_factor
        subtile_offset = iter_idx % subtile_factor
        m_block = curr_q_idx[sparse_idx] * subtile_factor + subtile_offset

        if m_block < m_block_max:
            consumer_state_Q, consumer_state_dO = mma_one_m_block_fn(
                m_block,
                consumer_state_Q,
                consumer_state_dO,
                mask_fn=mask_fn_partial,
                score_mod_fn=score_mod_fn,
                score_mod_bwd_fn=score_mod_bwd_fn,
                dKV_accumulate=dKV_accumulate,
            )
            dKV_accumulate = True

    if const_expr(full_cnt is not None):
        for iter_idx in cutlass.range(curr_full_cnt * subtile_factor, unroll=1):
            sparse_idx = iter_idx // subtile_factor
            subtile_offset = iter_idx % subtile_factor
            m_block = curr_full_idx[sparse_idx] * subtile_factor + subtile_offset

            if m_block < m_block_max:
                consumer_state_Q, consumer_state_dO = mma_one_m_block_fn(
                    m_block,
                    consumer_state_Q,
                    consumer_state_dO,
                    mask_fn=mask_fn_full,
                    score_mod_fn=score_mod_fn,
                    score_mod_bwd_fn=score_mod_bwd_fn,
                    dKV_accumulate=dKV_accumulate,
                )
                dKV_accumulate = True

    return consumer_state_Q, consumer_state_dO


# =============================================================================
# SM90 backward N64-metadata / N128-MMA chunking
#
# Backward sparse metadata is transposed: each N64 KV block has a list of Q
# blocks.  Pairing two adjacent N64 KV blocks therefore requires iterating the
# union of their Q lists.  A Q block present in only one list is still executed
# as one N128 MMA tile, with the other N64 score half masked to -inf.  This is
# the backward analogue of the forward sparse M64N128 chunked path, and avoids
# reloading Q/dO when both source blocks select the same Q block.
# =============================================================================


@cute.jit
def get_total_q_block_count_bwd_m64n128(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
):
    """Return a nonzero upper bound for the two N64 two-list union."""
    q_cnt, _, full_cnt, _ = blocksparse_tensors
    source_n0 = n_block * Int32(2)
    source_n1 = source_n0 + Int32(1)
    return (
        q_cnt[batch_idx, head_idx, source_n0]
        + q_cnt[batch_idx, head_idx, source_n1]
        + full_cnt[batch_idx, head_idx, source_n0]
        + full_cnt[batch_idx, head_idx, source_n1]
    )


@cute.jit
def produce_block_sparse_q_loads_bwd_m64n128_sm90(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    producer_state_Q,
    producer_state_dO,
    pipeline_Q,
    pipeline_dO,
    load_K,
    load_V,
    load_Q,
    load_dO,
    load_LSE,
    load_dPsum,
    tma_copy_bytes_K,
    tma_copy_bytes_V,
    Q_stage_eq_dO_stage: cutlass.Constexpr,
    m_block_max: int,
):
    """Load Q/dO once for a sorted four-way union of N64 sparse lists."""
    q_cnt, q_idx, full_cnt, full_idx = blocksparse_tensors
    source_n0 = n_block * Int32(2)
    source_n1 = source_n0 + Int32(1)
    kv_loaded = False
    cq0 = q_cnt[batch_idx, head_idx, source_n0]
    cq1 = q_cnt[batch_idx, head_idx, source_n1]
    cf0 = full_cnt[batch_idx, head_idx, source_n0]
    cf1 = full_cnt[batch_idx, head_idx, source_n1]
    iq0 = q_idx[batch_idx, head_idx, source_n0, None]
    iq1 = q_idx[batch_idx, head_idx, source_n1, None]
    if0 = full_idx[batch_idx, head_idx, source_n0, None]
    if1 = full_idx[batch_idx, head_idx, source_n1, None]
    pq0, pq1, pf0, pf1 = Int32(0), Int32(0), Int32(0), Int32(0)
    sentinel = Int32(0x7FFFFFFF)
    while pq0 < cq0 or pq1 < cq1 or pf0 < cf0 or pf1 < cf1:
        mq0, mq1, mf0, mf1 = sentinel, sentinel, sentinel, sentinel
        if pq0 < cq0:
            mq0 = iq0[pq0]
        if pq1 < cq1:
            mq1 = iq1[pq1]
        if pf0 < cf0:
            mf0 = if0[pf0]
        if pf1 < cf1:
            mf1 = if1[pf1]
        m_block = min(mq0, mq1, mf0, mf1)
        if m_block < m_block_max:
            producer_state_Q, producer_state_dO = _load_q_do_block_sm90(
                m_block, producer_state_Q, producer_state_dO, pipeline_Q, pipeline_dO,
                load_K, load_V, load_Q, load_dO, load_LSE, load_dPsum,
                tma_copy_bytes_K, tma_copy_bytes_V, Q_stage_eq_dO_stage,
                load_kv=not kv_loaded,
            )
            kv_loaded = True
        if mq0 == m_block:
            pq0 = pq0 + Int32(1)
        if mq1 == m_block:
            pq1 = pq1 + Int32(1)
        if mf0 == m_block:
            pf0 = pf0 + Int32(1)
        if mf1 == m_block:
            pf1 = pf1 + Int32(1)
    return producer_state_Q, producer_state_dO


@cute.jit
def sparse_bwd_m64n128_half_mask(
    acc_S: cute.Tensor,
    m_block,
    thr_mma,
    swap_AB: cutlass.Constexpr,
    low_active,
    high_active,
):
    """Mask either N64 half of an N128 backward score tile."""
    acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=swap_AB)
    acc_shape = (64, 128)
    cS = cute.make_identity_tensor(acc_shape if not swap_AB else acc_shape[::-1])
    tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS), transpose=swap_AB)
    t0ScS_mn = layout_utils.reshape_acc_to_mn(
        thr_mma.get_slice(0).partition_C(cS), transpose=swap_AB
    )
    COL = 1 if const_expr(not swap_AB) else 0
    thr_col_offset = tScS_mn[0][COL]
    for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
        col = thr_col_offset + t0ScS_mn[0, c][COL]
        keep = low_active if col < Int32(64) else high_active
        for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
            acc_S_mn[r, c] = acc_S_mn[r, c] if keep else -Float32.inf


@cute.jit
def consume_block_sparse_mma_bwd_m64n128_sm90(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    consumer_state_Q,
    consumer_state_dO,
    mma_one_m_block_fn,
    thr_mma_SdP,
    swap_AB: cutlass.Constexpr,
    m_block_max: int,
):
    """Consume a sorted four-way union as native N128 backward MMA tiles."""
    q_cnt, q_idx, full_cnt, full_idx = blocksparse_tensors
    source_n0 = n_block * Int32(2)
    source_n1 = source_n0 + Int32(1)
    dKV_accumulate = False
    cq0 = q_cnt[batch_idx, head_idx, source_n0]
    cq1 = q_cnt[batch_idx, head_idx, source_n1]
    cf0 = full_cnt[batch_idx, head_idx, source_n0]
    cf1 = full_cnt[batch_idx, head_idx, source_n1]
    iq0 = q_idx[batch_idx, head_idx, source_n0, None]
    iq1 = q_idx[batch_idx, head_idx, source_n1, None]
    if0 = full_idx[batch_idx, head_idx, source_n0, None]
    if1 = full_idx[batch_idx, head_idx, source_n1, None]
    pq0, pq1, pf0, pf1 = Int32(0), Int32(0), Int32(0), Int32(0)
    sentinel = Int32(0x7FFFFFFF)
    while pq0 < cq0 or pq1 < cq1 or pf0 < cf0 or pf1 < cf1:
        mq0, mq1, mf0, mf1 = sentinel, sentinel, sentinel, sentinel
        if pq0 < cq0:
            mq0 = iq0[pq0]
        if pq1 < cq1:
            mq1 = iq1[pq1]
        if pf0 < cf0:
            mf0 = if0[pf0]
        if pf1 < cf1:
            mf1 = if1[pf1]
        m_block = min(mq0, mq1, mf0, mf1)
        low_active = mq0 == m_block or mf0 == m_block
        high_active = mq1 == m_block or mf1 == m_block
        if m_block < m_block_max:
            consumer_state_Q, consumer_state_dO = mma_one_m_block_fn(
                m_block, consumer_state_Q, consumer_state_dO,
                mask_fn=partial(
                    sparse_bwd_m64n128_half_mask, thr_mma=thr_mma_SdP,
                    swap_AB=swap_AB, low_active=low_active, high_active=high_active,
                ),
                score_mod_fn=None, score_mod_bwd_fn=None,
                dKV_accumulate=dKV_accumulate,
            )
            dKV_accumulate = True
        if mq0 == m_block:
            pq0 = pq0 + Int32(1)
        if mq1 == m_block:
            pq1 = pq1 + Int32(1)
        if mf0 == m_block:
            pf0 = pf0 + Int32(1)
        if mf1 == m_block:
            pf1 = pf1 + Int32(1)
    return consumer_state_Q, consumer_state_dO


@cute.jit
def _store_one_dQaccum_sm90(
    m_block,
    sdQaccum: cute.Tensor,
    gdQaccum: cute.Tensor,
    num_dQ_warp_groups: cutlass.Constexpr,
    num_threads_per_warp_group: cutlass.Constexpr,
    tma_copy_bytes_dQ,
):
    """Store dQaccum for a single m_block."""
    for warp_group_idx in cutlass.range_constexpr(num_dQ_warp_groups):
        cute.arch.cp_async_bulk_wait_group(num_dQ_warp_groups - 1 - warp_group_idx, read=True)
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
            number_of_threads=num_threads_per_warp_group + cute.arch.WARP_SIZE,
        )
    for warp_group_idx in cutlass.range_constexpr(num_dQ_warp_groups):
        cute.arch.barrier(
            barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
            number_of_threads=num_threads_per_warp_group + cute.arch.WARP_SIZE,
        )
        with cute.arch.elect_one():
            copy_utils.cpasync_reduce_bulk_add_f32(
                sdQaccum[None, warp_group_idx].iterator,
                gdQaccum[(None, warp_group_idx), m_block].iterator,
                tma_copy_bytes_dQ,
            )
        cute.arch.cp_async_bulk_commit_group()


@cute.jit
def dQaccum_store_block_sparse_bwd_sm90(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    sdQaccum: cute.Tensor,
    gdQaccum: cute.Tensor,
    subtile_factor: cutlass.Constexpr,
    m_block_max: int,
    num_dQ_warp_groups: cutlass.Constexpr,
    num_threads_per_warp_group: cutlass.Constexpr,
    tma_copy_bytes_dQ,
):
    """SM90 backward block sparse dQaccum store with separate partial/full loops.

    Iterates partial blocks first, then full blocks, matching producer/consumer order.
    """
    q_cnt, q_idx, full_cnt, full_idx = blocksparse_tensors
    curr_q_cnt = q_cnt[batch_idx, head_idx, n_block]
    curr_q_idx = q_idx[batch_idx, head_idx, n_block, None]

    if const_expr(full_cnt is not None):
        curr_full_cnt = full_cnt[batch_idx, head_idx, n_block]
        curr_full_idx = full_idx[batch_idx, head_idx, n_block, None]
    else:
        curr_full_cnt = Int32(0)
        curr_full_idx = None

    for iter_idx in cutlass.range(curr_q_cnt * subtile_factor, unroll=1):
        sparse_idx = iter_idx // subtile_factor
        subtile_offset = iter_idx % subtile_factor
        m_block = curr_q_idx[sparse_idx] * subtile_factor + subtile_offset

        if m_block < m_block_max:
            _store_one_dQaccum_sm90(
                m_block,
                sdQaccum,
                gdQaccum,
                num_dQ_warp_groups,
                num_threads_per_warp_group,
                tma_copy_bytes_dQ,
            )

    if const_expr(full_cnt is not None):
        for iter_idx in cutlass.range(curr_full_cnt * subtile_factor, unroll=1):
            sparse_idx = iter_idx // subtile_factor
            subtile_offset = iter_idx % subtile_factor
            m_block = curr_full_idx[sparse_idx] * subtile_factor + subtile_offset

            if m_block < m_block_max:
                _store_one_dQaccum_sm90(
                    m_block,
                    sdQaccum,
                    gdQaccum,
                    num_dQ_warp_groups,
                    num_threads_per_warp_group,
                    tma_copy_bytes_dQ,
                )


@cute.jit
def dQaccum_store_block_sparse_bwd_m64n128_sm90(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    sdQaccum: cute.Tensor,
    gdQaccum: cute.Tensor,
    m_block_max: int,
    num_dQ_warp_groups: cutlass.Constexpr,
    num_threads_per_warp_group: cutlass.Constexpr,
    tma_copy_bytes_dQ,
):
    """Store dQ contributions in the same sorted four-way union order."""
    q_cnt, q_idx, full_cnt, full_idx = blocksparse_tensors
    source_n0 = n_block * Int32(2)
    source_n1 = source_n0 + Int32(1)
    cq0 = q_cnt[batch_idx, head_idx, source_n0]
    cq1 = q_cnt[batch_idx, head_idx, source_n1]
    cf0 = full_cnt[batch_idx, head_idx, source_n0]
    cf1 = full_cnt[batch_idx, head_idx, source_n1]
    iq0 = q_idx[batch_idx, head_idx, source_n0, None]
    iq1 = q_idx[batch_idx, head_idx, source_n1, None]
    if0 = full_idx[batch_idx, head_idx, source_n0, None]
    if1 = full_idx[batch_idx, head_idx, source_n1, None]
    pq0, pq1, pf0, pf1 = Int32(0), Int32(0), Int32(0), Int32(0)
    sentinel = Int32(0x7FFFFFFF)
    while pq0 < cq0 or pq1 < cq1 or pf0 < cf0 or pf1 < cf1:
        mq0, mq1, mf0, mf1 = sentinel, sentinel, sentinel, sentinel
        if pq0 < cq0:
            mq0 = iq0[pq0]
        if pq1 < cq1:
            mq1 = iq1[pq1]
        if pf0 < cf0:
            mf0 = if0[pf0]
        if pf1 < cf1:
            mf1 = if1[pf1]
        m_block = min(mq0, mq1, mf0, mf1)
        if m_block < m_block_max:
            _store_one_dQaccum_sm90(
                m_block, sdQaccum, gdQaccum, num_dQ_warp_groups,
                num_threads_per_warp_group, tma_copy_bytes_dQ,
            )
        if mq0 == m_block:
            pq0 = pq0 + Int32(1)
        if mq1 == m_block:
            pq1 = pq1 + Int32(1)
        if mf0 == m_block:
            pf0 = pf0 + Int32(1)
        if mf1 == m_block:
            pf1 = pf1 + Int32(1)
