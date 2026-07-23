"""Bit-exact SM90 dual-query-tile forward checks.

Run under an outer timeout so a device-side barrier regression is reported as
a hang rather than mistaken for slow compilation:

    timeout -k 5 300 python test_dual_tile.py
"""

import os

import torch

from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from flash_attn.cute.interface import _flash_attn_fwd


BLOCK = 64
DEVICE = "cuda"


def make_case(
    *,
    batch: int,
    q_heads: int,
    kv_heads: int,
    blocks: int,
    topk: int,
    causal: bool,
    broadcast_sparse_heads: bool = False,
    seed: int = 0,
):
    torch.manual_seed(seed)
    seqlen = blocks * BLOCK
    q = torch.randn(batch, seqlen, q_heads, 128, dtype=torch.bfloat16, device=DEVICE)
    k = torch.randn(batch, seqlen, kv_heads, 128, dtype=torch.bfloat16, device=DEVICE)
    v = torch.randn(batch, seqlen, kv_heads, 128, dtype=torch.bfloat16, device=DEVICE)
    sparse_heads = 1 if broadcast_sparse_heads else q_heads
    selected = torch.empty(
        batch, sparse_heads, blocks, topk, dtype=torch.int32, device=DEVICE
    )
    for b in range(batch):
        for h in range(sparse_heads):
            for m in range(blocks):
                if causal:
                    # Always include block zero so every causal row has at
                    # least one finite score. Remaining entries are distinct.
                    rest = torch.randperm(max(blocks - 1, 1), device=DEVICE)[: topk - 1] + 1
                    idx = torch.cat(
                        (torch.zeros(1, dtype=torch.int64, device=DEVICE), rest)
                    )
                else:
                    idx = torch.randperm(blocks, device=DEVICE)[:topk]
                selected[b, h, m] = torch.sort(idx).values.to(torch.int32)
    count = torch.full(
        (batch, sparse_heads, blocks), topk, dtype=torch.int32, device=DEVICE
    )
    mask_count = torch.zeros_like(count)
    mask_idx = torch.zeros(
        batch, sparse_heads, blocks, topk, dtype=torch.int32, device=DEVICE
    )
    sparse = BlockSparseTensorsTorch(
        mask_block_cnt=mask_count,
        mask_block_idx=mask_idx,
        full_block_cnt=count,
        full_block_idx=selected,
        block_size=(BLOCK, BLOCK),
    )
    return q, k, v, sparse


def set_baseline_env():
    os.environ.update(
        FLASH_ATTN_SM90_DUAL_TILE="0",
        FLASH_ATTN_SM90_NUM_STAGES="2",
        FLASH_ATTN_SM90_MIN_BLOCKS="2",
    )


def set_dual_env(*, scheduler_barrier: bool = True):
    os.environ.update(
        FLASH_ATTN_SM90_DUAL_TILE="1",
        FLASH_ATTN_SM90_DUAL_SCHEDULER_BARRIER="1" if scheduler_barrier else "0",
        FLASH_ATTN_SM90_NUM_STAGES="3",
        FLASH_ATTN_SM90_MIN_BLOCKS="1",
        FLASH_ATTN_SM90_MMA_REGS="216",
        FLASH_ATTN_SM90_PRODUCER_REGS="64",
        FLASH_ATTN_SM90_TILE_PAIRING="adjacent_sparse",
    )


def run_bit_exact(name: str, **case_kwargs):
    q, k, v, sparse = make_case(**case_kwargs)
    set_baseline_env()
    ref_o, ref_lse = _flash_attn_fwd(
        q,
        k,
        v,
        causal=case_kwargs["causal"],
        block_sparse_tensors=sparse,
        tile_mn=(BLOCK, BLOCK),
        return_lse=True,
    )
    torch.cuda.synchronize()

    set_dual_env()
    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        causal=case_kwargs["causal"],
        block_sparse_tensors=sparse,
        tile_mn=(BLOCK, BLOCK),
        return_lse=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(out, ref_o), (
        f"{name}: output differs; max abs error "
        f"{(out.float() - ref_o.float()).abs().max().item()}"
    )
    assert torch.equal(lse, ref_lse), (
        f"{name}: LSE differs; max abs error "
        f"{(lse - ref_lse).abs().max().item()}"
    )
    print(f"PASS {name}")


def check_nonuniform_guard():
    q, k, v, sparse = make_case(
        batch=1,
        q_heads=2,
        kv_heads=2,
        blocks=3,
        topk=2,
        causal=False,
        seed=99,
    )
    sparse.full_block_cnt[0, 0, 0] = 1
    set_dual_env()
    try:
        _flash_attn_fwd(
            q, k, v, block_sparse_tensors=sparse, tile_mn=(BLOCK, BLOCK)
        )
    except ValueError as exc:
        assert "uniform sparse block counts" in str(exc)
    else:
        raise AssertionError("nonuniform full_block_cnt did not trip the dual-tile guard")
    print("PASS nonuniform_full_block_cnt_guard")


def main():
    cases = [
        (
            "gqa_even_topk1",
            dict(
                batch=1, q_heads=8, kv_heads=1, blocks=4, topk=1,
                causal=False, seed=1,
            ),
        ),
        (
            "packed_gqa_odd",
            dict(
                batch=1, q_heads=8, kv_heads=1, blocks=3, topk=2,
                causal=False, broadcast_sparse_heads=True, seed=2,
            ),
        ),
        (
            "mha_even_multihead",
            dict(
                batch=1, q_heads=3, kv_heads=3, blocks=4, topk=3,
                causal=False, seed=3,
            ),
        ),
        (
            "mha_odd_multibatch_multihead",
            dict(
                batch=2, q_heads=2, kv_heads=2, blocks=3, topk=1,
                causal=False, seed=4,
            ),
        ),
        (
            "causal_odd",
            dict(
                batch=1, q_heads=2, kv_heads=2, blocks=3, topk=2,
                causal=True, seed=5,
            ),
        ),
    ]
    for name, kwargs in cases:
        run_bit_exact(name, **kwargs)
    check_nonuniform_guard()


if __name__ == "__main__":
    main()
