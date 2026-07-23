"""Correctness tests for two-64-block SM90 KV pairing."""

import os

import torch

from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from flash_attn.cute.interface import _flash_attn_fwd


BLOCK = 64


def make_sparse(selected, num_kv_blocks):
    batch, heads, num_q_blocks, topk = selected.shape
    full_idx = torch.zeros(
        (batch, heads, num_q_blocks, num_kv_blocks),
        dtype=torch.int32,
        device=selected.device,
    )
    full_idx[..., :topk] = selected
    full_cnt = torch.full(
        (batch, heads, num_q_blocks),
        topk,
        dtype=torch.int32,
        device=selected.device,
    )
    return BlockSparseTensorsTorch(
        mask_block_cnt=torch.zeros_like(full_cnt),
        mask_block_idx=torch.zeros_like(full_idx),
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        block_size=(BLOCK, BLOCK),
    )


def reference(q, k, v, selected):
    batch, _, heads, head_dim = q.shape
    kv_heads = k.shape[2]
    num_q_blocks = q.shape[1] // BLOCK
    group = heads // kv_heads
    out = torch.zeros_like(q, dtype=torch.float32)
    for batch_idx in range(batch):
        for head_idx in range(heads):
            kv_head = head_idx // group
            for q_block in range(num_q_blocks):
                q_tile = q[
                    batch_idx,
                    q_block * BLOCK : (q_block + 1) * BLOCK,
                    head_idx,
                ].float()
                indices = selected[batch_idx, head_idx, q_block].tolist()
                k_tiles = torch.cat(
                    [
                        k[
                            batch_idx,
                            idx * BLOCK : (idx + 1) * BLOCK,
                            kv_head,
                        ]
                        for idx in indices
                    ]
                ).float()
                v_tiles = torch.cat(
                    [
                        v[
                            batch_idx,
                            idx * BLOCK : (idx + 1) * BLOCK,
                            kv_head,
                        ]
                        for idx in indices
                    ]
                ).float()
                scores = q_tile @ k_tiles.T * head_dim**-0.5
                out[
                    batch_idx,
                    q_block * BLOCK : (q_block + 1) * BLOCK,
                    head_idx,
                ] = scores.softmax(-1) @ v_tiles
    return out


def run_case(batch, heads, kv_heads, num_q_blocks, num_kv_blocks, topk, seed):
    torch.manual_seed(seed)
    head_dim = 128
    q = torch.randn(
        batch,
        num_q_blocks * BLOCK,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.randn(
        batch,
        num_kv_blocks * BLOCK,
        kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v = torch.randn_like(k)
    selected = torch.stack(
        [
            torch.sort(torch.randperm(num_kv_blocks, device="cuda")[:topk]).values
            for _ in range(batch * heads * num_q_blocks)
        ]
    ).view(batch, heads, num_q_blocks, topk).to(torch.int32)
    sparse = make_sparse(selected, num_kv_blocks)

    os.environ.update(
        FLASH_ATTN_SM90_KV_PAIR="0",
        FLASH_ATTN_SM90_DUAL_TILE="0",
        FLASH_ATTN_SM90_MIN_BLOCKS="1",
        FLASH_ATTN_SM90_NUM_STAGES="2",
    )
    stock, _ = _flash_attn_fwd(
        q, k, v, block_sparse_tensors=sparse, tile_mn=(64, 64)
    )
    os.environ.update(
        FLASH_ATTN_SM90_KV_PAIR="1",
        FLASH_ATTN_SM90_NUM_STAGES="1",
    )
    paired, _ = _flash_attn_fwd(
        q, k, v, block_sparse_tensors=sparse, tile_mn=(64, 128)
    )
    expected = reference(q, k, v, selected)
    stock_error = (stock.float() - expected).abs().max().item()
    paired_error = (paired.float() - expected).abs().max().item()
    cross_error = (paired.float() - stock.float()).abs().max().item()
    passed = paired_error < max(2.5 * stock_error, 1e-2) and paired_error < 0.1
    print(
        f"{'PASS' if passed else 'FAIL'} B={batch} H={heads} Hkv={kv_heads} "
        f"nq={num_q_blocks} nkv={num_kv_blocks} topk={topk}: "
        f"paired={paired_error:.3e} stock={stock_error:.3e} "
        f"cross={cross_error:.3e}",
        flush=True,
    )
    return passed


def main():
    cases = (
        (2, 8, 1, 16, 16, 4, 0),
        (2, 8, 1, 16, 16, 3, 1),
        (1, 8, 1, 8, 8, 1, 2),
        (1, 8, 1, 12, 33, 33, 3),
        (2, 2, 2, 8, 16, 5, 4),
    )
    passed = [run_case(*case) for case in cases]
    print("ALL PASS" if all(passed) else "SOME FAILED")
    raise SystemExit(0 if all(passed) else 1)


if __name__ == "__main__":
    main()
