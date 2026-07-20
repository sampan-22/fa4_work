"""Fine-attention-only latency: stock (1 CTA/SM) vs 2-CTA/SM
(FLASH_ATTN_SM90_MIN_BLOCKS=2) on the SM90 forward, block-sparse VSA
shapes. Exact VSA bench fine-stage shapes (per-GPU after ulysses 8):
B=1, D=128, Hq=8, Hkv=1, num_blocks cubes of 64 tokens, topk selected per
query cube. FLOPs = 4 * 64^2 * D * num_blocks * topk * heads.

Usage: python bench_min_blocks.py [sizes]
  sizes: comma list from {16k,32k,64k,128k} (default all)
"""

import os
import sys
import torch

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

BLK = 64
PEAK = 990e12

SIZES = {
    "16k": (264, 33),
    "32k": (528, 66),
    "64k": (1056, 132),
    "128k": (2160, 270),
}


def make_inputs(nkv, topk, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    B, H, Hkv, D = 1, 8, 1, 128
    S = nkv * BLK
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, S, Hkv, D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, S, Hkv, D, dtype=torch.bfloat16, device=dev)
    sel = torch.stack(
        [torch.sort(torch.randperm(nkv, device=dev)[:topk]).values for _ in range(B * H * nkv)]
    ).view(B, H, nkv, topk).to(torch.int32)
    full_block_idx = torch.zeros((B, H, nkv, nkv), dtype=torch.int32, device=dev)
    full_block_idx[..., :topk] = sel
    full_block_cnt = torch.full((B, H, nkv), topk, dtype=torch.int32, device=dev)
    zeros_i = torch.zeros((B, H, nkv, nkv), dtype=torch.int32, device=dev)
    zeros_c = torch.zeros((B, H, nkv), dtype=torch.int32, device=dev)
    bs = BlockSparseTensorsTorch(
        mask_block_cnt=zeros_c,
        mask_block_idx=zeros_i,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        block_size=(BLK, BLK),
    )
    return q, k, v, bs


def time_kernel(q, k, v, bs, min_blocks, warmup=10, rep=50):
    os.environ["FLASH_ATTN_SM90_MIN_BLOCKS"] = str(min_blocks)
    for _ in range(warmup):
        _flash_attn_fwd(q, k, v, block_sparse_tensors=bs, tile_mn=(64, 64))
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(rep):
        _flash_attn_fwd(q, k, v, block_sparse_tensors=bs, tile_mn=(64, 64))
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / rep


def main():
    sizes = sys.argv[1].split(",") if len(sys.argv) > 1 else list(SIZES)
    print(f"{'size':>6} {'cfg':>8} {'ms':>9} {'TFLOP/s':>9} {'MFU':>7}")
    for name in sizes:
        nkv, topk = SIZES[name]
        q, k, v, bs = make_inputs(nkv, topk)
        flops = 4 * BLK * BLK * 128 * nkv * topk * 8
        for label, min_blocks in [("1cta", 1), ("2cta", 2)]:
            ms = time_kernel(q, k, v, bs, min_blocks)
            tf = flops / (ms * 1e-3) / 1e12
            print(f"{name:>6} {label:>8} {ms:9.3f} {tf:9.1f} {tf / (PEAK / 1e12) * 100:6.1f}%")


if __name__ == "__main__":
    main()
