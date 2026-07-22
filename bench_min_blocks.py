"""Fine-attention-only latency: baseline and chunked, 1 vs 2 CTAs/SM.

``FLASH_ATTN_SM90_MIN_BLOCKS=2`` applies the Hopper launch bound.  The
chunked specialization also uses one K/V pipeline stage in that mode so its
shared-memory allocation permits two resident CTAs.  Exact VSA bench
fine-stage shapes (per-GPU after ulysses 8):
B=1, D=128, Hq=8, Hkv=1, num_blocks cubes of 64 tokens, topk selected per
query cube. FLOPs = 4 * 64^2 * D * num_blocks * topk * heads.

Usage: python bench_min_blocks.py [sizes]
  sizes: comma list from {16k,32k,64k,128k} (default all)
"""

import os
import statistics
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


def time_kernel(q, k, v, bs, *, chunked, min_blocks, warmup, rep, trials):
    os.environ["FLASH_ATTN_SM90_SPARSE_M64N128_CHUNKED_MMA"] = str(int(chunked))
    os.environ["FLASH_ATTN_SM90_FIXED_REF_SOFTMAX"] = "0"
    os.environ["FLASH_ATTN_SM90_MIN_BLOCKS"] = str(min_blocks)
    for _ in range(warmup):
        _flash_attn_fwd(q, k, v, block_sparse_tensors=bs, tile_mn=(64, 64))
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        for start, end in zip(starts, ends):
            start.record()
            _flash_attn_fwd(q, k, v, block_sparse_tensors=bs, tile_mn=(64, 64))
            end.record()
        torch.cuda.synchronize()
        samples.extend(start.elapsed_time(end) for start, end in zip(starts, ends))
    samples.sort()
    return statistics.median(samples), samples[min(len(samples) - 1, int(0.9 * len(samples)))]


def main():
    sizes = sys.argv[1].split(",") if len(sys.argv) > 1 else list(SIZES)
    warmup = int(os.environ.get("BENCH_WARMUP", 10))
    rep = int(os.environ.get("BENCH_REP", 50))
    trials = int(os.environ.get("BENCH_TRIALS", 3))
    print(f"warmup={warmup} repetitions={rep} trials={trials} online_softmax=True")
    print(
        f"{'size':>6} {'kernel':>9} {'cfg':>8} {'stages':>7} "
        f"{'median':>9} {'p90':>9} {'TFLOP/s':>9} {'MFU':>7}"
    )
    for name in sizes:
        nkv, topk = SIZES[name]
        q, k, v, bs = make_inputs(nkv, topk)
        flops = 4 * BLK * BLK * 128 * nkv * topk * 8
        configs = [
            ("baseline", "1cta", False, 1, 2),
            ("baseline", "2cta", False, 2, 2),
            ("chunked", "1cta", True, 1, 2),
            ("chunked", "2cta", True, 2, 1),
        ]
        for kernel, label, chunked, min_blocks, stages in configs:
            ms, p90 = time_kernel(
                q,
                k,
                v,
                bs,
                chunked=chunked,
                min_blocks=min_blocks,
                warmup=warmup,
                rep=rep,
                trials=trials,
            )
            tf = flops / (ms * 1e-3) / 1e12
            print(
                f"{name:>6} {kernel:>9} {label:>8} {stages:7d} "
                f"{ms:9.3f} {p90:9.3f} {tf:9.1f} {tf / (PEAK / 1e12) * 100:6.1f}%"
            )


if __name__ == "__main__":
    main()
