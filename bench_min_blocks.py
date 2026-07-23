"""Fine-attention-only latency: stock 1-CTA, 2-CTA/SM, and the optimized
SM90 forward on block-sparse VSA shapes. Exact VSA bench fine-stage shapes
(per-GPU after ulysses 8):
B=1, D=128, Hq=8, Hkv=1, num_blocks cubes of 64 tokens, topk selected per
query cube. FLOPs = 4 * 64^2 * D * num_blocks * topk * heads.

Usage: python bench_min_blocks.py [sizes]
  sizes: comma list from {16k,32k,64k,128k,256k,512k}
  default: 16k,32k,64k,128k
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
    "256k": (4320, 540),
    "512k": (8640, 1080),
}
DEFAULT_SIZES = ("16k", "32k", "64k", "128k")


def make_inputs(nkv, topk, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    B, H, Hkv, D = 1, 8, 1, 128
    S = nkv * BLK
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, S, Hkv, D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, S, Hkv, D, dtype=torch.bfloat16, device=dev)
    rows = B * H * nkv
    sel = torch.empty((rows, topk), dtype=torch.int32, device=dev)
    generator = torch.Generator(device=dev)
    generator.manual_seed(seed)
    chunk_rows = int(os.environ.get("SAMPLE_CHUNK", 256))
    for row_start in range(0, rows, chunk_rows):
        row_end = min(row_start + chunk_rows, rows)
        scores = torch.rand(
            (row_end - row_start, nkv), device=dev, generator=generator
        )
        indices = torch.topk(
            scores, topk, dim=1, largest=False, sorted=False
        ).indices
        sel[row_start:row_end] = torch.sort(indices, dim=1).values.to(
            torch.int32
        )
    full_block_idx = sel.view(B, H, nkv, topk)
    full_block_cnt = torch.full((B, H, nkv), topk, dtype=torch.int32, device=dev)
    zeros_i = torch.zeros((B, H, nkv, 1), dtype=torch.int32, device=dev)
    zeros_c = torch.zeros((B, H, nkv), dtype=torch.int32, device=dev)
    bs = BlockSparseTensorsTorch(
        mask_block_cnt=zeros_c,
        mask_block_idx=zeros_i,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        block_size=(BLK, BLK),
    )
    return q, k, v, bs


def time_kernel(
    q,
    k,
    v,
    bs,
    min_blocks,
    dual_tile=False,
    kv_pair="0",
    stages="2",
    warmup=10,
    rep=50,
):
    os.environ["FLASH_ATTN_SM90_MIN_BLOCKS"] = str(min_blocks)
    os.environ["FLASH_ATTN_SM90_DUAL_TILE"] = (
        dual_tile if isinstance(dual_tile, str) else "1" if dual_tile else "0"
    )
    os.environ["FLASH_ATTN_SM90_KV_PAIR"] = kv_pair
    os.environ["FLASH_ATTN_SM90_NUM_STAGES"] = stages
    kernel_kwargs = {
        "intra_wg_overlap": os.environ.get("VSA_INTRA_WG_OVERLAP", "1") == "1",
        "mma_pv_is_rs": os.environ.get("VSA_MMA_PV_RS", "1") == "1",
        "pack_gqa": os.environ.get("VSA_PACK_GQA", "1") == "1",
    }
    for _ in range(warmup):
        _flash_attn_fwd(
            q, k, v, block_sparse_tensors=bs, tile_mn=(64, 64), **kernel_kwargs
        )
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(rep):
        _flash_attn_fwd(
            q, k, v, block_sparse_tensors=bs, tile_mn=(64, 64), **kernel_kwargs
        )
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / rep


def main():
    sizes = sys.argv[1].split(",") if len(sys.argv) > 1 else list(DEFAULT_SIZES)
    warmup = int(os.environ.get("BENCH_WARMUP", 10))
    rep = int(os.environ.get("BENCH_REP", 50))
    print(f"warmup={warmup} repeats={rep}")
    print(f"{'size':>6} {'cfg':>10} {'ms':>9} {'TFLOP/s':>9} {'MFU':>7} {'vs-1cta':>9}")
    for name in sizes:
        nkv, topk = SIZES[name]
        q, k, v, bs = make_inputs(nkv, topk)
        flops = 4 * BLK * BLK * 128 * nkv * topk * 8
        baseline_ms = None
        for label, min_blocks, dual_tile, kv_pair, stages in [
            ("1cta", 1, False, "0", "2"),
            ("2cta", 2, False, "0", "2"),
            ("optimized", 1, False, "auto", "auto"),
        ]:
            ms = time_kernel(
                q,
                k,
                v,
                bs,
                min_blocks,
                dual_tile,
                kv_pair,
                stages,
                warmup,
                rep,
            )
            if baseline_ms is None:
                baseline_ms = ms
            tf = flops / (ms * 1e-3) / 1e12
            speedup = baseline_ms / ms
            print(
                f"{name:>6} {label:>10} {ms:9.3f} {tf:9.1f} "
                f"{tf / (PEAK / 1e12) * 100:6.1f}% {speedup:8.3f}x"
            )


if __name__ == "__main__":
    main()
