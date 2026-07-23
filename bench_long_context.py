"""Long-context SM90 VSA benchmark: 1 CTA, 2 CTA, and auto.

The sparse selections have the same distribution as ``bench_min_blocks.py``
(uniform sampling without replacement, sorted by block index), but are built
in chunks and stored at their actual ``topk`` width.  This keeps the 256k and
512k cases practical without changing the kernel workload.

Usage:
    python -u bench_long_context.py [16k,32k,64k,128k,256k,512k]

Environment:
    BENCH_TRIALS   Per-configuration timing trials (default: 3)
    BENCH_WARMUP   Warmup launches per configuration (default: 3)
    BENCH_REP      Timed launches per trial (default: 10)
    SAMPLE_CHUNK   Query rows sampled together (default: 256)
"""

import os
import statistics
import sys

import torch

from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from flash_attn.cute.interface import _flash_attn_fwd


BLOCK = 64
HEAD_DIM = 128
HEADS = 8
PEAK = 990e12

SIZES = {
    "16k": (264, 33),
    "32k": (528, 66),
    "64k": (1056, 132),
    "128k": (2160, 270),
    "256k": (4320, 540),
    "512k": (8640, 1080),
}

CONFIGS = (
    ("1cta", 1, "2", 64, "0"),
    ("2cta", 2, "2", 64, "0"),
    ("optimized", "auto", "auto", 64, "auto"),
)
if os.environ.get("BENCH_KV_PAIR", "0") == "1":
    pair_stages = os.environ.get("BENCH_PAIR_STAGES", "2")
    CONFIGS = (
        ("1cta", 1, "2", 64, "0"),
        ("paired", 1, pair_stages, 128, "1"),
    )
if os.environ.get("BENCH_SHORT_SWEEP", "0") == "1":
    CONFIGS = (
        ("1cta", 1, "2", 64, "0"),
        ("2cta", 2, "2", 64, "0"),
        ("paired", 1, "2", 128, "1"),
    )


def sample_sparse_blocks(nkv: int, topk: int, seed: int, chunk_rows: int):
    """Uniform subsets without replacement, sorted like randperm(...)[0:topk]."""
    rows = HEADS * nkv
    selected = torch.empty((rows, topk), dtype=torch.int32, device="cuda")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    for row_start in range(0, rows, chunk_rows):
        row_end = min(row_start + chunk_rows, rows)
        scores = torch.rand(
            (row_end - row_start, nkv),
            device="cuda",
            generator=generator,
        )
        indices = torch.topk(scores, topk, dim=1, largest=False, sorted=False).indices
        selected[row_start:row_end] = torch.sort(indices, dim=1).values.to(torch.int32)
    return selected.view(1, HEADS, nkv, topk)


def make_inputs(nkv: int, topk: int, seed: int = 0):
    torch.manual_seed(seed)
    seqlen = nkv * BLOCK
    q = torch.randn(
        1, seqlen, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    k = torch.randn(
        1, seqlen, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    v = torch.randn_like(k)
    chunk_rows = int(os.environ.get("SAMPLE_CHUNK", 256))
    full_idx = sample_sparse_blocks(nkv, topk, seed, chunk_rows)
    full_cnt = torch.full(
        (1, HEADS, nkv), topk, dtype=torch.int32, device="cuda"
    )
    mask_cnt = torch.zeros_like(full_cnt)
    # No masked blocks are used. A one-column placeholder is sufficient.
    mask_idx = torch.zeros(
        (1, HEADS, nkv, 1), dtype=torch.int32, device="cuda"
    )
    sparse = BlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        block_size=(BLOCK, BLOCK),
    )
    return q, k, v, sparse


def set_config(
    min_blocks: int,
    stages: str,
    kv_pair: str,
):
    os.environ.update(
        FLASH_ATTN_SM90_MIN_BLOCKS=str(min_blocks),
        FLASH_ATTN_SM90_NUM_STAGES=stages,
        FLASH_ATTN_SM90_KV_PAIR=kv_pair,
    )
    os.environ.pop("FLASH_ATTN_SM90_MMA_REGS", None)
    os.environ.pop("FLASH_ATTN_SM90_PRODUCER_REGS", None)


def launch(q, k, v, sparse, tile_n: int):
    return _flash_attn_fwd(
        q,
        k,
        v,
        block_sparse_tensors=sparse,
        tile_mn=(BLOCK, tile_n),
        intra_wg_overlap=True,
        mma_pv_is_rs=True,
        pack_gqa=True,
    )


def time_trial(q, k, v, sparse, tile_n: int, warmup: int, repeats: int):
    for _ in range(warmup):
        launch(q, k, v, sparse, tile_n)
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        launch(q, k, v, sparse, tile_n)
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) / repeats


def main():
    sizes = sys.argv[1].split(",") if len(sys.argv) > 1 else list(SIZES)
    trials = int(os.environ.get("BENCH_TRIALS", 3))
    warmup = int(os.environ.get("BENCH_WARMUP", 3))
    repeats = int(os.environ.get("BENCH_REP", 10))
    print(
        f"trials={trials} warmup={warmup} repeats={repeats}",
        flush=True,
    )
    print(
        f"{'size':>6} {'config':>8} {'trials_ms':>32} "
        f"{'median_ms':>10} {'TFLOP/s':>9} {'MFU':>7} {'vs-1cta':>9}",
        flush=True,
    )
    for name in sizes:
        if name not in SIZES:
            raise ValueError(f"unknown size {name!r}; choose from {tuple(SIZES)}")
        nkv, topk = SIZES[name]
        q, k, v, sparse = make_inputs(nkv, topk)
        flops = 4 * BLOCK * BLOCK * HEAD_DIM * nkv * topk * HEADS
        samples = {}
        # Compile each specialization before collecting comparable trials.
        for (
            label,
            min_blocks,
            stages,
            tile_n,
            kv_pair,
        ) in CONFIGS:
            set_config(min_blocks, stages, kv_pair)
            time_trial(q, k, v, sparse, tile_n, warmup=warmup, repeats=1)
        samples = {label: [] for label, *_ in CONFIGS}
        # Alternate the order to reduce bias from clock or temperature drift.
        for trial in range(trials):
            configs = CONFIGS if trial % 2 == 0 else tuple(reversed(CONFIGS))
            for (
                label,
                min_blocks,
                stages,
                tile_n,
                kv_pair,
            ) in configs:
                set_config(min_blocks, stages, kv_pair)
                samples[label].append(
                    time_trial(
                        q, k, v, sparse, tile_n, warmup=warmup, repeats=repeats
                    )
                )
        baseline_label = "normal" if "normal" in samples else "1cta"
        baseline_ms = statistics.median(samples[baseline_label])
        for label, *_ in CONFIGS:
            median_ms = statistics.median(samples[label])
            tflops = flops / (median_ms * 1e-3) / 1e12
            sample_text = ",".join(f"{value:.3f}" for value in samples[label])
            speedup = baseline_ms / median_ms
            print(
                f"{name:>6} {label:>8} {sample_text:>32} "
                f"{median_ms:10.3f} {tflops:9.1f} "
                f"{100 * tflops / (PEAK / 1e12):6.1f}% {speedup:8.3f}x",
                flush=True,
            )


if __name__ == "__main__":
    main()
