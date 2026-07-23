"""Focused tuning sweep for the long-context one-CTA SM90 VSA kernel.

Usage:
    python -u bench_long_tuning.py [128k|256k|512k]

Environment:
    TUNE_TRIALS  Timing trials per candidate (default: 2)
    TUNE_WARMUP  Warmups per timing trial (default: 2)
    TUNE_REP     Launches per timing trial (default: 5)
    TUNE_ONLY    Comma-separated candidate labels (optional)
"""

import os
import statistics
import sys

import torch

import bench_long_context as bench
from flash_attn.cute.interface import _flash_attn_fwd


CANDIDATES = (
    ("s1-overlap-rs", 1, True, True),
    ("s2-overlap-rs", 2, True, True),
    ("s3-overlap-rs", 3, True, True),
    ("s2-serial-rs", 2, False, True),
    ("s3-serial-rs", 3, False, True),
    ("s3-overlap-ss", 3, True, False),
)


def set_candidate(stages: int):
    os.environ.update(
        FLASH_ATTN_SM90_MIN_BLOCKS="1",
        FLASH_ATTN_SM90_DUAL_TILE="0",
        FLASH_ATTN_SM90_NUM_STAGES=str(stages),
    )
    # Exercise the kernel's native one-CTA register split.
    os.environ.pop("FLASH_ATTN_SM90_MMA_REGS", None)
    os.environ.pop("FLASH_ATTN_SM90_PRODUCER_REGS", None)


def launch(q, k, v, sparse, overlap: bool, pv_rs: bool):
    return _flash_attn_fwd(
        q,
        k,
        v,
        block_sparse_tensors=sparse,
        tile_mn=(bench.BLOCK, bench.BLOCK),
        intra_wg_overlap=overlap,
        mma_pv_is_rs=pv_rs,
        pack_gqa=True,
    )


def time_candidate(
    q, k, v, sparse, overlap: bool, pv_rs: bool, warmup: int, repeats: int
):
    for _ in range(warmup):
        launch(q, k, v, sparse, overlap, pv_rs)
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        launch(q, k, v, sparse, overlap, pv_rs)
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) / repeats


def main():
    size = sys.argv[1] if len(sys.argv) > 1 else "256k"
    if size not in bench.SIZES:
        raise ValueError(f"unknown size {size!r}; choose from {tuple(bench.SIZES)}")
    only = set(filter(None, os.environ.get("TUNE_ONLY", "").split(",")))
    candidates = tuple(c for c in CANDIDATES if not only or c[0] in only)
    if not candidates:
        raise ValueError(f"TUNE_ONLY selected no candidates from {CANDIDATES}")

    trials = int(os.environ.get("TUNE_TRIALS", 2))
    warmup = int(os.environ.get("TUNE_WARMUP", 2))
    repeats = int(os.environ.get("TUNE_REP", 5))
    nkv, topk = bench.SIZES[size]
    q, k, v, sparse = bench.make_inputs(nkv, topk)
    flops = 4 * bench.BLOCK**2 * bench.HEAD_DIM * nkv * topk * bench.HEADS

    print(
        f"size={size} trials={trials} warmup={warmup} repeats={repeats}",
        flush=True,
    )
    samples = {label: [] for label, *_ in candidates}
    for label, stages, overlap, pv_rs in candidates:
        set_candidate(stages)
        time_candidate(q, k, v, sparse, overlap, pv_rs, warmup, 1)
    for trial in range(trials):
        order = candidates if trial % 2 == 0 else tuple(reversed(candidates))
        for label, stages, overlap, pv_rs in order:
            set_candidate(stages)
            samples[label].append(
                time_candidate(
                    q, k, v, sparse, overlap, pv_rs, warmup, repeats
                )
            )

    best_ms = min(statistics.median(values) for values in samples.values())
    print(
        f"{'candidate':>18} {'trials_ms':>22} {'median_ms':>10} "
        f"{'TFLOP/s':>9} {'vs-best':>9}",
        flush=True,
    )
    for label, *_ in candidates:
        median_ms = statistics.median(samples[label])
        values = ",".join(f"{value:.3f}" for value in samples[label])
        tflops = flops / (median_ms * 1e-3) / 1e12
        print(
            f"{label:>18} {values:>22} {median_ms:10.3f} "
            f"{tflops:9.1f} {best_ms / median_ms:8.3f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
