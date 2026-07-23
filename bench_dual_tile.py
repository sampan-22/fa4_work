"""SM90 block-sparse baseline/dual-tile attribution benchmark.

Methodology matches the recorded baseline: three trials, each with 10 warmup
launches followed by 50 timed launches; the reported latency is the median of
the three per-trial means.

Usage: python bench_dual_tile.py [16k,32k,64k,128k]
"""

import os
import statistics
import sys

import torch

from bench_min_blocks import PEAK, SIZES, make_inputs
from flash_attn.cute.interface import _flash_attn_fwd


CONFIGS = (
    ("baseline_s2", 2, False, False, 2, 216, 64),
    ("baseline_s3", 3, False, False, 2, 216, 64),
    ("dual_no_barrier", 3, True, False, 1, 216, 64),
    ("dual_barrier", 3, True, True, 1, 216, 64),
)


def set_config(stages, dual, scheduler_barrier, min_blocks, mma_regs, producer_regs):
    os.environ.update(
        FLASH_ATTN_SM90_NUM_STAGES=str(stages),
        FLASH_ATTN_SM90_DUAL_TILE="1" if dual else "0",
        FLASH_ATTN_SM90_DUAL_SCHEDULER_BARRIER="1" if scheduler_barrier else "0",
        FLASH_ATTN_SM90_MIN_BLOCKS=str(min_blocks),
        FLASH_ATTN_SM90_MMA_REGS=str(mma_regs),
        FLASH_ATTN_SM90_PRODUCER_REGS=str(producer_regs),
        FLASH_ATTN_SM90_TILE_PAIRING="adjacent_sparse",
    )


def time_trial(q, k, v, sparse, warmup=10, repeats=50):
    for _ in range(warmup):
        _flash_attn_fwd(q, k, v, block_sparse_tensors=sparse, tile_mn=(64, 64))
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        _flash_attn_fwd(q, k, v, block_sparse_tensors=sparse, tile_mn=(64, 64))
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def main():
    sizes = sys.argv[1].split(",") if len(sys.argv) > 1 else list(SIZES)
    print(
        f"{'size':>6} {'config':>18} {'trials_ms':>29} "
        f"{'median_ms':>10} {'TFLOP/s':>9} {'MFU':>7}",
        flush=True,
    )
    for size in sizes:
        n_blocks, topk = SIZES[size]
        q, k, v, sparse = make_inputs(n_blocks, topk)
        flops = 4 * 64 * 64 * 128 * n_blocks * topk * 8
        for name, stages, dual, barrier, min_blocks, mma_regs, producer_regs in CONFIGS:
            set_config(stages, dual, barrier, min_blocks, mma_regs, producer_regs)
            trials = [time_trial(q, k, v, sparse) for _ in range(3)]
            median_ms = statistics.median(trials)
            tflops = flops / (median_ms * 1e-3) / 1e12
            trials_text = ",".join(f"{trial:.3f}" for trial in trials)
            print(
                f"{size:>6} {name:>18} {trials_text:>29} "
                f"{median_ms:10.3f} {tflops:9.1f} {100 * tflops / (PEAK / 1e12):6.1f}%",
                flush=True,
            )


if __name__ == "__main__":
    main()

