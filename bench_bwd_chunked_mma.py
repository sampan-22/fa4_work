"""Benchmark forced-N64 versus chunked-N128 SM90 sparse backward."""

import gc
import os
import statistics
import sys
import time

import torch

from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd


BLK = 64
D = 128
ENV = "FLASH_ATTN_SM90_SPARSE_BWD_M64N128_CHUNKED_MMA"
SIZES = {
    "16k": (264, 33),
    "32k": (528, 66),
    "64k": (1056, 132),
    "128k": (2160, 270),
    "256k": (4320, 540),
    "512k": (8640, 1080),
    "1m": (16320, 2040),
}


def make_metadata(nblocks, topk, heads=8):
    selected = torch.stack(
        [
            torch.sort(torch.randperm(nblocks, device="cuda")[:topk]).values
            for _ in range(heads * nblocks)
        ]
    ).view(1, heads, nblocks, topk).to(torch.int32)
    fwd_cnt = torch.full((1, heads, nblocks), topk, dtype=torch.int32, device="cuda")
    fwd = BlockSparseTensorsTorch(
        torch.zeros_like(fwd_cnt),
        torch.zeros_like(selected),
        fwd_cnt,
        selected,
        (BLK, BLK),
    )

    # GPU transpose from Q->KV adjacency to KV->Q adjacency.
    counts = torch.zeros((1, heads, nblocks), dtype=torch.int32, device="cuda")
    per_head = []
    max_count = 0
    m_ids = torch.arange(nblocks, device="cuda", dtype=torch.int32).repeat_interleave(topk)
    for h in range(heads):
        n_ids = selected[0, h].reshape(-1)
        order = torch.argsort(n_ids, stable=True)
        n_sorted = n_ids[order]
        m_sorted = m_ids[order]
        count = torch.bincount(n_sorted.long(), minlength=nblocks).to(torch.int32)
        counts[0, h] = count
        max_count = max(max_count, int(count.max()))
        per_head.append((n_sorted, m_sorted, count))
    bwd_idx = torch.zeros((1, heads, nblocks, max_count), dtype=torch.int32, device="cuda")
    for h, (n_sorted, m_sorted, count) in enumerate(per_head):
        starts = torch.cumsum(count, 0) - count
        rank = torch.arange(n_sorted.numel(), device="cuda", dtype=torch.int32)
        rank -= torch.repeat_interleave(starts, count.long())
        bwd_idx[0, h, n_sorted.long(), rank.long()] = m_sorted
    bwd = BlockSparseTensorsTorch(
        torch.zeros_like(counts),
        torch.zeros_like(bwd_idx),
        counts,
        bwd_idx,
        (BLK, BLK),
    )
    return fwd, bwd


def make_case(nblocks, topk):
    torch.manual_seed(0)
    seqlen = nblocks * BLK
    q = torch.randn(1, seqlen, 8, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, seqlen, 1, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    dout = torch.randn_like(q)
    fwd_sparse, bwd_sparse = make_metadata(nblocks, topk)
    os.environ["FLASH_ATTN_SM90_SPARSE_M64N128_CHUNKED_MMA"] = "0"
    out, lse = _flash_attn_fwd(
        q, k, v, tile_mn=(BLK, BLK), block_sparse_tensors=fwd_sparse,
        causal=False, return_lse=True,
    )
    return q, k, v, out, dout, lse, bwd_sparse


def launch(case, chunked):
    os.environ[ENV] = "1" if chunked else "0"
    return _flash_attn_bwd(
        *case[:-1], causal=False, block_sparse_tensors=case[-1]
    )


def first_call(case, chunked):
    torch.cuda.synchronize()
    start = time.perf_counter()
    launch(case, chunked)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def sample(case, chunked, warmup, repetitions, trials):
    for _ in range(warmup):
        launch(case, chunked)
    torch.cuda.synchronize()
    values = []
    for _ in range(trials):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
        for start, end in zip(starts, ends):
            start.record()
            launch(case, chunked)
            end.record()
        torch.cuda.synchronize()
        values.extend(start.elapsed_time(end) for start, end in zip(starts, ends))
    values.sort()
    return statistics.median(values), values[min(len(values) - 1, int(0.9 * len(values)))]


def main():
    sizes = sys.argv[1].split(",") if len(sys.argv) > 1 else list(SIZES)
    warmup = int(os.environ.get("BENCH_WARMUP", 5))
    repetitions = int(os.environ.get("BENCH_REP", 20))
    trials = int(os.environ.get("BENCH_TRIALS", 3))
    prop = torch.cuda.get_device_properties(0)
    print(
        f"GPU={prop.name} SMs={prop.multi_processor_count} Hq=8 Hkv=1 D=128 "
        f"density=.125 warmup={warmup} reps={repetitions} trials={trials}"
    )
    print("size baseline_ms/p90 chunked_ms/p90 speedup")
    for name in sizes:
        case = make_case(*SIZES[name])
        baseline_compile = first_call(case, False)
        chunked_compile = first_call(case, True)
        baseline = sample(case, False, warmup, repetitions, trials)
        chunked = sample(case, True, warmup, repetitions, trials)
        print(
            f"{name} {baseline[0]:.3f}/{baseline[1]:.3f} "
            f"{chunked[0]:.3f}/{chunked[1]:.3f} {baseline[0] / chunked[0]:.3f}x "
            f"first={baseline_compile:.2f}s/{chunked_compile:.2f}s"
        )
        del case
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
