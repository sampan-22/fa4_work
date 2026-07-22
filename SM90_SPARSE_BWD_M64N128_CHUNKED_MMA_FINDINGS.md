# SM90 sparse backward M64N128 chunked-MMA findings

## Result

The backward path now accepts N64 VSA metadata while scheduling adjacent KV
cubes as one N128 work tile.  It is opt-in with:

```text
FLASH_ATTN_SM90_SPARSE_BWD_M64N128_CHUNKED_MMA=1
```

With the flag off, N64 backward metadata selects the forced `tile_n=64`
baseline.  The experiment is beneficial at the long VSA sequence lengths that
motivated it: the clean-GPU rerun is 1.086x at 64K.  It is 3-4% slower at 16K
and 32K, so it remains default-off rather than changing all N64 calls.

## Backward mapping

Backward metadata is transposed compared with forward metadata: every N64 KV
cube owns a sorted list of Q blocks.  Two adjacent cubes can have overlapping,
disjoint, or empty Q lists.  The implementation merges the partial/full lists
of both cubes in one linear four-way merge:

- a Q block in both cubes is loaded once and both N64 score halves remain live;
- a Q block in one cube is loaded once and the other score half is set to
  `-inf` before the probability calculation;
- dS is consequently zero for an inactive half, so dQ, dK, and dV naturally
  receive only valid edge contributions;
- dQ stores follow exactly the same merged order as the producer and MMA
  consumer, preserving the named-barrier contract;
- dK/dV epilogues store the ordinary contiguous N128 tile.

This removes redundant Q/dO/stat loads for Q blocks shared by the paired KV
cubes.  The first implementation used membership scans and was correct but
quadratic in list length; it measured only 0.504x at 16K.  The final sorted
merge is O(k) and removes that regression.

## Correctness

Command:

```bash
FLASH_ATTN_SM90_FIXED_REF_SOFTMAX=0 python test_bwd_chunked_mma.py
```

All cases pass for bf16 D=DV=128, including a single head, GQA (Hq=2/Hkv=1),
overlapping and half-only edges, and mixed partial/full metadata lists.

| Case | max chunked vs N64 (dQ, dK, dV) | max chunked vs FP32 reference |
|---|---:|---:|
| single head | 6.10e-5, 0, 0 | 2.69e-3, 2.60e-3, 2.65e-3 |
| GQA | 3.05e-5, 0, 0 | 3.82e-3, 6.01e-3, 3.30e-3 |
| mixed lists | 6.10e-5, 0, 0 | 2.69e-3, 2.60e-3, 2.65e-3 |

No non-finite gradients occurred.  The chunked and baseline FP32-reference
errors are identical at the displayed precision.

## Benchmark

Command:

```bash
BENCH_WARMUP=5 BENCH_REP=20 BENCH_TRIALS=3 \
  python bench_bwd_chunked_mma.py 16k,32k,64k,128k
```

Environment: NVIDIA H100 80GB HBM3 (132 SMs), Hq=8, Hkv=1, bf16 D=DV=128,
fixed-length noncausal attention, N64 cubes, and 0.125 active block density.
Times include backward preprocess, main kernel, dQ conversion, and GQA dK/dV
postprocess, matching `_flash_attn_bwd` end to end.

| Size | N64 baseline median / p90 | Chunked N128 median / p90 | Speedup |
|---|---:|---:|---:|
| 16K | 1.274 / 1.277 ms | 1.322 / 1.405 ms | 0.963x |
| 32K | 4.781 / 4.866 ms | 4.935 / 5.018 ms | 0.969x |
| 64K | 23.441 / 24.352 ms | 21.583 / 23.133 ms | 1.086x |

These numbers were rerun on an idle H100 selected with
`CUDA_VISIBLE_DEVICES=2`. GPU 0 had an unrelated hidden workload (about 63 GB
resident and 529 W at idle), so measurements made there were discarded.

## Ursa end-to-end integration

Ursa's `fa4_hybrid` now accepts true 64-element cubes and constructs the
sorted, transposed K-to-Q metadata required by backward. Previously the bench
adapter reused Q-to-K forward metadata; its tensor shapes fit self-attention,
but the gradients and per-KV workload were wrong. The process-wide chunking
flag now leaves dense attention calls alone, allowing the bench's dense
reference and sparse variant to run in one process.

Command (run once with the flag `0`, then with `1`):

```bash
CUDA_VISIBLE_DEVICES=2 \
FLASH_ATTN_SM90_SPARSE_BWD_M64N128_CHUNKED_MMA=1 \
python -m ursa.models.omni.sparse_attn_kernels.bench.bench \
  --task t2v --sizes 16k,32k,64k --ulysses 8 --cube 64 \
  --vsa fa4_hybrid --warmup 5 --rep 20 --backward
```

This measures the complete registered VSA operation (coarse selection, fine
forward, backward, and gate), so the isolated backward speedup is diluted:

| Size | N64 FA4 | Chunked-backward FA4 | Chunked / N64 | Triton baseline |
|---|---:|---:|---:|---:|
| 16K | 2.732 ms | 2.453 ms | 1.114x | 2.800 ms |
| 32K | 7.415 ms | 7.220 ms | 1.027x | 7.985 ms |
| 64K | 32.133 ms | 29.508 ms | 1.089x | 28.041 ms |

Against Triton, chunked FA4 is 1.141x faster at 16K and 1.106x faster at
32K. Triton remains 1.052x faster at 64K. Thus the requested cube-64 FA4 path
wins two of the three target sizes while preserving the finer selection
granularity.

Profiling attributes the remaining 64K gap to the main backward kernels:
chunked FA4 spends about 20.8 ms in its SM90 main kernel, while Triton's dQ
and dK/dV kernels total about 16.3 ms. FA4 forward is faster (about 4.9 vs
6.4 ms). Runtime skipping of inactive N64 halves was correct but serialized
the divergent warpgroup WGMMA schedule and regressed backward by more than
2x, so that experiment was reverted. Stage-count and single-warpgroup-dQ
sweeps were also neutral or slower.

### Split-backward production candidate

The registered `fa4_fwd_triton_bwd` variant keeps FA4's faster native-GQA
sparse forward, but feeds FA4's saved output and natural-log LSE into the
existing split Triton dQ and dK/dV backward kernels. The wrapper converts LSE
to log2, supplies precomputed Q-to-K and K-to-Q indices directly, and reduces
the expanded GQA dK/dV heads back to the original KV head count. At 8,192 or
more KV cubes it dispatches the fine forward to Triton as well: profiling at
512K showed a forward crossover (FA4 668.8 ms, Triton 621.0 ms). This tail
still beats the baseline because it avoids rebuilding Q-to-K indices in both
forward and backward.

Final idle-H100 training-step results:

| Size | Triton baseline | FA4/direct-index split variant | Speedup |
|---|---:|---:|---:|
| 16K | 2.841 ms | 2.385 ms | 1.191x |
| 32K | 8.222 ms | 6.952 ms | 1.183x |
| 64K | 28.461 ms | 26.105 ms | 1.090x |
| 128K | 119.093 ms | 113.289 ms | 1.051x |
| 256K | 533.888 ms | 482.761 ms | 1.106x |
| 512K | 2304.058 ms | 2255.394 ms | 1.022x |

The 16K-128K rows use the full Omni command with warmup=5/rep=20. The 256K
row uses warmup=2/rep=5 and 512K uses warmup=1/rep=3 through the same
`_measure_kernel` path without rerunning the unrelated dense reference.
`fa4_fwd_triton_bwd_test.py` forces both fine-forward backends and compares
the complete output plus q/k/v/gate gradients against `triton_baseline`; both
cases pass at a 1e-2 relative-L2 bf16 tolerance. The observed relative RMS
differences on the FA4-forward branch were 0.18% dQ, 0.26% dK, and 0.32% dV.

For completeness, the earlier pure-FA4 long-tail sweep was:

| Size | N64 FA4 | Chunked-backward FA4 | Chunked / N64 | Triton baseline |
|---|---:|---:|---:|---:|
| 128K | 169.611 ms | 157.902 ms | 1.074x | 122.622 ms |
| 256K | 746.129 ms | 699.123 ms | 1.067x | 531.206 ms |
| 512K | 3417.373 ms | 3250.880 ms | 1.051x | 2281.659 ms |

The 16K-64K rows use warmup=5/rep=20. To keep the long sweep practical,
128K-512K use warmup=2/rep=5. Their cube-padded actual sequence lengths are
138,240, 276,480, and 552,960 tokens. The large chunked rows skip remeasuring
the unchanged dense reference; all sparse timings still use the same Ursa
`_measure_kernel` path.

The existing cube-128 FA4 path measured 2.060, 5.294, and 22.174 ms at
16K/32K/64K. It remains faster, but changes the VSA selection granularity and
therefore is not a semantic replacement for the requested cube-64 path.
Triton's current backward is physically fixed at 64: requesting cube 128 is
rejected with a Q-block shape mismatch, so its valid comparison is the
cube-64 column above.

## Instruction check

The dumped chunked main-kernel PTX contains 8 static
`wgmma.mma_async...m64n128k16` sites and 24 `m64n64k16` sites; `cuobjdump`
reports the matching 8 `HGMMA.64x128x16.F32.BF16` and 24
`HGMMA.64x64x16.F32.BF16` instructions.  Backward contains five differently
oriented GEMMs and two MMA warpgroups, so N64 instructions remain expected;
the N128 tile is not implemented as two separate scheduler work tiles or two
Q/dO pipeline iterations.

## Current specialization boundary

The opt-in path requires SM90, D=DV=128, N64 sparse metadata, tile_m=64,
fixed sequence lengths divisible by Q64/KV128, noncausal/nonlocal attention,
sorted partial and full lists, and no custom score/mask modification.  GQA is
supported.  A full-block tensor must be present (its counts may be zero).
