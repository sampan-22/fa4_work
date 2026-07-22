# SM90 sparse M64N128 chunked-MMA findings

## Verdict

The opt-in experiment is correct for the requested VSA configuration and emits
genuine Hopper `m64n128k16` WGMMA operations.  It does not call the N64 QK path
twice and it performs one N128 online-softmax update and one N128 PV tile call
per pair.  Median speedup at `.875` sparsity is positive but below the hoped-for
roughly 10%: **1.5%, 6.5%, 7.1%, and 4.6%** at 16K, 32K, 64K, and 128K.
Consequently the experiment remains default-off behind:

```text
FLASH_ATTN_SM90_SPARSE_M64N128_CHUNKED_MMA=1
```

A follow-up occupancy specialization is also available with
`FLASH_ATTN_SM90_MIN_BLOCKS=2`. It reduces the chunked K/V pipelines from two
stages to one so that two CTAs fit on an H100 SM. This improves the chunked
kernel through 64K but regresses sharply at 128K, so it is opt-in rather than
an automatic/default dispatch.

`FLASH_ATTN_SM90_FIXED_REF_SOFTMAX` was forced to `0` in every correctness and
benchmark launch.  The chunked dispatch rejects it when nonzero.

## Files intentionally changed

- `flash_attn/cute/interface.py`
- `flash_attn/cute/flash_fwd_sm90.py`
- `flash_attn/cute/block_sparse_utils.py`
- `test_fixed_ref_softmax.py` (existing VSA harness, repurposed without fixed-ref)
- `bench_fixed_ref_softmax.py` (existing VSA benchmark, repurposed without fixed-ref)
- `bench_min_blocks.py` (1-CTA/2-CTA baseline and chunked occupancy matrix)
- `SM90_SPARSE_M64N128_CHUNKED_MMA_FINDINGS.md`

No public `_flash_attn_fwd` argument changed.  The VSA call remains
`tile_mn=(64, 64)` with its existing `BlockSparseTensorsTorch` format.

## Phase-1 shape and resource audit

The concrete VSA specialization is bf16, D=DV=128, noncausal, fixed length,
one MMA warpgroup plus one producer warpgroup (256 threads), register-sourced
PV, intra-warpgroup overlap enabled, and two K plus two V pipeline stages.

| Property | Baseline | Chunked experiment |
|---|---:|---:|
| API-visible tile | M64 N64 | M64 N64 |
| Sparse metadata block | M64 N64 | M64 N64 |
| Internal QK/score/P tile | M64 N64 | M64 N128 |
| Internal PV K extent | N64 | N128 |
| D / DV | 128 / 128 | 128 / 128 |
| Consumer warpgroups / threads | 1 / 256 | 1 / 256 |
| K stages / V stages | 2 / 2 | 2 / 2 |
| Intra-WG overlap | enabled | enabled |
| Q payload | 16 KiB | 16 KiB |
| K payload | 32 KiB | 64 KiB |
| V payload | 32 KiB | 64 KiB |
| Dynamic shared memory (Nsight) | 82,944 B | 148,480 B |
| Registers/thread | 255 | 255 |
| Theoretical occupancy | 12.5% | 12.5% |
| Achieved occupancy (profile launch) | 7.68% | 7.67% |

Both variants are one CTA/SM in this original launch configuration. The
optional chunked `MIN_BLOCKS=2` specialization uses one K and one V stage,
reducing its allocation to approximately 83 KiB/CTA. Together with the
existing 128-register launch cap, this permits two 256-thread CTAs to reside on
an H100 SM. The compile key contains `MIN_BLOCKS`, and the stage count is a
deterministic consequence of that key (`2 stages` for 1CTA, `1 stage` for
2CTA).

## Logical/physical mapping and list order

Metadata remains a compact ordered stream of N64 block indices.  The producer
walks the masked list first and the full list second, as baseline does.  It
never pairs across the list boundary.

For an even list `[i0, i1, i2, i3]`, baseline consumes it in reverse and the
packed stages are:

```text
stage 0 low/high N64 = [i3, i2]
stage 1 low/high N64 = [i1, i0]
```

For an odd list `[i0, i1, i2]`, the padded stage is issued first:

```text
stage 0 low/high N64 = [i2, i2]  # high half masked before softmax
stage 1 low/high N64 = [i1, i0]
```

After the invalid duplicate is excluded, the N128 column stream is exactly the
baseline reverse traversal.  The two source blocks may be arbitrarily far
apart in KV memory.

The canonical `(128, 128, 2 stages)` SW128 K/V allocation is re-carved only for
TMA as canonical 64x64 chunks.  At D=128, each N128 stage receives four 8 KiB
transactions for K and four for V:

```text
K: block0/D0, block1/D0, block0/D1, block1/D1
V: block0/D0, block1/D0, block0/D1, block1/D1
```

All four transactions arrive on the stage's existing 32 KiB transaction
barrier.  WGMMA sees the ordinary canonical N128 stage, not the chunk view.

## Producer/consumer schedule

The non-overlap interpretation for one pair is:

```text
producer: acquire K stage; issue K0/K1 chunk TMAs
producer: acquire V stage; issue V0/V1 chunk TMAs
consumer: wait complete K stage
consumer: one QK tile call over M64xN128 (D128 = 8 m64n128k16 steps)
consumer: wait QK; release K stage
consumer: mask only an odd duplicated half; one N128 online-softmax update
consumer: convert/publish one N128 P fragment; rescale O once
consumer: wait complete V stage
consumer: one PV tile call over P(M64xN128) x V(N128xDV)
consumer: wait PV; release V stage
```

The enabled intra-WG schedule retains the existing Hopper pipeline:

1. Pair 0 runs the QK/softmax/P prologue.
2. For pair `i>0`, QK(i) is issued, then PV(i-1) is issued on the other
   completed stage.
3. `wait_group(1)` proves QK(i) complete before K(i) is released and before
   its score fragment is consumed.
4. `wait_group(0)` proves PV(i-1) complete before V(i-1) is released and
   before the shared probability operand is overwritten with P(i).
5. The tail issues and waits for the final PV.

Thus overlap changes temporal ordering, not tile count: every pair still has
one N128 QK tile call, one N128 softmax fragment, and one N128 PV tile call.

## Lifetime and barrier proof

- A K or V stage becomes visible only after all four half/chunk TMA
  transactions satisfy its full-tile mbarrier transaction count.
- K is released only after the pair's QK WGMMA completion wait, so neither
  source half can be overwritten while QK reads it.
- V is released only after the pair's PV WGMMA completion wait.
- `acc_S` is read for masking and normal online softmax only after QK
  completion.  Its N128 probabilities are converted into the PV A fragment.
- In overlap, PV(i-1) completes before that probability fragment is reused for
  P(i); this is the existing `wait_group(0)` dependency, not source ordering.
- Sparse indices remain in global metadata and are read only while issuing
  their corresponding source transactions; metadata storage is never aliased
  by the kernel.
- An odd upper half is `-inf` before max/sum.  It therefore contributes zero
  probability, no output, and no LSE mass.  The physical duplicate is only a
  legal fixed-shape TMA payload.
- Empty rows produce no pipeline traffic and retain baseline zero-output / LSE
  `-inf` behavior.

## Native instruction gate

Final generated PTX was dumped from the implemented opt-in specialization.
It contained **72 static `wgmma.mma_async...m64n128k16` sites and zero
`m64n64k16` sites** (the count includes prologue/steady/tail control paths).
The D128 pair loop has eight QK k16 steps.  PV is one tile call with eight k16
steps over K=N128.  `cuobjdump` reports only `HGMMA.64x128x16.F32.BF16` for the
specialization.  This passes the hard rejection gate: there is no pair of N64
QK calls hidden under a grouped loop.

## Correctness

Command:

```bash
FLASH_ATTN_SM90_FIXED_REF_SOFTMAX=0 python test_fixed_ref_softmax.py
```

All tests use Hq=8/Hkv=1, bf16 D=DV=128, N64 metadata, and 12.5% active block
density for every non-empty row.  They compare baseline, chunked, and an FP32
reference for output and LSE.

| Case | max chunked-ref O | max chunked-baseline O | max chunked-ref LSE | Result |
|---|---:|---:|---:|---|
| even topk=2 | 3.012e-3 | 3.906e-3 | 9.537e-7 | pass |
| odd topk=3 | 2.017e-3 | 1.953e-3 | 9.537e-7 | pass |
| single topk=1 | 3.794e-3 | 0 | 9.537e-7 | pass |
| one empty row | 3.457e-3 | 1.953e-3 | 9.537e-7 | pass |
| masked-list + full-list | 2.178e-3 | 1.953e-3 | 9.537e-7 | pass |

No NaNs occurred.  Empty-row output was exactly zero and LSE was `-inf`.

## Benchmark

Command:

```bash
BENCH_WARMUP=10 BENCH_REP=50 BENCH_TRIALS=3 \
  python bench_fixed_ref_softmax.py 16k,32k,64k,128k
```

Environment: NVIDIA H100 80GB HBM3 (132 SMs), driver 580.126.09, CUDA 12.9,
torch 2.11.0+cu129, CUTLASS DSL 4.4.2, nvcc 12.9.86.  Each row has density
0.125 (sparsity 0.875).  First-call wall time, including compile/load and one
execution, was 3.113 s baseline and 3.324 s chunked.

| Size | Baseline median / p90 | Chunked median / p90 | Speedup |
|---|---:|---:|---:|
| 16K | 0.344 / 0.345 ms | 0.339 / 0.345 ms | 1.015x |
| 32K | 1.344 / 1.356 ms | 1.262 / 1.310 ms | 1.065x |
| 64K | 5.292 / 5.595 ms | 4.939 / 5.788 ms | 1.071x |
| 128K | 22.345 / 24.657 ms | 21.371 / 22.168 ms | 1.046x |

The benchmark uses ordinary online softmax and identical inputs/metadata for
both variants.  It does not average away the weaker 16K and 128K results.

### 2-CTA occupancy follow-up

Command:

```bash
BENCH_WARMUP=10 BENCH_REP=50 BENCH_TRIALS=3 \
  python bench_min_blocks.py 16k,32k,64k,128k
```

The baseline 2CTA configuration retains two N64 K/V stages. Chunked 2CTA uses
one N128 K/V stage; otherwise its N128 MMA, metadata, online-softmax, and
overlap semantics are unchanged.

| Size | Baseline 1CTA | Baseline 2CTA | Chunked 1CTA | Chunked 2CTA |
|---|---:|---:|---:|---:|
| 16K | 0.345 ms | **0.292 ms** | 0.340 ms | 0.304 ms |
| 32K | 1.313 ms | **1.140 ms** | 1.263 ms | 1.181 ms |
| 64K | 5.343 ms | **4.684 ms** | 5.172 ms | 4.960 ms |
| 128K | 22.509 ms | 25.665 ms | **21.585 ms** | 25.199 ms |

Relative to chunked 1CTA, chunked 2CTA is 1.118x, 1.069x, and 1.043x faster
at 16K, 32K, and 64K, then 0.857x as fast at 128K. It also remains slower
than baseline 2CTA at every size where 2CTA is beneficial. This reproduces the
existing `bench_min_blocks.py` conclusion: independent CTA concurrency hides
latency for the smaller working sets, while at 128K its larger concurrent
random-KV footprint increases cache/TMA pressure enough to erase the gain.

## Profile attribution

One 16K `.875` launch was profiled for each path with identical stage count,
overlap mode, thread count, register launch bound, and online softmax.

| Metric | Baseline | Chunked |
|---|---:|---:|
| Nsight duration | 426.688 us | 407.136 us |
| Tensor-pipe active | 45.76% | 49.98% |
| L2 throughput | 60.98% | 64.88% |
| DRAM throughput | 4.79% | 5.03% |
| wait stalls / issue | 0.99 | 1.28 |
| long-scoreboard stalls / issue | 0.88 | 1.11 |
| barrier stalls / issue | 0.29 | 0.52 |

The tensor-pipe increase is direct evidence for the intended wider-QK effect.
The higher L2, scoreboard, and barrier pressure is consistent with issuing four
small TMA transactions per K/V stage and limits the net gain.  There was no
stage-count, overlap, occupancy, fixed-softmax, or `MIN_BLOCKS` change in the
comparison, so the gain is attributable to the chunked layout/control change,
not an accidental configuration shortcut.

## Unsupported configurations and remaining risks

The opt-in dispatch fails clearly unless all of the following hold: SM90,
forward block-sparse attention, API tile `(64,64)`, metadata block size
`(64,64)`, bf16 D=DV=128, fixed Q/K lengths divisible by 64, noncausal/nonlocal,
register-sourced PV, no packed-GQA metadata, no page table, no split-KV, no
score/mask mod, no learnable sink/qv, normal online softmax, and
`FLASH_ATTN_SM90_MIN_BLOCKS` set to either `1` or `2`.

Masked and full lists remain distinct and ordered.  With `mask_mod` unsupported
and sequence lengths block-aligned, masked-list entries need no per-token mask;
the test covers a nonempty masked list followed by a full list.  Arbitrary
partial-token tails and custom mask semantics deliberately fall back to the
default path by leaving the experiment disabled (or fail clearly if explicitly
requested).

Risks that remain:

- The two-stage chunked path rises by 65,536 B/CTA and cannot run at two
  CTAs/SM. The one-stage 2CTA specialization fits, but loses pipeline depth
  and has the documented 128K locality collapse.
- The four-transaction gather raises barrier and scoreboard pressure; this is
  the likely reason measured gain is 1.5-7.1% rather than roughly 10%.
- The benchmark uses the existing synthetic random VSA construction, not saved
  model activations.
- The floating-point reduction/accumulation order changes.  Observed output and
  LSE errors remain within the existing bf16 conventions and no tolerance was
  loosened globally.

## Final checklist

- **Metadata granularity:** N64.
- **One logical N128 group loads intended data:** yes; two reverse-list entries,
  or one entry plus a masked transport duplicate.
- **QK column order matches baseline:** yes, after invalid columns are removed.
- **Score dead before reuse:** yes, completion wait then softmax/P conversion.
- **K/V release after final consumer:** yes, QK-backed K release and PV-backed V
  release.
- **Masked/full paths:** separate list traversal is retained; custom mask_mod is
  explicitly unsupported.
- **Tail exclusion:** duplicated high half is `-inf` before max/sum/PV.
- **`.875` indices:** tested with the existing compact `full_block_idx` format.
- **Compile key:** contains the experiment flag and `MIN_BLOCKS`; internal tile
  N128 is also part of the key, covering MMA/TMA/SMEM layout changes. The
  chunked stage count is derived from `MIN_BLOCKS`.
- **Reproducible requested lengths:** measured separately above.
- **Gain source:** native N128 QK plus paired control/softmax, with fixed-ref and
  unrelated configuration changes disabled.
