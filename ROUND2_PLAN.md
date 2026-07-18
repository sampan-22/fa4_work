# Plan: fa4_new_hybrid round 2 — verify the pairing kernel's implementation and close the MFU gap

Written for a fresh agent with zero context on this repo, this bench
harness, or the FA4 CuTe DSL kernel. Read this whole document before
touching anything. A previous round already implemented and validated a
first version of this kernel; your job is NOT to build it from scratch —
it is to (a) verify with your own eyes that the round-1 implementation
actually does what it claims at the SASS/profile level, (b) find what it
left on the table, and (c) fix it. Treat round 1's *reasoning* with
suspicion and its *measurements* as reproducible facts (repro commands
below).

## 0. Background — what this is about

`lib/ursa` benchmarks Video Sparse Attention (VSA) kernels for the Omni
video model. VSA splits video tokens into 3D `4x4x4=64` token boxes
("cubes"), pools each cube to one vector, does cheap coarse attention
between pooled cubes, keeps the top-K highest-affinity cubes per query
cube (87.5% sparsity — e.g. 33 of 264 cubes at 16k tokens), and runs
real fine attention only between each query cube and its selected key
cubes, via FA4 (flash-attn-4, CuTe DSL) with a `BlockSparseTensorsTorch`
whose `full_block_idx`/`full_block_cnt` list the selected cube indices,
`block_size=(64,64)`.

The idea (from a Baseten worklog on a Blackwell VSA kernel): keep the
selection granularity at 64 (model quality depends on it), but feed the
tensor core TWO independently selected 64-token cubes as one 128-wide
K/V tile per MMA (`m64n128k16` instead of `m64n64k16`). Their B200
measurement: an `m64n64k16` retires 2^16 MACs in ~46 cycles (35% of the
~4096 MACs/cycle/SM tcgen05 peak) while `m64n128k16` retires 2^17 in
~64 cycles (50%) — i.e. a fixed per-instruction cost that wide N
amortizes.

## 1. What already exists (round 1) — do not rebuild, verify

### The working tree (IMPORTANT — do not use ~/flash-attention)

All kernel work lives in **`/fsx/sampan/fa4_work`** (git repo, 4
commits; first commit = pristine tree). It is a copy of the kuma venv's
installed `flash-attn-4 4.0.0b11` `flash_attn/cute` tree. The
`/fsx/sampan/flash-attention` checkout must NOT be used: it requires
`nvidia-cutlass-dsl==4.6.0.dev0` while this pod's venv has 4.4.2, and
its kernels fail JIT argument conversion here (`AuxData` NamedTuple).
This was discovered the hard way in round 1 — don't re-litigate it.

Environment (before running anything):

```bash
source /fsx/sampan/fa4_env.sh
```

This puts `/fsx/sampan/fa4_work` first on `PYTHONPATH` (namespace
package: `flash_attn.cute` resolves there, everything else — quack,
cutlass DSL, torch — from the kuma venv) and remaps ~15 stale
`/root/lumaverse/lib/*` editable installs to `/fsx/sampan/lumaverse/lib/*`.
Verify: `python -c "import flash_attn.cute.interface as fi; print(fi.__file__)"`
must print a path under `/fsx/sampan/fa4_work`.

### The round-1 implementation (kv_pair_factor = 2)

Design: run the SM90 forward with `tile_mn=(64,128)` while the sparse
metadata stays at `block_size=(64,64)`. The consumer/MMA/smem machinery
is the stock dense `tile_n=128` path (the QK tiled-MMA shape is
literally `(64, tile_n, 16)`, so this emits a native `m64n128k16`; the
PV GEMM was ALWAYS `m64n128k16` — `tiler_mn=(64, hdimv=128)` — in both
kernels). What changed:

- `flash_attn/cute/block_sparsity.py` — `normalize_block_sparse_config`
  gained `allow_kv_pairing` and returns `kv_pair_factor =
  tile_n // sparse_block_size_kv` (2). Index/count tensor shapes stay in
  sparse-block (64) units.
- `flash_attn/cute/block_sparse_utils.py` (bottom of file) —
  `load_paired_block_list` / `produce_block_sparse_paired_loads` /
  `consume_block_sparse_paired_loads` / `paired_tail_mask`. Producer
  walks the flat selected-cube list two at a time (in reverse, K/V
  overlapped, mirroring `load_block_list`); consumer runs
  `ceil(cnt/2)` ordinary 128-wide tiles. Odd count: the last cube is
  duplicated into both halves of the tail pair (which is consumed
  FIRST, since iteration is reversed) and `paired_tail_mask` sets its
  columns to -inf via a runtime `col_limit` (64 if odd else 128 — same
  compiled code either way). Full-block list only; mask blocks
  unsupported (VSA passes mask counts = 0).
- `flash_attn/cute/flash_fwd_sm90.py` — ctor takes `kv_pair_factor`;
  `__call__` builds "chunk" smem layouts + (64,64)-box TMA atoms;
  `load()` builds per-hdim-chunk TMA copy fns and `load_K_pair`/
  `load_V_pair`; `mma()` dispatches to the paired consumer. The key
  smem fact (verified numerically in round 1): the canonical
  `(128, hdim, stages)` SW128 staged K/V layout decomposes exactly into
  contiguous canonical `(64,64)` chunks at
  `chunk = stage*(2*hdim_chunks) + hdim_chunk*2 + row_half`, so one
  128-wide stage is filled by 4 TMA copies (2 per cube × 2 hdim chunks
  at hdim=128) all arriving on that stage's single mbarrier with
  `tx_count` unchanged (4 × 8KB = 32KB).
- `flash_attn/cute/interface.py` — plumbs `kv_pair_factor` (compile key
  + Sm90 ctor), asserts (no causal/local/score_mod/mask_mod/paged/
  varlen; `seqlen_k % 64 == 0`), and a `FLASH_ATTN_SM90_NUM_STAGES` env
  knob (default 2; 3 was tried — see measurements — no gain).

ursa side (done, registered as `fa4_new_hybrid`):
`lib/ursa/ursa/models/ray3/sparse_attn_kernels/bench/kernels/fa4_new_hybrid/`
(`__init__.py`, `forward.py`, `autograd.py`) + one import line in
`kernels/__init__.py`. Python selection is byte-identical to
`fa4_hybrid` (same coarse pool/top-K/tensors); only the FA4 call passes
`tile_mn=(64,128)`. Forward-only v1.

### Correctness status — green, keep it green

`python /fsx/sampan/fa4_work/test_kv_pairing.py` (needs the env, a GPU,
~3 min JIT): compares paired vs unpaired vs an fp32 torch reference.
Even topk, odd topk (incl. 33), topk=1 (pure duplicated pair), GQA
Hq=8/Hkv=1, MHA — ALL PASS, paired matches the reference at the same
bf16 tolerance as unpaired. Any change you make must keep this passing.

## 2. Measured facts (reproduce, don't re-derive)

### WGMMA microbenchmarks (raw inline PTX, H100, this pod)

Sources in the round-1 scratchpad are gone-able; regenerate from
`/fsx/sampan/fa4_work/` if present or rewrite (~100 lines; nvcc
`-gencode arch=compute_90a,code=sm_90a`). Two experiments were run:

Throughput (2 accumulator groups in flight, `wait_group 1`, SS
operands, 1 WG/SM): `m64n64k16` 976.6 TFLOP/s (98.6% of 990),
`m64n128k16` 978.0 (98.8%). Both saturate — no per-instruction win in
the pipelined limit.

Cadence (cycles via `clock64()`, one SM; H100 peak = 2048 MACs/cyc/SM,
so ideal is 32 cyc for n64, 64 for n128):

| regime                          | m64n64k16       | m64n128k16      |
|---------------------------------|-----------------|-----------------|
| pipelined (wait_group 1)        | 32.0 cyc (100%) | 64.0 cyc (100%) |
| drain per 8-instr k-loop (tile) | 38.9 cyc (82.2%)| 70.9 cyc (90.2%)|
| drain per instruction (serial)  | 85.3 cyc (37.5%)| 117.3 cyc (54.6%)|

The serial row is H100's analogue of Baseten's 46/64-cycle numbers
(fixed ~53-cycle chain cost here vs ~30 there) — the premise DOES hold
on Hopper, but the FA4 QK k-loop pipelines 8 k16-instructions per tile
and drains once per tile (before softmax reads acc_S), so the in-kernel
regime is the "tile" row: pairing buys ~8 points of QK pipe utilization
(82→90%), on ~half the FLOPs (PV is already n128 in both kernels).
Predicted end-to-end gain from the MMA pipe alone: ~4–5%.

### Kernel measurements (fine attention only, exact bench shapes)

`python /fsx/sampan/fa4_work/bench_fine_only.py 16k,32k,64k,128k`
(B=1, Hq=8, Hkv=1, D=128, bf16, uniform-random sorted selections;
FLOPs = `4·64²·128·num_blocks·topk·8`, peak 990; reproduced twice
within ~2%):

| size | unpaired ms / MFU | paired ms / MFU | paired speedup |
|------|-------------------|-----------------|----------------|
| 16k  | 0.347 / 42.5%     | 0.334 / 44.3%   | +3.9%          |
| 32k  | 1.265 / 46.7%     | 1.225 / 48.2%   | +3.3%          |
| 64k  | 5.303 / 44.5%     | 4.904 / 48.2%   | +8.1%          |
| 128k | 22.903 / 43.2%    | 21.344 / 46.3%  | +7.3%          |

`FLASH_ATTN_SM90_NUM_STAGES=3` for the paired path: no change (±1%).

End-to-end ursa bench (`t_vsa` includes the Python coarse stage — at
16k the FA4 kernel is only ~0.35ms of the 0.74ms t_vsa) has a ±5%
noise floor on this pod (the DENSE baseline swings that much between
adjacent rows), so use `bench_fine_only.py` for kernel comparisons:

```bash
source /fsx/sampan/fa4_env.sh
python -m ursa.models.omni.sparse_attn_kernels.bench.bench \
    --task t2v --sizes 16k,32k,64k,128k --ulysses 8 --cube 64 \
    --vsa fa4_hybrid,fa4_new_hybrid --warmup 10 --rep 50
```

Reference points: `fa4_hybrid_b128` (cube=128, quality-risky) = 43.4%
MFU at 128k (t_vsa basis). Dense FA4 SM90 hdim128 ≈ 75% MFU.

## 3. The gap, quantified — this is your target

If the QK pipe ran at the tile-regime 90.2% and PV at ~100% with all
non-MMA work hidden, the paired kernel's ceiling would be ~95% MFU. It
measures 44–48%. So ~half the kernel's time is NOT hidden tensor-core
work. Candidate accounting per 128-wide pair-tile (tile_m=64,
hdim=128, one consumer warpgroup):

- MMA: QK + PV = 2 × 64·128·128 MACs = 2.10M MACs ≈ 1024 cycles at peak.
- softmax exp2: 64×128 = 8192 MUFU.EX2 ≈ 512 cycles at 16/cyc/SM.
- P conversion (f32→bf16), row max/sum reductions, acc_O rescale
  (64×128 f32 FMA), pipeline waits/releases, wgmma fences: several
  hundred more cycles.

A single consumer warpgroup must overlap ALL of that behind 1024 cycles
of async MMA. The unpaired kernel has the same per-element scalar work
but runs 2 CTAs/SM (80KB smem each) = two independent consumer WGs
interleaving on the SM, which hides scalar work better; the paired
kernel is 144KB/CTA → 1 CTA/SM → one consumer WG. That structural
difference is the prime suspect for why pairing nets only 3–8% instead
of the MMA-side ~5% PLUS the per-tile softmax/rescale savings.

## 4. Task, in order

### 4.1 Verify round 1 did what it says (fresh eyes, no trust)

- Dump the SASS/PTX of the paired kernel and confirm the QK GEMM is a
  single `wgmma.mma_async.sync.aligned.m64n128k16` per k-step, not two
  n64s or something degenerate. `CUTE_DSL_KEEP_PTX=1` and/or
  `CUTE_CUBIN_PATH=<dir>` (see `flash_attn/cute/cute_dsl_utils.py` /
  `cache_utils.py`) while running `bench_fine_only.py 16k`; grep the
  PTX, `nvdisasm`/`cuobjdump -sass` the cubin (look for QGMMA/HGMMA
  64x128).
- Confirm launch shape + occupancy: paired should be 256 threads/CTA,
  ~144KB smem, 1 CTA/SM; unpaired 2 CTAs/SM. (`ncu` if available —
  `which ncu` — else `cudaFuncGetAttributes`-style inspection or
  `CUDA_LAUNCH_BLOCKING` + torch profiler shows grid/block.)
- Profile one shape (64k is the cleanest signal). If `ncu` works:
  `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active`,
  MUFU throughput, warp stall reasons (`smsp__average_warps_issue_stalled_*`),
  achieved occupancy, for BOTH paired and unpaired. The stall breakdown
  decides which hypothesis below you chase first. If ncu is not
  available, do ablations instead: temporarily hack the fa4_work tree
  (it's a git repo — branch) to (a) skip exp2/softmax (garbage output,
  timing only), (b) skip acc_O rescale, (c) skip P conversion — each
  ablation's time delta tells you what that stage costs unhidden.

### 4.2 Diagnose and fix — ranked hypotheses

1. **Single consumer WG can't hide softmax (occupancy).** Cheapest
   experiment: asymmetric K/V stage counts. K stages=2, V stages=1
   gives Q16 + K64 + V32 = 112KB < 113.6KB → **2 CTAs/SM again**. The
   code currently shares `num_stages` for K and V
   (`flash_fwd_sm90.py` `__call__`, `_get_shared_storage_cls`,
   pipeline creation, chunk layouts) — split it into `num_stages_k` /
   `num_stages_v`. V single-buffering serializes V-load vs PV of the
   previous pair, but K stays double-buffered so QK keeps flowing;
   whether the trade wins is exactly what to measure.
2. **FA3-style ping-pong inside one CTA.** Two consumer warpgroups over
   the SAME shared K/V pipeline, alternating pairs (WG0 takes pairs
   0,2,4…, WG1 takes 1,3,5…), each with its own softmax stats + acc_O,
   merged at the end (intra-CTA split-K over the selected list: merge
   O with per-WG row_max/row_sum, standard LSE combine, via smem).
   smem unchanged; tile_m stays 64. This is the structural fix if (1)
   is insufficient — bigger surgery, prototype only if profiling shows
   tensor-pipe idle waiting on softmax.
3. **Per-m_block prologue/epilogue amortization.** Each m_block loads
   Q, runs only ceil(topk/2) tiles (17 at 16k), and runs a full
   epilogue. Dense amortizes over 100+ tiles. Check what fraction of
   time is prologue/epilogue (first/last-tile timestamps via one
   `cute.printf`-guarded clock, or ncu). If large, persistent CTAs /
   processing multiple m_blocks per CTA without re-entering the
   scheduler loop is the lever.
4. **Producer-side TMA issue overhead.** 4 small copies per stage vs 1
   (and 2 per stage for V). Unlikely (single producer warp, ~20 cyc
   issue each), but cheap to bound: time a variant with topk halved vs
   full to see if load-issue scales into the critical path.
5. **n192 (three cubes per tile).** Valid wgmma N; throughput bench
   showed n192 at 94–97%. Only worth it if (1)/(2) get the kernel to
   where per-tile overhead again dominates.

### 4.3 Keep honest

- Rerun `test_kv_pairing.py` after every kernel change (all cases must
  PASS).
- Compare with `bench_fine_only.py` (50 reps, same GPU, same process
  for both variants); ignore end-to-end deltas smaller than the ±5%
  bench noise.
- The pristine baseline is git commit `2d4559f` in `/fsx/sampan/fa4_work`
  ("pristine flash-attn-4 4.0.0b11 cute tree from kuma venv"); the
  round-1 pairing implementation is `b08c654..c0d2edf`. Branch from
  HEAD; commit as you go.

## 5. Gotchas learned in round 1 (will bite you otherwise)

- CuTe DSL rewrites ALL `for` loops: building or indexing a Python
  list inside a loop needs `cutlass.range_constexpr(...)`, not
  `range(...)`.
- `compile_key` in `interface.py` must contain anything that changes
  codegen (kv_pair_factor is already in it; add your new knobs).
- The consumer walks lists in REVERSE; producer/consumer pairing order
  must match exactly (see `load_paired_block_list` docstring).
- `first_half_block_overlap` calls `mask_fn` unconditionally — the
  paired path passes `paired_tail_mask` with a runtime col_limit that
  no-ops for even counts; don't pass None there.
- Sparse tensors are 4D `[B, H=8, nq, nkv]` (full Q-head dim), which
  silently disables pack_gqa in the interface — that's the existing
  fa4_hybrid behavior, keep it.
- WGMMA smem descriptors need the canonical SW128 stage layout; you
  cannot TMA into "half of a 128-row stage" as one box — that's why
  the (64,64) chunk carving exists. If you change smem layouts,
  re-verify the chunk offset math (round 1 did it by evaluating both
  layouts on the host and comparing offsets — 10-line script).

## 6. Deliverable

- The verification results from 4.1 (SASS instruction confirmed or
  not; profile/ablation breakdown of where the paired kernel's cycles
  go).
- At least hypothesis (1) implemented and measured; (2) if profiling
  says softmax-hiding is the limiter and (1) didn't recover it.
- Updated `bench_fine_only.py` table (paired-v2 vs paired-v1 vs
  unpaired) + the end-to-end ursa table for the record.
- `test_kv_pairing.py` green on the final tree.
- A short writeup: measured cycle budget per pair-tile (MMA vs softmax
  vs other), which fix moved the number and by how much, and what the
  realistic MFU ceiling for tile_m=64 sparse attention on H100 is.
