# Plan: fa4_new_hybrid round 3 — two consumer warpgroups + overlap for the paired VSA kernel

Written for a fresh agent with zero context. Read this whole document
before touching anything. Two previous rounds exist: round 1 built and
validated a "KV pairing" kernel (two sparse 64-token cubes per 128-wide
MMA tile) that gains 3–8% over the baseline; round 2's plan
(`ROUND2_PLAN.md`, same directory) was a broad diagnosis plan. This
round has ONE focused mission:

**Get two consumer warpgroups' worth of concurrency (plus the existing
intra-warpgroup overlap) working for the paired kernel, because the
measured cycle budget says roughly 20% more throughput is sitting there
— unless you find a concrete issue that makes it unreachable, in which
case document that issue precisely.**

Keep everything that exists. Branch; don't delete.

## 0. Background (condensed)

VSA fine attention for the Omni video model: tokens in 4x4x4=64-token
"cubes"; a coarse stage picks top-K cubes per query cube (e.g. 33 of
264 at 16k tokens); fine attention runs each 64-row query cube against
its selected cubes via FA4 (flash-attn-4 CuTe DSL, SM90/H100) with
`BlockSparseTensorsTorch` (`full_block_idx`/`full_block_cnt`,
`block_size=(64,64)`, full blocks only, mask lists all zero).
Shapes: B=1, Hq=8, Hkv=1 (GQA), D=128, bf16. Selection is per Q-head
(sparse tensors are `[B, 8, nq, nkv]`, which silently disables
pack_gqa in the interface — existing behavior).

Round 1 ("KV pairing", `kv_pair_factor=2`): kernel runs
`tile_mn=(64,128)`; each 128-wide K/V pipeline stage is filled with TWO
independently selected cubes by 4 (64,64)-chunk TMA copies on one
mbarrier; the consumer runs ceil(topk/2) ordinary 128-wide tiles; odd
counts duplicate the trailing cube and mask its columns.

## 1. Environment and repos (exact, do not improvise)

```bash
source /fsx/sampan/fa4_env.sh
python -c "import flash_attn.cute.interface as fi; print(fi.__file__)"
# must print a path under /fsx/sampan/fa4_work
```

- Kernel tree: `/fsx/sampan/fa4_work` (git). Branch `kv-pairing-round1`
  == `master` = round-1 state. Create your branch from it (e.g.
  `round3-2wg`). Commit as you go.
- Do NOT use `/fsx/sampan/flash-attention` (needs cutlass-dsl 4.6; this
  pod has 4.4.2 and it fails JIT). fa4_work is a copy of the venv's
  working flash-attn-4 4.0.0b11 tree.
- ursa bench kernel `fa4_new_hybrid` already registered:
  `lumaverse/lib/ursa/.../bench/kernels/fa4_new_hybrid/` (branch
  `sampan/fa4-new-hybrid` in lumaverse). Python side identical to
  `fa4_hybrid` except `tile_mn=(64,128)`.
- Correctness gate (must stay green after every change):
  `python /fsx/sampan/fa4_work/test_kv_pairing.py`
- Kernel-level timing (use THIS for comparisons; the end-to-end ursa
  bench has a ±5% noise floor and a large Python coarse stage):
  `python /fsx/sampan/fa4_work/bench_fine_only.py 16k,32k,64k,128k`

## 2. Measured facts motivating this round

WGMMA cycle measurements on this H100 (peak 2048 MACs/cycle/SM;
`clock64()`, one SM; SS operands):

| regime                              | m64n64k16        | m64n128k16       |
|-------------------------------------|------------------|------------------|
| pipelined (wait_group 1, 2 groups)  | 32.0 cyc (100%)  | 64.0 cyc (100%)  |
| drain per 8-instr k-loop (per tile) | 38.9 cyc (82.2%) | 70.9 cyc (90.2%) |
| drain per instruction               | 85.3 cyc (37.5%) | 117.3 cyc (54.6%)|

Fine-attention kernel, exact bench shapes (FLOPs =
`4*64^2*128*num_blocks*topk*8`, peak 990 TFLOP/s):

| size | unpaired (fa4_hybrid path) | paired (round 1) |
|------|----------------------------|------------------|
| 16k  | 0.347 ms / 42.5%           | 0.334 ms / 44.3% |
| 32k  | 1.265 ms / 46.7%           | 1.225 ms / 48.2% |
| 64k  | 5.303 ms / 44.5%           | 4.904 ms / 48.2% |
| 128k | 22.903 ms / 43.2%          | 21.344 ms / 46.3%|

The gap analysis, per 128-wide pair-tile (tile_m=64, hdim=128):

- MMA (QK m64n128k16 x8 + PV m64n128k16 x8): ~2.10M MACs ~= 1024 cycles.
- softmax exp2: 64x128 = 8192 MUFU.EX2 ~= 512 cycles (16/cyc/SM).
- P f32->bf16 conversion, row max/sum reductions, acc_O rescale
  (64x128 f32), pipeline waits, wgmma fence/drain: several hundred more.

So non-MMA work is comparable to MMA work per tile. The tensor pipe can
run at 90–100% (table above); the kernel runs at 44–48%. With ONE
consumer warpgroup per SM (see below), every cycle of scalar/SFU work
that the intra-warpgroup overlap fails to hide is a dead tensor-pipe
cycle. Dense FA4 SM90 at hdim 128 — which runs TWO consumer warpgroups
— reaches ~75% MFU on this pod. That difference is the ~20% (or more)
this round is after.

Why the paired kernel has only one consumer WG per SM:
`tile_m=64` -> `atom_layout_mnk=(tile_m//64,1,1)=(1,1,1)` -> one
128-thread MMA warpgroup + one producer warpgroup = 256 threads/CTA.
SMEM = Q 16KB + K 2x32KB + V 2x32KB = 144KB -> 1 CTA/SM. The UNPAIRED
kernel is the same structure but 80KB -> 2 CTAs/SM, i.e. two
independent consumer WGs interleaving on the SM — and it still only
hits 43–47%, which tells you cross-CTA interleaving alone is not
sufficient; understand why as part of this round (per-m_block
prologue/epilogue? scheduler? L2?) before assuming any given fix works.

Also verify, don't assume, that intra-warpgroup overlap is actually
engaged and effective in the paired path: `interface.py` defaults
`intra_wg_overlap=True` when `tile_mn` is explicit (check
`FwdConfig`), the paired consumer uses
`mma_one_n_block_intrawg_overlap` + `first_half_block_overlap` /
`last_half_block_overlap` (see `flash_fwd_sm90.py`), and
`use_scheduler_barrier` is False for one WG. Confirm in PTX/SASS
(`CUTE_DSL_KEEP_PTX=1`, `CUTE_CUBIN_PATH=<dir>`; look for
`wgmma.wait_group 1` between tiles, and a single
`wgmma.mma_async...m64n128k16` per k-step) that the overlap structure
survived compilation. If something is broken there, fixing it may be
the whole win — that is the "unless we are running into an issue
there" branch of this mission.

## 3. Candidate designs (ranked by effort; measure after each)

### A. Two CTAs/SM via asymmetric K/V staging (small change)

Q 16KB + K 2x32KB + V 1x32KB = 112KB < 113.6KB -> 2 CTAs/SM, giving two
independent consumer WGs per SM with zero consumer-logic changes.
Cost: V single-buffered — the producer's V load for pair j+1 stalls
until PV of pair j releases the stage; K stays double-buffered so QK
keeps flowing. Touch: `flash_fwd_sm90.py` (`num_stages` -> separate
`num_stages_k`/`num_stages_v` in `__call__` layouts, chunk layouts,
`_get_shared_storage_cls` mbar counts, both pipeline creations,
`tma_copy_bytes` unchanged) and `interface.py` (knob + compile key).
The unpaired kernel's 2-CTA result (43–47%) is your realism check on
this option — cross-CTA interleaving hides softmax only partially. If
A lands ~5% it is still worth keeping; do not stop there.

### B. Intra-CTA ping-pong: two consumer WGs split the pair list (the real target)

One CTA, 384 threads (1 producer WG + 2 consumer WGs), SMEM unchanged
(shared K/V pipeline, 144KB; stages can go to 3–4 within 227KB if it
helps alternation). WG0 consumes pairs 0,2,4,...; WG1 consumes pairs
1,3,5,... of the SAME query cube's list. Each WG keeps its own softmax
stats (row_max/row_sum) and its own acc_O; at the end the two partials
merge exactly like SplitKV combine (rescale both O by
exp2(row_max - global_max), add, sum row_sums) via smem + a named
barrier, then one WG runs the epilogue. While WG0 does softmax/rescale
for its tile, WG1's wgmma occupies the tensor pipe, and vice versa —
this is the FA3 ping-pong idea adapted to a sparse per-cube list.

Implementation notes / where things live:

- `flash_fwd_sm90.py::mma()` — today `tidx` spans one WG; you need
  `num_wg_mma=2` WITHOUT changing the tiled MMA shape (both WGs compute
  full 64-row tiles independently; do NOT use `atom_layout_mnk=(2,..)`,
  that is the cooperative-same-tile path and needs tile_m=128).
  Cleanest is probably: keep tiled MMA as-is, launch 2 consumer WGs,
  give each its own `kv_consumer_state` starting offset (WG index) and
  stride-2 `advance()` over stages, its own Softmax/acc_O, and guard
  the shared-Q pipeline release + epilogue.
- Pipeline consumer groups: `pipeline_k/v` are created with
  `consumer_group=mma_warps` (all consumer warps). With stride-2
  consumption each stage is waited/released by only ONE of the two WGs
  — check `PipelineTmaAsync` consumer arrive counts
  (`flash_attn/cute/pipeline.py` + cutlass.pipeline) and set the
  consumer group / arrive count so a single-WG release fully frees a
  stage. This is the fiddliest part; get it right or you hang.
- Producer side needs NO change (it already just fills stages in
  order); `load()`/`produce_block_sparse_paired_loads` untouched.
- Odd/even split details: pairs are consumed in REVERSE list order and
  the tail (possibly duplicated+masked) pair is first — decide which WG
  owns it by parity of `num_pairs`, keep producer/consumer agreement
  (see `load_paired_block_list` docstring).
- Empty/short lists: num_pairs can be 1 (one WG idles) or 0; the idle
  WG must still produce neutral stats for the merge (row_max=-inf,
  row_sum=0, acc_O=0) and hit all barriers.
- Merge: reuse `sO` (16KB) or a small new smem buffer for the partial;
  stats via smem array; `NamedBarrierFwd` has spare enum slots.
- Registers: 2 consumer WGs + 1 producer at 384 threads — revisit
  `num_mma_regs`/`num_producer_regs` ({1:(256,56),2:(240,24),...} table
  in `__call__`; you now genuinely have num_wg_mma=2).

Expected payoff if the pipe analysis holds: fine-stage MFU from ~46–48%
toward 55–65% (~15–30% speedup). If measured payoff is small, profile
WHERE the second WG's time goes before iterating (ncu if available:
tensor-pipe active %, MUFU throughput, stall reasons; else git-branch
ablations: skip exp2 / skip rescale / skip P-convert and time).

### C. tile_m=128 cooperative 2-WG via head-shared selection (ceiling probe; algorithm asterisk)

If the two heads' selections were IDENTICAL, 128 query rows could share
one key list and the bone-stock dense 2-WG path would apply. You can
get this TODAY without pack_gqa by Python-side row packing: with
Hkv=1, all 8 Q-heads attend the same K/V, so fold pairs of heads into
rows — permute q from (B,S,8,D) to rows ordered
[cube c, head 2g, 64 rows; cube c, head 2g+1, 64 rows] giving
Hq'=4 "virtual heads" and 128-row blocks that each cover ONE cube;
sparse tensors become [B,4,nq,nkv] with block_size=(128,64) and a
selection shared across each head pair (e.g. top-K of the mean of the
two heads' coarse scores). Then `tile_mn=(128,128)` + kv_pair_factor=2
runs the existing cooperative 2-WG machinery unchanged (Q 32KB + K/V
128KB = 176KB, 1 CTA/SM, 2 consumer WGs + scheduler barrier). Un-permute
the output after.

This CHANGES WHAT IS COMPUTED (selection shared per head-pair instead
of per head) — it is a perf ceiling probe and a quality question for
the model owner, not a drop-in. Build it as a separate ursa bench
kernel (e.g. `fa4_new_hybrid_m128`) so it can be A/B'd. It is also the
cheapest way to learn what 2 cooperative WGs are worth on this exact
workload before/while doing the surgery in B. Correctness check: fp32
reference with the shared selection (adapt `test_kv_pairing.py`).

## 4. Gotchas (learned rounds 1–2; will bite you)

- CuTe DSL rewrites ALL `for` loops: building/indexing a Python list
  inside a loop needs `cutlass.range_constexpr(...)`.
- Anything that changes codegen must go into `compile_key` in
  `interface.py` (kv_pair_factor already is; add num_stages_v, ping-pong
  flags, etc.).
- Producer and consumer walk lists in REVERSE and must agree exactly on
  pairing and order.
- `first_half_block_overlap` calls `mask_fn` unconditionally — the
  paired path passes `paired_tail_mask` with a runtime `col_limit`
  (no-op when even); don't pass None there.
- WGMMA smem descriptors need canonical SW128 stage layouts; TMA cannot
  write "half of a 128-row stage" as one box — that's why the
  (64,64)-chunk carving exists (`chunk = stage*4 + hdim_chunk*2 +
  row_half` at hdim 128). If you touch smem layouts, re-verify offsets
  on the host (10-line layout-evaluation script).
- End-to-end `bench.py` noise is ±5% (the dense baseline itself swings
  that much); only `bench_fine_only.py` deltas count.
- Interactive `git rebase -i` etc. unavailable; plain commits fine.

## 5. Deliverables

- Verification result for the current paired kernel's overlap
  structure (SASS evidence that m64n128k16 + wait_group(1) overlap is
  as designed, or the bug you found).
- Option A implemented + measured (small, do it regardless).
- Option B implemented + measured, OR a precise writeup of the blocker
  (e.g. pipeline arrive-count semantics, register pressure, smem) with
  numbers. B is the core of this round.
- Option C measured if time permits (flagged as algorithm-changing).
- Updated `bench_fine_only.py` table: unpaired vs paired-r1 vs each new
  variant, all four sizes; `test_kv_pairing.py` green on the final
  branch (extend it for any new variant).
- Short writeup: cycles budget per tile (MMA vs softmax vs other, from
  profile or ablation), what each option bought, and the realistic MFU
  ceiling for cube-64 sparse attention on H100.
