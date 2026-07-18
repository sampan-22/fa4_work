# Plan: fa4_hybrid_v2 — raise VSA fine-attention MFU on H100, clean slate

Written for a fresh agent with zero context. Read this whole document
before touching anything. This is a **restart**: a previous attempt
(KV-pairing, packing 2 cubes into one m64n128 MMA tile) was implemented,
measured, and **ruled out by the owner — do not pursue it, do not build
on it, do not "improve" it**. Its measurements and microbenchmarks are
kept below as established findings so you don't re-derive them. Ignore
`ROUND2_PLAN.md` and `ROUND3_PLAN.md` in this repo — they belong to the
abandoned experiment.

## 0. Mission

`lib/ursa` (in the lumaverse monorepo) benchmarks Video Sparse Attention
(VSA) kernels for the Omni video model. VSA splits video tokens into 3D
`4x4x4=64`-token boxes ("cubes"), pools each cube, does cheap coarse
attention between pooled cubes, keeps the top-K key cubes per query cube
(87.5% sparsity — e.g. 33 of 264 cubes at 16k tokens), and runs real
fine attention only on the selected pairs via FA4's SM90 CuTe-DSL
forward kernel with `BlockSparseTensorsTorch` (`full_block_idx` /
`full_block_cnt`, `block_size=(64,64)`, full blocks only, mask lists
empty).

The fine stage today (`fa4_hybrid`) runs at **42–47% MFU** (kernel-only,
H100, bf16, peak 990 TFLOP/s). Dense FA4 forward on the same GPU/head
geometry reaches roughly **~75% MFU**. The mission: **find where those
cycles actually go, then close a meaningful part of that gap** — while
keeping the algorithm bit-identical: cube=64 selection granularity,
per-query-cube top-K, same attended set, forward only. Output must
match `fa4_hybrid` to bf16 tolerance on identical inputs/selection.

Success bar: >52% fine-only MFU at 128k (baseline 43.2%), with the
correctness harness passing. Stretch: 60%+.

## 1. Hard constraints — read before forming any plan

1. **No KV pairing / no widening the MMA tile by packing two selected
   cubes.** Already tried (branch `kv-pairing-round1` in this repo):
   only +3–8% fine-only, and the owner has ruled the whole direction
   out. A raw-PTX microbench (finding F1 below) explains why the
   Blackwell rationale doesn't transfer to Hopper.
2. Selection stays cube=64 per query cube. Anything that changes
   *which* K/V tokens a query attends to (bigger cubes, shared
   selection across heads, unions of lists) changes model quality and
   is out of scope for the main deliverable. (A clearly-labeled
   ceiling *measurement* that breaks this is allowed as data, never as
   the deliverable kernel.)
3. Forward only.
4. Kernel work happens in `/fsx/sampan/fa4_work` (a git-tracked copy of
   the venv's flash-attn-4 4.0.0b11 `flash_attn/cute` tree). Do NOT
   use `/fsx/sampan/flash-attention` — that checkout needs
   `nvidia-cutlass-dsl==4.6.0.dev0`, the venv has 4.4.2, and its
   `AuxData` NamedTuple fails JIT arg conversion. Already verified;
   don't burn time on it.
5. Start from the **pristine** commit, not master:
   ```bash
   cd /fsx/sampan/fa4_work
   git checkout -b v2-clean 2d4559f          # pristine b11 tree, no pairing code
   git checkout master -- CLEAN_SLATE_PLAN.md bench_fine_only.py test_kv_pairing.py
   ```
   The two harness files contain paired-path arms that don't exist on
   this branch — adapt them (§4), don't run them as-is.

## 2. Established findings — treat as ground truth, do not re-measure

**F1 — WGMMA instruction shape is NOT the bottleneck.** Raw inline-PTX
microbench on this H100 (`wgmma.mma_async.m64nNk16.f32.bf16.bf16`, SS
operands): pipelined (2 accumulator groups, `wait_group 1`) —
m64n64k16 hits **98.6%** of the 990 TFLOP/s peak, m64n128k16 98.8%.
Per-instruction serial drain: n64 = 85.3 cyc (37.5% of 2048
MACs/cyc/SM), n128 = 117.3 cyc (54.6%) — that mirrors Baseten's B200
numbers, but FA4's k-loop pipelines 8 WGMMAs per drain, so the regime
that matters (8 back-to-back + `wait_group 0`) is 82.2% (n64) vs 90.2%
(n128). Conclusion: the win channel on Hopper is **concurrency and
per-tile overhead**, not tensor-core instruction efficiency. Bench
generators: scratchpad `gen_wgmma_bench.py` / `gen_wgmma_cycles.py`
(build with `nvcc -gencode arch=compute_90a,code=sm_90a`).

**F2 — the structural handicap vs dense.** Block-sparse VSA forces
`tile_m = sparse_block_size_q = 64`, and in `flash_fwd_sm90.py` the
consumer warpgroup count is `tile_m // 64` → **one consumer warpgroup
per CTA**. Dense FA4 runs `tile_m=128` → two consumer WGs ping-ponging
softmax against MMA. SMEM at (64,64,hdim128,2 stages) is ~80KB → 2
CTAs/SM, so the SM does host two consumer WGs — but from *different*
CTAs with separate producers/pipelines, and measured MFU is still
42–47%. So "two consumer WGs somewhere on the SM" is demonstrably not
sufficient; whatever you build must explain (with profile data) why
cross-CTA interleaving underperforms dense's intra-CTA ping-pong before
trusting a fix.

**F3 — per-tile cycle budget (hdim=128, per 64×64 selected tile).**
QK MMA = 64·64·128/2048 = 256 cyc; PV = 256 cyc; exp2 = 4096 MUFU ops
@ 16/cyc = 256 cyc; plus row-max/row-sum reductions, P fp32→bf16
convert, acc_O rescale, pipeline barrier waits. Non-MMA work is
comparable to MMA time, so a single consumer WG can only hide it via
the intra-WG overlap path (`mma_one_n_block_intrawg_overlap`,
`first_half_block_overlap`/`last_half_block_overlap` in
`flash_fwd_sm90.py`). **Nobody has yet verified in SASS that this
overlap actually engages for the block-sparse path** — that is your
first job (§5).

**F4 — measurement noise.** The end-to-end ursa bench (`bench.py`)
swings ±5% run-to-run (the dense baseline itself moved 2.7–5% between
identical rows). Kernel comparisons must use the fine-only harness
(§4). E2E numbers are only for the final table.

**F5 — misc.** `num_stages=3` measured no faster than 2 (in the paired
config; unknown elsewhere — cheap to retest for yours). Known-good
e2e forward latencies (fa4_hybrid, cube=64, ulysses 8): 16k 0.755 ms,
32k 2.203 ms, 64k 7.830 ms, 128k 26.919 ms.

### Baseline fine-only numbers your kernel must beat

`bench_fine_only.py` (unpaired `fa4_hybrid` path; Hq=8, Hkv=1,
hdim=128, bf16; FLOPs = `4·64²·128·nkv·topk·8`; peak 990 TFLOP/s):

| size | nkv cubes | topk | fa4_hybrid MFU |
|---|---:|---:|---:|
| 16k  | 264  | 33  | 42.5% |
| 32k  | 528  | 66  | 46.7% |
| 64k  | 1056 | 132 | 44.5% |
| 128k | 2160 | 270 | 43.2% |

(For reference only — the abandoned paired kernel got 44.3/48.2/48.2/
46.3; `fa4_hybrid_b128` cube=128 quality-risky reference is 43.4% at
128k. Both are bars to clear incidentally, not targets.)

## 3. Environment (known-working, exact)

```bash
source /fsx/sampan/fa4_env.sh
python -c "import flash_attn.cute.interface as fi; print(fi.__file__)"
# MUST print a path under /fsx/sampan/fa4_work — stop and fix if not.
```

`fa4_env.sh` puts `/fsx/sampan/fa4_work` first on PYTHONPATH (namespace
shim — site-packages `flash_attn` has no `__init__.py`, so `flash_attn.
cute` resolves to fa4_work) and remaps ~15 stale `/root/lumaverse/lib/*`
editable paths to `/fsx/sampan/lumaverse/lib/*`. E2E bench:

```bash
python -m ursa.models.omni.sparse_attn_kernels.bench.bench \
    --task t2v --sizes 16k,32k,64k,128k --ulysses 8 --cube 64 \
    --vsa fa4_hybrid --warmup 5 --rep 20
```

Kernel registry: `lumaverse/lib/ursa/ursa/models/ray3/sparse_attn_kernels/
bench/kernels/` (self-registering; `fa4_hybrid.py` is the Python-side
coarse-pool + top-K + `BlockSparseTensorsTorch` construction — reuse
verbatim). There is an existing `fa4_new_hybrid/` subpackage there from
the abandoned experiment — leave it alone; register yours as
`fa4_hybrid_v2` in a new sibling subpackage (§8).

## 4. Build the measurement rig FIRST (before any kernel edit)

The previous rounds' biggest process failure: optimizing against
arithmetic models instead of profiles. Deliverable #1 is a **baseline
attribution report** for stock `fa4_hybrid`, produced before you change
a line of kernel code.

1. **Correctness harness** — adapt `test_kv_pairing.py` into
   `test_fine_v2.py`: strip the paired arm, keep (your kernel) vs
   (stock fa4_hybrid path) vs fp32 torch reference on identical
   synthetic Q/K/V/top-K (same seed). Cases: even topk, odd topk,
   topk=1, GQA 8/1, MHA, at least two sizes. Bit-different softmax
   orderings are fine; assert max-abs error vs fp32 ref is ≤ ~2× the
   stock kernel's own error.
2. **Fine-only perf harness** — adapt `bench_fine_only.py` (strip the
   paired arm; keep SIZES = {16k:(264,33), 32k:(528,66), 64k:(1056,132),
   128k:(2160,270)}, the FLOP formula, warmup + many-rep CUDA-event
   timing). Add a kernel-selector flag so baseline and v2 run in one
   invocation, interleaved, to cancel clock drift.
3. **SASS/PTX inspection recipe** — the DSL can keep intermediates
   (`CUTE_DSL_KEEP_PTX=1`; there is also a cubin-path knob — grep the
   installed `cutlass` DSL package for the exact env var names rather
   than trusting these). Disassemble with `cuobjdump -sass` /
   `nvdisasm`. You need this for §5.1.
4. **Profile recipe** — try `ncu` first:
   `ncu --section SchedulerStats --section WarpStateStats --section
   Occupancy --section SpeedOfLight ...` on a bench_fine_only run
   (use `--launch-count`/`--kernel-name` to hit the fine kernel).
   Key questions: tensor-pipe active % (find the exact metric via
   `ncu --query-metrics | grep -i tensor`), warp-stall breakdown for
   the consumer WG, achieved CTAs/SM. If perf counters are blocked on
   this pod (`ERR_NVGPUCTRPERM`), fall back to **ablation
   attribution**: env-guarded hack variants of the kernel that delete
   work to attribute time — (a) skip exp2 (identity "softmax"),
   (b) skip PV accumulation, (c) skip rescale/reductions. Wrong
   results, valid timing deltas. Never commit the hacks to the real
   path; guard with env vars and delete before the final diff.

## 5. Investigation order (read code with these questions, then profile)

Files, all under `/fsx/sampan/fa4_work/flash_attn/cute/`:
`interface.py` (compile_key gate, `FlashAttentionForwardSm90(...)`
ctor args, `normalize_block_sparse_config` call — `block_size=(tile_m,
tile_n)` must stay (64,64)); `flash_fwd_sm90.py` (`_get_tiled_mma`,
`_get_shared_storage_cls`, the producer `load()` loop, the consumer
`mma()` loop and its intra-WG overlap branches); `block_sparse_utils.py`
(`produce_block_sparse_loads`, `load_block_list` — note both walk the
selected list in **reverse**); `block_sparsity.py`
(`normalize_block_sparse_config`).

1. **Does intra-WG overlap engage for the block-sparse path?** Read
   which `mma_one_n_block` variant the block-sparse branch actually
   selects, then confirm in SASS that per-n-block WGMMA groups overlap
   softmax of the previous block (look for `wgmma.wait_group 1`-style
   deferred waits rather than immediate `wait_group 0` after every QK
   group). If overlap silently doesn't engage (a compile-time branch
   on q_stage / sparsity / mask presence), that alone could be the
   whole gap — fix that before anything structural.
2. **Profile stock fa4_hybrid** (§4.4) at 32k and 128k. Write the
   baseline report: where do the consumer WG's stalls sit (MMA wait?
   smem barrier? MUFU throughput? issue-bound scalar stretch?), is the
   producer ever the limiter, are 2 CTAs/SM actually resident.
3. Only then pick a direction from §6 — justified by the profile, not
   by this document's ordering.

## 6. Candidate directions (pick by evidence)

- **A. Fix overlap if broken** (§5.1). Cheapest possible win; do first
  if SASS shows the block-sparse path serializing softmax after MMA.
- **B. Two consumer WGs per CTA over one query cube's list** —
  intra-CTA ping-pong: `tile_m` stays 64 rows *of output per WG*? No —
  keep the sparse block (64,64); give the CTA 2 consumer WGs that
  split the selected-block list (WG0 takes blocks 0,2,4…, WG1 takes
  1,3,5…) over one shared K/V pipeline, each keeping private softmax
  stats + acc_O, merged at the end SplitKV-style (LSE merge via smem +
  a named barrier). Dangers (learned the hard way in the abandoned
  round, still apply): pipeline consumer-arrive counts must match the
  new consumer thread count or it hangs; reverse-order list walking
  means "the tail block" is consumed first; an idle WG on short lists
  must contribute neutral stats; register budget for 2 accumulators +
  producer WG in 384 threads.
- **C. Shrink per-tile overhead.** If the profile says issue-bound
  scalar/SFU work: full blocks need no masking at all in this workload
  (mask lists are empty) — confirm no mask function is being applied
  per tile; check P convert + rescale scheduling; retest num_stages.
- **D. Occupancy/staging.** If smem or CTA residency shows up:
  asymmetric K/V staging (K double-, V single-buffered) or similar to
  hold 2 CTAs/SM under whatever you build in B.
- **(Ceiling probe only, algorithm-changing, clearly labeled):** Hkv=1
  means head-pairs could share a 128-row block via Python-side row
  packing to unlock the stock `tile_m=128` 2-WG cooperative path — but
  that shares selection across a head pair, which violates constraint
  §1.2. Allowed only as an A/B *measurement* of what 2-WG buys, never
  as the shipped kernel.

## 7. Correctness before performance

Same rule as ever: a fast wrong number is worse than no number. The §4
harness must pass (even/odd topk, topk=1, GQA, MHA) before any bench
number is reported. Also re-run stock `fa4_hybrid` e2e once at the end
to confirm you haven't broken the shared source for the baseline.

## 8. Wiring into ursa

Mirror the existing structure — new subpackage
`.../bench/kernels/fa4_hybrid_v2/` with `__init__.py` (registers
`fa4_hybrid_v2` via `register(KernelImpl(...))`), `forward.py`
(coarse-pool + top-K, copy of fa4_hybrid's Python side), `autograd.py`
(forward-only wrapper). One import line in `kernels/__init__.py`.
Gate availability on whatever knob you add to `interface.py`, and add
that knob to `compile_key` — stale-cache bugs from missing compile-key
entries are silent and maddening.

## 9. Deliverables

1. **Baseline attribution report first** (§4/§5.2): overlap verdict
   from SASS + profile/ablation table naming the dominant stall for
   stock `fa4_hybrid`. Post this before optimizing.
2. `fa4_hybrid_v2` on branch `v2-clean`, correctness harness passing.
3. Fine-only bench table (interleaved same-run): `fa4_hybrid` vs
   `fa4_hybrid_v2`, all four sizes — latency, TFLOP/s, MFU. Plus one
   final e2e table via `bench.py` (noting the ±5% noise).
4. One-paragraph writeup: what the profile said, what you changed,
   what it bought, and what still separates the result from dense
   FA4's ~75% — named mechanism, not "it didn't work."

## Gotchas carried over (each cost real time once)

- The CuTe DSL rewrites plain `for` loops as dynamic loops; iterating
  a Python list of copy fns/layouts inside the kernel needs
  `cutlass.range_constexpr(...)` or it fails with
  "update to list<N> inside `for` not supported".
- Every codegen-affecting knob goes into `compile_key` in
  `interface.py`, or you'll bench a stale kernel.
- Producer and consumer both walk `full_block_idx` in reverse; any
  per-position special-casing (tail handling) happens at the *start*
  of the loop, not the end.
- Kernel signature order and call-site order in `flash_fwd_sm90.py`
  must be edited in both places; the DSL error for a mismatch is
  cryptic.
- The e2e bench's `import ursa` only resolves thanks to `fa4_env.sh` —
  never trust an import that only works from a particular cwd.
