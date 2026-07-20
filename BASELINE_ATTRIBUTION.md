# Baseline attribution: stock `fa4_hybrid` fine kernel (SM90, H100)

Deliverable 1 of `CLEAN_SLATE_PLAN.md`. All measurements on this pod's
H100 (990 TFLOP/s bf16 peak), stock unpaired path
(`tile_mn=(64,64)`, `BlockSparseTensorsTorch`, full lists only), exact
bench shapes (Hq=8, Hkv=1, hdim 128, bf16).

## Verdict in one paragraph

The intra-warpgroup overlap **is engaged and correct** — that is not the
problem. The problem is that the kernel launches with **REGCOUNT = 255**
(256 threads x 255 regs = the entire 64K-register file), so **only ONE
CTA — one consumer warpgroup — is resident per SM**. The plan's F2
premise ("80KB smem → 2 CTAs/SM co-resident") is false: occupancy is
register-limited, not smem-limited, and cross-CTA interleaving never
existed. A single consumer WG cannot hide its own softmax/reduction
latency behind the tensor pipe, so the tensor pipe idles ~half the
time: `sm__pipe_tensor_op_hmma_cycles_active` = **48.3%** at 32k /
**50.1%** at 128k — within a point of measured MFU (46.7% / 43.2%).
The gap to dense FA4 (~75%) is a concurrency deficit, and the fix must
add a second *independent* consumer stream per SM.

## 1. SASS: overlap engages for the block-sparse path (§5.1)

`CUTE_DSL_KEEP_PTX=1 CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=...`, then
`cuobjdump -sass`. The block-sparse full-list loop body compiles to the
canonical ping-pong-within-one-WG structure:

```
8x HGMMA.64x64x16   (QK, k-loop, commit to group)   <- S_i issue
4x HGMMA.64x128x16  (PV, k-loop, commit to group)   <- O_{i-1} issue
WARPGROUP.DEPBAR.LE gsb0, 0x1                       <- wgmma.wait_group 1: S_i done, O_{i-1} in flight
~32x MUFU.EX2 + reductions + FMUL/FFMA (softmax_i, rescale, P convert)
WARPGROUP.DEPBAR.LE gsb0, 0x0                       <- O_{i-1} done, release V stage
```

First block: QK → `DEPBAR 0x0` → softmax (nothing to overlap, expected).
Tail: PV → `DEPBAR 0x0` → epilogue. One HGMMA per k-step, no
serialization per instruction. **Candidate A (broken overlap) is ruled
out.**

## 2. Occupancy: register file, not smem, caps residency

- `cuobjdump -res-usage`: `REG:255` (also on the abandoned paired
  kernel — both were always 1 CTA/SM).
- Driver check (`cuOccupancyMaxActiveBlocksPerMultiprocessor`, 256
  threads, 82,944 B dyn smem): **max 1 CTA/SM**.
- ncu Occupancy section at 32k: `Block Limit Registers: 1`, achieved
  4.97 active warps/SM (of 64).

Cause: the DSL emits `.reqntid 256 / .minnctapersm 1`, so ptxas budgets
65536/256 = 255 regs/thread at launch. The `setmaxregister` dance
(producer ↓56, consumer ↑256) then redistributes a file that was
already fully allocated. Dense FA4 gets two consumer WGs not via
occupancy but via 384-thread CTAs (`.reqntid 384` → 168 regs at launch,
consumers ↑240, producer ↓24 — exactly balanced).

## 3. ncu profile, stock kernel (`--launch-skip 3`, 1 launch)

| metric | 32k | 128k |
|---|---:|---:|
| tensor pipe active (`sm__pipe_tensor_op_hmma`) | 48.3% | 50.1% |
| Compute (SM) throughput | 47.9% | — |
| XU/MUFU pipe active (exp2 lives here) | 25.7% | 26.6% |
| FMA pipe | 13.4% | 13.9% |
| ALU pipe | 12.5% | 12.4% |
| DRAM throughput | 3.0% | 11.6% |
| L2 throughput | 65.7% | — |
| active warps / scheduler (of 16) | 1.25 | — |
| issue rate | 1 per 3.0 cyc | — |
| stalls per issue: wait / long-sb / short-sb / barrier | 0.99 / 0.80 / 0.32 / 0.27 | 0.98 / 0.75 / 0.32 / 0.25 |

Reading: nothing is *throughput*-saturated (MUFU 26%, DRAM ≤12%, issue
33%). The dominant stalls are `wait` (fixed-latency dependencies —
wgmma group waits, exp2 results) and `long_scoreboard` (smem/mbarrier
waits). This is a **latency-bound single consumer warpgroup**: per-tile
non-MMA work (F3: exp2 ≈ 256 cyc + reductions + P-convert + rescale vs
512 cyc of MMA) can only overlap with the WG's *own* previous MMA, and
whatever the intra-WG overlap fails to hide is a dead tensor-pipe
cycle. With 4.97 warps/SM there is no other work for the SM to issue.

## 4. Concurrency experiment: forcing 2 CTAs/SM (candidate D, measured)

On branch `round3-optionA` (`FLASH_ATTN_SM90_MIN_BLOCKS=2` →
`.minnctapersm 2` → REGCOUNT 128, consumer WG ↑232, producer ↓24;
smem already fits): driver-confirmed 2 CTAs/SM, `REG:128`, zero spills.

| size | stock (1 CTA/SM) | stock forced 2 CTAs/SM |
|---|---:|---:|
| 16k  | 43.2% | **50.7%** |
| 32k  | 46.6% | **52.2%** |
| 64k  | 44.5% | **51.0%** |
| 128k | 43.1% | **38.0%** (collapse) |

Two independent consumer WGs per SM (from different CTAs) buy **+7-8
MFU points** at ≤64k — direct proof the gap is concurrency. But at 128k
it *collapses*: K+V working set is 2×2160×64×128×2B = 141 MB (vs 17 MB
at 32k, which fits the 50MB L2 — hence DRAM 3% at 32k). Doubling
resident CTAs doubles the concurrent random-cube footprint per SM,
L2 hit rate drops, and the un-hidable TMA latency eats the win. The
success bar lives at 128k, so 2-CTA occupancy is the wrong vehicle.

## 5. Direction history

Candidate B (intra-CTA ping-pong: two consumer warpgroups in one CTA
splitting the block list by pipeline-stage parity, SplitKV-style merge)
was implemented and measured on branch history now removed from this
tree — it recovered the ≤64k win without the 2-CTA's 128k L2 penalty,
but the code path was judged not worth keeping (issue-slot contention
between the two warpgroups capped delivered MFU well below the tensor
pipe's measured occupancy, and the implementation complexity/fragility
— e.g. an unexplained Xid-43 fault at odd pipeline-stage counts — wasn't
worth the ~1pp it bought at 128k). The current 128k direction is
KV-pairing (`kv_pair_factor=2`, see the repo's kv-pairing docs/history)
instead.
