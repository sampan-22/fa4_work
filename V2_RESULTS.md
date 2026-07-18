# fa4_hybrid_v2 results (SM90 H100, fine-only bench, MFU @ 990 TFLOP/s peak)

All numbers from `bench_fine_v2.py` (interleaved same-run where applicable),
exact VSA shapes (B=1, Hq=8, Hkv=1, hdim 128, bf16, cube 64), CUDA-event
timing over 50 reps after 10 warmups. "stock" = fa4_hybrid path, pristine
kernel. Knobs: `K/V` = num_stages / num_stages_v, `HM` = head-major
rasterization, `pp` = ping-pong (two consumer WGs), `2CTA` =
minnctapersm=2 occupancy config (one consumer WG per CTA, two CTAs/SM).

## Fine-only MFU

| config | threads/CTA | CTAs/SM | 16k | 32k | 64k | 128k |
|---|---|---|---:|---:|---:|---:|
| stock (s2)            | 256 | 1 | 43.1 | 46.4 | 44.7 | 43.1 |
| stock (s4)            | 256 | 1 | 42.3 | 45.9 | 44.6 | 43.0 |
| pp K2/V2              | 384 | 1 | 39.8 | 43.8 | 44.5 | 44.2 |
| pp K4/V4              | 384 | 1 | 43.1 | 46.2 | **48.2** | 41.6 |
| pp K4/V4 + HM         | 384 | 1 | 43.3 | 47.1 | **48.2** | 41.6 |
| pp K4/V2 + HM         | 384 | 1 | 39.9 | 44.8 | 44.8 | 44.1 |
| pp K2/V2 + HM         | 384 | 1 | 39.8 | 44.3 | 44.5 | 44.0 |
| pp K3/V3 + HM (racy!) | 384 | 1 | —    | —    | —    | 45.0 |
| stock 2CTA (s2)       | 256 | 2 | **50.8** | **52.2** | **50.7** | 37.8 |
| stock 2CTA + HM       | 256 | 2 | 50.9 | 52.2 | 50.7 | 37.8 |

(K3/V3 hit a reproducible Xid 43 at 128k without HM — sanitizers clean;
odd stage counts are now asserted out for pp.)

**fa4_hybrid_v2 = per-size dispatch** (2CTA below nkv=1500, ping-pong
K4/V2+HM at/above): **50.8 / 52.2 / 50.7 / 44.1** vs stock
43.1 / 46.4 / 44.7 / 43.1 → **+7.7 / +5.8 / +6.0 / +1.0 points**
(1.18x / 1.12x / 1.13x / 1.02x fine-stage speedup). The
CLEAN_SLATE success bar (>52% at 128k) is NOT met — 32k exceeds it,
128k lands at 44.1%; the two capping mechanisms are quantified below.

Reference points: dense FA4 SM90 hdim128 ~75%; abandoned KV-pairing
kernel 44.3 / 48.2 / 48.2 / 46.3.

## What the profiles say (ncu, `--launch-skip 3`)

- Stock, 32k/128k: tensor pipe active 48/50%, XU (exp2) 26%, DRAM 3/19%,
  L2 hit 83% (128k), 1.25 active warps/scheduler, stalls dominated by
  `wait` (~1.0/issue) and `long_scoreboard` (~0.8/issue). Latency-bound
  single consumer WG; REGCOUNT=255 caps occupancy at 1 CTA/SM
  (see BASELINE_ATTRIBUTION.md).
- pp K4/V4, 64k: tensor pipe active **62.5%** but delivered MFU 48.2 —
  `not_selected` stalls jump 0.07 → 0.47/issue (two WGs competing for
  issue slots). The FA3-style scheduler barrier (PP_SB) did not change
  the numbers.
- 128k DRAM-read totals (per launch): stock-s2 15.4 GB / stock-s4
  10.1 GB / pp-s2 9.7 GB / pp-s4 **40.5 GB** (L2 hit 61%). The wider
  in-flight position window of pp-s4 pushes the 132 co-resident CTAs'
  combined K/V front past the 50MB L2; K4/V2 (V holds the oldest
  positions) restores 128k to 44.1 while keeping per-WG K
  double-buffering. Under ncu's clamped clocks pp-s4 is the *fastest*
  config at 128k (20.2 ms vs 24.5 stock) — its compute side wins; DRAM
  is the wall at boost clocks.

## Ceiling discussion

Two independent mechanisms cap the current numbers:

1. **Issue/latency inside the SM (≤64k regime).** Adding the second
   consumer stream (pp, or a second CTA) lifts tensor-pipe occupancy
   from ~48% to ~62%, but the two streams contend for issue slots and
   L1/smem bandwidth during softmax bursts; delivered MFU lands at
   48–52%. Dense FA4 avoids this with tile_m=128 cooperative WGs where
   the per-WG softmax work is HALF (one 128-row softmax split across
   two WGs) — block-sparse VSA cannot use that shape without giving up
   per-64-row selection (the banned ceiling probe).
2. **L2 capacity at 128k.** 141MB K/V vs 50MB L2 with ~12% cross-CTA
   list overlap (synthetic random top-K): every additional in-flight
   position per CTA costs L2 hit rate. Anything that widens concurrency
   (2 CTAs, deep pipelines) trades against it. Head-major rasterization
   buys ~1pp at 32k and nothing at 128k with random selections;
   production VSA selections are spatially correlated, so HM should be
   re-evaluated on real traces.

## Recommended fa4_hybrid_v2 configuration

Per-size dispatch in the ursa wrapper (all knobs are in the compile key,
so mixed use is safe):

- nkv < ~1500 (≤ ~96k tokens): `MIN_BLOCKS=2` (stock kernel, 2 CTAs/SM).
- larger: `PP=1, NUM_STAGES=4, NUM_STAGES_V=2, HEAD_MAJOR=1`.
