# Resident-A SpMV (`--encoding resident`)

**One resident kernel: the COO scatter**, chosen on hardware among three local kernels (table in
"The local kernel", below). It compiled, ran in the 16-PE simulator, and passed 20/20 on hardware at
16 PEs and 65K; its full-wafer run is part of the v2.0 benchmarks (the full-wafer runs logged below used the first kernel). **Read the log sections with care: the numbers up to and including
"Full wafer, runs 2 and 3" were taken with the first kernel, a CSR row loop ("kernel 0"), which no
longer exists; "kernel 1" (a CSR loop on u16 state) is also gone.** The window of the shipped kernel
is the one in the table below and in the later sections.

## Why

On 2026-10-13 a 32-thread CSR on the EPCC user node beat our streamed wafer SpMV (`docs/results.md`): **61.7 ms vs 458.8 ms** at the full-wafer size (171M NNZ), **3.5 ms vs 45.2 ms** at 65K.
Streaming A moves 4 B/NNZ over a ~2 GB/s link on every call; the host reads the same bytes
from DRAM. Streaming can't win. The wafer's case is **A resident in SRAM**, which is what CG
needs (ADR-005). The January v1 resident numbers (5–30 µs) had no on-wafer reduction, a
host-side gather and unsynced clocks, and were never measured with the current kernel.
This is that measurement.

## What it does

```
setup, once (copy mode, 32 strips):   x_vec, col_indices, vals, row_ids, row_ptrs  → every PE
per call (host):                      f_tic ; spmv ; f_toc        (all queued, f_toc blocks)
per call (each PE):                   spmv() → ew_start: local COO scatter over resident vals / x
                                      → EW chain into y_result on column 0 → chain_done
                                      → unblock_cmd_stream → f_toc stamps
```

`src/csl/pe_resident.csl`, `ENCODING == 3`, picked by `layout.csl`. Differences from
`pe_valtask.csl`, and only these:

- **No stream.** No H2D data task, no `val_meta`, no `n_recv`/`draining`/`tsc_armed`:
  nothing arrives per call, so there is no count to wait for and no word that can beat
  `spmv()` to the PE.
- **`vals[TOTAL_NNZ]` is exported** and written by the setup copy (valtask filled it from the
  stream). `layout.csl` exports `vals` for `ENCODING == 3` only.
- **`spmv()` holds the command stream.** It stamps `tsc_start`, activates `ew_start`, and
  returns *without* `unblock_cmd_stream`; `chain_done` releases it. So the host's `spmv`
  completes, for the command stream, exactly when that PE's pass is done, and `f_toc` (queued
  behind it) stamps right after. This is what `wait_ew` did for the streaming kernels, folded
  in; `wait_ew` is not exported for `ENCODING == 3`.
- `local_spmv` and `ew_start` are **verbatim** `pe_valtask.csl` (a test pins that), as is the
  sync block (a test pins it across the kernels). `pe.csl` and `pe_valtask.csl` were not touched
  by the resident addition.
- **Under `SYNC_TIMING` the kernel never enables or disables the tsc**: every
  `enable_tsc`/`disable_tsc` sits inside `if (SYNC_TIMING == 0)` (a test checks it). The sync
  module enables it once at startup; one `f_sync` per run.

**Host (`cluster/worker_runtime.py`).** A resident branch in the existing pass loop (the
streaming path is unchanged), `build_resident_vals` (PE-major `(W·H, max_nnz)` float32, the
same CSR order as `col_indices`, a test checks it against `build_val_stream_buffer`), and the
setup array list `x_vec, col_indices, vals, row_ptrs` (`row_ptrs` last: the setup barrier
reads it). `--h2d-mode`, `--memcpy-order`, `--serialize-column-rects` do nothing.
`run.py run --encoding resident` defaults `--loop-count` to 20; sweeps keep their TOML's.

### Host-fed x (an experiment that was removed)

Until v2.0 the worker had `--x-per-pass` (H2D a new x before every pass, barrier-closed) and
`--x-row0-probe` (the link cost of n unique floats into PE row 0). They were meant to price a
CG whose x comes from the host. The 16-PE simulator died on the 3rd pass with the refresh
(3/3 runs) and the hardware hung, the numbers were never measured, and the estimate from the
setup rate already settles the question (below). The code is on the branch `archive/x-per-pass`.

**The on-wafer y → x transpose is the missing piece** (ADR-005). Without it CG's
iteration is `y` D2H → host → `x` H2D, which loses to the host CPU alone. The wafer's number only
exists if x never leaves it.

## CSV columns

Reused, with resident meaning:

| column | resident meaning |
|---|---|
| `wafer_cycles_*`, `wafer_time_*` | the synced window of **one SpMV**: max(toc) − min(tic) on the common clock, min/avg/max/std over passes |
| `host_pass_time_{avg,std}` | host time per pass from before `f_tic` to the barrier D2H returning (true upper bound) |
| `stream_time_corrected` | the same bracket minus the idle barrier D2H: **barrier-closed host time per SpMV** (name kept for schema stability) |
| `wafer_host_ratio_max` | synced window / `host_pass_time`; must be ≤ 1 (but see Risk 6: here it is a weak check) |
| `compute_hw_cycles_*` | per-PE `spmv()` entry → `chain_done` span (last pass), the per-PE view of the same pass |
| `setup_h2d_*` | **includes vals**: 4·W·H·(KX + 2·max_nnz + KY + 1) bytes |
| `h2d_bytes_total`, `memcpy_h2d_time` | 0: nothing streams per pass |
| `loop_count`, `sync_timing`, `encoding` | 20, True, `resident` |

New:

| column | meaning |
|---|---|
| `wafer_gflops_avg` | 2·NNZ / `wafer_time_avg`, GFLOP/s (0 without sync timing) |
| `null_spmv` | the kernel's `spmv()` was a no-op (launch floor); `wafer_*` is the launch machinery alone, y not validated |

Old CSVs migrate on append (new columns backfilled with 0).

## Expected numbers, and why

All at 850 MHz, `WSE_CLOCK_HZ`. Sizes are the host-baseline matrices (n = 14,300 / 114,000 /
414,000 at d = 1e-3; the weak-scaling runs used 114,487 / 414,007, ≤ 0.4% more NNZ).

| | grid | KX = KY | NNZ/PE avg / max | `TOTAL_NNZ` | SRAM (of 38 KB) |
|---|---|---|---|---|---|
| smoke (d = 0.01) | 4×4 | 250 | 625 / 730 (compiled bound) | 730 | 9.7 KB |
| 65K | 256×256 | 446 | 198 / 259 | 314 | 9.5 KB |
| full wafer | 742×1155 | KX 558, KY 359 | 200 / ~272 | 326 | 9.0 KB |

**SRAM is not the constraint.** Resident = valtask − `val_meta` (one word): `vals[TOTAL_NNZ]`
already existed, and x_vec already lived on the PE. At the full wafer the PE uses ~9 KB with
`sync_timing` on, even at the compiled `TOTAL_NNZ` = 326 (above the measured max of ~272); a
test prices 272 explicitly. The budget leaves room for roughly an order of magnitude more NNZ per PE (3.7K more
`col_indices` + `vals` slots), which is a
separate experiment (S4, words-per-PE).

### The on-wafer window (`wafer_time_*`)

The window is the slowest PE's chain: local compute, the `ew_buf` copy, then the EW chain
west across W columns, plus the tic/toc dispatch skew.

1. **Local CSR.** The loop does, per row, two `row_ptrs` loads and a store, and per NNZ a `vals`,
   a `col_indices` and a gathered `x_vec` load plus an FMAC. Scalar code, no DSD can do the
   gather. Estimate **4–8 cycles/row, 6–12 cycles/NNZ**. (`val-only-encoding.md` budgets
   ~5k cycles for 446 rows + 259 NNZ, inside this range. None of it has ever been measured on
   this loop.) The slowest PE has the max NNZ.
2. **`ew_buf` copy**, KY elements at 2–4 cycles. (A cheap later win: let `local_spmv` write
   `ew_buf` directly and skip it, ~1 µs; not done, to keep `local_spmv` verbatim.)
3. **EW chain: measured, not estimated** (cs3-microbench M6–M8, ADR-005 amendment): **13.7
   cycles per PE + 1.13 per word** along a row. That reproduces the 4.7 µs recorded at
   256 PEs × 446 words (13.7·256 + 1.13·446 = 4.0k cycles).
4. **Skew.** `f_tic` stamps when each PE's command stream reaches it; commands reach PEs over
   the fabric, ~1 cycle/hop, so up to (W+H) cycles between the first tic and the last: 0.5 µs
   (65K), 2.2 µs (full wafer). `f_toc` stamps right behind each PE's own pass.

| | local CSR | `ew_buf` copy | EW chain | skew | **window** | rate |
|---|---|---|---|---|---|---|
| smoke 4×4 | 5.2–10.4k (730 NNZ) | 0.5–1k | 0.3k | ~0 | **7–14 µs** | — |
| 65K | 3.4–6.7k | 0.9–1.8k | 4.0k (4.7 µs) | 0.5–2 µs | **10–17 µs, ~13** | ~2 TFLOP/s |
| full wafer | 3.0–6.2k | 0.7–1.4k | 13.7·742 + 1.13·359 = 10.6k (**12.4 µs**) | ≤ 2.2 µs | **18–30 µs, ~21** | ~16 TFLOP/s |

(Cycles; the window is their sum at 850 MHz plus skew.) At the full wafer the EW chain is
**more than half the window**: 742 hops × 13.7 cycles, which is why the window grows with W
and not with NNZ/PE. Against the host baseline that is **61.7 ms / ~21 µs ≈ 3,000×** at the
full wafer and **3.5 ms / ~13 µs ≈ 270×** at 65K, consistent with January v1's 6–34 µs.
`wafer_gflops_avg` should read ~2,000 at 65K and ~16,000 at the full wafer, against 5.6 for the
32-thread host.

**What would falsify it:** a window above ~50 µs at the full wafer means the 13.7 cycles/PE
chain figure does not extrapolate from W = 256 to 742 (fabric backpressure at KY = 359 words,
or a route longer than the fit saw), or the scalar loop is 3× slower than budgeted. Compare
`compute_hw_cycles_max` (the per-PE span) against the window: if the chain is the cost, the
spans of column-0 PEs sit ~10k cycles above column W−1's.

### The host's view (`host_pass_time_*`, `stream_time_corrected`)

A pass is three launches and a one-word D2H. A host round trip is ≥ 167 µs (M7: one-word D2H
sync), a blocking launch returns in ~13 µs without waiting for the device. Expect
`host_pass_time_avg` **~0.2–0.6 ms** and `stream_time_corrected` **~0.05–0.4 ms**, at every
size: it is the host runtime's latency, not the wafer. So "one SpMV per host call" is **~200×
faster than the host CPU at the full wafer (0.3 ms vs 61.7 ms)** but *slower* at the 14K size
(0.14 ms); the
~21 µs only exists if the host stops mediating every call. That is the CG kernel, and it is
the argument for the on-wafer loop.

### What feeding x from the host would cost (an estimate, never measured)

The setup rate is the model: ~12.0 GB/s at the full wafer and 10.9 GB/s at 65K (`docs/results.md`),
plus the ~1.3 ms fixed cost.

| | replicated x | estimated H2D time | host CPU SpMV | n unique floats (row 0) | estimated H2D time |
|---|---|---|---|---|---|
| 65K | 117 MB | **~11–12 ms** | 3.5 ms | 0.46 MB | ~0.5–0.7 ms |
| full wafer | 1,913 MB | **~160 ms** | 61.7 ms | 1.66 MB | ~1.6–2 ms |

The row-0 time assumes a one-PE-tall rectangle sits on one memcpy channel (≤ 1.13 GB/s,
M1) plus ~0.1–0.2 ms fixed cost. It is the least certain number here: M6 found one-PE-tall
*D2H* rectangles ~70× slower, and no H2D one-tall shape has been measured.

**Reading it, if the prediction holds:** the host-fed iteration is **slower than the host CPU
alone with the replicated layout** (~11 ms vs 3.5 ms at 65K; ~160 ms vs 61.7 ms at the full
wafer) and, with n unique floats, still ~100× its own SpMV (~2 ms vs ~21 µs at the full wafer).
Feeding x from the host forfeits the wafer's advantage; the transpose has to run on the wafer.
Break-even against the host CPU, setup included, at the full wafer: load 217 s + tiling 159 s
+ setup copy 0.4 s ≈ 376 s over (61.7 ms − t_iter): **~6,100 SpMVs** with x on the wafer
(t_iter ≈ 0.02 ms), **~6,300** with host-fed unique x (≈ 2 ms), **never** with replicated x.

### Setup

`setup_h2d_bytes` grows by the vals: full wafer 4·857,010·(558 + 2·272 + 360) ≈ **5.0 GB** →
~420 ms at 12.0 GB/s (valtask's setup was 341 ms); 65K ≈ 370 MB → ~34 ms.

### First hardware numbers (16 PEs, no refresh, 20/20 PASSED at 5.32e-08)

| | measured | predicted above |
|---|---|---|
| synced window | 41.66 / 41.70 / 41.71 µs (35,414–35,453 cycles, std 8) | 7–14 µs |
| `--null-spmv` window (launch floor) | 0.49–0.53 µs (417–454 cycles) | 0.5–3 µs |
| per-PE span (`compute_hw_cycles`) | 34,077–35,086 | |
| `host_pass_time` | 208.5 µs (null run: 199.8 µs) | 0.2–0.6 ms |
| **SpMV alone** (window − floor) | **≈ 41.2 µs** | |

The launch floor is small and the host's per-call cost (~0.2 ms) is as predicted. **The local
compute is 3–6× slower than I budgeted.** At 4×4 the EW chain is ~340 cycles, so ~34.7k cycles
are `local_spmv` plus the `ew_buf` copy on a PE with KY = 250 rows and up to 730 NNZ: ~48
cycles per NNZ if rows are free, ~139 per row if NNZ are. The 65K and full-wafer runs (same
loop, different KY and NNZ) will separate the two; the fit is

```
window − floor − chain  ≈  a·KY + b·NNZ_max          chain = 13.7·W + 1.13·KY cycles
  16 PEs : KY 250, NNZ 730, chain 0.34k   →  34.7k cycles
  65K    : KY 446, NNZ 259, chain 4.0k
  full   : KY 359, NNZ 272, chain 10.6k
```

Three equations, two unknowns: solve (a, b), and the third point checks the model. Until then,
the two extreme readings predict **65K 19–73 µs** and **full wafer 28–60 µs** (b = 48 and a = 0,
or a = 139 and b = 0), against the 13 / 21 µs above, which should be discarded. The wafer
still beats the host CPU at either end (61.7 ms / 28–60 µs ≈ 1,000–2,200×), but the kernel is
the thing to fix, not the fabric.

### First fit (16 PEs and 65K run 1, hardware, floor taken as 450 cycles; refine with the null twins)

65K, 256×256, run 1 of 3 (system not recorded): window **38,598 cycles = 45.4 µs**, 20/20 PASSED at 1.49e-07, per-PE span
30,747 / 35,098 avg / 37,808 max, `host_pass_time` 1.04 ms (std 0.70), setup 31.6 ms.

```
  window − floor − chain = a·KY + b·NNZ_max,   chain = 13.7·W + 1.13·KY
  16 PEs:  35,449 − 448 − 337   = 34,664 = a·250 + b·730
  65K   :  38,598 − 450 − 4,011 = 34,137 = a·446 + b·259
  →  a ≈ 61 cycles per row,  b ≈ 27 cycles per NNZ
```

Two equations, two unknowns: **the solve is exact, so it reproduces both inputs by construction
and that is no evidence the model is right.** Both coefficients are also **conditional on the
assumed chain model** (13.7·W + 1.13·KY from W ≤ 256; 4,011 of the 38,598 cycles at 65K) and on
the floor (450 cycles assumed at 65K). The full-wafer window is the first real test.

**The per-row term dominates**: 61 cycles/row (empty rows, the `row_ptrs` loads, the store, and
the `ew_buf` copy iteration) is 80% of the local time at 65K (446 × 61 = 27k of 34k cycles)
and 44% at 16 PEs; the per-NNZ cost is ~27 cycles, about half of my 48-cycle reading. This is
a two-point fit with an estimated floor; the full-wafer run is the independent check.
Prediction for the **full wafer** (KY 359, NNZ 272, chain 10.6k): 359 × 61 + 272 × 27 + 10.6k +
0.45k ≈ **40k cycles ≈ 47 µs** (45–50 if the chain figure extrapolates), ~1,300× the host CPU.
**Written down before the full-wafer result exists: 40k cycles = 47 µs (45–50 µs).** When it lands,
refit all three points by least squares and record the residual here next to that prediction.
For the variants (the flag is gone; this was the prediction for the COO scatter): `--local-kernel 2` removes the row loop and the copy loop (two DSD ops, ~2·KY
cycles) and should leave ~NNZ × 27: **65K ≈ 8k + 4k chain ≈ 12.5k cycles ≈ 15 µs (3×), full
wafer ≈ 7.3k + 0.7k + 10.6k ≈ 19k cycles ≈ 22 µs (2×)**; variant 1 helps in proportion to how
much of the 61 and 27 is `i32` arithmetic, which this fit cannot say.

**Refit with the measured 65K floor (null twin: 1,485 cycles = 1.75 µs on every pass; 16 PEs:
448):** 65K net = 38,598 − 1,485 − 4,011 = 33,102, so (still an exact two-point solve, still
conditional on the chain model) **a ≈ 58 cycles/row, b ≈ 27.6 cycles/NNZ**. The floor grows with
the grid: 448 → 1,485 cycles for (W+H) 8 → 512, i.e. ≈ 430 + 2.06·(W+H) (two points, an
exact line, **not** a measured mechanism; consistent with command dispatch at ~2 cycles/hop).
That puts the **full-wafer floor near 4.3k cycles (5 µs)** (W+H = 1,897), which my 47 µs
prediction above did not include. **Revised prediction, also on record before the result:**
4.3k + 359·58 + 272·27.6 + 10.6k ≈ 43k cycles ≈ **51 µs** (range 46–56). The 47 µs entry stays
as first written; the full-wafer window is compared with both.

### Full wafer, run 1 (857,010 PEs, 20/20 PASSED at 2.75e-07): the predictions meet the result

Window **43,293 cycles = 50.93 µs** (min 43,293, max 43,296, std 1); per-PE span 21,656 / 33,497
avg / 39,714 max; `host_pass_time` 1.11 ms; setup 5,012 MB in 416.8 ms (12.0 GB/s). Against
what was written down beforehand:

| prediction (made when) | said | result |
|---|---|---|
| first table (before any hardware) | 18–30 µs | **wrong by 1.7–2.8×**: the scalar loop costs 4× the budget (a ≈ 58 cycles/row, b ≈ 27.6/NNZ) and the launch floor grows with the grid |
| after 65K run 1, assumed 450-cycle floor | 40k cycles = 47 µs | 43.3k: −8% |
| after the 65K null twin (floor 1,485) | 43k cycles = **51 µs** (46–56) | **43,293 = 50.9 µs: within 1%** |

Three-point least squares on `window − floor − chain = a·KY + b·NNZ_max` (16 PEs, 65K, full wafer;
full-wafer floor 4.34k from the `430 + 2.06·(W+H)` line and chain 10.57k from `13.7·W + 1.13·KY`,
both **extrapolated, not measured**): a = 58.2, b = 27.55, residuals **−0.8, −5.4, +7.3 cycles
(rms 5)**; the two-point fit on 16 PEs and 65K predicted the third point to 11 cycles. That is the
strongest result so far for this model, with one limit: **only the sum floor + chain (14.9k of
43.3k cycles at the full wafer) is tested.** A floor that is too high and a chain that is too low by
the same amount would pass. The full-wafer `--null-spmv` twin measures the floor alone and settles
it. Corroboration, not proof: the window sits 3.6k cycles above the slowest PE's span (0.8k at 65K),
the same order as the dispatch skew the floor line predicts (4.3k and 1.5k).

### Full wafer, runs 2 and 3, and the null twin: the floor undershot

n = 3 at the full wafer: window **43,293 cycles = 50.93 µs** median (min over all 60 passes 43,293;
run averages 43,293 / 43,341 / 43,293; one outlier pass of 44,249, +956 cycles, in run 2); per-PE
spans identical in all three runs (21,656 / 33,497 / 39,714), so the kernel is deterministic and
only the launch jitters. Quote the **median**, not the mean. 65K, n = 3: 38,598 / 38,598 / 38,594.

**Launch floor (`--null-spmv`), measured at three sizes** (min = median over 20 passes; the
full-wafer twin also had outlier passes of 18,085 / 6,682 / 5,871 cycles):

| grid | W + H | floor (cycles) | µs |
|---|---|---|---|
| 4×4 | 8 | 448 | 0.53 |
| 256×256 | 512 | 1,485 | 1.75 |
| 742×1155 | 1,897 | **4,843** | **5.70** |

The `430 + 2.06·(W+H)` line I wrote down from the first two points **undershot** the third: it
said 4,338, the wafer says 4,843 (+505 cycles, +12%). Slope: 2.06 cycles per unit of W+H up to
65K, 2.43 beyond. Three points fit several shapes exactly and **say nothing about mechanism**; all
three of these reproduce the data:

| shape | fit | predicts at 262,144 PEs (512×512) | at 428,505 PEs (735×583) |
|---|---|---|---|
| piecewise linear in W+H | 2.06 up to 512, 2.43 after | 2,726 | 3,439 |
| `432 + 0.83·W + 3.29·H` (W and H apart) | exact on 3 points | 2,538 | 2,956 |
| `432 + 1.96·(W+H) + 1.9e-4·(W+H)²` | exact on 3 points | 2,640 | 3,349 |

A null twin at 428,505 PEs (the weak-scaling size already run) separates them by 400–480 cycles
against a jitter of ~1 cycle; at 262,144 PEs by 90–190. Until one of those exists, the floor at an
unmeasured size is an interpolation, not a prediction.

**Refit with all three measured floors.** Net of floor, with the chain model `13.7·W + 1.13·KY`
(from M6–M8, W ≤ 256) still assumed, `net − chain = a·KY + b·NNZ_max` on 16 PEs, 65K and the full
wafer gives 34,664 / 33,102 / 27,879 cycles:

```
  least squares:  a = 57.6 cycles/row,  b = 27.7 cycles/NNZ
  residuals:      −36,  −245,  +329 cycles   (rms 238 = 0.7% of the net)
  16 PEs + 65K alone (exact) predicted the full wafer 516 cycles HIGH (28,394 vs 27,879)
```

The residual that was 5 cycles with the modelled floor is now ~240: **the 51 µs hit was partly
cancellation**. The floor was 505 cycles higher than modelled, and the chain plus row / NNZ terms
came out about 515 lower; the total window was right to 1% by accident of the sum. Does the chain
model still hold? It is the only extrapolated piece left. Leaving the W coefficient free:

```
  a = 58.9,  b = 27.3,  c = 12.8 cycles per PE   (chain = c·W + 1.13·KY; M6–M8 said 13.7)
```

This is an exact solve of three equations in three unknowns, so it validates nothing, but it
says the data want a per-PE chain cost 7% under the M6–M8 fit (at W = 742: 9.5k cycles of chain
instead of 10.6k). Same size of effect as the residuals above: **the data are consistent with the
13.7 figure only if a, b drift by a few percent**, and equally with c = 12.8 at fixed a, b.
Telling these apart needs a point that varies W independently of KY and NNZ (a non-square grid
at fixed matrix, or a different height at the full width), which the weak-scaling sizes do not give.

**Predictions on record, and how each did:**

| made | said | result |
|---|---|---|
| first table, before hardware | 18–30 µs | wrong, 1.7–2.8× low |
| after 65K run 1 | 47 µs (floor assumed 450) | 43,293 = 50.93 µs: 8% low |
| after the 65K null twin | **51 µs** (46–56), floor 4,338 | 50.93 µs: within 1%, **partly by cancellation** (floor +505, rest −515) |
| same, floor line | 4,338 cycles at W+H = 1,897 | 4,843: +12% |
| net SpMV on the wafer (window − floor) | | **38,450 cycles = 45.2 µs** at the full wafer; 43.7 µs at 65K; 41.2 µs at 16 PEs |

### Pre-registered predictions for the u16 and COO kernels ("kernel 1" and "kernel 2") at 65K and the full wafer

Written down **before any 65K or full-wafer run of either kernel.** Data in hand (16 PEs, KY 250,
NNZ_max 730, chain 337 cycles, hardware window; floor 448):

| kernel | simulator | hardware window | net of floor | local compute (net − chain) |
|---|---|---|---|---|
| 0 (baseline) | 36,967 | 35,449 | 35,001 | 34,664 |
| 2 (COO scatter) | 27,787 | 27,857 | 27,409 | 27,072 |
| 1 (u16 loop + DSD copy) | 26,739 | pending | | |

The simulator tracked hardware within 4% for kernel 0 and 0.3% for kernel 2, so kernel 1's
hardware window is expected at **26.7k ± 1.1k**. Kernel 2 is 21.7% faster than kernel 0 net at
16 PEs, **far less than the 2× I predicted from the row/NNZ fit**; the reason is that its
per-NNZ cost is not kernel 0's.

**Unknowns, stated plainly.** Each kernel has the form `local = d·KY + e·NNZ_max`. One 16-PE point
per kernel gives one equation in two unknowns; kernel 0 had three sizes, so only it has (a, b) =
(57.6, 27.7). Everything below is a chosen assumption plus the one point, not a fit.

**(a) Kernel 2.** The row loop and the copy loop are two DSD operations (zero `y`, copy to
`ew_buf`), so the per-row term is small: assume `d` = 2–4 cycles per row (1–2 cycles/element per
DSD op plus set-up; an assumption, not measured). Then `e` = (27,072 − d·250) / 730 = **36.0–37.1
cycles per NNZ**. Higher than kernel 0's 27.7 because each NNZ now also loads `row_ids[k]` and does
a read-modify-write of `y[row]`. Chain model as before, `c·W + 1.13·KY` with c = 12.8–13.7 (the
chain is still the least certain piece), measured floors 1,485 (65K) and 4,843 (full wafer):

| | KY | NNZ_max | local | chain | net | **window = net + floor** |
|---|---|---|---|---|---|---|
| 65K | 446 | 259 | 10.2–11.4k | 4.0k | **14.2–15.4k (≈ 14.8k)** | **15.7–16.9k cycles = 18.5–19.9 µs** |
| full wafer | 359 | 272 | 10.5–11.5k | 9.9–10.6k | **20.4–22.1k (≈ 21.2k)** | **25.3–27.0k cycles = 29.8–31.8 µs** |

That is about **2.5× faster than kernel 0 at 65K** (net 37.1k → 14.8k) and **1.8× at the full wafer**
(38.5k → 21.2k), although only 1.28× at 16 PEs: kernel 0's cost is mostly per-row, and 16 PEs
have 730 NNZ for 250 rows while 65K has 259 NNZ for 446. Kernel 2 is then **NNZ-bound, with the
EW chain nearly half of the full-wafer net**. It falsifies the model if the 65K net lands outside
14–16k, because that would mean `d` is not small (a larger per-row term) or `e` depends on
something other than NNZ_max.

**(b) Kernel 1.** Its hardware point is pending. Using the simulator point
(26,739 → about 25.9k local), the saving over kernel 0 is **8.8k cycles at 16 PEs (25%)**, and one
point cannot say whether it came from the per-row term (a), the per-NNZ term (b), or both. The
three bracketing hypotheses, each reproducing the 16-PE point, give:

| kernel 1 saves | a1, b1 | 65K local | 65K net | full-wafer local | full-wafer net |
|---|---|---|---|---|---|
| all of it per row (H1) | 23.1, 27.5 | 17.4k | 21.4k | 15.7k | 25.6–26.3k |
| proportionally (H3) | 43.4, 20.5 | 24.7k | 28.7k | 21.2k | 31.1–31.8k |
| all of it per NNZ (H2) | 58.2, 15.4 | 30.0k | 34.0k | 25.1k | 35.0–35.7k |

Kernel 1's 65K window (floor 1,485 added) therefore falls in **22.9–35.5k cycles (27–42 µs)** and
the full wafer in **30.4–40.5k cycles (36–48 µs)**: a bracket, not a prediction, until its hardware
point arrives and, after that, its 65K point fixes (a1, b1) exactly. The full wafer is then the
test.

**Update, kernel 1's hardware point (16 PEs): 25,757 cycles** (min 25,756, max 25,758; 20/20 valid;
simulator 26,739). Inside the band written above (25.6–27.8k). With the 448-cycle floor assumed
(its null twin is still to come): net 25,309, local 24,972 cycles, a **saving of 9,692 cycles
(28.0%)** over kernel 0 (34,664). Ranking at 16 PEs: kernel 1 25,757 < kernel 2 27,857 < kernel 0
35,449. Kernel 0's coefficients are taken as the exact two-point values (a, b) = (58.22, 27.54).
The same one point still cannot say where the saving came from, so the three bracketing cases are
restated with the hardware number (this replaces the simulator-based table above; chain 13.7·W
+ 1.13·KY at 65K, and 12.8–13.7 at the full wafer):

| kernel 1 saves | a1, b1 | 65K local | 65K net | **65K window** | full-wafer local | full-wafer net | **full-wafer window** |
|---|---|---|---|---|---|---|---|
| all of it per row (H1) | 19.5, 27.5 | 15.8k | 19.8k | **21.3k = 25.1 µs** | 14.5k | 24.4–25.0k | **29.2–29.9k = 34.4–35.2 µs** |
| proportionally (H3) | 41.9, 19.8 | 23.8k | 27.9k | **29.3k = 34.5 µs** | 20.5k | 30.4–31.0k | **35.2–35.9k = 41.4–42.2 µs** |
| all of it per NNZ (H2) | 58.2, 14.3 | 29.7k | 33.7k | **35.2k = 41.4 µs** | 24.8k | 34.7–35.4k | **39.5–40.2k = 46.5–47.3 µs** |

Kernel 1's **65K window is bracketed at 21.3–35.2k cycles (25–41 µs)** and its **full-wafer window at
29.2–40.2k cycles (34–47 µs)**. The width is the unknown one equation cannot remove: kernel 1's 65K
point is the next number that fixes (a1, b1) exactly (with the 16-PE point), and the full wafer is
then the test. This is a bracket, written before kernel 1 has run at 65K.

**Kernel choice, as it stands.** At 16 PEs kernel 1 (25.8k hardware) is ahead of kernel 2
(27.9k, 27.4k net) by 2.1k cycles. At 65K and the full wafer kernel 2's net (14.8k / 21.2k) is below
even the best case of kernel 1 (H1: 19.8k / 24.4k), so **kernel 2 is the expected winner at scale
under every hypothesis above**, unless its `d` or `e` assumptions fail. Kernel 1 can only win at
scale if kernel 2's per-NNZ cost grows with something the 16-PE point does not show.

### Result: kernel 2 at 65K, a miss on the low side, and the new full-wafer prediction

Kernel 2 at 65K (n = 1, 20/20 valid, rel err 1.49e-07): window **15,316 cycles = 18.0 µs** (min
15,315, max 15,319), net of the 1,485 floor **13,831**; per-PE span 6,360 / 11,670 / 14,654.
**Against the criterion written down beforehand (net 14–16k, window 15.7–16.9k): a miss.** It
landed *below* the band, by 169 cycles of net (about 400 in the window), in the favourable
direction, and is recorded as a miss, not a confirmation. It is 2.68× faster than kernel 0's net at
65K (37,113 → 13,831); I had said about 2.5×.

**Refit of kernel 2, exact on 16 PEs and 65K** (`local = d·KY + e·NNZ_max`; local = net − chain):

```
  16 PEs:  27,409 − 337   = 27,072 = 250·d + 730·e
  65K   :  13,831 − 4,011 =  9,820 = 446·d + 259·e
  →  d = 0.60 cycles/row,  e = 36.9 cycles/NNZ
```

I had assumed d = 2–4; the point says 0.6. **This solve is exact, so it is not a test, and d is
badly determined**: the two equations are nearly parallel (determinant 1,043 for coefficients of
size 250–730). A ±400-cycle error in the 65K chain term (the model is extrapolated) moves d
from −0.5 to +1.7 while e only moves 36.5–37.3. Equally consistent: d = 2 (two DSD passes at one
element per cycle) with e = 36.4 at 16 PEs and 34.5 at 65K, a 5% difference in e.

**What d ≈ 0.6 implies about kernel 2's cost.** It is almost entirely a per-NNZ loop at
~35–37 cycles/NNZ, and the per-row work (zeroing `y`, copying to `ew_buf`) is negligible, at most
a few cycles per row. Two readings, which these points cannot separate:
(i) the two DSD operations run faster than one element per cycle (a mem-to-mem `@fmovs` of f32 is
cheap, d < 1 is the data's central value); (ii) the per-NNZ cost is not constant: at 16 PEs there
are 2.9 NNZ per row, so consecutive NNZ often update the *same* `y[row]`, a read-modify-write
that must wait for the previous store; at 65K there are 0.58 NNZ per row, consecutive NNZ almost
never share a row, and the loop can run faster per NNZ. Reading (ii) is a hypothesis about a
hazard I have not seen in a trace; it predicts e at the full wafer (0.76 NNZ per row, similar to
65K) to be nearer 34.5 than 36.9. Either way, kernel 2's window is now **NNZ-bound**: at 65K
~9.8k of the 13.8k net is the NNZ loop, ~4.0k the chain.

**New full-wafer prediction for kernel 2, written before it runs.** KY 359, NNZ_max 272 (the
kernel-0 full-wafer runs on the same matrix; ± 5 NNZ would move the result ± 185 cycles),
measured floor 4,843, chain 12.8–13.7 cycles per PE:

| (d, e) used | local | chain | net | window |
|---|---|---|---|---|
| (0.6, 36.9): the exact solve | 10.25k | 10.57k | 20.8k | 25.7k |
| (2, 34.5): the reading-(ii) 65K value | 10.10k | 9.90–10.57k | 20.0–20.7k | 24.8–25.5k |
| (2, 36.4): the 16-PE value | 10.62k | 9.90–10.57k | 20.5–21.2k | 25.4–26.0k |

**Prediction: window 24.8–26.0k cycles = 29.2–30.6 µs, central 25.7k = 30.2 µs** (net ~20.8k, 1.85×
faster than kernel 0's net of 38,450). The earlier band (25.3–27.0k) is superseded by this one,
not erased: it was made before the 65K point. It is falsified by a window outside 24.5–26.5k:
low would say the chain is below 12.8 cycles/PE at W = 742 or e keeps falling with sparsity,
high would say the chain exceeds 13.7 or the floor shifts.

### Pre-registered: the resident weak-scaling grids, for kernel 2

Written **before bench runs any of them.** Kernel 2 ships, so the earlier grid table (which used
kernel 0's coefficients) is replaced by this one. All new grids have KX = KY = 446, d = 1e-3
(avg 198.9 NNZ/PE); `--num-pes W·H --grid-height H`, rows = 446·H, cols = 446·W.

Model: `window = floor + local + chain`, `local = d·446 + e·NNZ_max` with (d, e) ∈ {(0.6, 36.9),
(2, 34.5), (2, 36.4)} (the three readings of the 16-PE and 65K points; d is only determined to
± 1.1), `chain = c·W + 1.13·446` with c ∈ [12.8, 13.7] (central 13.25), floor from one of the three
shapes above (A: piecewise linear in W + H, B: `432 + 0.83·W + 3.29·H`, C: the quadratic).
NNZ_max is **estimated** as `198.9 + 0.95·sqrt(2 ln N)·sqrt(198.9)` (calibrated on 65K = 259 and
the full wafer = 272); **± 6 NNZ moves local by ± 220 cycles**. The worker records the true
`max_nnz_per_pe`; a result is judged with the measured value, not this estimate.

| # | W×H | PEs | NNZ_max (est.) | local | chain | **net** (range; central) | floor A / B / C | **window, central (floor A / B / C)** |
|---|---|---|---|---|---|---|---|---|
| 1 | 64×256 | 16,384 | 258 | 9.8–10.3k | 1.3–1.4k | **11.1–11.7k; 11.1k** | 1,090 / 1,326 / 1,078 | 12.2k / 12.5k / 12.2k |
| 2 | 742×256 | 189,952 | 265 | 10.0–10.5k | 10.0–10.7k | **20.0–21.2k; 20.4k** | 2,663 / 1,887 / 2,578 | 23.0k / 22.3k / 23.0k |
| 3 | 256×64 | 16,384 | 258 | 9.8–10.3k | 3.8–4.0k | **13.6–14.3k; 13.7k** | 1,090 / 854 / 1,078 | 14.8k / 14.5k / 14.8k |
| 4 | 256×1155 | 295,680 | 266 | 10.1–10.6k | 3.8–4.0k | **13.9–14.6k; 14.0k** | 3,665 / 4,441 / 3,580 | 17.6k / 18.4k / 17.6k |
| 5 | 512×512 | 262,144 | 266 | 10.1–10.6k | 7.1–7.5k | **17.1–18.1k; 17.4k** | 2,726 / 2,538 / 2,640 | 20.1k / 19.9k / 20.0k |
| 6 | 742×100 | 74,200 | 262 | 9.9–10.4k | 10.0–10.7k | **19.9–21.1k; 20.3k** | 2,285 / 1,374 / 2,217 | 22.6k / 21.6k / 22.5k |
| 7 | 64×1155 | 73,920 | 262 | 9.9–10.4k | 1.3–1.4k | **11.3–11.8k; 11.3k** | 3,199 / 4,282 / 3,106 | 14.5k / 15.6k / 14.4k |
| (4×4) | | 16 | 730 | | | measured 27,409 | 448 (measured) | measured 27,857 |
| (65K) | 256×256 | 65,536 | 259 | | | measured 13,831 | 1,485 (measured) | measured 15,316 |
| (full) | 742×1155 | 857,010 | 272 | | | 20.0–21.2k | 4,843 (measured) | 24.8–26.0k (see above) |

(Cycles; ÷ 850 for µs, e.g. #2 under floor A: 23.0k = 27.1 µs.)

**What would falsify each floor shape.** The null twin gives the floor directly; call a shape
falsified at a grid when the measured floor is more than **50 cycles** from its value (the 16-PE floor
reproduced exactly, 448 and 448, in two sessions; the larger sizes were measured once, and a twin's
outlier passes are excluded by taking the median of 20).

- **B (W and H apart)** is the most separable: it differs from A and C by 236–1,083 cycles at
  grids 1, 2, 3, 4, 6 and 7 (grid 6: 1,374 against 2,285; grid 7: 4,282 against 3,199). It is
  falsified by any of those grids landing at A's or C's value, and A and C are falsified by landing
  at B's.
- **A against C cannot be separated by this set.** They agree at the three fitted points by
  construction and differ by 12–93 cycles at the seven new grids (largest: grid 7, 93; grid 5, 86;
  grid 2 and 4, 85; grid 6, 68), around the 50-cycle tolerance. Telling them apart needs a size well beyond
  the fitted range or a smaller tolerance; I do not claim the set does it.
- **None of the three** if a floor lands more than 50 cycles from all of them: then the shape is
  something else and ten points suffice to fit a two- or three-term model; this is the outcome I
  would not be surprised by (three points fitted three shapes exactly, which says nothing about a
  fourth).
- Transposed pair (grids 1 and 3, same PE count): A and C predict **equal** floors (1,090 / 1,078),
  B predicts 1,326 against 854, a 472-cycle difference.

**What would falsify each chain value.** At **fixed KY = 446** the compute term varies between grids
only through NNZ_max (258–266, measured exactly), so the chain slope is read from the W series,
`W = 64` (grid 1), `256` (the 65K point, measured) and `742` (grid 2):

```
  c = [ net(742×256) − net(64×256) − e·(NNZ_max(742) − NNZ_max(64)) ] / 678
```

Criterion: **c within 12.8 ± 0.15 supports 12.8; within 13.7 ± 0.15 supports 13.7; anything else
(13.25 would be the midpoint) supports neither.** The error on c is ± 0.03 from the e·ΔNNZ term
(e ± 2.4, ΔNNZ ≈ 7) plus the measurement jitter (a few cycles over 678), so ± 0.15 is conservative
(the null twins fix the floors; an outlier pass in a null twin of 18k cycles is why its median is
used). The third point (65K: net 13,831 measured) must lie on the line through the other two
within the NNZ_max term: if it does not, the chain is not linear in W. Separately, at **W = 256, KY
= 446** the three grids 3, 4 and the measured 65K point (H = 64, 1155, 256) must have equal net
within NNZ_max (13.6–14.3k, 13.9–14.6k, 13.8k measured): nets differing by more than ~500 cycles
falsify "the net does not depend on H".

**How much better kernel 2 discriminates the chain, stated carefully.**
- The chain's *share* of the net rises: 29% at 65K (4.0k of 13.8k) against 11% with kernel 0, and
  about 51% at the full wafer (10.6k of 20.8k) against 27%.
- For a **single point** that matters: the 0.9 cycles/PE difference between c = 12.8 and 13.7 is 670
  cycles at W = 742, **3.2% of kernel 2's net against 1.7% of kernel 0's**, and kernel 2's compute
  model spread at the full wafer is ± 260 cycles (10.1–10.6k) where kernel 0's was about ± 850.
  So the full-wafer window alone separates the two values at ~2.6σ for kernel 2 and not usefully for
  kernel 0.
- The **differential** W-series estimate above does not improve with the kernel: compute cancels at
  fixed KY and measured NNZ_max for either, and its error (± 0.03) is already small. What kernel 2 buys
  is that the single full-wafer run and the 742×100 and 742×256 grids can also be read, as
  cross-checks, without leaning on a poorly-known compute model.
- What this set does **not** test: **d** (every new grid has KY = 446, so d is not separable from
  the constant; only the full-wafer run, KY = 359, adds a KY), and **whether e depends on NNZ per
  row** (every new grid has 0.58 NNZ per row at NNZ_max, like 65K).

### Scorecard after kernel 2 at the full wafer (release candidate, n = 3)

Kernel 2, full wafer (857,010 PEs, rel err 2.75e-07, 60/60 passes valid): window **25,193 cycles
= 29.64 µs**, min = median in all three runs; outlier passes 27,141 / 25,235 / 25,197. Null twin:
**4,845** on all 20 passes (kernel 0's twin gave 4,843: the floor reproduces across kernels and
runs). Net = **20,348 cycles = 23.9 µs**. 65K, n = 3: median 15,316 (60/60 valid), one outlier
pass at 25,726.

| prediction (made when) | said | result | verdict |
|---|---|---|---|
| kernel 2, 65K, before it ran | net 14–16k, window 15.7–16.9k | net 13,831, window 15,316 | **miss**, 169 net cycles low |
| kernel 2, full wafer, before it ran (after the 65K point) | window **24.8–26.0k** (falsified outside 24.5–26.5k), net ≈ 20.8k (20.0–21.2k) | window **25,193**, net **20,348** | **hit**; net −475 cycles (−2.3%) from the central value, inside its range |
| the earlier kernel-2 full-wafer band, made before the 65K point | 25.3–27.0k | 25,193 | superseded; it would have missed by 107 cycles |
| kernel 1, 65K, from the 16-PE point (three cases) | 21.3–35.2k | 23,756 (reported by the bench) | inside the bracket, between "all per row" (21.3k) and "proportional" (29.3k) |
| floor at the full wafer | line 4,338; shapes A/B/C fitted to 3 points | 4,843 and 4,845 | the line **missed** (+12%); the floor itself is reproducible |

Relative to kernel 0: kernel 2 is **1.27× faster at 16 PEs, 2.52× at 65K and 1.72× at the full
wafer** (window; 1.89× on the net, 38,450 → 20,348).

**What the three kernel-2 points say about d, e and the chain.** Using the measured floors and
`net − 1.13·KY = d·KY + e·NNZ_max + c·W` (16 PEs: KY 250, NNZ 730, net 27,409; 65K: 446, 259,
13,831; full wafer: 359, 272, 20,348), three equations, **two unknowns at a fixed c**, so the
residuals are a real test of the chain value (unlike the exact solves earlier):

| chain c (cycles/PE) | least-squares d, e | residuals (16 PEs / 65K / full wafer) | rms |
|---|---|---|---|
| 12.8 | d = 1.28, e = 36.66 | +2 / +12 / −17 | **12** |
| 13.25 | 0.64, 36.85 | −15 / −106 / +142 | 102 |
| 13.7 (the M6–M8 fit) | 0.01, 37.04 | −33 / −223 / +300 | **217** |

Freeing c (exact 3 × 3 in d, e, c) gives **d = 1.21, e = 36.68, c = 12.85**. Kernel 0's three points,
refitted the same way with the measured floors, **independently** prefer c: 12.8 gives (a, b) =
(58.8, 27.3) with residuals −1 / −9 / +12 (rms 9), 13.7 gives rms 238, and the exact solve gives
c = 12.76. One fit sharing a single c across both kernels' six points (five parameters, one degree
of freedom) gives **c = 12.81 with residuals within ±15 cycles** (rms 11). So the data say **the EW
chain costs about 12.8 cycles per PE at W up to 742, not 13.7**; the 13.7 extrapolated from
W ≤ 256 over-predicted the full wafer by ~670 cycles. This is a consistency result with one
degree of freedom, not a proof, and it is conditional on the compute model being `d·KY + e·NNZ_max`.
The W-series grids above test it directly.

**Do 0.6 cycles/row and 34.5–36.9 cycles/NNZ survive?**
- **d:** the point estimate moves to **d ≈ 1.0–1.3** (0.6 was within the stated ± 1.1), consistent
  with two DSD passes over KY at close to two elements per cycle; still not directly measured.
- **e:** **36.7 ± 0.3, constant.** At d = 1.28 and c = 12.8, e per grid is 36.65 (16 PEs, 2.9 NNZ
  per row), 36.6 (65K, 0.58) and 36.7 (full wafer, 0.76).
- **The NNZ-per-row hazard hypothesis is not supported.** It predicted a lower e at the sparse
  grids (nearer 34.5 than 36.9 at the full wafer); the data show e flat across a 5× range in NNZ
  per row. A lower sparse-grid e reappears only if d is forced to 2 *and* c to 13.25 (e = 36.4, 34.9,
  34.5), the combination the chain residuals above disfavour. Both d and e are partly degenerate
  with c at three points; a 16-PE run at another density with the same NNZ_max would test the
  hazard directly and has not been run.

**Status for the release candidate.** Kernel 2 is the shipping kernel: it is validated at 16 PEs,
65K and the full wafer (rel err 1.5e-07 to 2.8e-07 on every pass, 60 of 60 passes at both large
sizes), the on-wafer window is **18.0 µs at 65K and 29.6 µs at the full wafer** (net of the
launch floor: 16.3 µs and 23.9 µs), deterministic to a few cycles with an occasional single-pass
outlier, so quote the median. That is about 190× (65K: 3.5 ms) and 2,100× (full wafer: 61.7 ms) the
32-thread host CPU on the same matrices, for a per-call host cost of ~1 ms through the runtime. The
cost is now ~37 cycles per NNZ plus an EW chain that is half of the full-wafer net and costs about
12.8 cycles per PE, with a launch floor that grows with the grid (448 / 1,485 / 4,843 cycles) and
whose shape is not known. What is not done: the on-wafer y → x transpose (so no CG iteration
yet), the weak-scaling grids that pin the floor shape and the chain, the one-PE-tall H2D rate, and
the host RAM peak at the full wafer.

### Real matrices on the release candidate: two misses at 16 PEs, and predictions before the 65K and full-wafer runs

**New data, 16 PEs (4×4), n = 1, kernel 2** (floor 448): nos6 (675 × 675, 3,255 NNZ; KX = KY = 169;
NNZ_max 786, 3.9× the mean) window **32,182** (min 32,181, max 32,184); bcsstk01 (48 × 48, 400 NNZ;
KX = KY = 12; NNZ_max 60) window **2,832**. Against the uniform-matrix model (d = 1.2, e = 36.7,
c = 12.8, measured floor; the busiest PE sets the window):

| matrix | NNZ_max (per row at the busiest PE) | model | measured | miss | effective e |
|---|---|---|---|---|---|
| nos6 | 786 (4.65) | 29,739 | 32,182 | **+2,443 (+8.2%)** | 39.8 |
| bcsstk01 | 60 (5.0) | 2,704 (2,729 with the chain placed at W) | 2,832 | **+128 (+4.8%); +103 (+3.8%)** | 38.8 (38.4) |
| uniform, 16 PEs (fitted) | 730 (2.92) | 27,857 | 27,857 | 0 (by construction) | 36.7 |

Both misses are in the same direction (slower than modelled) at more NNZ per row than any uniform
point (65K: 0.58, full wafer: 0.76, 16-PE uniform: 2.92). **The earlier statement that the
NNZ-per-row hazard hypothesis is "not supported" is withdrawn as too strong:** it rested on
uniform points at 0.58–2.9 NNZ per row, which the new points do not cover. It is open again.

**What else the two points are consistent with.** Three points cannot settle it, so the
candidates, with what each explains:

1. **e rises with NNZ per row** (a read-modify-write on the same `y[row]`): `e = 33.0 + 1.30·r` by
   least squares on the three 16-PE points (r = 2.92, 4.65, 5.0 with e_eff = 36.66, 39.81, 38.84;
   residuals +0.13 / −0.77 / +0.64): three points, two parameters, **one degree of freedom, and
   the points are not monotone** (5.0 sits below 4.65), so this is weak.
2. **A constant per-call cost of about 100–130 cycles** (task activation, two DSD op start-ups).
   It explains bcsstk01 completely (+128) and **about 5% of nos6's residual**, so it is not the
   cause of the nos6 miss; it would be invisible in the uniform fits (absorbed in d).
3. **Structure of the matrix, not NNZ per row:** nos6 and bcsstk01 are banded, so consecutive NNZ
   have nearby columns (`x_vec` loads at adjacent addresses) and the same row (`y[row]` loads and
   stores at one address); the uniform random matrix has neither. A cost that grows with
   locality (memory-bank conflicts, store-to-load waits) would raise e for any banded matrix at
   any NNZ per row. This is a hypothesis with no trace behind it.
4. **Not the explanation:** a different per-row cost d (it would need 8.6 and 14.5 extra cycles
   per row, inconsistent between the two, and the uniform fit fixes d ≈ 1.2); the imbalance across
   PEs at 16 PEs (the chain is 4 hops, ~50 cycles).

**Model used for the predictions below.** Per PE: `T = d·KY + e·NNZ_pe + c·(px + 1)`; the window is
`floor + max over PEs of T + 1.13·KY`. For a uniform matrix this is the model above (the east-end
PE has the longest chain). **For a banded matrix the maximum is not at the busiest PE:** the
chain runs through all W columns, and a PE with no NNZ at the east end already costs `c·W`, so a
matrix that occupies only the western PEs still pays the full chain. d = 1.2, c = 12.8, floors
measured (1,485 at 65K, 4,845 at the full wafer), NNZ per PE computed by tiling the actual files
(the fit report's counts; the 4 × 4 NNZ_max reproduce 786 and 60). Three e models:

- **(a) flat:** e = 36.7.
- **(b) hazard in r:** `e = 33.0 + 1.30·(NNZ_pe / KY)`, extrapolated beyond the fitted range 2.9–5.0.
- **(c) structure:** e = 39.8 for every banded matrix (the nos6 value), whatever r.

**Pre-registered predictions** (before the 65K and full-wafer runs; window = net of nothing, i.e.
floor included; µs = cycles / 850):

| matrix, grid | KX = KY | NNZ_max (r) | critical PE | **(a) flat** | **(b) hazard** | **(c) structure** |
|---|---|---|---|---|---|---|
| nos6, 256 × 256 | 3 | 7 (2.33) | an empty east-end PE | **4,769 (5.6 µs)** | 4,769 | 4,769 |
| bcsstk01, 256 × 256 | 1 | 1 (1.0) | an empty east-end PE | **4,764 (5.6 µs)** | 4,764 | 4,764 |
| bcsstk14, 256 × 256 | 8 | 64 (8.0) | PE (224, 224), 62 NNZ | **6,659 (7.8 µs)** | **7,053 (8.3 µs)** | **6,851 (8.1 µs)** |
| parabolic_fem, 742 × 1155 | KX 709, KY 456 | 1,368 (3.0; mean 4.3) | PE (717, 461), 1,368 NNZ | **62,027 (73.0 µs)** | **62,292 (73.3 µs)** | **66,267 (78.0 µs)** |

How each matrix is read:

- **nos6 and bcsstk01 at 65K (a sharp check of the chain, not of e):** both occupy only the western
  PEs (nos6: 225 of 256 columns; bcsstk01: 48), so the window is the empty east-end PE's
  `floor + c·256 + d·KY + 1.13·KY`, **the same for both within ~5 cycles** (their KY differ by
  2). Prediction **4,769 and 4,764, tolerance ± 150** (± 40 from c = 12.8 ± 0.15, ± ~100 for a
  constant per-call cost of the kind in candidate 2). Outside it, the chain does not extrapolate
  to KY → 1–3 (the chain figure was fitted at KY ≥ 250), or a per-call constant is larger than
  thought. The two windows differing by more than ~30 cycles would say the busiest PE matters,
  i.e. the east-end argument is wrong.
- **bcsstk14 at 65K (the one that separates e models):** `e·62` is 2.3k of the 6.7k window, so
  (a) 6,659, (c) 6,851, (b) 7,053: **a window below 6,810 falsifies (b) and (c); above 6,900
  falsifies (a); 6,810–6,900 favours (c) but does not settle it** (model tolerance ± 150: the d,
  c and constant-cost uncertainties).
- **parabolic_fem at the full wafer:** its busiest PE has 1,368 NNZ, **5× the uniform wafer's 272**,
  because a mesh matrix on block tiling keeps its NNZ near the diagonal (17,125 of 857,010 PEs
  hold any; the mean over those is 215). Its mean is 7.0 NNZ per row for the matrix, **but the
  busiest PE holds exactly 3 per row** (1,368 / 456), because each stencil row is split across PEs
  by column; the hazard model (b) therefore predicts almost no change (e = 36.9), and **(c) is
  the model that separates**: (a) and (b) 62.0–62.3k, (c) 66.3k, a **~4k gap (6%)**, against a
  model uncertainty of ± 0.3 in e (±410 cycles) and ± 0.15 in c (±110): ± ~600 in all
  (61.5–62.5k for (a)). **A window above 63.5k falsifies (a) and (b); below 63k falsifies (c).**
  It also tests the model at 5× the NNZ_max and at a location of the critical PE
  (px = 717) away from the extreme east end; it fits in SRAM (about 6.2k words of 9.7k).
- **What would falsify the pair (b)/(a) overall:** bcsstk14 decides between them, parabolic_fem
  cannot (r = 3.0 is inside the fitted range).

### Scoring the 65K real matrices (kernel 2, 256 × 256, n = 1, 20/20 valid, floor 1,485)

| matrix | registered window (a) / (b) / (c) | measured | span min / avg / max | verdict |
|---|---|---|---|---|
| nos6 (KX = KY = 3, NNZ_max 7) | 4,769 ± 150 | **4,940** (min = median = avg, max 4,943) | 50 / 1,932 / 3,783 | **miss, +171 (21 outside tolerance)** |
| bcsstk01 (KX = KY = 1, NNZ_max 1) | 4,764 ± 150 | **4,939** (average 5,231: one outlier pass of 10,767) | 50 / 1,932 / 3,783 (identical to nos6) | **miss, +175 (25 outside)** |
| bcsstk14 (KX = KY = 8, NNZ_max 64) | 6,659 / 7,053 / 6,851 | **6,243** on all 20 passes | 53 / 2,223 / 5,803 | **all three falsified**: 416 below the lowest |

The structural claim made for the first two **held**: the two windows are within **1 cycle** of each
other (registered: within ~5; falsified if more than 30 apart), with identical spans, so the matrix
is barely visible: window = floor + chain. The levels missed by the same ~173 cycles.

**What the +173 is.** A constant. Fitting `net = K + d·KY + e·NNZ + h·(W − 1) + 1.13·KY` by least
squares to the five uniform or empty points (16 PEs, 65K and the full wafer uniform, plus nos6 and
bcsstk01 at 65K, which have no compute and so measure the chain directly) gives **K = 186,
d = 0.93, e = 36.5, h = 12.80 cycles per hop, with residuals 0 / 0 / 0 / +2 / −2**: five equations,
four unknowns, one degree of freedom, the first real test of the chain at KY → 1 and it passes. The
16-PE estimate of the constant (100–130) was low because it was confounded with e at 12 rows; the
65K points, where the chain dominates, give 175–186. So there is a **fixed ~186-cycle cost per
call** on top of everything fitted before (task activation and DSD start-up), and the chain value
**c = 12.8 stands** (the same value both kernels' uniform points preferred).

**bcsstk14: all three e-models falsified; what went wrong is the model's structure, not (only) e.**
The per-PE spans (spmv entry to chain done, last pass) are measured relative to each PE's own
start and carry the chain's ramp, so they can be decomposed with the real per-PE NNZ (tiled from
the file; the busiest PE has 62 NNZ at PE (224, 224) in a 256 × 256 grid, 8 rows):

- **Ramp.** nos6's spans fall linearly from 3,783 at the west end to 50 at the east end: **14.64
  cycles per hop** in span terms (this includes the dispatch skew along x; the true hop, from the
  windows, is ~13.4 including the constant, or 12.8 with K = 186).
- **Span of the slow PE.** A PE finishes at the maximum over PEs east of it of `compute + hops·h'`.
  With the actual NNZ per PE and a free e, **the maximum span (5,803) gives e = 39.9 and the mean
  span (2,223, against 1,932 for the empty matrix, a rise of 291) gives e = 40.3**: two independent
  statistics of the same decomposition agree to 1%, so the structure "compute at a PE, then the chain
  from that PE's position" is right.
- **Why the registered windows were wrong then.** They added the full 1,485-cycle floor to every
  PE's `compute + chain`. The floor is the *dispatch skew* across the grid (the span between the
  earliest and the latest PE to receive its commands), and it is **position-dependent**: PE (x, y)
  starts at about `σx·x + σy·(H − 1 − y)` after the earliest, so a PE away from the late corner
  carries less of it. The slow PE of bcsstk14 sits 31 rows from that corner and loses
  `31·σy` (about 90–100 cycles) and, relative to the east-end reference, 31 hops of chain; added
  to the constant floor the model credited it ~600 cycles it does not have. With the skew
  per-position (σx = 0.83–1.29, σy = 2.85–3.29 cycles per PE, from the floor shapes and the span
  ramp), the window fits bcsstk14 to −20 / +64 cycles at e = 39.2 and to −188 / −104 at e = 36.5.
  **The vertical sense of the skew (later starts at *smaller* y) is read off this one point and
  rests on it.**

**What this says about e.** The effective e at the busiest PE is **~39–40 on all three real
matrices** (nos6 39.8 at 4.65 NNZ per row, bcsstk01 38.8 at 5.0, bcsstk14 39.9–40.3 at 8.0) and
**36.5 on uniform matrices** at 0.6–2.9. Registered e-models: (a) flat 36.7 is **falsified as an
e-model** (too low by ~3); (b) `e = 33.0 + 1.30·r` is **falsified** (it predicted 43.4 at r = 8 where
the data say 40; the dependence on r saturates or is not about r); **(c) e = 39.8 for banded
matrices is right on e** although its window was wrong (structure). Data fit equally "a step between
r ≈ 3 and r ≈ 4.65" and "banded structure", and cannot tell them apart; neither has a trace behind it.
The earlier reading that bcsstk14 contradicts a higher e was an artefact of the position-blind floor.

**Parabolic_fem on the full wafer.** The registration above stands unchanged (62,027 / 62,292 /
66,267). Revised under the corrected model, **post-hoc, written before the run lands (2026-10-15,
12:28 local, UTC-3)**:

| e used | σx = 0.83, σy = 3.29 | σx = 1.29, σy = 2.85 |
|---|---|---|
| 36.5 (uniform) | 59.2k | 59.4k |
| 38.2 | 61.5k | 61.7k |
| **39.2 (real matrices, central)** | **62.9k = 74.0 µs** | **63.1k = 74.2 µs** |
| 40.3 | 64.4k | 64.6k |

**Post-hoc prediction: 63.0k cycles (74.1 µs); 61.5–64.6k over e = 38.2–40.3 and both skew splits;
59.2–59.4k if e were the uniform value.** The critical PE is (717, 461) with 1,368 NNZ: with the
skew per position the full floor is not added, which is why the model with e = 36.5 falls ~2.8k below
the registered (a). Consequence for reading the run: the registered rule (above 63.5k falsifies (a)
and (b), below 63k falsifies (c)) now **lies in the middle of this prediction**, so the run
discriminates less than registered; it can still say whether e is the uniform 36.5 (59.2–59.4k,
which would end the real-matrix e ≈ 39 reading) or ≈ 39 (61.5–64.6k), and it re-tests the skew
reading at a 1155-row grid.

### Scoring parabolic_fem on the full wafer (kernel 2, RC, n = 1)

parabolic_fem (525,825², 3,674,625 NNZ; grid 742 × 1155, KX = 709, KY = 456; NNZ_max **1,368**,
319× the mean; 24.2 KB per PE): window **64,080 cycles = 75.4 µs** (min = median = avg, max 64,082),
20/20 valid, rel err 9.32e-08; per-PE span min / avg / max = 884 / 27,788 / 62,352. Only **17,125 of
857,010 PEs hold any NNZ.**

| prediction | said | measured 64,080 | verdict |
|---|---|---|---|
| (a) flat e = 36.7, registered | 62,027 (± 600) | +2,053 (+3.3%) | **falsified** (above 63.5k) |
| (b) hazard in r, registered | 62,292 | +1,788 | **falsified** (above 63.5k) |
| (c) structure e = 39.8, registered | 66,267 (± 600) | −2,187 (−3.3%) | outside its own tolerance, though not by its stated rule (below 63k) |
| post-hoc row (written 12:28, before the run) | 63.0k (61.5–64.6k) | +1,080 (+1.7%) | **inside the range** |

**Effective e from the per-PE spans** (the decomposition used for bcsstk14: a PE's span is
`B + max over PEs east of it of (e·NNZ + h′·hops)`, with B = 884 the empty-PE span, h′ = 14.64
from nos6, and the real NNZ grid tiled from the file): the **maximum span gives e = 40.0**, the
**mean span gives e = 41.4**; the two statistics disagree by 3.5% here (they agreed to 1% for
bcsstk14), and e = 40.0 predicts a mean of 26,969 against 27,788 measured (−3%). Take **e ≈ 40–41**:
about **10–13% above the uniform 36.5**, at the same 3.0 NNZ per row as the uniform 16-PE point
(2.92) where e was 36.5. **So e is not a function of NNZ per row**: the hazard reading (b)
(`e = 33.0 + 1.30·r`, which says 36.9 at r = 3.0) is falsified by this point too, and the
real-matrix pattern (nos6 39.8, bcsstk01 38.8, bcsstk14 39.9–40.3, parabolic_fem 40.0–41.4) is a
**property of these matrices' structure** (a mesh or banded matrix against a random one), not
of their density per row. What in the structure is not known.

**The position-dependent skew holds, and now has a number.** `window − slowest span` is 1,728, not
the 4,845 of the full floor. The decomposition `window = Q + max_y[σy·(H − 1 − y) + span₀(y)]` has
two constants that the 65K matrices fix: nos6 (1,157 = Q + 255·σy) and bcsstk14 (440 = Q + 31·σy,
for its slow row y* = 224) give **σy = 3.2 cycles per PE row and Q = 341**. Applied to this grid
*without refitting*: the row of the maximum span is y = 717 (from the real NNZ grid), `341 + 3.2 ·
(1154 − 717) = 1,739` against **1,728 measured**; going the other way, 1,728 implies y* = 720.6, four
rows from 717. With e = 40.0 the model window is 64,091 against 64,080. The check is the row (and
the 1,728), not the final window, which used the measured maximum span. With the full-wafer floor
4,845 = `Q + σx·741 + σy·1,154` this gives **σx ≈ 1.1**; the three floors then follow
`341 + 1.09·(W − 1) + 3.2·(H − 1)` to −94 / −50 / 0 cycles (16 PEs, 65K, full wafer): a
form of floor shape **B** (H-dominated: σy about 3× σx). That is consistent with B, **not a test** of
it; the registered weak-scaling grids are. (The null twin on this run will check the 4,845
this uses.)

**What it means for the model.** The structure (compute at a PE, then the chain from that PE's
position, with the dispatch skew carried by position rather than added once) is confirmed at a
second grid size, and the registered "floor + max(compute + chain)" is superseded by it: for a
matrix whose work sits away from the corner where commands arrive last, only part of the floor
shows (here 1,728 of 4,845). What is **not** pinned: e (40–41 here against 38.8–40.3 on the 65K
matrices: a ±3% scatter across matrices that one n = 1 each cannot attribute), and the 3% under-
prediction of the mean span.

**What can be claimed about real matrices on the wafer.** One real matrix (parabolic_fem, 3.67M NNZ)
ran at the full wafer, validated to 9.3e-08, in **75.4 µs** (net of the launch floor, 59,235
cycles = 69.7 µs), and the 65K matrices ran validated at 4,940–6,243 cycles; the uniform and real
matrices differ in e by ~10%, and the decomposition predicts the real-matrix window to ~2%
(63.0k against 64.1k post-hoc). Only **2% of the PEs hold any NNZ** and the busiest PE holds 319×
the mean, so the window is set by **one PE's compute** (about 55k of the 64k cycles), not by
the fabric: if the same NNZ were spread evenly (4.3 per PE; block tiling cannot do this, a cyclic
distribution that the tiler does not implement could) the same model gives about **15.6k cycles
(18 µs)**, 4× lower. That is a model estimate, not a measurement. It is also the answer to "is it
fast on real matrices": the kernel is, the block tiling of a banded or mesh matrix is what costs.
No host CPU time exists for this matrix; at the uniform host rate (5.6 GFLOP/s) it would be about
1.3 ms, so roughly 17× the wafer window, an estimate that stays one until measured.

### The launch itself is inside the window: `--null-spmv`

The window runs from `f_tic`'s stamp to `f_toc`'s, and both stamps are taken when *that PE's*
command stream reaches the function. So the window contains, besides the SpMV, the **command
stream's latency**: the launches reach PEs over the fabric at ~1 cycle/hop, so PE (x, y) sees
`f_tic` up to (x + y) cycles after PE (0, 0), and each of the three launched functions costs
the RPC dispatch on every PE. That is the dispatch skew in the table (0.5 µs at 65K, up to
2.2 µs at the full wafer), plus a few hundred cycles per PE, **if** the stream behaves like a
1-cycle/hop broadcast. Nobody has measured it; if the stream is slower the "skew" column is
the part that is wrong, and it grows with W + H, not with NNZ.

It shows up as an offset that **does not scale with NNZ or KY** and that the window has even
for an empty SpMV. To separate it, `--encoding resident --null-spmv` compiles the same kernel
with `spmv()` as a no-op that calls `unblock_cmd_stream` at once (`ENCODING == 4`: the same
kernel file and artifact layout, a different compile param, so no existing option string
moves). The identical host pass (`f_tic`, `spmv`, `f_toc`, the barrier) then times *only* the
launch machinery: `wafer_time_avg` of the null run is the floor, and

```
SpMV on the wafer  ≈  wafer_time_avg(resident)  −  wafer_time_avg(null-spmv)     (same size)
```

Treat the difference as a close estimate, not exact: the skew is not additive with the PE's
work (a late-tic PE finishes late, but the chain is pipelined behind it). The null run skips
validation (y is never written; `exit_reason` is `SUCCESS_NULL_SPMV`, `null_spmv` is True),
and reports `wafer_gflops_avg` 0. Prediction: **0.5–3 µs at 16 PEs
and 65K, 1–5 µs at the full wafer**; a value above ~10 µs means the command stream, not the
kernel, sets the floor, and every resident window should be read net of it.

### Why the x finding matters: the case for an on-wafer transpose

Resident SpMV makes y in ~20 µs. CG needs that y as the next x, and with the 2-D layout x is
replicated down every PE column, so the next SpMV needs `y` re-laid-out as `x` on every PE
row: a transpose. If the host does it, the iteration is `y` D2H → host → `x` H2D, and the
prediction above says that **loses to the host CPU alone**: the replicated x is ~160 ms at the
full wafer (61.7 ms for the host's entire SpMV), and even n unique floats into one PE row is
~2 ms, ~100× the SpMV. A D2H of y adds a further ≥ 167 µs round trip. So resident A is
necessary but not sufficient: the wafer wins only if **x never leaves it**, i.e. an on-wafer
y → x transpose (south conveyor / column broadcast, ADR-005), which is the one piece
of CG that is neither built nor measured (the transpose is not a broadcast; ADR-005 amendment).
The estimate above is the gap to `wafer_time_avg` that the on-wafer transpose has to close.

## Commands for bench

Stop at the first failure. Compile is per job (`ENCODING:3`, `SYNC_TIMING:1`). Cancel at 2×
the usual: 60 s for 16 PEs, 5 min at 65K, 45 min at the full wafer; a
hang is diagnosed in the simulator at 16 PEs, never on the wafer.

```bash
# 0. the simulator, 16 PEs: compile, setup layout, 20 passes. Do this first.
uv run python run.py run --rows 1000 --cols 1000 --density 0.01 --num-pes 16 --simulator --encoding resident --sync-timing --loop-count 20 --output cluster/results/data/sim_resident.csv
# 1. smoke, 16 PEs, hardware. Expect rel_err ~5e-8 (the 16-PE figure), 20/20 passes PASSED.
uv run python run.py run --rows 1000 --cols 1000 --density 0.01 --num-pes 16 --encoding resident --sync-timing --loop-count 20 --output cluster/results/data/resident.csv
# 1b. the launch floor, 16 PEs (--null-spmv: spmv() is a no-op; validation is skipped)
uv run python run.py run --rows 1000 --cols 1000 --density 0.01 --num-pes 16 --encoding resident --null-spmv --sync-timing --loop-count 20 --output cluster/results/data/resident_null.csv
# 2. 65K PEs (114,000^2, 13.0M NNZ), n = 3: repeat this line three times.
uv run python run.py run --rows 114000 --cols 114000 --density 0.001 --num-pes 65536 --encoding resident --sync-timing --loop-count 20 --output cluster/results/data/resident.csv
# 3. full wafer (857,010 PEs, 414,000^2, 171.4M NNZ), n = 3: repeat three times, after 2 is clean.
uv run python run.py run --rows 414000 --cols 414000 --density 0.001 --num-pes 857010 --encoding resident --sync-timing --loop-count 20 --output cluster/results/data/resident.csv
```

The same three sizes as one sweep: `run.py sweep bench/configs/resident.toml --max-runs 1`
(then 2, then 3; never all blind). `bench/configs/sim-resident.toml` is the simulator version
(16, 64 and 256 PEs).

Once each size passes, run its `--null-spmv` twin (the 1b line with that size's `--rows --cols
--density --num-pes`, once each, n = 3 at 65K) so every window can be read net of the launch.

**Check in the worker log.** Per pass, `Pass k/20: ... on-wafer N cycles = X ms (r of
host-through-barrier ...)` and `PASSED`; no `RECORDED AS 0`, no `EXCEEDS`; `Validation PASSED` with every pass listed. `relative_error` is the worst pass.
Then the table above, with the *unsynced-vs-synced* sanity: `compute_hw_cycles_max` (per-PE
span, last pass) ≤ the synced window, and within ~2.5 µs of it.

## The local kernel: COO scatter (chosen on hardware)

The first resident kernel ran the local SpMV as the CSR row loop `pe_valtask.csl` has always had
(`for r < KY: acc = 0; for k in row_ptrs[r]..row_ptrs[r+1]: acc += vals[k]·x[col[k]]; y[r] = acc`). It was written for
correctness and costs about 48 cycles per NNZ on a hypersparse tile (0.5 NNZ per row). Two replacements were built as
compile-time variants, run on the same matrices (every run 20/20 valid, relative error 5.3e-08 at 16 PEs, 1.5e-07 at
65K), and the one with the lowest valid window at scale was kept:

| kernel | local SpMV | 16 PEs (250 rows, NNZ_max 730) | 65K PEs (446 rows, NNZ_max 259) | extra SRAM |
|---|---|---|---|---|
| 0 | CSR row loop, `i32` loop state | 35,449 cycles = 41.7 µs | 38,598 cycles = 45.4 µs | none |
| 1 | the same loop on `u16` loop state; `ew_buf` copy as one DSD move | **25,757** = 30.3 µs | 23,756 = 27.9 µs | none |
| **2** | **COO scatter: `y` zeroed by one DSD op, then every NNZ adds into its row (`row_ids`)** | 27,857 = 32.8 µs | **15,316 = 18.0 µs** | + TOTAL_NNZ words (`row_ids`) |

Windows are the synced on-wafer window (max toc − min tic, cycles at 850 MHz). **Kernel 2 ships**, by the rule "lowest valid
window at scale": kernel 1 was fastest at 16 PEs, but at 65K kernel 2 is 1.55x faster than kernel 1 and 2.5x faster than kernel 0.
Kernel 2's window is NNZ-bound (about 37 cycles per NNZ, a per-row term near zero); the fit and its caveats are in the
sections above. All three give bitwise the same `y`: a row's products are added into a zeroed `y` in CSR order, which is what the CSR
loop's accumulator does from 0.0 (a test compares the numpy model of the shipped kernel bit for bit with an independent CSR reference).

**What the shipped kernel is.** `pe_resident.csl` has one `local_spmv`: `@fmovs(mem_y, 0.0)`, then for `k < row_ptrs[KY]`
(this PE's NNZ count) `y[row_ids[k]] += vals[k] * x[col[k]]`, then `@fmovs(mem_ew, mem_y)` into the EW chain. Per PE it holds
`x_vec[KX]`, `col_indices[N]`, `vals[N]`, `row_ids[N]`, `row_ptrs[KY+1]` (only its last word, the NNZ count, is read) and
`y[KY]`, `ew_buf[KY]`: `KX + 3N + 3KY + 1` words, so SRAM is now tighter in N than the CSR kernel (which had no `row_ids`).
That is what `SpMVConfig`, `run.py --mtx-file` and `docs/results.md` ("Supported matrices") count. The loop state is `u16`, so the
kernel asserts at compile time `TOTAL_NNZ < 65536 and KY < 65535 and KX < 65536` (far from binding: SRAM allows at most a few thousand
NNZ). `row_ptrs` could be replaced by a one-word count (saving KY+1 words); it was not, to keep the layout that was measured.

**History, in short.** There was a compile parameter `LOCAL_KERNEL` (0, 1, 2) and a `--local-kernel` flag; both are removed,
with kernels 0 and 1, their tests and their sweep configs. An artifact compiled with `LOCAL_KERNEL` 0 or 1 is refused by the worker
(recompile); `LOCAL_KERNEL:2` artifacts are the shipped kernel. The code of the removed kernels is in the git history of the
private development repository (`feat/resident-a`).

**What the compiler and hardware have seen.** The constructs the shipped kernel relies on (`@as(u16, <i32 array element>)`
narrowing, `@fmovs(dsd, 0.0)`, the mem-to-mem `@fmovs(mem_ew, mem_y)`) compiled and ran in the simulator and on hardware, at 16 PEs,
65K and the full wafer.

**Still hardware-only:** the exact cost split (d·KY versus e·NNZ) is ill-determined from the points taken (see "Result: kernel 2 at
65K" and the later fit); per-operation cycle costs are known only in aggregate.

## When a job hangs: `progress.log`

The worker's stdout reaches the host only when the job completes, so a cancelled hang used to
leave nothing. The worker now appends one fsynced line per phase to `progress.log` (`start`,
`host matrix and tiles ready`, `runner.load / run`, `setup copy start` / `... barrier done`,
`f_sync`, then per pass `start`,
`launches start`, `launches returned, barrier D2H start`, `y read back`, `end`), each with
`+seconds` since the worker started. `--progress-file ''` disables it.
`sdk/launcher.py` downloads it in the `finally` next to `benchmark_results.csv`
(`cluster_work/.../progress.log`) and logs its contents. **Unverified:** whether SdkLauncher can
still `download_artifact` from a job the user cancelled with `csctl` (the worker's directory may
be gone with it). If it cannot, the log is only fetched when the run raises (a timeout or an
SDK error), not on a manual cancel; in that case the next-best tool is to cancel only after
the SDK's own error. Tell me which it is from the first hang.

## Risks

1. **CSL never compiled.** What cslc has not seen, ranked by how likely it is to bite:
   `spmv()` returning without `unblock_cmd_stream` while `@activate`ing a local task (the
   `wait_ew` pattern, but the activation is new in an RPC function); the comptime
   `if (ENCODING != 3)` around `@export_name("wait_ew")` in the layout block (precedent: the
   `val_meta` gate); a kernel that never binds the H2D queue under `--memcpy`; and the
   `sync_pe_file()` path-selecting function, which was already unverified when synced timing landed.
   **Run step 0 first.**
2. **Hang = `f_toc` never returns.** If any PE's chain never completes (a lost wavelet, an EW
   color clash with the sync colors, never exercised together), its `chain_done` never
   releases the command stream and the host blocks on `f_toc`. EW and sync both ran at every
   size separately; together at one width only in the 65K synced-timing run. Simulator at 16 PEs.
3. **Host RAM at the full wafer.** The setup arrays are ~12 GB at peak (x replicated 1.9 GB
   plus its flattened strips, col_indices, vals, row_ptrs each twice, the COO), about 2 GB
   above valtask's peak. The worker frees
   those arrays after setup. **`free -g` on the user node before step 3.**
4. **The window includes dispatch skew** (up to (W+H) cycles between the first tic and the
   last PE's), the same definition as bandwidth-test. It is a ≤ 10% effect here, but it is
   in the number.
5. **Scalar-loop cost is a guess** (6–12 cycles/NNZ, 4–8/row). If compute dominates by 3×,
   the window triples at 65K but barely moves at the full wafer, where the chain dominates;
   the shape of the weak-scaling curve (window vs W) separates the two.
6. **`wafer_host_ratio_max` is a weak check here.** It is a bound (window ≤ host time through
   the barrier) and the host time is ~300 µs, so a ratio of 0.05 is correct and a wrong clock
   by 5× still passes. The real cross-checks are `compute_hw_cycles_*` (an independent stamp
   pair per PE) agreeing with the window, and the chain's W-dependence.
7. **The EW chain figure is extrapolated** from the M6–M8 fit (W ≤ 256) to W = 742.
## What only hardware can confirm (refreshed)

| item | status |
|---|---|
| compiles, does not hang | **done**: 16 PEs, 65K, full wafer, all three kernels at 16 PEs |
| on-wafer window | **measured**: 16 PEs, 65K (n = 3) and the full wafer (n = 3) for kernel 0; 16 PEs and 65K for kernel 2; kernel 2 at the full wafer predicted above |
| per-row and per-NNZ cost of the loop | **measured as fits**: kernel 0 (a, b) = (57.6, 27.7); kernel 2 d ≈ 0.6 ± 1.1, e ≈ 34.5–36.9 (d ill-determined) |
| the launch floor | **measured** at three sizes (448 / 1,485 / 4,843); **its shape is not known** (three shapes fit); the grids above are the test |
| the 13.7 cycles/PE chain at W = 742 | **still open**: the data are consistent with 12.8–13.7; the W series above settles it |
| the host's per-call latency | **measured**: ~0.2 ms at 16 PEs, ~0.7–1.1 ms per pass (std 0.5–0.7) at 65K and the full wafer |
| the one-PE-tall H2D rate | **never measured** (the probe was removed) |
| host RAM at the full wafer | the full-wafer runs completed; the peak was **not recorded** |
| whether e depends on NNZ per row (the read-modify-write hypothesis) | **open**; a 16-PE run at a different density with the same NNZ_max would test it |
