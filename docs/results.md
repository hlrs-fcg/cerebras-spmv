# Results (development runs, October 2026)

These are the measurements the v2.0 design rests on. They were taken on the CS-3 during development, from
the commits named in the (private) development log; **the v2.0 release benchmarks are re-run from the tagged
commit and supersede every number below** (each CSV row of those runs records its commit, `git describe`
and whether the tree was clean, `docs/adr/ADR-008-host-wafer-transfer-and-timing.md`).

## Timing basis

Only barrier-closed times count: `stream_time_corrected` and `setup_h2d_time_corrected` (see
`docs/host-timing-barrier.md`). A blocking launch or H2D returns when the *host* is done with the call, which
can be before the device is; every earlier host-return time is a lower bound. Compare runs within one
session, n ≥ 3, and report min / median / max: single runs vary by up to 3x in the noisiest configuration
(full-mode streaming at 16 channels) and a session can run 3-4x slower than another day.

## Streamed SpMV (A sent on every call)

Matrices: random, d = 1e-3, seed 42. 65K = 65,536 PEs, 114,000², 13.0M NNZ. Full wafer = 857,010 PEs,
414,000², 171.4M NNZ. All runs validated against a numpy reference (relative error < 3e-7).

| | Config | Stream | Setup copy |
|---|---|---|---|
| 65K, best | valtask, column, col-major setup, 8 ch | **40.5 / 40.6 / 41.6 ms** (62.8 MB, 1.51-1.55 GB/s) | 54-56 ms |
| 65K | valtask, column, col-major setup, 16 ch | 43.5 / 46.3 / 48.9 ms | 42-48 ms |
| 65K | header, column, 16 ch | 61-70 ms (112.1 MB) | 41-58 ms |
| Full wafer | valtask, column, col-major setup, 16 ch | **443.3 ms** (851.8 MB, 1.92 GB/s) | 8.70 GB/s |
| Full wafer | header, same | 669.7 ms (1,476.6 MB, 2.20 GB/s) | 8.35 GB/s |
| 65K, defaults | valtask, column, col setup, 16 ch, **32 setup strips** | 44.6 / 46.0 / 46.1 ms | **26.0 / 27.7 / 28.8 ms = 10.9 GB/s** (1 strip: 41.0 ms, 7.4 GB/s) |
| Full wafer, defaults (n=3) | same | **417.7 / 458.8 / 480.1 ms** (1.86 GB/s median) | **341 ms = 12.0 GB/s** |

- **valtask is the default** (ADR-003): 1.4-1.6x faster than header at 65K and 1.51x at the full wafer.
- **Column mode beats full mode**: narrow transfers reach the link ceiling, one full rectangle gets 3.5x less.
- **Column-major setup** roughly halves the setup time; **32 setup strips** are 1.48x faster than one at 65K.
  Setup saturates at 12.0 GB/s from 262K PEs up, 96% of one 100 GbE link.
- Serialising column rectangles does not help header; valtask serialises them automatically.
- The stream rate does not collapse at scale: 1.39 GB/s at 65K rising to 1.86 GB/s at the full wafer, so 13.1x
  the PEs costs 10.2x the time.
- The on-wafer window measured with synced timing (`docs/synced-timing.md`) agrees with the barrier: 40.7 /
  40.8 / 41.4 ms against a 45.2 ms barrier-closed time; the sync build costs about 1 ms.
- Bulk `--encoding val` hung under concurrent streams and was removed in v2.0 (ADR-003).

## Weak scaling

About 200 NNZ per PE, d = 1e-3, defaults, median of n = 3, all 18 runs validated.

| PEs | 1K | 4K | 16K | 65K | 262K | 428K | 857K |
|---|---|---|---|---|---|---|---|
| Stream (ms) | 2.2 | 5.4 | 14.8 | 45.2 | 158.2 | 248.8 | 458.8 |
| Stream rate (GB/s) | 0.37 | 0.6 | 0.9 | 1.39 | 1.62 | 1.69 | 1.86 |
| Setup rate (GB/s) | 0.6 | 2.0 | 5.0 | 10.9 | 11.9 | 12.0 | 12.0 |

**Stream model** (fit over all 21 runs, within about 3%): T ≈ W · (57 µs + bytes per column / 2.07 GB/s), where W
is the grid width: W serialized column transfers, each with a fixed cost of 56.5 µs. **Setup**, from 65K PEs up:
about 1.3 ms + bytes / 12.0 GB/s (within about 5%).

## Wall time of a run

Medians. Host stream encoding is not timed in the CSV; it appears as total minus the timed phases.

| | wafer I/O | host stream encoding | host tiling | program load | total |
|---|---|---|---|---|---|
| 65K, header, full, 1 ch, row setup | 1.44 s | 95.7 s | 11.3 s | 36.9 s | **146.6 s** |
| 65K, defaults + 32 strips | 0.072 s | 0.7 s | 10.5 s | 90.4 s | **102.1 s** |
| 857K, header | 1.16 s | 1,211.8 s | 158.9 s | 216.3 s | **1,592.8 s** |
| 857K, defaults + 32 strips | 0.80 s | 8.1 s | 159.4 s | 217.3 s | **390.0 s** |

Host tiling has since been rewritten (a fused numba or ahead-of-time kernel, `docs/worker-numba.md`): 159 s
to 1.05 s of tiling at the full wafer, plus a fixed ~12 s numba JIT per job or none with the ahead-of-time
module. Program load (about 210 s at the full wafer) is then 90% of a run.

## Against the host CPU

Same matrices, a 32-core Cascade Lake VM on the user node (`bench/host_spmv_baseline.py`):

| | 65K size (13M NNZ) | full-wafer size (171M NNZ) |
|---|---|---|
| CS-3, A streamed every call (barrier-closed) | 45.2 ms | 458.8 ms |
| host, scipy CSR fp32, 1 thread | 20.9 ms | 886 ms |
| host, numba CSR fp32, 32 threads | **3.5 ms** | **61.7 ms** |

Re-measured on the same node for the release (`cluster/results/v2.0-data/2026-10-15_host-cpu-random.txt`): at the 65K size the 32-thread
numba time was a median of 5.38 ms (min 2.28, max 11.10: noisy on a shared VM; the 3.48 ms above is one earlier run) and 5.0 ms at 8
threads; at the full-wafer size 58.2 ms (min 58.0, max 85.7).

Streaming A on every call cannot win: it moves A over a ~2 GB/s link while the host reads it from DRAM. The
case for the wafer is **A resident** (ADR-009).

## Real matrices against the host CPU

The four SuiteSparse matrices of "Supported matrices" ran on the wafer with the shipped kernel (RC1, `4739dc4`) and on the host CPU
(`bench/host_spmv_baseline.py --mtx`, the 32-core user node, the steady state of a loop of calls; raw rows in `cluster/results/v2.0-data/`,
the matrices' hashes in `matrices.md`). The wafer column is the synced on-wafer window, every run validated. "Host best" is the fastest of
numpy / scipy 1 thread / numba 1, 8, 32 threads.

| matrix | NNZ | PEs | wafer window | host best | host, scipy 1 thread | wafer vs host best | vs one core |
|---|---|---|---|---|---|---|---|
| bcsstk01 | 400 | 16 | 3.3 µs | 2.0 µs | 3.3 µs | 0.6x (slower) | 1.0x (level) |
| bcsstk01 | 400 | 65K | 5.8 µs | 2.0 µs | 3.3 µs | 0.35x (slower) | 0.6x |
| nos6 | 3,255 | 16 | 37.8 µs | 5.7 µs | 5.7 µs | 0.15x (slower) | 0.15x |
| nos6 | 3,255 | 65K | 5.8 µs | 5.7 µs | 5.7 µs | level | level |
| bcsstk14 | 63,454 | 65K | 7.3 µs | 36.2 µs | 57.1 µs | **4.9x** | **7.8x** |
| parabolic_fem | 3.67M | full wafer | 75.4 µs (69.7 µs net of the launch floor) | 1.55 ms | 4.03 ms | **21x** | **53x** |

So the wafer **loses on the tiny matrices** (nos6 on 16 PEs, bcsstk01: the grid floor of a few µs and the NNZ-bound chain exceed what a core
needs for a few hundred non-zeros), **ties at 3K NNZ on 65K PEs**, is about **5 to 8x** ahead at 63K NNZ and **21 to 53x** ahead for the
3.7M-NNZ matrix on the full wafer; the random 171M-NNZ matrix gives about 2,000x (the table above). The 32-thread numba time is no use
below a few thousand NNZ (3.3 to 3.5 ms for the first calls: thread launch). These are windows of one product; through the host runtime a
call still costs about a millisecond (the resident section below), so the iteration loop has to run on the wafer to keep these margins.

## Resident A

One resident kernel ships: a COO scatter, chosen on hardware among three local kernels (`docs/resident-spmv.md`). **Headline
(release candidate RC1, commit `4739dc4`, the shipped kernel):** the synced on-wafer window of one SpMV, 20 passes, every pass
validated (relative error 5.3e-08 at 16 PEs, 1.5e-07 at 65K, 2.75e-07 at the full wafer, 60 of 60 passes valid at both large sizes),
median of n = 3:

| PEs | window | net of the launch floor | host CPU (32 threads) | window vs host |
|---|---|---|---|---|
| 16 | 27,857 cycles = 32.8 µs | 32.2 µs | n/a | n/a |
| 65,536 (13.0M NNZ) | **15,316 cycles = 18.0 µs** | 16.3 µs | 3.48 ms | about 190x |
| 857,010 (171.4M NNZ) | **25,193 cycles = 29.6 µs** | 23.9 µs | 61.7 ms | about 2,100x |

The windows are deterministic to a few cycles with an occasional single-pass outlier, so quote the median. The earlier numbers in this
repository's history (45.4 µs at 65K, 41.7 µs at 16 PEs, the first full-wafer runs at about 51 µs) are the **first kernel**, a CSR
row loop, which no longer exists. The three kernels on the same matrices:

| kernel | 16 PEs | 65K PEs (13.0M NNZ) |
|---|---|---|
| CSR row loop (first kernel, removed) | 35,449 cycles = 41.7 µs | 38,598 cycles = 45.4 µs |
| CSR on `u16` loop state (removed) | 25,757 = 30.3 µs | 23,756 = 27.9 µs |
| **COO scatter (ships)** | 27,857 = 32.8 µs | **15,316 = 18.0 µs** |

The launch-only twin (`--null-spmv`) prices the launch machinery: 0.5 µs at 16 PEs, 1.75 µs at 65K and about 5.7 µs at the full
wafer (floors of 448 / 1,485 / 4,843 cycles: it grows with the grid). The barrier-closed host time per call is 0.7 to 0.9 ms at 65K
(the host runtime's latency, not the wafer), and the setup copy takes 31.5 ms. The kernel's window is NNZ-bound, about 37 cycles per
NNZ, plus an east-west chain of about 12.8 cycles per PE. A resident SpMV needs the iteration loop on the wafer to keep its advantage
over the host (ADR-009).

## Host-to-wafer link (microbenchmarks M1-M10, summarised)

The microbenchmarks live in their own (private) repository; the rules they established are in ADR-008.

- **H2D ceiling:** 1.13 / 5.15 / 10.3 GB/s at 1 / 8 / 16 channels, with narrow strips issued serially.
  D2H: 0.89 / 3.8 / 6.3 GB/s.
- **Streaming costs nothing over copy mode;** a bulk receive is as fast as a per-word task.
- **Host timing is wrong:** a blocking launch returns after 13 µs; a blocking H2D returns with about 29% still
  in flight (8 and 16 channels).
- **Sync and collectives:** a one-word D2H sync costs 167 µs; an on-wafer allreduce costs 31 µs at 256×256.
  The east-west reduction here beats the vendor library.

## Supported matrices

The resident kernel keeps `x_vec[KX]`, `col_indices[N]`, `vals[N]`, `row_ids[N]`, `row_ptrs[KY+1]`, `y[KY]` and `ew_buf[KY]` on every PE
(about 38 KB), where N is the busiest PE's NNZ under the block tiling (PE (px, py) holds rows py·KY.. and columns px·KX..). That is
`KX + 3N + 3KY` words, so a matrix has to be square-ish, sparse **and balanced**: a banded or stencil matrix in natural order puts its
NNZ on the PEs near the diagonal (the busiest PE holds 300 to 700 times the mean), and a large n makes `KX + 3·KY` alone too big.
`run.py run --mtx-file` counts this on the login side and refuses before compiling, naming the smallest grid that fits. The release
benchmarks stay with uniform random matrices and the matrices below.

SuiteSparse matrices, sized without compiling (`NNZ_max` is the busiest PE; KB is per PE; sync words included):

| matrix | n, NNZ | 16 PEs | 65K PEs (256²) | full wafer (742×1155) |
|---|---|---|---|---|
| bcsstk01 | 48, 400 | fits | fits | fits |
| nos6 | 675, 3.3K | fits | fits | fits |
| bcsstk14 | 1,806, 63K | no (174.5 KB) | fits | fits |
| parabolic_fem | 525,825, 3.67M | no | no (104 KB) | **fits** (NNZ_max 1,368; 24.2 KB) |
| apache2 | 715,176, 4.82M | no | no (202 KB) | no (NNZ_max 2,838; 44.3 KB) |
| ecology2 | 999,999, 4.99M | no | no | no (NNZ_max 2,945; 50.0 KB) |
| thermal2 | 1,228,045, 8.58M | no | no | no (NNZ_max 6,932; 100.2 KB) |
| G3_circuit | 1,585,478, 7.66M | no | no | no (NNZ_max 6,613; 102.0 KB) |

So: the three small matrices run on 16 PEs and 65K (`bcsstk14` not on 16); `parabolic_fem` only on the full wafer; the rest need a
different distribution of the matrix over the PEs, which is planned for later (below). (With the earlier CSR kernel, which had no
`row_ids[N]`, `apache2` also fit the full wafer; the COO scatter is faster but costs N more words.)

**Planned for v3.0, not built:** a row/column-cyclic distribution, or a fixed random symmetric permutation applied on the host
before tiling (for CG everything stays in permuted space; only b and the final x are permuted once), would balance these
matrices. Neither needs a kernel change; the host tiling, `x_grid`, `gather_y` and the validation reference would change.
Sizing showed all five large matrices fitting the full wafer under a cyclic mapping (NNZ_max 17 to 30).
