# ADR-009: Resident A as the Primary Mode

**Status:** Accepted
**Date:** 2026-10-13 (v2.0)

## Context

ADR-001 designed for A streamed from the host on every call, on the assumption of a 220 GB/s host-to-wafer
link. The link was measured (ADR-008): 1.13 / 5.15 / 10.3 GB/s at 1 / 8 / 16 I/O channels. Streamed SpMV at
the best configuration is then limited by the link, and a host CPU doing the same product from DRAM is faster
(same matrices, 32-core host, barrier-closed wafer times):

| | 65K PEs (13.0M NNZ) | full wafer (171.4M NNZ) |
|---|---|---|
| wafer, A streamed every call (`valtask`, column, 16 ch) | 45.2 ms | 458.8 ms |
| host, numba CSR, 32 threads | 3.5 to 5.4 ms (median of the final run 5.38 ms) | 58.2 ms |

An iterative solver multiplies by the same A hundreds of times, so A should be loaded once and stay on the
wafer. A resident-A kernel (`src/csl/pe_resident.csl`) was built and measured in development runs.

## Decision

**Resident A is the primary mode of v2.0** (`--encoding resident`). The setup copy puts `col_indices`,
`row_ptrs` and the values on the wafer once; a call is one launched function, a local CSR loop over the
resident arrays followed by the east-west reduction. Nothing streams per call. It is also the base of the
conjugate-gradient solver planned for v3.0.

Streaming stays, for matrices that do not fit on the wafer and as the basis of the transfer measurements:
`valtask` is the default streaming encoding (ADR-003), `header` and `--h2d-mode full` are kept as comparison
modes. ADR-001's design rule (data flow designed for the streaming case) still holds for those.

## Evidence (release candidate RC1, commit `4739dc4`, shipped kernel; the numbers are the final RC1 runs in `cluster/results/v2.0-data`, the same data as the paper and `analysis/v1_vs_v2.ipynb`)

Synced on-wafer window of one SpMV (`--sync-timing`, max(toc) - min(tic) across PEs on a common clock), 20
passes, every pass validated, median of n = 3. Three local kernels were built and measured on the same matrices; the one with
the lowest valid window at scale ships (a COO scatter; the other two are removed, `docs/resident-spmv.md`):

| PEs | CSR loop | CSR, u16 state | **COO scatter (ships)** | launch-only twin | host CPU |
|---|---|---|---|---|---|
| 16 | 41.7 µs | 30.3 µs | 32.8 µs | 0.5 µs | |
| 65,536 (114,000², 13.0M NNZ) | 45.4 µs | 27.9 µs | **18.0 µs** | 1.75 µs | 5.38 ms |
| 857,010 (414,000², 171.4M NNZ) | | | **29.6 µs** (net of the floor 23.9 µs) | 5.7 µs | 58.2 ms |

That is about 300x (65K; 127 to 616x over the host's run-to-run spread) and 2,000x (full wafer) the 32-thread host CPU on the window. At 65K the barrier-closed host time per call
is 0.7 to 0.9 ms (the host runtime's latency, not the wafer), and the setup copy takes 31.5 ms. The shipped kernel costs about 37
cycles per NNZ. The earlier, slower numbers in the history are the first kernel.

## Consequences

- **One SpMV per host call is not the point.** At 0.7 to 0.9 ms per call the host round trip dominates the
  45 us window; the wafer wins only if the iteration loop runs on it.
- **x must stay on the wafer.** Feeding x from the host each iteration costs about 11 ms (replicated, 65K) to
  160 ms (full wafer) by the setup-rate arithmetic, more than the host CPU's whole SpMV. CG therefore needs the
  on-wafer y to x transpose (ADR-005), the one piece of CG not yet built or measured.
- Break-even against the host CPU, setup included, at the full wafer is several thousand SpMVs, because program
  load and host tiling cost minutes. Resident A pays off for solvers, not for a single product.
- **Removed in v2.0:** the bulk `val` kernel (hung under concurrent streams; `valtask` replaced it) and the
  host-fed-x experiments `--x-per-pass` / `--x-row0-probe` (broken in the simulator and on hardware, and
  answered by the arithmetic above). The ENCODING compile-parameter ids are kept (id 1 is retired).

## Real matrices (added after the release-candidate benchmarks)

Resident kernel 2 with block tiling, same synced window, against the best host CPU time on the same matrix (32-core host, numba and scipy):

| matrix | PEs | wafer window | host best | |
|---|---|---|---|---|
| bcsstk01 | 16 | 3.3 µs | 2.0 µs | wafer loses |
| nos6 | 65K | 5.8 µs | 5.7 µs | even |
| bcsstk14 | 65K | 7.3 µs | 36.2 µs | 4.9x (7.8x vs scipy) |
| parabolic_fem | full wafer | 75.4 µs (net 69.7) | 1.55 ms | 21x (53x vs scipy) |

The wafer pays off with size, not on small problems. The cost per NNZ on real matrices is 39 to 41 cycles against 36.5 for random ones; it follows structure, not NNZ per row. apache2, ecology2, thermal2 and G3_circuit do not fit under block tiling (ecology2 needs KY = 250,000 rows per PE, above the 65,535 limit); the sweep refuses them at configuration load. A cyclic or permuted mapping is deferred to v3.0.
