# ADR-004: Parallel Host Preprocessing Pipeline (CSR to Column Slices)

**Status:** Accepted  
**Date:** 2026-05-31

## Context

The input matrix $A$ is in CSR format (row-major). The wafer consumes $A$ as column slices in row-header encoding (ADR-003). The host must transform CSR into $W = 750$ encoded column-slice buffers before or during H2D.

### Timing analysis (100 M NNZ, 10 M rows, 220 GB/s H2D)

| Step | Single-core | 64-core parallel |
|---|---|---|
| Phase 1: Scatter CSR to $W$ buckets | ~40 ms | ~0.6 ms |
| Phase 2: Encode buckets (row-header) | ~46 ms | ~0.7 ms |
| H2D (4.6 B/NNZ, 460 MB) | 2.1 ms | 2.1 ms |

Single-threaded preprocessing is **22x slower** than H2D and would fully stall the pipeline. Parallelisation is mandatory on the worker side.

### What "scatter" means here

For each NNZ at `(row_i, col_j, val)` in CSR: compute `px = col_j // Kx` and write `(local_row_id, val)` into `bucket[px]`. This distributes values from the global row-major order into $W$ per-column destinations. Because CSR iterates rows $0 \to N_\text{rows}$ in order, each bucket receives entries in ascending row order automatically; no sort is needed after scatter.

## Decision

Use a **two-phase parallel pipeline** on the 64-core worker node:

### Phase 1: Row-parallel scatter (all 64 cores)

Each core owns a contiguous row range of $N_\text{rows} / 64$. It maintains **thread-local** sub-buckets (one per PE column), avoiding lock contention. After all cores finish, sub-buckets are merged per column (parallel reduction, $O(W \times 64)$ pointer operations).

### Phase 2: Column-parallel encode + async H2D (all 64 cores)

Each core owns $W / 64 \approx 12$ PE columns. It encodes its merged bucket into the row-header wire format and immediately issues `memcpy_h2d(..., nonblock=True)`. The non-blocking H2D allows the core to start encoding the next column while the DMA engine transfers the previous one, overlapping encode and H2D.

```
Timeline (64 cores):

Phase 1:  |== scatter rows 0..156K ==|== scatter 156K..312K ==| ... (~0.6 ms, all cores parallel)
Phase 2:  |encode col 0..11 + H2D   | encode col 12..23 + H2D | ... (~2.1 ms, H2D dominates)
```

Total wall time: ~0.6 ms (scatter) + ~2.1 ms (H2D) = **~2.7 ms** for 100 M NNZ.

## Rationale

- H2D is the bottleneck (2.1 ms); CPU phases must finish faster, achieved with 64-core parallelism
- Thread-local buckets in Phase 1 avoid false sharing and lock overhead
- Non-blocking H2D in Phase 2 allows compute-DMA overlap at no code complexity cost
- The natural row-ordering property of CSR scatter eliminates the need for a sort step, saving $O(\text{NNZ} \log \text{NNZ})$ work

## Consequences

- Memory allocation: $64 \times W$ thread-local sub-buckets per scatter pass; at 100 M NNZ this is ~800 MB working set (acceptable on a worker node)
- For matrices that fit on the wafer, Phase 1 runs once at startup; Phase 2 runs once; subsequent iterations only update $x$ (ADR-005)
- Implementation: can prototype with `scipy.sparse.csr_matrix.tocsc()` (which implements exactly this scatter internally) and column iteration; replace with a parallel C extension or `numba.prange` for production throughput
