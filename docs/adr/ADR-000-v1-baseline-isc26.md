# ADR-000: v1.0 Baseline: ISC26 Poster Architecture

**Status:** Superseded (by ADR-001 through ADR-006)  
**Date:** 2026-05-31  
**Reference:** Renschler et al., "SpMV for the Cerebras Wafer-Scale Engine", ISC26 poster, June 2026

## Purpose

This record documents the architecture as presented at ISC26 (v1.0). It serves as the baseline against which all subsequent ADRs are measured. The poster itself acknowledged the key shortcomings; this ADR collects them formally and maps each to the ADR that addresses it.

## v1.0 Architecture

### Pipeline (7 steps, as described in the poster)

1. **(Host)** Problem specifications and runtime parameters transmitted to worker node
2. **(Worker, Distribution)** Matrix $A$ partitioned using histogram-based bucketing of NNZ; each PE receives its assigned rows, column indices, **corresponding elements of $x$** (pre-gathered by the worker), and a row index for result collection
3. **(Communication)** Data packets transmitted **per-PE** from worker to WSE (blocking memcpy per PE)
4. **(WSE, Parallel)** PEs compute partial dot products independently; **no inter-PE NoC communication**
5. **(Communication)** Results collected **per-PE** from WSE to worker (blocking memcpy per PE)
6. **(Worker, Parallel)** Partial results aggregated on worker to form complete vector $b$
7. **(Communication)** Final result returned to host

### Key properties

| Property | v1.0 value |
|---|---|
| Matrix format on host | CSR |
| $x$ handling | Host pre-gathers `x[col_indices]` per PE (matched-pairs) |
| H2D pattern | Per-PE blocking memcpy, $O(4 \times \text{num\_PEs})$ calls |
| Load balancing | Histogram bucketing by NNZ magnitude, round-robin from largest bucket |
| NoC usage | None |
| Aggregation | Worker node (off-chip) |
| Assumed problem size | Fits entirely on wafer |

### What the poster measured

Strong scaling (16 384 to 524 288 PEs, up to 40 M NNZ) and weak scaling (1 000 NNZ/PE, 256 to 65 536 PEs). Key findings:

- **Compute time is negligible compared to SDK / communication overhead** by orders of magnitude
- Communication dominates; compute scales perfectly when measured in isolation
- Strong scaling efficiency collapses when communication is included (constant comm cost, decreasing compute)
- Weak scaling: compute scales perfectly; communication, partitioning, and aggregation are the bottlenecks

### Problems explicitly identified in the poster

> "the current implementation redundantly sends data over the worker-WSE interconnect, which lacks the bandwidth of the WSE NoC"

> "two key capabilities of the WSE are not yet exploited: explicit NoC communication between PEs, and minimizing the worker-WSE communication"

> "We plan to optimize the communication bottleneck by moving the partitioning and aggregation steps of vectors $b$ and $x$ to the WSE"

## What was correct and kept

- **Histogram-based load balancing** for row assignment to PEs (retained in all subsequent versions)
- **CSR as the host-side input format** (retained)
- **Basic PE grid sizing** (fabric dims, offsets, padding) (retained in `config.py`)

## Mapping: v1.0 problems to ADRs

| v1.0 Problem | Root Cause | Addressed by |
|---|---|---|
| Per-PE blocking H2D, $O(\text{num\_PEs})$ calls | No bulk rectangle memcpy | Implemented before ADR series (bulk H2D in `worker_runtime.py`) |
| $x$ re-sent every iteration | Matched-pairs: host gathers `x[col_indices]` | ADR-002 ($x$ permanent on wafer) |
| 8 B/NNZ wire format | Explicit (val, row\_id) per NNZ | ADR-003 (row-header: ~4.6 B/NNZ) |
| Sequential worker-side partition | Single-threaded Python | ADR-004 (64-core parallel scatter) |
| Aggregation on worker (off-chip) | No NoC usage | ADR-005 (south conveyor on wafer) |
| NoC unused | No CSL color/message design | ADR-006 (colors for PE roles) |
| Fits-on-wafer assumption | Design choice | ADR-001 (streaming-first) |

## Quantified gap (v1.0 to target)

For 100 M NNZ, 10 M rows, $W = 750$ PE columns:

| Metric | v1.0 | Target (ADR-001 to 006) |
|---|---|---|
| H2D calls per SpMV | $O(\text{num\_PEs}) \approx 3.5\,\text{M}$ | $O(W) = 750$ |
| Bytes/NNZ over H2D | 8 B (matched pairs, every iter) | ~4.6 B (row-header, $A$ once + $x$ once) |
| Aggregation location | Worker CPU | On-wafer south conveyor |
| NoC usage | 0 | East-west reduce + south conveyor |
| Preprocessing parallelism | Single-threaded | 64-core parallel scatter |
| Host preprocessing time | ~86 ms (single-core) | ~1.3 ms (64-core) |
