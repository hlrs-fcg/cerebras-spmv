# ADR-001: Streaming-First Architecture

**Status:** Accepted  
**Date:** 2026-05-31

## Context

The WSE-3 fabric provides up to 857 K PEs with 48 KB SRAM each, totalling ~42 GB on-chip. For many HPC-relevant sparse matrices this is insufficient:

- A 10 M × 10 M matrix at 1 % density requires ~80 GB (values + column indices)
- The dense vector x for a 3 M-column problem requires 12 MB, which cannot fit in a single PE (48 KB)

An earlier prototype assumed A fits entirely on the wafer and treated the single-SpMV case as the primary target. This limits applicability to small or very sparse matrices.

## Decision

Design all on-wafer data flow and host preprocessing for the **streaming case**, where A is too large to reside on the wafer simultaneously. The assumption that A fits is treated as a special case (zero streaming passes) rather than the base case.

## Rationale

- H2D bandwidth (220 GB/s) is the system bottleneck for SpMV regardless of problem size; designing for streaming does not penalise the fits-on-wafer case
- Streaming enables problems that are scientifically relevant (large FEM, graph, ML sparse matrices) where the wafer would otherwise be idle
- The streaming abstraction naturally supports iterative Krylov solvers: stream A once per SpMV call, keep x and partial y on-chip across iterations

## Consequences

- A must be decomposed into column slices on the host before transmission (see ADR-003, ADR-004)
- Per-PE SRAM budget must allocate space for a single in-flight A tile plus permanent x slice and partial y (see ADR-002)
- PE logic must process arriving data incrementally rather than operating on a static array
