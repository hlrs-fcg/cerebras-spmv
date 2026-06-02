# Architecture Decision Records

This directory documents the key architectural decisions for the WSE-3 SpMV implementation.
Each ADR captures the context, decision, and trade-offs at the time the decision was made.

## Index

| ADR | Title | Status |
|---|---|---|
| [ADR-000](ADR-000-v1-baseline-isc26.md) | v1.0 Baseline — ISC26 Poster Architecture | Superseded |
| [ADR-001](ADR-001-streaming-first-architecture.md) | Streaming-First Architecture | Accepted |
| [ADR-002](ADR-002-2d-data-layout.md) | 2D Data Layout — x Permanent on Wafer, A Streamed in Column Slices | Accepted |
| [ADR-003](ADR-003-row-header-encoding.md) | Row-Header Encoding for NNZ Streams | Accepted |
| [ADR-004](ADR-004-host-preprocessing-pipeline.md) | Parallel Host Preprocessing Pipeline (CSR → Column Slices) | Accepted |
| [ADR-005](ADR-005-aggregation-and-cg-compatibility.md) | On-Wafer Aggregation and CG Compatibility | Accepted |
| [ADR-006](ADR-006-csl-colors-for-pe-roles.md) | CSL Colors for PE Role Differentiation | Accepted |
| [ADR-007](ADR-007-toml-sweep-config.md) | TOML Sweep Configuration (replacing CSV + argparse + Makefile) | Accepted |

## Design Summary

The architecture targets large sparse matrices that exceed the wafer's 42 GB SRAM and dense vectors that cannot fit in a single PE's 48 KB. The key ideas:

1. **Streaming-first** (ADR-001): A is always treated as a stream; fits-on-wafer is the zero-pass special case.
2. **2D tiling** (ADR-002): x is distributed permanently across PE columns (16 KB/PE, up to 3.1 M entries total); A is streamed as column slices so every x lookup during multiply is local.
3. **Row-header encoding** (ADR-003): ~4.6 B/NNZ instead of 8 B/NNZ; headers are free from CSR's row\_ptr.
4. **64-core parallel scatter** (ADR-004): host preprocessing finishes in ~1.3 ms vs 2.1 ms H2D — H2D remains the bottleneck.
5. **South conveyor** (ADR-005): east-west reduction then south-flowing D2H; CG dot products ride alongside at zero extra cost.
6. **CSL colors** (ADR-006): single binary, PE roles determined at runtime via `@get_id()`.
