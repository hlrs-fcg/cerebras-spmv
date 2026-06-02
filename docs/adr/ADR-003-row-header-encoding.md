# ADR-003: Row-Header Encoding for NNZ Streams

**Status:** Accepted  
**Date:** 2026-05-31

## Context

Each column slice of $A$ must be transmitted from the host to the corresponding PE column via H2D. Each NNZ in the slice needs (at minimum):

- The floating-point value (irreducible: 4 B float32)
- An identifier for which row it belongs to (so the PE knows which $y$ entry to accumulate into)

### Encoding options considered

| Scheme | Bytes/NNZ | Notes |
|---|---|---|
| `(val:f32, global_row_id:i32)` | 8 B | naive, no wafer logic |
| `(val:f32, local_row_id:i16)` | 6 B | 25 % saving; host precomputes `row - py*Ky` |
| Row-header + val stream | ~4.6 B | 43 % saving at $K = 10$ NNZ/row |
| Val-only (row\_offsets preloaded) | 4 B | max saving; requires 40 MB row\_offsets overhead |

**Row-header format:**
`[(local_row_id:i16, count:i16), val, val, ..., (local_row_id, count), val, ...]`
One 4 B header per non-empty row in the PE's slice, then `count` raw float32 values.

### CSR compatibility

CSR stores NNZ in row-major order. A one-pass scatter over CSR rows naturally produces data in row-sorted order within each PE-column bucket; no secondary sort is required. The `count` field for each row-header comes directly from `row_ptr[i+1] - row_ptr[i]` (filtered to the column slice), so the header is built at zero additional cost during the scatter.

## Decision

Use **row-header encoding with local `i16` row\_id**:

- Header: `(local_row_id: uint16, count: uint16)`, 4 B, emitted once per non-empty row in the slice
- Payload: `count` $\times$ float32 values, 4 B each, no per-value metadata
- Effective cost: $4 + 6/K$ B/NNZ, where $K$ is the average NNZ/row in this slice

The local `row_id` is `global_row_i - py*Ky`, computed by the host during scatter. `uint16` supports up to 65 535 rows per PE; with $H = 1\,155$ and $N_\text{rows} = 10\,\text{M}$, each PE handles ~8 658 rows, comfortably within range.

## Rationale

- **43 % H2D reduction** vs naive (at $K = 10$): from 800 MB to ~460 MB for 100 M NNZ
- **Simple on-wafer decode:** a two-state machine (awaiting header / consuming values); ~10 CSL lines
- **float32 preserved:** numerical precision is not compromised, which is critical for CG convergence
- **Natural fit with CSR:** the scatter pass produces row-ordered buckets; headers build themselves

## Consequences

- On-wafer PE logic must implement a small streaming state machine (header vs. value mode)
- The host scatter pass must compute `local_row_id` and group headers correctly (see ADR-004)
- Val-only encoding (4 B/NNZ) remains an option if $A$ fits on the wafer and `row_offsets` can be preloaded; the 40 MB overhead per column-slice pass makes it impractical for the streaming case
