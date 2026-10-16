# ADR-003: Row-Header Encoding for NNZ Streams

**Status:** Superseded as the default by `valtask` (2026-10-08); the bulk `val` kernel was removed in v2.0. `header` stays as a comparison mode  
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

## Amendment (2026-10-06): val-only implemented behind `--encoding val`

The objection above, that val-only needs a 40 MB `row_offsets` preload, no
longer applies. The streaming kernel as built already keeps the CSR `row_ptrs`
and `col_indices` resident, so val-only adds just one preloaded word per PE (its
per-pass word count) plus a `vals[TOTAL_NNZ]` landing buffer. Hardware showed the
header stream to be consumption-bound (commit 4040625). Val-only is now
implemented as an alternative kernel (`src/csl/pe_val.csl`), with row-header
still the default. Design, the cycles-per-word reasoning and the A/B plan are in
[`docs/val-only-encoding.md`](../val-only-encoding.md). This ADR's decision
stands until that A/B is measured.

## Amendment (2026-10-08): values-only (valtask) measured faster; becomes the default candidate

The A/B referred to above is measured, barrier-verified, at 65,536 PEs (114,000², 13.0M NNZ). Medians, n=3:
- Full mode: valtask beats header 2.6× / 1.9× / 1.8× at 1 / 8 / 16 channels.
- Column mode (S1, `bed16fe`): valtask stream 40.6 ms (8 ch) / 46.3 ms (16 ch); header 59.1 / 62.7–70.0 ms.

Values-only sends 1.76× fewer words (16.97M vs 29.88M), and the gain exceeds that ratio at 1 and 8 channels.

- **`--encoding valtask`** (per-word data task, 4 B/NNZ) is correct everywhere it has run. Column mode serializes its column rects automatically.
- **`--encoding val`** (one bulk `@fmovs` per PE) hangs on hardware under concurrent streams: 8+ channels, or overlapping column rects. It is correct only at 1 channel. Don't use it. Microbenchmark M3 could not reproduce the hang with a plain bulk receive, so the cause lies elsewhere in that kernel (open).
- The 2026-10-06 claim that the header stream is "consumption-bound" rested on host-return times and is **withdrawn** (see `docs/host-timing-barrier.md`).

**Decision update:** valtask replaces row-header as the intended default once the full-wafer run confirms it. Row-header stays available as `--encoding header` for comparison.

## Amendment (2026-10-08, evening): valtask is the default

The full wafer confirms it (857,010 PEs, 414,000², 171.4M NNZ, column mode, col-major setup, 16 channels, barrier-verified, both validated at 2.75e-07): **valtask streams in 443.3 ms vs 669.7 ms for header (1.51×)**, in line with 1.4–1.6× at 65,536 PEs (S1, n=3). Header's link rate is higher (2.20 vs 1.92 GB/s), but it sends 1.76× more words.

**Decision:** the default becomes `--encoding valtask --h2d-mode column --setup-order col --channels 16` (row-major stream). Row-header stays selectable as `--encoding header`. Bulk `--encoding val` stays available only for experiments (it hangs under concurrent streams). Data: `cluster/results/data/2026-10-07_s1-column-matrix.csv`, `2026-10-08_fullwafer.csv`.
