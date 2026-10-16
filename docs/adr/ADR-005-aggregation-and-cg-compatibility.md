# ADR-005: On-Wafer Aggregation and CG Compatibility

**Status:** Accepted, amended 2026-10-08. The on-wafer y to x transpose that CG needs is not built (ADR-009)  
**Date:** 2026-05-31

## Context

After each PE $(p_x, p_y)$ computes its partial result `partial_y[py]` (contributions from column slice $p_x$ to row group $p_y$), two things must happen:

1. **Reduce** `partial_y` across PE columns to obtain complete $y$ row stripes
2. **Deliver** $y$ to the host (D2H) or feed it into the next Krylov iteration

The design of this aggregation step must not prevent future implementation of a **Conjugate Gradient (CG)** solver, which requires, beyond SpMV, the following per-iteration operations:

| Operation | Formula | Communication |
|---|---|---|
| SpMV | $y = Ap$ | (this ADR) |
| Dot product | $\alpha_\text{num} = r^\top r$ | reduce-to-scalar |
| DAXPY | $x = x + \alpha p$ | purely local |
| Dot product | $\beta = r_\text{new}^\top r_\text{new}$ | reduce-to-scalar |

Dot products are the only additional communication primitives CG needs beyond SpMV.

## Decision

### Aggregation: east-west reduction + south conveyor

**Step 1, East-West reduction (along PE rows):**
After computing `partial_y`, PEs reduce across PE columns using a scan along their row. Each PE receives the partial sum from its eastern neighbour, adds its own contribution, and passes the accumulated value west. After $W - 1$ steps, PE column 0 holds the complete $y$ stripe for each PE row.

**Step 2, South conveyor (along PE column 0):**
PE $(0, p_y)$ passes its complete $y$ stripe southward. Each PE $(0, p_y)$ appends its stripe as data flows past. The bottom PE row ($p_y = H - 1$) streams the fully assembled $y$ to the D2H output queue.

```
  <- partial_y reduce (east to west) <-<-<-<-<-
  +------+------+------+------+
  |(0,0) |(1,0) |(2,0) |(3,0) |  y[0..Ky] assembled at (0,0)
  +------+------+------+------+         | south
  |(0,1) |(1,1) |(2,1) |(3,1) |  y[Ky..2Ky] at (0,1)
  +------+------+------+------+         |
  |(0,2) | ...  | ...  | ...  |  ...
  +------+------+------+------+
     | D2H output queue
```

The conveyor is **pipelined**: PE $(0, 0)$ can begin D2H while PE $(0, H-1)$ is still computing.

### CG dot products: carried alongside the conveyor

PE $(p_x, p_y)$ holds both its $x$ slice and `partial_y` at the end of SpMV. It can compute a local partial dot product `local_dot += x_slice[i] * partial_y[i]` at zero additional H2D cost. This scalar travels south with the $y$ stripe and is summed as it passes through each PE row. The bottom row emits both $y$ and $x^\top y$ together.

For CG's second dot product ($r^\top r$) the same mechanism applies to $r$, which is also maintained per-PE in local SRAM.

DAXPY ($x = x + \alpha p$, $r = r - \alpha y$) is purely local; each PE updates its own slice with no communication.

### $x$ update between iterations

After SpMV, $y$ becomes (after normalisation) the new $p$ vector, which must be distributed as $x$ across PE columns for the next iteration. For square matrices ($N_\text{rows} = N_\text{cols}$, $K_y = K_x = K$), the $y$ stripe at PE row $p_y$ becomes the $x$ slice for PE column $p_y$, a 90 degree transpose. This is implemented as a one-time NoC scatter per iteration: PE $(p_x, p_y)$ sends its partial result to PE $(p_y, p_x)$.

## Rationale

- The south conveyor costs no extra H2D calls; it reuses the D2H path already needed for $y$
- CG dot products are computed for free during the conveyor pass, no additional communication round
- DAXPY being local means CG adds zero inter-PE traffic beyond SpMV plus the conveyor
- The transpose for $x$ update is $O(N)$ data over NoC, comparable to one H2D pass of $x$, and overlaps with host-side preprocessing of the next $A$ column slice

## Consequences

- The bottom PE row ($p_y = H - 1$) has a different role (D2H gateway) from interior rows, handled via CSL colors (ADR-006)
- The $x$-transpose step requires a NoC all-to-all within the PE grid; at 857 K PEs this must be implemented as a structured wavefront, not arbitrary point-to-point, to avoid deadlock
- The conveyor serialises $y$ output through a single PE column; for very large $N_\text{rows}$ this could add latency. Alternative: all PE $(0, p_y)$ write to D2H independently ($W$ parallel D2H streams); decision deferred to implementation

## Amendment (2026-10-08): measured collective and sync costs

From microbenchmarks M6–M8 (65,536 PEs unless stated; `cs3-microbench`):
- **East-west reduction (this ADR's step 1) is cheap:** 13.7 cycles per PE + 1.13 per word, i.e. 4.1 µs (1 word) to 6.8 µs (2,048 words) across a 256-PE row, ≈4.7 µs at the SpMV's KY ≈ 446. The vendor allreduce library's row reduce is 3.6× slower for scalars and ~100× slower at 2,048 words. Keep our `@fadds` chain.
- **Dot products: do them on the wafer.** A scalar allreduce over 256×256 costs ~31 µs (2.4 µs at 16×16, 8 µs at 64×64). Any host round trip costs ≥ 167 µs (one-word D2H read, M7). This supports computing CG's dot products on the wafer, either carried by the conveyor as decided above, or as allreduce.
- **Host synchronisation:** a blocking `launch` returns after ~13 µs without waiting for the device (M7). Synchronise with a D2H read, never a launch.
- **D2H shape:** a full col-major rectangle or 1-wide columns are best (0.89 / 3.8 / 6.3 GB/s at 1 / 8 / 16 ch, M6). One-PE-tall rectangles are ~70× slower; the conveyor's bottom-row D2H must not be issued that way.
- **Still open:** the south conveyor isn't implemented; y is read as a 1×H column-0 rectangle. The x-transpose is untested as such: broadcasts are cheap (≈4 cycles per PE), but a transpose isn't a broadcast.

## Amendment (2026-10-16): status at v2.0

v2.0 builds and measures the resident SpMV with the east-west reduction (about 13 cycles per PE) and returns y through the column-0 D2H read. Not built: the south conveyor, the on-wafer dot products and the y-to-x transpose. The transpose decides whether the iteration loop can stay on the wafer. Without it each iteration pays a host round trip of 0.86 ms (65K) or 0.93 ms (full wafer) plus feeding x (about 11 ms at 65K, 160 ms at full wafer), which is more than the host CPU's whole product. It is the first work item of v3.0.
