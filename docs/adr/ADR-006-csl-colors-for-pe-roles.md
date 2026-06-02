# ADR-006: CSL Colors for PE Role Differentiation

**Status:** Accepted  
**Date:** 2026-05-31

## Context

The aggregation architecture (ADR-005) requires three distinct PE behaviours:

| Role | PEs | Behaviour |
|---|---|---|
| **Compute** | all $(p_x, p_y)$ | receive $A$ tile stream, multiply by local $x$, accumulate `partial_y` |
| **Accumulate** | $p_y < H - 1$ (interior rows) | additionally reduce `partial_y` east to west, pass south |
| **Output** | $p_y = H - 1$ (bottom row) | accumulate final $y$, write to D2H queue |

One approach is to stage separate CSL programs for each role. This is operationally complex: the SDK must address individual PEs and upload multiple binaries.

## Decision

Use a **single CSL program** that branches on PE position at runtime using `@get_id()` and **CSL colors** to route messages:

```csl
// Each PE queries its own coordinates at startup
const my_x: i16 = @get_id(.{ .dim = .x });
const my_y: i16 = @get_id(.{ .dim = .y });

const IS_BOTTOM_ROW = (my_y == HEIGHT - 1);

// Colors define message types and routing:
// COLOR_A_STREAM : incoming A tile values (from H2D fabric)
// COLOR_PARTIAL_Y: east-to-west partial_y reduction
// COLOR_SOUTH    : south-bound conveyor (y stripe + partial dot)
// COLOR_D2H      : output to host (bottom row only)
```

Interior compute PEs activate handlers for `COLOR_A_STREAM` and `COLOR_PARTIAL_Y`. The bottom row additionally activates `COLOR_D2H`. Because colors are compile-time constants in CSL but handlers are selected by runtime position, the branching compiles to a single binary with dead code eliminated per PE by the CSL compiler.

## Rationale

- **Single binary:** one compile artefact, one `SdkLauncher.stage()` call, no per-PE addressing complexity
- **CSL-idiomatic:** colors are the native NoC routing mechanism on the WSE; using them for role differentiation is the intended pattern
- **Compile-time optimisation:** the CSL compiler can eliminate unreachable color handlers per PE tile when position is statically known at the fabric level, keeping per-PE code size small
- **Extensible:** adding a new role (e.g., a dedicated reduction PE for CG) requires adding a new color and a position predicate; no staging changes needed

## Consequences

- PE position predicates (`IS_BOTTOM_ROW`, `IS_LEFT_COLUMN`, etc.) must be consistently defined across the CSL program; any mismatch silently produces wrong results
- The bottom row ($p_y = H - 1$) consumes slightly more SRAM for the D2H buffer; this must be accounted for in the memory budget (ADR-002)
- Colors are a finite resource in CSL (typically ~24 usable); the current design uses 4, well within budget, leaving room for CG additions
