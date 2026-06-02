# ADR-002: 2D Data Layout: x Permanent on Wafer, A Streamed in Column Slices

**Status:** Accepted  
**Date:** 2026-05-31

## Context

SpMV computes $y = Ax$ where $A$ is sparse (CSR), $x$ is dense. Both $A$ and $x$ must reach the PEs. Two axes of distribution must be decided:

1. **How are matrix rows assigned to PEs?** (row axis = PE-row dimension)
2. **How is $x$ made available?** (column axis = PE-column dimension)

An earlier matched-pairs prototype had the host pre-gather `x[col_indices]` and send `(A_val, x_val)` pairs per PE. This is correct but host-heavy: the host does $O(\text{NNZ})$ random-access work per iteration, and $A$ plus $x$ must be retransmitted every iteration.

### Memory budget per PE row ($W = 750$ PEs)

| Allocation | Size |
|---|---|
| Total SRAM per PE | 48 KB |
| Runtime / code / stack | ~10 KB |
| Available for data | ~38 KB |
| Per row of PEs ($\times 750$) | ~27.8 MB |
| $x$ budget (proposed) | 12 MB, **3.1 M float32 entries** |
| Remaining for $A$ tile | ~21 KB per PE, ~2 700 NNZ |

## Decision

Adopt a **2D tiling layout**:

- **PE column $p_x$** permanently stores $x[p_x K_x \,..\, (p_x{+}1) K_x]$, where $K_x = N_\text{cols} / W$. All $H$ PE rows in a column hold the same $x$ slice, sent once via H2D at startup.
- **PE $(p_x, p_y)$** processes the $A$ tile for rows $[p_y K_y \,..\, (p_y{+}1) K_y]$ and columns $[p_x K_x \,..\, (p_x{+}1) K_x]$. The $A$ tile is streamed in; $x[j]$ is always local (no NoC lookup during multiply).
- After local accumulation, partial $y$ is reduced **east-to-west** along each PE row. The leftmost PE column ($p_x = 0$) collects complete $y$ stripes.

### Why $x$ must be distributed, not broadcast

Broadcasting the full $x$ vector to every PE requires $x$ to fit in each PE's 48 KB SRAM. For $N_\text{cols} = 3.1\,\text{M}$ that would be 12 MB, which is impossible. With $W = 750$ PE columns, each PE holds only $12\,\text{MB} / 750 = 16\,\text{KB}$ of $x$, well within budget.

## Rationale

- **No NoC during multiply:** $A$ tile columns align exactly with the local $x$ slice, so every $x$ lookup is in local SRAM, with zero inter-PE communication during the compute phase
- **$x$ amortised over iterations:** for iterative solvers (CG, GMRES) $x$ is loaded once; only the east-west reduction and the iteration update (DAXPY) incur further traffic
- **Scales with wafer size:** adding PE columns increases $x$ capacity proportionally

## Consequences

- $N_\text{cols}$ is bounded by available per-PE-row SRAM allocated to $x$ (currently 3.1 M float32)
- The east-west reduction adds $O(W)$ NoC messages per PE row per SpMV; at $W = 750$ this is negligible compared to H2D time (~0.003 ms vs ~2 ms)
- Updating $x$ between Krylov iterations requires a NoC transpose for square matrices: $y$ lives on PE rows, new $x$ must live on PE columns (see ADR-005)
