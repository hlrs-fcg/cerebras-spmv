# Sketch: Streaming SpMV with On-Wafer Reduction

## Current state (pe.csl) — batch model

```
H2D: load all buffers (a_values, a_col_ids, a_row_ptr) → launch compute → D2H: download all partial_y
```

The PE stores the entire A tile in SRAM. No inter-PE NoC. Host sums partial results.

## Target — streaming model

```
H2D: stream A tiles → compute on-the-fly → east-west reduce → south conveyor → D2H
```

The PE never buffers the full A tile. Data arrives incrementally; multiply happens as values land. Results reduce on-wafer.

---

## 1. PE roles (ADR-006)

Every PE runs the same binary; `@get_id()` determines role:

| Role | PEs | Handlers |
|---|---|---|
| **Compute** | all `(px, py)` | stream A data, compute multiply, participate in EW reduce |
| **Left column** | `(0, py)` | same as compute + start south conveyor (px=0 has complete y after EW) |
| **Bottom row** | `(px, H-1)` | compute + EW + optionally write to D2H |
| **Origin** | `(0, H-1)` | final D2H output |

```
const my_x = @get_id(.{ .dim = .x });
const my_y = @get_id(.{ .dim = .y });
const IS_LEFT   = (my_x == 0);
const IS_BOTTOM = (my_y == HEIGHT - 1);
```

---

## 2. Data format on the stream (per ADR-003)

Row-header encoding — one header per non-empty row, then `count` raw float32 values:

```
row₀ header: (local_row_id:u16, count:u16)   4 B
row₀ vals:   val₁:f32, val₂:f32, ...         4 B × count
row₁ header: (local_row_id:u16, count:u16)   4 B
row₁ vals: ...
```

Effective cost: `4 + 6/K` B/NNZ ≈ ~4.6 B/NNZ at K = 10 NNZ/row.
(The 6 accounts for the 4 B value + a share of structural overhead.)

**There is no per-value column index in the stream.** Columns are resolved through a pre-loaded `col_indices` array on each PE — see §3 below.

The PE runs a two-phase state machine: **await header** → **consume `count` values** → **await header** → ...

### 3. Where column indices live

Matrix structure is static across SpMV iterations (as in CG). So column indices are **pre-loaded once** into each PE's SRAM during setup:

| Data | Loaded | Per PE cost | Notes |
|---|---|---|---|
| `x_vec[KX]` | One-time bulk H2D | 16 KB | Permanent across iterations |
| `col_indices[MAX_TILE_NNZ]` | One-time bulk H2D | ~8-10 KB | Same order as stream values per row |
| `partial_y[KY]` | Zeroed per iteration | ~1-4 KB | Accumulator for this PE's rows |

The streaming data for each SpMV iteration is _only_ the row headers + values. The PE processes each row:

```
// Pre-loaded once at setup:
var col_indices: [MAX_TILE_NNZ]i32;   // column → which x_vec entry
var row_ptr: [KY + 1]i32;            // row → start offset in col_indices

// Per-iteration stream reader:
fn a_stream_handler(word: u32) void {
    if (!phase) {
        // Header: (local_row_id, count)
        current_row = @as(i16, word & 0xFFFF);
        remaining   = @as(i16, (word >> 16) & 0xFFFF);
        pos = row_ptr[current_row];
        phase = true;
    } else {
        // Value phase: multiply by local x at pre-known column
        var col = col_indices[pos];
        partial_y[current_row] += @bitcast(f32, word) * x_vec[col];
        pos += 1;
        remaining -= 1;
        if (remaining == 0) phase = false;
    }
}
```

This is the key insight: the stream carries **no structural metadata per value** — just the values. The structure (which row, which column) is pre-loaded once. The `+ 6/K` in the byte formula covers the one-time load of `col_indices` amortized over the iteration count.

---

## 4. CSL color map (ADR-006)

Four colors route the data pipeline:

| Color | Direction | Contents | Set by |
|---|---|---|---|
| `COLOR_A_STREAM` | H2D fabric → PE | Row-header data | SDK memcpy (`streaming=True`) |
| `COLOR_EW` | East → West | Partial y stripe | NoC send from PE (px+1, py) |
| `COLOR_SOUTH` | North → South | Completed y stripe + dot | NoC send from PE (0, py-1) |
| `COLOR_D2H` | PE (0, H-1) → Host | Final y | Bottom-left PE |

```
// In pe.csl comptime block:
const COLOR_A_STREAM = @get_color(8);
const COLOR_EW       = @get_color(9);
const COLOR_SOUTH    = @get_color(10);
const COLOR_D2H      = @get_color(11);

// Only bottom PE activates D2H:
if (IS_BOTTOM and IS_LEFT) {
    // Color handler for COLOR_D2H writes to SDK output queue
}
```

---

## 5. pe.csl streaming sketch

```
param memcpy_params;
param KX: i32;
param KY: i32;
param MAX_TILE_NNZ: i32;

const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

// ── Permanent state (loaded once at setup) ──
var x_vec       = @zeros([KX]f32);            // x slice, permanent
var col_indices = @zeros([MAX_TILE_NNZ]i32);  // matrix structure, permanent
var row_ptrs    = @zeros([KY + 1]i32);        // row start/end in col_indices

// ── Per-iteration state ──
var partial_y = @zeros([KY]f32);              // accumulator, zeroed each SpMV
var y_buf     = @zeros([KY]f32);              // EW receive / south send buffer

// Stream state machine
var current_row: i16 = 0;
var remaining:   i16 = 0;
var pos:         i32 = 0;
var phase: bool = false;   // false = awaiting header, true = consuming values

// ── Color: incoming A stream from H2D fabric ──
fn a_stream_handler(data: u32) void {
    if (!phase) {
        // Header: (local_row_id:u16, count:u16) packed in one u32
        current_row = @as(i16, data & 0xFFFF);
        remaining   = @as(i16, (data >> 16) & 0xFFFF);
        pos = row_ptrs[current_row];
        phase = true;
    } else {
        // Value: multiply by local x at pre-loaded column index
        var col = col_indices[pos];
        partial_y[current_row] += @bitcast(f32, data) * x_vec[col];
        pos += 1;
        remaining -= 1;
        if (remaining == 0) phase = false;
    }
}

// ── Color: east-west reduction ──
// Receive partial_y from east (px+1), add own, send west (px-1)
fn ew_handler(src_data: u32) void {
    // Stride-based EW reduce over KY elements
    // After W-1 steps, px=0 has the complete stripe
}

// ── Color: south conveyor (px = 0 only) ──
// Receive completed y stripe from north, append own, send south
fn south_handler(src_data: u32) void { ... }

// ── D2H output (px = 0, py = H-1 only) ──
fn d2h_handler() void {
    // Write completed y to host output queue
}

comptime {
    @bind_color_handler(a_stream_handler, COLOR_A_STREAM);
    if (!IS_LEFT) {
        @bind_color_handler(ew_handler, COLOR_EW);
    }
    if (IS_LEFT and !IS_BOTTOM) {
        @bind_color_handler(south_handler, COLOR_SOUTH);
    }
    if (IS_LEFT and IS_BOTTOM) {
        @bind_color_handler(d2h_handler, COLOR_D2H);
    }
}
```

---

## 6. layout.csl additions

```
// Pass HEIGHT as a param so pe.csl can use @get_id()
// for role detection:

layout {
    @set_rectangle(width, height);
    var py: i16 = 0;
    while (py < height) : (py += 1) {
        var px: i16 = 0;
        while (px < width) : (px += 1) {
            @set_tile_code(px, py, "pe.csl", .{
                .memcpy_params = memcpy.get_params(px),
                .KX = KX,
                .KY = KY,
                .HEIGHT = height,    // <-- new: for role detection
            });
        }
    }
}
```

---

## 7. Data flow timeline

```
Time ──────────────────────────────────────────────────────►

H2D:  |== stream A col 0 ===|== stream A col 1 ===| ... 
       │                     │
PE:    │a_stream_handler()   │
       │  accumulate         │
       │  partial_y[0..KY-1] │
       │                     │
EW:    │ ←─── EW reduce ──── │  (pipelined: PE (px,py) sends
                            │   while PE (px+1,py) still computing)
South: │  y[0..KY-1] ↓      │  (column 0 only)
       │  y[KY..2KY-1] ↓    │
       │  ...               │
D2H:   │  y[0..Ky] → host   │
```

Key: the EW pipeline can start as soon as the rightmost PE (px=W-1) finishes. Subsequent PEs join the pipeline leftward as they finish, overlapping compute with reduction.

---

## 8. Host-side changes

The host now has two phases: **setup** (one-time) and **iterate** (per SpMV).

### Setup phase (one-time per matrix)

Pre-load matrix structure onto PEs via bulk H2D:

```python
# One-time, before the SpMV loop:
runner.memcpy_h2d(sym_x_vec,       x_slices,        ..., streaming=False)
runner.memcpy_h2d(sym_col_indices, all_col_indices,  ..., streaming=False)
runner.memcpy_h2d(sym_row_ptrs,    all_row_ptrs,     ..., streaming=False)
```

No more `a_values` in the setup — values stream each iteration.

### Iterate phase (per SpMV)

Stream only headers + values:

```python
encoded = encode_values_stream(row_idx, col_idx, data, W, H, KX, KY)
runner.memcpy_h2d(sym_a_stream, encoded,
                  0, 0, W, H, len(encoded),
                  streaming=True, ...)
```

`a_stream` is the only per-iteration H2D symbol. D2H reads `y_result` from PE (0, H-1) — size N_rows, not W×H×KY. The gather at `worker_runtime.py:237` is eliminated.

### coo_tiling.py — two encoding paths

```python
def extract_tile_structure(
    row_idx, col_idx, W, H, KX, KY
) -> tuple[list, list, list]:
    """Compute col_indices and row_ptrs per PE tile (one-time)."""
    # Returns (col_indices_per_pe, row_ptrs_per_pe, nnz_counts)

def encode_values_stream(
    data, tile_nnz, tile_starts, order
) -> np.ndarray:
    """Pack values in row-header format: header, val, val, ..., header, val, ..."""
    words = []
    for pe in range(num_pes):
        s, e = tile_starts[pe], tile_starts[pe + 1]
        if s == e:
            continue
        # data[s:e] is already in row-major, sorted by (row, col)
        # Find row boundaries and emit headers
        # ...
    return np.array(words, dtype=np.uint32)
```

The scatter phase (CSR → per-tile sorted buckets) still uses the same `lexsort` approach from the current `tile_coo`. The difference is the output format: instead of producing `a_values[TOTAL_NNZ]` + `a_col_ids[TOTAL_NNZ]` + `a_row_ptr[KY+1]` for all PEs, it produces `col_indices` + `row_ptrs` once, and per-iteration `values_stream`.

---

## 9. Summary of changes

| Component | Current (batch) | Streaming target |
|---|---|---|
| `pe.csl` | Pre-loaded buffers + `@activate(compute)`, per-NNZ col indices in bulk arrays | Color handlers + row-header state machine; col indices pre-loaded once |
| `layout.csl` | 2D grid, same binary per PE | + `HEIGHT` param for `@get_id()` role detection |
| `sdk/launcher.py` | Stages `x_vec`, `a_values`, `a_col_ids`, `a_row_ptr`, `partial_y` | Stages `x_vec`, `col_indices`, `row_ptrs` (once) + `a_stream` (per iter) + `y_result` (D2H) |
| `worker_runtime.py` | 4x bulk H2D per iter; host `y_flat.sum(axis=1)` gather | 1x streaming H2D per iter + D2H from bottom-left PE; no host gather |
| `coo_tiling.py` | `tile_coo()` → per-tile CSR arrays | `extract_tile_structure()` once + `encode_values_stream()` per iter |
| `benchmark.py` | Unchanged | Unchanged |

The streaming model eliminates the per-PE bulk H2D calls (O(num_PEs) → O(W)) and moves aggregation from worker CPU to on-wafer NoC. The one-time pre-load of `col_indices` + `row_ptrs` is amortized over iterations (CG).
