"""Pure-numpy COO matrix generation and 2-D PE-grid tiling.

No scipy or SDK imports — safe to import on login node and in tests.
"""

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


def generate_random_coo(
    rows: int,
    cols: int,
    density: float,
    seed: int = 42,
    cache_dir: str = "matrices",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (row_idx, col_idx, data) as int32/float32 arrays.

    Duplicates are allowed — tile_coo and spmv_reference both sum them correctly.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"coo_{rows}x{cols}_d{density}_s{seed}.npz"

    if cache_file.exists():
        npz = np.load(cache_file)
        row_idx, col_idx, data = npz["row"], npz["col"], npz["data"]
        logger.info(f"Loaded cached COO: ({rows},{cols}) NNZ={len(data):,}")
        return row_idx, col_idx, data

    logger.info(f"Generating matrix (caching to {cache_file.name})...")
    rng = np.random.default_rng(seed)
    nnz = max(1, int(rows * cols * density))
    row_idx = rng.integers(0, rows, size=nnz, dtype=np.int32)
    col_idx = rng.integers(0, cols, size=nnz, dtype=np.int32)
    data = rng.random(size=nnz).astype(np.float32)

    np.savez(cache_file, row=row_idx, col=col_idx, data=data)
    logger.info(f"Generated and cached: ({rows},{cols}) NNZ={nnz:,}")
    return row_idx, col_idx, data


def stable_tile_order(pe_id: np.ndarray, loc_row: np.ndarray, loc_col: np.ndarray) -> np.ndarray:
    """``np.lexsort((loc_col, loc_row, pe_id))``, computed by LSD radix passes.

    numpy sorts 16-bit keys with a radix sort (O(n)) when ``kind="stable"``, but 32-bit keys with
    a comparison sort.  So every key is cut into 16-bit digits, least significant first, and the
    order is refined digit by digit; stability makes the result identical to lexsort (duplicates
    keep their input order).  A digit that is zero everywhere (e.g. the high half of pe_id on a
    grid under 65,536 PEs) costs no pass.  ~25x faster than lexsort at 13M NNZ.
    """
    order = None
    for key in (loc_col, loc_row, pe_id):
        key = key.astype(np.uint32, copy=False)
        top = int(key.max()) if len(key) else 0
        for shift in (0, 16):
            if shift and top >> shift == 0:
                break
            digit = ((key >> np.uint32(shift)) & np.uint32(0xFFFF)).astype(np.uint16)
            if order is None:
                order = np.argsort(digit, kind="stable")
            else:
                order = order[np.argsort(digit[order], kind="stable")]
    return order if order is not None else np.arange(0, dtype=np.int64)


def tile_coo(
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    data: np.ndarray,
    x: np.ndarray,
    rows: int,
    cols: int,
    W: int,
    H: int,
    KX: int,
    KY: int,
    TOTAL_NNZ: int,
    num_stripes: int = 1,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], list[int]]:
    """Distribute a COO matrix onto a W×H PE grid, optionally split into column stripes.

    Returns (stripes, tile_nnz_counts) where:
      stripes[s] = (all_x_vec_s, all_a_values_s, all_a_col_ids_s, all_a_row_ptr_s)
      tile_nnz_counts — full-tile NNZ per PE (sum across all stripes), for load balance

    For num_stripes=1 the list has one element and behaviour is identical to the
    single-pass case. KX_STRIPE = ceil(KX / num_stripes) must equal the CSL KX param.
    TOTAL_NNZ is the compile-time buffer size per stripe (not per full tile).
    """
    num_pes   = W * H
    nnz       = len(data)
    KX_STRIPE = math.ceil(KX / num_stripes)

    if nnz == 0:
        empty = (
            np.zeros((num_pes, KX_STRIPE), dtype=np.float32),
            np.zeros((num_pes, TOTAL_NNZ), dtype=np.float32),
            np.zeros((num_pes, TOTAL_NNZ), dtype=np.int32),
            np.zeros((num_pes, KY + 1),    dtype=np.int32),
        )
        return [empty] * num_stripes, [0] * num_pes

    bulk_bytes = num_pes * (KX_STRIPE * 4 + TOTAL_NNZ * 8 + (KY + 1) * 4) * num_stripes
    logger.info(f"Tile bulk allocation: {bulk_bytes / 1e9:.2f} GB (S={num_stripes})")

    # Assign each NNZ to a PE and compute local tile coordinates
    logger.debug("Computing PE assignments...")
    pe_id   = ((row_idx // KY) * W + col_idx // KX).astype(np.int32)
    loc_row = (row_idx % KY).astype(np.int32)
    loc_col = (col_idx % KX).astype(np.int32)

    # Single global sort by (pe_id, loc_row, loc_col) — reused for all stripes
    logger.debug("lexsort...")
    order       = stable_tile_order(pe_id, loc_row, loc_col)
    pe_id       = pe_id[order]
    loc_row     = loc_row[order]
    loc_col     = loc_col[order]
    data_sorted = data[order]
    logger.debug("lexsort done")

    # Full tile NNZ (summed over all stripes) — for load balance reporting
    tile_nnz = np.bincount(pe_id, minlength=num_pes).astype(np.int64)

    CHUNK   = 4096
    stripes = []

    for s in range(num_stripes):
        col_lo = s * KX_STRIPE
        col_hi = min((s + 1) * KX_STRIPE, KX)

        # Filter to NNZ in this stripe's column range
        mask      = (loc_col >= col_lo) & (loc_col < col_hi)
        s_pe_id   = pe_id[mask]
        s_loc_row = loc_row[mask]
        s_loc_col = (loc_col[mask] - col_lo).astype(np.int32)
        s_data    = data_sorted[mask]

        # Per-stripe tile boundaries — searchsorted works because s_pe_id is sorted
        stripe_starts = np.searchsorted(s_pe_id, np.arange(num_pes + 1))
        stripe_nnz    = np.diff(stripe_starts).astype(np.int64)

        max_stripe_nnz = int(stripe_nnz.max()) if len(s_pe_id) > 0 else 0
        if max_stripe_nnz > TOTAL_NNZ:
            bad = int(np.argmax(stripe_nnz))
            raise ValueError(
                f"Stripe {s}: PE ({bad % W},{bad // W}) has {max_stripe_nnz} NNZ > "
                f"TOTAL_NNZ={TOTAL_NNZ}; recompile with larger TOTAL_NNZ or more stripes"
            )

        all_x_vec_s    = np.zeros((num_pes, KX_STRIPE), dtype=np.float32)
        all_a_values_s  = np.zeros((num_pes, TOTAL_NNZ), dtype=np.float32)
        all_a_col_ids_s = np.zeros((num_pes, TOTAL_NNZ), dtype=np.int32)
        all_a_row_ptr_s = np.zeros((num_pes, KY + 1),    dtype=np.int32)

        # Build row_ptr via chunked bincount
        logger.debug(f"Stripe {s}: building row pointers...")
        for c0 in range(0, num_pes, CHUNK):
            c1 = min(c0 + CHUNK, num_pes)
            n  = c1 - c0
            cs = int(stripe_starts[c0])
            ce = int(stripe_starts[c1])
            if cs < ce:
                flat   = (s_pe_id[cs:ce].astype(np.int64) - c0) * (KY + 1) + s_loc_row[cs:ce] + 1
                counts = np.bincount(flat, minlength=n * (KY + 1)).reshape(n, KY + 1)
                all_a_row_ptr_s[c0:c1] = np.cumsum(counts, axis=1, dtype=np.int32)

        # Scatter values and col_ids
        logger.debug(f"Stripe {s}: scattering values...")
        if len(s_pe_id) > 0:
            local_pos = np.arange(len(s_pe_id), dtype=np.int64) - stripe_starts[s_pe_id]
            all_a_values_s[s_pe_id,  local_pos] = s_data
            all_a_col_ids_s[s_pe_id, local_pos] = s_loc_col

        # Fill x_vec: broadcast this stripe's x slice to all H PE rows in each PE column
        logger.debug(f"Stripe {s}: filling x_vec...")
        x_grid = all_x_vec_s.reshape(H, W, KX_STRIPE)
        for px in range(W):
            x_g_start = px * KX + col_lo
            x_g_end   = min(px * KX + col_hi, cols)
            if x_g_start < cols:
                n = x_g_end - x_g_start
                x_grid[:, px, :n] = x[x_g_start:x_g_end]

        stripes.append((all_x_vec_s, all_a_values_s, all_a_col_ids_s, all_a_row_ptr_s))

    min_nnz   = int(tile_nnz.min())
    max_nnz   = int(tile_nnz.max())
    avg_nnz   = float(tile_nnz.mean())
    imbalance = (max_nnz - min_nnz) / avg_nnz * 100 if avg_nnz > 0 else 0.0
    logger.info(
        f"Tile NNZ — min={min_nnz} avg={avg_nnz:.0f} max={max_nnz} "
        f"imbalance={imbalance:.1f}% S={num_stripes} KX_STRIPE={KX_STRIPE}"
    )

    return stripes, tile_nnz.tolist()


def spmv_reference(
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    data: np.ndarray,
    x: np.ndarray,
    rows: int,
) -> np.ndarray:
    """Reference SpMV via bincount — handles duplicates, no scipy needed."""
    contrib = data.astype(np.float64) * x[col_idx].astype(np.float64)
    return np.bincount(row_idx, weights=contrib, minlength=rows).astype(np.float32)


# ---------------------------------------------------------------------------
# Row-header streaming encoder (ADR-003)
# ---------------------------------------------------------------------------

RowHeaderBounds = list[list[tuple[int, int, int]]]
"""Per-PE list of (local_row, start_within_pe, count) for each non-empty row (legacy form,
still returned by parallel_csr_scatter)."""


class RowBounds(NamedTuple):
    """The same information as RowHeaderBounds, as flat arrays (CSR-like).

    Row ``j`` of PE ``pe`` is entry ``ptr[pe] + j``; ``row``, ``start`` (offset inside the PE's
    tile) and ``count`` are parallel arrays over every non-empty row of every PE, PE-major.
    """
    ptr: np.ndarray      # (num_pes + 1,) int64
    row: np.ndarray      # (R,) int32 local row
    start: np.ndarray    # (R,) int32 offset of the row's first NNZ within its PE
    count: np.ndarray    # (R,) int32 NNZ in the row

    @property
    def rows_per_pe(self) -> np.ndarray:
        return np.diff(self.ptr)

MAX_ROW_NNZ = 32767
"""Largest per-row NNZ a stream header may carry.

The header packs the count into 16 bits, so 65535 is the format limit.  We
enforce 32767 instead: the kernel's ``remaining`` counter is 16-bit and
decremented to zero, and staying inside the signed range keeps the decode
correct however that counter happens to be typed (it was ``i16`` once, where
32768 decoded as -32768 and the ``== 0`` test never fired).
"""

MAX_LOCAL_ROW = 65535
"""Largest local row index a stream header may carry (16-bit field)."""


def _pack_header(local_row: int, count: int) -> np.uint32:
    """Pack one row header: (count:u16 << 16) | (local_row:u16)."""
    if count > MAX_ROW_NNZ:
        raise ValueError(
            f"local row {local_row} holds {count} NNZ on a single PE, but a "
            f"stream header carries at most {MAX_ROW_NNZ}. Use more PE columns "
            f"so each tile holds fewer NNZ per row."
        )
    return np.uint32(count << 16) | np.uint32(local_row & 0xFFFF)


def extract_tile_structure(
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    W: int, H: int, KX: int, KY: int,
    want_row_bounds: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray, "RowBounds | None"]:
    """Extract per-PE column indices and row pointers from COO data.

    One-time setup for the streaming SpMV pipeline. Returns everything
    needed to pre-load matrix structure onto PEs and to encode values
    each iteration.

    Parameters
    ----------
    row_idx, col_idx : int32 arrays of NNZ positions.
    W, H : PE grid dimensions.
    KX, KY : tile dimensions.

    Returns
    -------
    all_col_indices : (num_pes, max_tile_nnz) int32
        Column indices per PE, in row-major order.
    all_row_ptrs : (num_pes, KY + 1) int32
        Row pointers into col_indices for each PE.
    tile_nnz : list[int]
        NNZ count per PE.
    order : (nnz,) int64
        Global sort index mapping input → sorted (pe_id, row, col) order.
        Pass to encode_values_stream() each iteration.
    row_bounds : RowBounds or None
        Per-PE (local_row, start_offset, count) of every non-empty row, as flat arrays
        (header encoding only; ``want_row_bounds=False`` skips building it and returns None).
    """
    if KY > MAX_LOCAL_ROW:
        raise ValueError(
            f"KY={KY} exceeds {MAX_LOCAL_ROW}: a stream header carries the "
            f"local row index in 16 bits, so larger tiles alias rows silently. "
            f"Use a taller PE grid (--grid-height) so each PE row owns fewer "
            f"matrix rows."
        )

    num_pes = W * H
    nnz = len(row_idx)

    if nnz == 0:
        max_nnz = 1
        all_col_indices = np.zeros((num_pes, 1), dtype=np.int32)
        all_row_ptrs = np.zeros((num_pes, KY + 1), dtype=np.int32)
        order = np.arange(0, dtype=np.int64)
        tile_nnz = [0] * num_pes
        row_bounds = _empty_row_bounds(num_pes) if want_row_bounds else None
        return all_col_indices, all_row_ptrs, tile_nnz, order, row_bounds

    pe_id   = ((row_idx // KY) * W + col_idx // KX).astype(np.int32)
    loc_row = (row_idx % KY).astype(np.int32)
    loc_col = (col_idx % KX).astype(np.int32)

    order = stable_tile_order(pe_id, loc_row, loc_col)
    pe_id_sorted   = pe_id[order]
    loc_row_sorted = loc_row[order]
    loc_col_sorted = loc_col[order]

    tile_nnz   = np.bincount(pe_id_sorted, minlength=num_pes).astype(np.int64)
    max_nnz    = int(tile_nnz.max())
    tile_starts    = np.empty(num_pes + 1, dtype=np.int64)
    tile_starts[0] = 0
    np.cumsum(tile_nnz, out=tile_starts[1:])

    all_col_indices = np.zeros((num_pes, max(max_nnz, 1)), dtype=np.int32)
    all_row_ptrs    = np.zeros((num_pes, KY + 1), dtype=np.int32)

    local_pos = np.arange(nnz, dtype=np.int64) - tile_starts[pe_id_sorted]
    all_col_indices[pe_id_sorted, local_pos] = loc_col_sorted

    CHUNK = 4096
    for c0 in range(0, num_pes, CHUNK):
        c1 = min(c0 + CHUNK, num_pes)
        n  = c1 - c0
        s, e = int(tile_starts[c0]), int(tile_starts[c1])
        if s < e:
            pe_chunk = pe_id_sorted[s:e].astype(np.int64) - c0
            flat = pe_chunk * (KY + 1) + loc_row_sorted[s:e] + 1
            counts = np.bincount(flat, minlength=n * (KY + 1)).reshape(n, KY + 1)
            all_row_ptrs[c0:c1] = np.cumsum(counts, axis=1, dtype=np.int32)

    row_bounds = (_compute_row_bounds(pe_id_sorted, loc_row_sorted, tile_starts, num_pes)
                  if want_row_bounds else None)

    return all_col_indices, all_row_ptrs, tile_nnz.tolist(), order, row_bounds


def _empty_row_bounds(num_pes: int) -> RowBounds:
    return RowBounds(np.zeros(num_pes + 1, dtype=np.int64), np.zeros(0, dtype=np.int32),
                     np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32))


def _compute_row_bounds(
    pe_id_sorted: np.ndarray,
    loc_row_sorted: np.ndarray,
    tile_starts: np.ndarray,
    num_pes: int,
) -> RowBounds:
    """Per-PE row boundaries for stream encoding: one entry per (pe, local_row) run."""
    n = len(pe_id_sorted)
    if n == 0:
        return _empty_row_bounds(num_pes)
    new = np.empty(n, dtype=bool)
    new[0] = True
    np.not_equal(pe_id_sorted[1:], pe_id_sorted[:-1], out=new[1:])
    new[1:] |= loc_row_sorted[1:] != loc_row_sorted[:-1]
    idx = np.flatnonzero(new)
    pe_of = pe_id_sorted[idx]
    ptr = np.zeros(num_pes + 1, dtype=np.int64)
    np.cumsum(np.bincount(pe_of, minlength=num_pes), out=ptr[1:])
    return RowBounds(
        ptr,
        loc_row_sorted[idx].astype(np.int32),
        (idx - tile_starts[pe_of]).astype(np.int32),
        np.diff(idx, append=n).astype(np.int32),
    )


def count_nonempty_rows(row_ptrs: np.ndarray, chunk: int = 8192) -> int:
    """Number of (PE, row) pairs holding at least one NNZ, from the (num_pes, KY+1) row_ptrs."""
    total = 0
    for c0 in range(0, len(row_ptrs), chunk):
        total += int(np.count_nonzero(np.diff(row_ptrs[c0:c0 + chunk], axis=1)))
    return total


def _encode_header_streams(
    data: np.ndarray,
    order: np.ndarray,
    tile_nnz: list[int],
    rb: RowBounds,
) -> tuple[np.ndarray, np.ndarray]:
    """Row-header wire format (ADR-003) for every PE, PE-major: ``(flat uint32, lengths int64)``.

    Each PE stream is ``[(count:u16 << 16 | local_row:u16), val, val, ...]`` per non-empty row.
    Fully vectorised: header ``j`` lands at ``tile_start + start + j``, the ``i``-th sorted value
    of row ``j`` at ``i + j + 1``, because every row before it added one header word.
    """
    num_pes = len(tile_nnz)
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.uint32), np.zeros(num_pes, dtype=np.int64)
    counts = np.asarray(tile_nnz, dtype=np.int64)
    if len(rb.count) and int(rb.count.max()) > MAX_ROW_NNZ:
        bad = int(np.argmax(rb.count > MAX_ROW_NNZ))
        _pack_header(int(rb.row[bad]), int(rb.count[bad]))      # raises with the usual message
    data_sorted = data[order].view(np.uint32)
    tile_starts = np.zeros(num_pes + 1, dtype=np.int64)
    np.cumsum(counts, out=tile_starts[1:])
    n_rows = len(rb.row)
    row_pe = np.repeat(np.arange(num_pes, dtype=np.int64), rb.rows_per_pe)
    row_first = tile_starts[row_pe] + rb.start                   # sorted index of each row's first NNZ
    flat = np.empty(n + n_rows, dtype=np.uint32)
    flat[row_first + np.arange(n_rows, dtype=np.int64)] = (
        (rb.count.astype(np.uint32) << np.uint32(16)) | (rb.row.astype(np.uint32) & np.uint32(0xFFFF))
    )
    row_of_val = np.repeat(np.arange(n_rows, dtype=np.int64), rb.count)
    flat[np.arange(n, dtype=np.int64) + row_of_val + 1] = data_sorted
    return flat, counts + rb.rows_per_pe


def encode_values_stream(
    data: np.ndarray,
    order: np.ndarray,
    tile_nnz: list[int],
    row_bounds: RowBounds,
) -> np.ndarray:
    """Encode values in row-header wire format (ADR-003).

    Parameters
    ----------
    data : float32 array of NNZ values (original unsorted order).
    order : int64 sort index from extract_tile_structure().
    tile_nnz : NNZ count per PE from extract_tile_structure().
    row_bounds : per-PE row boundaries from extract_tile_structure().

    Returns
    -------
    stream : uint32 array
        [(local_row_id:u16, count:u16), val:f32, val:f32, ...] per PE,
        concatenated in PE-row-major order.
    """
    return _encode_header_streams(data, order, tile_nnz, row_bounds)[0]


# ---------------------------------------------------------------------------
# Phase 2: stream encoder (ADR-004 — was column-parallel, now vectorised)
# ---------------------------------------------------------------------------


def encode_streams_parallel(
    data: np.ndarray,
    order: np.ndarray,
    tile_nnz: list[int],
    row_bounds: RowBounds,
    num_threads: int = 64,
) -> tuple[np.ndarray, list[int]]:
    """Encode all PE streams: ``(flat_buffer uint32, per-PE word counts)``.

    ``num_threads`` is accepted for API compatibility; the encoder is vectorised numpy
    (no per-PE Python loop), which beat the thread pool it replaced.
    """
    flat, lengths = _encode_header_streams(data, order, tile_nnz, row_bounds)
    return flat, lengths.tolist()


def build_stream_buffer(
    data: np.ndarray,
    order: np.ndarray,
    tile_nnz: list[int],
    row_bounds: RowBounds,
    W: int, H: int,
    num_threads: int = 64,
) -> tuple[np.ndarray, list[int]]:
    """Build a uint32 buffer for streaming H2D, one 0-terminated stream per PE.

    Every PE's words are followed by an **explicit 0 end-of-stream sentinel**,
    and the buffer is sized from the data (longest stream + 1) so that
    terminator always has room.  Do not reintroduce a caller-supplied bound such
    as ``max_nnz + KY``: that is a tight upper bound which a PE whose rows are
    all non-empty fills exactly, leaving no trailing zero for the kernel to stop
    on — the PE then never reaches its EW reduction and the run deadlocks.

    Returns ``(buf, col_words)``:

    ``buf``
        ``(W, H, max_words)`` uint32, PE ``(px, py)`` at ``buf[px, py, :]``.
    ``col_words``
        Exact word count column ``px`` must transfer: the longest stream in that
        column plus its sentinel.  Send ``buf[px, :, :col_words[px]]`` so no
        column carries padding sized for a busier column.

    A rectangle memcpy sends a uniform count to every PE it covers, so a PE
    whose stream is shorter than its column's longest still receives the
    difference; the kernel drains those words instead of blocking on them.
    """
    num_pes = W * H
    flat, lengths = _encode_header_streams(data, order, tile_nnz, row_bounds)

    # Per-column transfer length = longest stream in the column + 1 sentinel.
    col_words = (lengths.reshape(H, W).max(axis=0) + 1).tolist()
    max_words = int(max(col_words))

    buf = np.zeros((W, H, max_words), dtype=np.uint32)     # zeros double as the sentinels
    if len(flat):
        starts = np.zeros(num_pes + 1, dtype=np.int64)
        np.cumsum(lengths, out=starts[1:])
        pe_of_word = np.repeat(np.arange(num_pes, dtype=np.int64), lengths)
        local = np.arange(len(flat), dtype=np.int64) - starts[pe_of_word]
        buf[pe_of_word % W, pe_of_word // W, local] = flat
    return buf, col_words


# ---------------------------------------------------------------------------
# Val-only streaming encoder (ADR-003 "val-only", src/csl/pe_valtask.csl)
# ---------------------------------------------------------------------------
#
# The stream carries float32 values only, in the per-PE CSR order that
# extract_tile_structure() already produced for col_indices / row_ptrs, so the
# resident structure needs no new per-NNZ array.  The kernel stops by COUNT,
# not by sentinel (every 32-bit pattern is a legal float, 0x0 included), so the
# number of words each PE expects must equal exactly what the transfers
# deliver.  plan_val_transfers() is the single source of both.

ENCODING_IDS = {"header": 0, "valtask": 2, "resident": 3}
"""Encodings and their ENCODING compile params.  Id 1 was the removed bulk "val" kernel and is
not reused.  "resident" streams nothing: its values are copied once at setup (build_resident_vals).
Keep equal to config.ENCODING_IDS."""

ENCODINGS = tuple(ENCODING_IDS)

VAL_ENCODINGS = ("valtask",)
"""Values-only encodings: one plan_val_transfers() stream + val_meta, consumed by a per-word
data task (valtask, pe_valtask.csl)."""

NULL_SPMV_ENCODING_ID = 4
"""ENCODING param of the resident kernel with spmv() a no-op (--null-spmv); the encoding name
stays "resident".  Keep equal to config.NULL_SPMV_ENCODING_ID."""

RESIDENT_ENCODINGS = ("resident",)
"""A on the wafer: vals[] is loaded by the setup copy, no per-call stream (pe_resident.csl)."""

MAX_VAL_STREAM_WORDS = 65535
"""Largest per-PE val-only transfer (the per-PE word count is a 16-bit quantity on the wire
plan; kept from the bulk kernel it replaced)."""


class ValTransferPlan:
    """The val-only H2D transfers for one pass, and what each PE must expect.

    ``rects``
        ``(x0, width, n_words)`` per ``memcpy_h2d`` call, each covering columns
        ``x0 .. x0+width-1`` and all ``H`` rows.  Calls with ``n_words == 0``
        are omitted: a column with no NNZ anywhere sends nothing.
    ``pe_words``
        ``(W*H,)`` int32 in ``pe = py*W + px`` order -- the order the setup
        memcpy's ROW_MAJOR layout uses.  Preloaded into ``val_meta[0]``.
        Computed by summing ``rects``, so it is what the plan sends by
        construction, padding included.
    """

    def __init__(self, rects: list[tuple[int, int, int]], W: int, H: int):
        self.rects = rects
        grid = np.zeros((H, W), dtype=np.int64)
        for x0, width, n_words in rects:
            grid[:, x0:x0 + width] += n_words
        if grid.size and int(grid.max()) > MAX_VAL_STREAM_WORDS:
            raise ValueError(
                f"a PE would receive {int(grid.max())} val-only words in one pass, "
                f"but the limit is {MAX_VAL_STREAM_WORDS}. Spread the matrix over more PEs."
            )
        self.pe_words = grid.reshape(-1).astype(np.int32)

    @property
    def total_words(self) -> int:
        """Words sent by the whole plan, padding included (H2D size / 4)."""
        return int(self.pe_words.sum())


def column_rects_serialized(encoding: str, h2d_mode: str, requested: bool) -> bool:
    """Whether the streaming rects of one pass go one at a time.

    valtask + column: always, whatever the word count.  Their W
    one-column rects, issued back to back with nonblock=True, hang on hardware
    once a PE gets more words than the fabric buffers.  (Measured on the removed bulk kernel,
    4x4: ~650 words/PE hung twice, <= 76 passed, the threshold is unknown so no count is
    consulted; it passed with every rect but the last nonblock=False.)  valtask hung
    the same way (4x4, d=0.01, cancelled at 2m54s), so the per-word data task
    does not buy overlap tolerance here.  The simulator passes either way.
    Anything else (header): only if requested (--serialize-column-rects), so
    the header kernel's column path is unchanged by default.  Full mode is one
    rect, so serialising it is a no-op.  docs/val-only-encoding.md.
    """
    return requested or (encoding in VAL_ENCODINGS and h2d_mode == "column")


def plan_val_transfers(tile_nnz: list[int], W: int, H: int, h2d_mode: str) -> ValTransferPlan:
    """Plan the val-only stream for ``--h2d-mode`` (``column`` or ``full``).

    A rectangle memcpy sends the same count to every PE it covers, so:

    ``column``  one 1-wide call per column, of that column's longest tile.
    ``full``    one W-wide call, of the longest tile anywhere.

    No sentinel word is added (contrast build_stream_buffer's ``+ 1``).
    """
    nnz = np.asarray(tile_nnz, dtype=np.int64).reshape(H, W)
    col_words = nnz.max(axis=0) if nnz.size else np.zeros(W, dtype=np.int64)
    if h2d_mode == "column":
        rects = [(px, 1, int(n)) for px, n in enumerate(col_words) if n > 0]
    elif h2d_mode == "full":
        common = int(col_words.max()) if col_words.size else 0
        rects = [(0, W, common)] if common > 0 else []
    else:
        raise ValueError(f"unknown h2d_mode {h2d_mode!r}")
    return ValTransferPlan(rects, W, H)


def build_val_stream_buffer(
    data: np.ndarray,
    order: np.ndarray,
    tile_nnz: list[int],
    W: int, H: int,
) -> np.ndarray:
    """``(W, H, max_nnz)`` uint32: PE ``(px, py)``'s float32 values, bit-cast.

    Values are ``data[order]`` sliced per PE, i.e. exactly the CSR order of
    that PE's ``col_indices`` / ``row_ptrs``.  Trailing slots stay 0 -- they are
    the padding a rectangle memcpy delivers and the kernel never reads them
    (they land past ``row_ptrs[KY]``).  Vectorised: no per-PE Python loop, so
    it stays cheap at 10^8 NNZ.
    """
    num_pes = W * H
    counts = np.asarray(tile_nnz, dtype=np.int64)
    max_nnz = int(counts.max()) if counts.size else 0
    buf = np.zeros((W, H, max_nnz), dtype=np.uint32)
    if len(data) == 0 or max_nnz == 0:
        return buf

    starts = np.zeros(num_pes + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    pe_ids = np.repeat(np.arange(num_pes, dtype=np.int64), counts)
    local_pos = np.arange(len(pe_ids), dtype=np.int64) - starts[pe_ids]
    buf[pe_ids % W, pe_ids // W, local_pos] = data[order].astype(np.float32).view(np.uint32)
    return buf


def build_resident_vals(
    data: np.ndarray,
    order: np.ndarray,
    tile_nnz: list[int],
    W: int, H: int,
) -> np.ndarray:
    """``(W*H, max_nnz)`` float32 indexed ``pe = py*W + px``: each PE's values in CSR order.

    The same values as build_val_stream_buffer (``data[order]`` sliced per PE, so slot k
    matches ``col_indices[k]``), laid out the way the setup copy wants them -- PE-major
    like ``extract_tile_structure``'s col_indices, so setup_strip_calls flattens it with
    no transpose -- and as float32 rather than a bit-cast uint32.  Slots past a PE's
    count stay 0; pe_resident.csl never reads past ``row_ptrs[KY]``.
    """
    num_pes = W * H
    counts = np.asarray(tile_nnz, dtype=np.int64)
    max_nnz = int(counts.max()) if counts.size else 0
    vals = np.zeros((num_pes, max(max_nnz, 1)), dtype=np.float32)
    if len(data) == 0 or max_nnz == 0:
        return vals

    starts = np.zeros(num_pes + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    pe_ids = np.repeat(np.arange(num_pes, dtype=np.int64), counts)
    local_pos = np.arange(len(pe_ids), dtype=np.int64) - starts[pe_ids]
    vals[pe_ids, local_pos] = data[order].astype(np.float32)
    return vals


def build_row_ids(row_ptrs: np.ndarray, tile_nnz: list[int]) -> np.ndarray:
    """``(W*H, max_nnz)`` int32: the local row of every NNZ, in CSR order (the resident COO scatter).

    ``row_ptrs`` is extract_tile_structure's ``(W*H, KY + 1)``; slot k of PE ``pe`` is the row r
    with ``row_ptrs[pe, r] <= k < row_ptrs[pe, r + 1]``, so it pairs with ``col_indices[pe, k]``
    and ``vals[pe, k]``.  Slots past a PE's count are 0 and never read (the kernel loops to
    ``row_ptrs[KY]``).  Chunked over PEs: the per-row count matrix is (chunk, KY).
    """
    rp = np.asarray(row_ptrs)
    num_pes, ky1 = rp.shape
    ky = ky1 - 1
    counts = np.asarray(tile_nnz, dtype=np.int64)
    max_nnz = int(counts.max()) if counts.size else 0
    out = np.zeros((num_pes, max(max_nnz, 1)), dtype=np.int32)
    rows = np.arange(ky, dtype=np.int32)
    CHUNK = 4096
    for c0 in range(0, num_pes, CHUNK):
        c1 = min(c0 + CHUNK, num_pes)
        per_row = np.diff(rp[c0:c1].astype(np.int64), axis=1)          # (chunk, KY) NNZ per row
        total = int(per_row.sum())
        if total == 0:
            continue
        pe_of = np.repeat(np.arange(c1 - c0), per_row.sum(axis=1))
        start = np.zeros(c1 - c0 + 1, dtype=np.int64)
        np.cumsum(per_row.sum(axis=1), out=start[1:])
        pos = np.arange(total, dtype=np.int64) - start[pe_of]
        out[c0 + pe_of, pos] = np.repeat(np.tile(rows, c1 - c0), per_row.ravel())
    return out


# ---------------------------------------------------------------------------
# Parallel CSR scatter (ADR-004 — row-parallel, lock-free sub-buckets)
# ---------------------------------------------------------------------------


def _csr_scatter_one(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    row_start: int, row_end: int,
    W: int, H: int, KX: int, KY: int,
    num_pes: int,
) -> list[list[tuple[int, int, float]]]:
    """Scatter a CSR row range into per-PE thread-local sub-buckets.

    Returns sub_buckets[pe] = [(local_row, local_col, val), ...] in row-major order.
    """
    buckets: list[list[tuple[int, int, float]]] = [[] for _ in range(num_pes)]

    for g_row in range(row_start, row_end):
        s = int(indptr[g_row])
        e = int(indptr[g_row + 1])
        if s >= e:
            continue
        py = g_row // KY
        lr = g_row % KY
        for p in range(s, e):
            g_col = int(indices[p])
            px = g_col // KX
            lc = g_col % KX
            pe = py * W + px
            buckets[pe].append((lr, lc, float(data[p])))

    return buckets


def parallel_csr_scatter(
    csr_indptr: np.ndarray,
    csr_indices: np.ndarray,
    csr_data: np.ndarray,
    rows: int,
    W: int, H: int, KX: int, KY: int,
    num_threads: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[int], RowHeaderBounds]:
    """Parallel row scatter of CSR into PE-tile col_indices/row_ptrs.

    Each thread owns a contiguous row range and fills thread-local
    sub-buckets keyed by PE.  Sub-buckets are merged per PE after
    all threads finish — no lock contention, no global sort.

    Parameters
    ----------
    csr_indptr : int32, length rows+1 — CSR row pointers.
    csr_indices : int32 — NNZ column indices in CSR order.
    csr_data : float32 — NNZ values in CSR order (unused for structure,
                         but kept for API symmetry).
    rows : matrix row count.
    W, H : PE grid dimensions.
    KX, KY : tile dimensions.
    num_threads : max worker threads.

    Returns
    -------
    all_col_indices : (num_pes, max_tile_nnz) int32
    all_row_ptrs : (num_pes, KY + 1) int32
    tile_nnz : list[int]
    row_bounds : RowHeaderBounds — per-PE (local_row, start, count).
    """
    num_pes = W * H
    total_nnz = int(csr_indptr[rows])

    if total_nnz == 0:
        ci = np.zeros((num_pes, 1), dtype=np.int32)
        rp = np.zeros((num_pes, KY + 1), dtype=np.int32)
        return ci, rp, [0] * num_pes, [[] for _ in range(num_pes)]

    n_threads = min(num_threads, rows, 128)
    row_splits = np.linspace(0, rows, n_threads + 1, dtype=np.int64)

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = []
        for i in range(n_threads):
            rs, re = int(row_splits[i]), int(row_splits[i + 1])
            futures.append(ex.submit(
                _csr_scatter_one,
                csr_indptr, csr_indices, csr_data,
                rs, re, W, H, KX, KY, num_pes,
            ))
        buckets_list = [f.result() for f in futures]

    tile_nnz_arr = np.zeros(num_pes, dtype=np.int64)
    for buckets in buckets_list:
        for pe in range(num_pes):
            tile_nnz_arr[pe] += len(buckets[pe])

    max_nnz = int(tile_nnz_arr.max())
    all_col_indices = np.zeros((num_pes, max(max_nnz, 1)), dtype=np.int32)
    all_row_ptrs    = np.zeros((num_pes, KY + 1), dtype=np.int32)
    row_bounds: RowHeaderBounds = [[] for _ in range(num_pes)]

    all_row_ids = np.zeros((num_pes, max(max_nnz, 1)), dtype=np.int32)
    pe_cursor = np.zeros(num_pes, dtype=np.int64)
    for buckets in buckets_list:
        for pe in range(num_pes):
            entries = buckets[pe]
            n = len(entries)
            if n == 0:
                continue
            cur = int(pe_cursor[pe])
            for j, (lr, lc, _) in enumerate(entries):
                all_row_ids[pe, cur + j] = lr
                all_col_indices[pe, cur + j] = lc
            pe_cursor[pe] += n

    for pe in range(num_pes):
        total = int(tile_nnz_arr[pe])
        if total == 0:
            continue
        rows_pe = all_row_ids[pe, :total]
        cols_pe = all_col_indices[pe, :total]

        boundaries = np.where(np.diff(rows_pe) != 0)[0] + 1
        r_starts = np.concatenate([[0], boundaries])
        r_ends   = np.concatenate([boundaries, [total]])

        cum = 0
        for rs, re in zip(r_starts, r_ends):
            lr = int(rows_pe[rs])
            cnt = re - rs
            while cum < lr:
                cum += 1
            row_bounds[pe].append((lr, int(rs), cnt))
            cum += 1

        row_counts = np.bincount(rows_pe.astype(np.int64), minlength=KY)
        row_ptrs_cum = np.cumsum(row_counts, dtype=np.int32)
        all_row_ptrs[pe, 0] = 0
        all_row_ptrs[pe, 1:KY + 1] = row_ptrs_cum

    return all_col_indices, all_row_ptrs, tile_nnz_arr.tolist(), row_bounds
