"""Verbatim copy of the host tiling / encoding code as of 34d1fdd (before perf/host-tiling).

The oracle the fast paths are compared against, bit for bit.  Do not "improve" it.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

MAX_ROW_NNZ = 32767
MAX_LOCAL_ROW = 65535
RowHeaderBounds = list[list[tuple[int, int, int]]]


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
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray, RowHeaderBounds]:
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
    row_bounds : RowHeaderBounds
        Per-PE list of (local_row, start_offset, count) for fast encoding.
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
        row_bounds = [[] for _ in range(num_pes)]
        return all_col_indices, all_row_ptrs, tile_nnz, order, row_bounds

    pe_id   = ((row_idx // KY) * W + col_idx // KX).astype(np.int32)
    loc_row = (row_idx % KY).astype(np.int32)
    loc_col = (col_idx % KX).astype(np.int32)

    order = np.lexsort((loc_col, loc_row, pe_id))
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

    row_bounds = _compute_row_bounds(
        pe_id_sorted, loc_row_sorted, tile_starts, tile_nnz, num_pes, KY
    )

    return all_col_indices, all_row_ptrs, tile_nnz.tolist(), order, row_bounds


def _compute_row_bounds(
    pe_id_sorted: np.ndarray,
    loc_row_sorted: np.ndarray,
    tile_starts: np.ndarray,
    tile_nnz: np.ndarray,
    num_pes: int,
    KY: int,
) -> RowHeaderBounds:
    """Pre-compute per-PE row boundaries for fast stream encoding."""
    row_bounds: RowHeaderBounds = [[] for _ in range(num_pes)]
    for pe in range(num_pes):
        s, e = int(tile_starts[pe]), int(tile_starts[pe] + tile_nnz[pe])
        if e - s == 0:
            continue
        pe_rows = loc_row_sorted[s:e]
        boundaries = np.where(np.diff(pe_rows) != 0)[0] + 1
        row_starts = np.concatenate([[0], boundaries])
        row_ends   = np.concatenate([boundaries, [len(pe_rows)]])
        for rs, re in zip(row_starts, row_ends):
            local_row = int(pe_rows[rs])
            count = int(re - rs)
            row_bounds[pe].append((local_row, int(rs), count))
    return row_bounds




def encode_streams_parallel(
    data: np.ndarray,
    order: np.ndarray,
    tile_nnz: list[int],
    row_bounds: RowHeaderBounds,
    num_threads: int = 64,
) -> tuple[np.ndarray, list[int]]:
    """Encode all PE streams in parallel using column-parallel Phase 2.

    Each thread encodes a disjoint subset of PEs.  The per-PE streams
    are concatenated into a single flat uint32 buffer for a single
    rectangle memcpy_h2d call.

    Parameters
    ----------
    data : float32 array of NNZ values (original unsorted order).
    order : int64 sort index from extract_tile_structure().
    tile_nnz : NNZ count per PE.
    row_bounds : per-PE row boundaries.
    num_threads : max worker threads.

    Returns
    -------
    flat_buffer : uint32 array — all PE streams concatenated.
    stream_lengths : list[int] — per-PE word count into flat_buffer.
    """
    if len(data) == 0:
        return np.array([], dtype=np.uint32), [0] * len(tile_nnz)

    data_sorted = data[order].view(np.uint32)

    tile_starts = np.empty(len(tile_nnz) + 1, dtype=np.int64)
    tile_starts[0] = 0
    np.cumsum(tile_nnz, out=tile_starts[1:])

    num_pes = len(tile_nnz)
    n_threads = min(num_threads, num_pes, 128)
    pe_splits = np.linspace(0, num_pes, n_threads + 1, dtype=np.int64)

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = []
        for i in range(n_threads):
            ps, pe = int(pe_splits[i]), int(pe_splits[i + 1])
            futures.append(ex.submit(
                _encode_pe_range,
                data_sorted, tile_starts, tile_nnz, row_bounds,
                ps, pe,
            ))
        results = [f.result() for f in futures]

    lengths: list[int] = []
    parts: list[np.ndarray] = []
    for part_array, part_lengths in results:
        parts.append(part_array)
        lengths.extend(part_lengths)

    return np.concatenate(parts) if parts else np.array([], dtype=np.uint32), lengths




def build_stream_buffer(
    data: np.ndarray,
    order: np.ndarray,
    tile_nnz: list[int],
    row_bounds: RowHeaderBounds,
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
    flat, lengths = encode_streams_parallel(data, order, tile_nnz, row_bounds, num_threads=num_threads)

    # Per-column transfer length = longest stream in the column + 1 sentinel.
    col_len = np.zeros(W, dtype=np.int64)
    for pe in range(num_pes):
        px = pe % W
        if lengths[pe] > col_len[px]:
            col_len[px] = lengths[pe]
    col_words = (col_len + 1).tolist()
    max_words = int(max(col_words))

    buf = np.zeros((W, H, max_words), dtype=np.uint32)
    pos = 0
    for pe in range(num_pes):
        px = pe % W
        py = pe // W
        n  = lengths[pe]
        if n > 0:
            buf[px, py, :n] = flat[pos:pos + n]
            pos += n
        buf[px, py, n] = 0  # explicit end-of-stream sentinel (n < col_words[px])
    return buf, col_words




def _encode_pe_range(
    data_sorted: np.ndarray,
    tile_starts: np.ndarray,
    tile_nnz: list[int],
    row_bounds: RowHeaderBounds,
    pe_start: int, pe_end: int,
) -> tuple[np.ndarray, list[int]]:
    """Encode PE streams for PE range [pe_start, pe_end)."""
    sub_parts: list[np.ndarray] = []
    sub_lengths: list[int] = []
    for pe in range(pe_start, pe_end):
        bounds = row_bounds[pe]
        if not bounds:
            sub_lengths.append(0)
            continue
        base = int(tile_starts[pe])
        pe_words: list[np.ndarray] = []
        for local_row, rs, count in bounds:
            header = _pack_header(local_row, count)
            pe_words.append(np.array([header], dtype=np.uint32))
            pe_words.append(data_sorted[base + rs:base + rs + count])
        sub_parts.append(np.concatenate(pe_words))
        sub_lengths.append(len(sub_parts[-1]))
    return (np.concatenate(sub_parts) if sub_parts
            else np.array([], dtype=np.uint32)), sub_lengths




def build_val_stream_buffer(
    data: np.ndarray,
    order: np.ndarray,
    tile_nnz: list[int],
    W: int, H: int,
) -> np.ndarray:
    """``(W, H, max_nnz)`` uint32: PE ``(px, py)``'s float32 values, bit-cast.

    Values are ``data[order]`` sliced per PE, i.e. exactly the CSR order of
    that PE's ``col_indices`` / ``row_ptrs``.  Trailing slots stay 0 -- they are
    the padding a rectangle memcpy delivers and pe_val.csl never reads them
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


