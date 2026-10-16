"""Host-side layout of what the worker moves: y reassembly, memcpy flattening, setup strips.
Pure numpy; staged flat next to worker_runtime.py."""

import numpy as np


def gather_y(y_flat: np.ndarray, H: int, KY: int, rows: int) -> np.ndarray:
    """Reassemble the row-partitioned result vector from the column-0 D2H buffer.

    ``y_flat`` is ``H * KY`` floats: PE row ``py`` holds ``KY`` entries starting
    at matrix row ``py * KY``.  Only the first ``rows - py*KY`` of those are real
    when a tile straddles the bottom edge of the matrix.

    ``KY = ceil(rows / H)``, so ``H * KY`` can exceed ``rows`` by anything up to
    ``H - 1``.  When that excess is ``>= KY`` the final PE row (or rows) sits
    entirely past the matrix: ``rs = py*KY >= rows``.  Those rows are skipped
    outright, mirroring the ``if cs < args.cols`` guard on the x_grid build.
    ``n`` is additionally clamped to ``>= 0`` so a partly-empty straddling row
    can never index ``y_grid`` with a negative stop.
    """
    y = np.zeros(rows, dtype=np.float32)
    y_grid = y_flat.reshape(H, KY)
    for py in range(H):
        rs = py * KY
        if rs >= rows:
            break  # rs increases with py: this row and every later one is empty
        re = min((py + 1) * KY, rows)
        n = max(re - rs, 0)
        y[rs:re] = y_grid[py, :n]
    return y


def rect_nonblock_flags(n_rects: int, serialize: bool) -> list[bool]:
    """memcpy_h2d nonblock= for each streaming rect of one pass.

    Default: all True.  Serialised (coo_tiling.column_rects_serialized): False
    for every rect but the last, so each blocks until delivered before the next
    is issued, while the last stays True and wait_ew is still queued behind it.
    """
    return [not (serialize and i < n_rects - 1) for i in range(n_rects)]


def _flatten_whl(block: np.ndarray, order: str) -> np.ndarray:
    """Flatten a [w][h][l]-indexed block into the 1-D array memcpy expects.

    `memcpy_h2d` takes a rectangle (w, h) with `l` elements per PE plus a
    MemcpyOrder, and the order decides which axis varies fastest in the host
    array.  Both layouts are derived from Cerebras' own example rather than from
    the guide's prose, because the prose is wrong:

      ROW_MAJOR  l fastest, then w, then h  ->  host shape (h, w, l)
      COL_MAJOR  h fastest, then w, then l  ->  host shape (l, w, h)

    ROW_MAJOR is confirmed by csl-examples/benchmarks/bandwidth-test, whose
    ROW_MAJOR timestamp readback reshapes as `(height, width, 3), order="C"`,
    and it matches what our transfers already did.  COL_MAJOR is confirmed by
    `hwl_2_oned_colmajor` (run.appliance.py:178) in the same example, which
    loops `for l / for w / for h` with **h innermost** -- not `w` as
    https://sdk.cerebras.ai/tensor-streaming claims.  Getting this backwards
    transfers a transposed matrix and shows up only as a bad relative error, so
    validate numerically after any change here.
    """
    if order == "col":
        return np.ascontiguousarray(block.transpose(2, 0, 1)).reshape(-1)
    return np.ascontiguousarray(block.transpose(1, 0, 2)).reshape(-1)


def _flatten_pe_major(arr: np.ndarray, W: int, H: int, order: str) -> np.ndarray:
    """Flatten a (W*H, l) array indexed `py * W + px` for a W x H memcpy.

    That indexing is already ROW_MAJOR's [h][w][l], so "row" returns a plain
    ravel (a view, no copy).  "col" goes through _flatten_whl like the streaming
    transfer does.
    """
    if order == "row":
        return arr.ravel()
    return _flatten_whl(arr.reshape(H, W, -1).transpose(1, 0, 2), order)


def setup_strip_bounds(W: int, k: int) -> list[tuple[int, int]]:
    """(x0, width) of ``k`` vertical strips covering W columns, as even as possible (widths differ
    by at most 1, the wider ones first).  ``k`` is clamped to [1, W], so k need not divide W."""
    k = max(1, min(int(k), W))
    base, extra = divmod(W, k)
    bounds, x = [], 0
    for i in range(k):
        w = base + (1 if i < extra else 0)
        bounds.append((x, w))
        x += w
    return bounds


def _flatten_strip(arr: np.ndarray, W: int, H: int, x0: int, w: int, order: str) -> np.ndarray:
    """Flat host array for the memcpy_h2d of columns x0 .. x0+w-1 of a (W*H, l) array indexed
    `py * W + px`: _flatten_pe_major's layout applied to the sub-rectangle (w, H)."""
    block = arr.reshape(H, W, -1)[:, x0:x0 + w, :]          # [h][w][l]
    if order == "row":
        return np.ascontiguousarray(block).reshape(-1)
    return _flatten_whl(block.transpose(1, 0, 2), order)    # _flatten_whl takes [w][h][l]


def setup_strip_calls(arr: np.ndarray, W: int, H: int, order: str, k: int, last_array: bool):
    """The memcpy_h2d calls (x0, width, flat, nonblock) that load one setup array in ``k`` strips.

    Issued serially: every strip but the last has nonblock=False.  The last strip stays
    nonblocking, except for the final array of the whole setup, which blocks as it always did
    (the completion barrier follows it).  k = 1 is exactly the old single full-grid call.
    """
    bounds = setup_strip_bounds(W, k)
    flags = rect_nonblock_flags(len(bounds), True)
    if last_array:
        flags[-1] = False
    return [(x0, w, _flatten_strip(arr, W, H, x0, w, order), nb)
            for (x0, w), nb in zip(bounds, flags)]
