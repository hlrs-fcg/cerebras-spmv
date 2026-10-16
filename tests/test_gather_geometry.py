"""Pure-numpy tests for cluster/worker_runtime.py:gather_y (no cerebras import).

gather_y reassembles the row-partitioned result vector from the column-0 D2H
buffer.  PE row ``py`` owns matrix rows ``[py*KY, py*KY+KY)`` with
``KY = ceil(rows / H)``, so ``H*KY`` overshoots ``rows`` by up to ``H-1``.

When that overshoot (``excess = H*KY - rows``) is ``>= KY`` the last PE row (or
rows) sits *entirely* past the matrix: ``rs = py*KY >= rows``.  The pre-fix
gather computed ``n = re - rs`` with ``re = min((py+1)*KY, rows) = rows``, giving
a negative ``n``; ``y_grid[py, :n]`` then had shape ``KY + n`` while the
destination slice ``y[rs:re]`` was empty:

    ValueError: could not broadcast input array from shape (73,) into shape (0,)

That is the full-wafer crash at worker_runtime.py: rows=cols=414000, grid
742x1155, KY=359, excess=645 (> KY).  Every smaller run passed because its
excess (16, 176, 83) was below KY, so the last row only ever straddled the edge.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("cluster", "bench"):
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from worker_runtime import gather_y  # noqa: E402  (path shim above)


# (H, rows); KY is derived as ceil(rows / H).  excess = H*KY - rows >= KY here,
# i.e. at least one PE row lies wholly beyond the matrix.
EXCESS_GE_KY = [
    (1155, 414000),   # KY=359, excess=645   -> 1 empty PE row  (the reported crash)
    (1155, 413590),   # KY=359, excess=1055  -> 2 empty PE rows
    (1155, 414286),   # KY=359, excess=359   -> 1 empty PE row  (excess == KY, boundary)
    (10,   13),       # KY=2,   excess=7     -> 3 empty PE rows
    (5,    6),        # KY=2,   excess=4     -> 2 empty PE rows
    (7,    8),        # KY=2,   excess=6     -> 3 empty PE rows
    (100,  102),      # KY=2,   excess=98    -> 49 empty PE rows
]

# Control geometries the fix must not regress: exact fit, and a straddling last
# row whose excess stays below KY (the pre-fix code already handled these).
NORMAL = [
    (1000, 999000),   # KY=999,  excess=0    -> exact fit
    (1000, 999001),   # KY=1000, excess=999  -> partial last row only
    (1,    500),       # single PE row
    (8,    100),       # KY=13,   excess=4
    (742,  414000),    # the column geometry from the crash config (already guarded)
]


def _scatter(y_true: np.ndarray, H: int, KY: int, rows: int, fill: np.float32) -> np.ndarray:
    """Lay y_true into an (H, KY) PE grid the way the wafer produces it.

    PE row ``py`` holds ``KY`` entries; only those mapping to a matrix row
    ``< rows`` are real.  Cells past the edge get ``fill`` — garbage the wafer
    still computed (uncleared SRAM / EW-reduced junk) that must not survive the
    gather.
    """
    g = np.full((H, KY), fill, dtype=np.float32)
    for py in range(H):
        for k in range(KY):
            r = py * KY + k
            if r < rows:
                g[py, k] = y_true[r]
    return g


def _fully_empty_pe_rows(H: int, KY: int, rows: int) -> int:
    return sum(1 for py in range(H) if py * KY >= rows)


@pytest.mark.parametrize("H, rows", EXCESS_GE_KY + NORMAL)
def test_gather_round_trips(H, rows):
    KY = math.ceil(rows / H)
    rng = np.random.default_rng(rows * 1000 + H)
    y_true = rng.standard_normal(rows).astype(np.float32)

    y_flat = _scatter(y_true, H, KY, rows, fill=np.float32(1e30)).reshape(-1).copy()
    out = gather_y(y_flat, H, KY, rows)

    assert out.shape == (rows,)
    assert np.array_equal(out, y_true), "gather must be an exact reassembly"
    assert out.max() < 1e29, "beyond-the-edge garbage leaked into the result"


@pytest.mark.parametrize("H, rows", EXCESS_GE_KY)
def test_excess_cases_really_have_a_fully_empty_pe_row(H, rows):
    KY = math.ceil(rows / H)
    assert H * KY - rows >= KY, "this case does not exercise the excess>=KY class"
    assert _fully_empty_pe_rows(H, KY, rows) >= 1


def test_reported_full_wafer_crash_geometry():
    """rows=cols=414000, grid 742x1155 — reproduce the exact reported shape."""
    H, rows = 1155, 414000
    KY = math.ceil(rows / H)
    assert KY == 359
    assert H * KY - rows == 645 and 645 > KY            # excess exceeds KY
    last_rs = (H - 1) * KY
    assert last_rs == 414286 and last_rs - rows == 286  # the negative-n PE row

    y_true = np.arange(rows, dtype=np.float32)
    y_flat = _scatter(y_true, H, KY, rows, fill=np.float32(np.nan)).reshape(-1).copy()
    out = gather_y(y_flat, H, KY, rows)

    assert np.array_equal(out, y_true)
    assert not np.isnan(out).any()


def test_pre_fix_inline_logic_raised_on_this_geometry():
    """Guard-rail: the old inline loop really did raise on an excess>=KY grid.

    Needs ``KY - (rs - rows) >= 2`` on the fully-empty PE row so the shape
    mismatch is real: a size-1 RHS would just broadcast into the empty slice
    (which is why H=10/rows=13, KY=2 did *not* crash, but the full wafer did).
    H=5, rows=11, KY=3: py=4 has rs=12, n=-1, y_grid[4, :-1] is shape (2,).
    """
    H, rows = 5, 11
    KY = math.ceil(rows / H)
    assert KY == 3 and H * KY - rows == 4 >= KY
    y_grid = np.zeros(H * KY, dtype=np.float32).reshape(H, KY)

    with pytest.raises(ValueError):
        y = np.zeros(rows, dtype=np.float32)
        for py in range(H):
            rs = py * KY
            re = min((py + 1) * KY, rows)
            n = re - rs                      # negative once rs >= rows
            y[rs:re] = y_grid[py, :n]


def test_gather_never_reads_outside_y_grid():
    """n is clamped so a straddling row cannot index y_grid with a negative stop."""
    H, rows = 4, 10          # KY=3, H*KY=12, excess=2 (< KY): last row straddles
    KY = math.ceil(rows / H)
    y_true = np.arange(rows, dtype=np.float32)
    y_flat = _scatter(y_true, H, KY, rows, fill=np.float32(-1.0)).reshape(-1).copy()

    out = gather_y(y_flat, H, KY, rows)
    assert np.array_equal(out, y_true)
