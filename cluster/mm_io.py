"""MatrixMarket reading and per-PE sizing without scipy or fast_matrix_market (numpy only).

The worker's Python has numpy and little else, and ``--mtx-file`` used to need the optional
``fast_matrix_market`` there. This reader covers what SpMV needs and says clearly what it refuses:

* ``coordinate`` with ``real``, ``integer`` or ``pattern`` fields (pattern entries become 1.0);
* ``general``, ``symmetric`` and ``skew-symmetric`` (``hermitian`` of a real matrix is symmetric):
  the stored triangle is expanded, the diagonal once;
* duplicates and explicit zeros are kept as separate entries (SpMV sums them; they cost PE slots);
* refused with a ValueError that names the file's problem: ``array`` (dense) format, ``complex`` fields,
  indices outside the declared shape, an entry count that disagrees with the header.

``tile_nnz_counts`` / ``fit_report`` size a matrix on a grid without tiling it: the same block mapping as
cluster/coo_tiling.py (PE (px, py) holds rows py*KY.. and columns px*KX..), so the busiest PE's NNZ is known
before anything is compiled. Staged flat next to worker_runtime.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PE_SRAM_WORDS = 38 * 1024 // 4        # config.PE_SRAM_BUDGET_BYTES in 32-bit words
SYNC_TIMING_WORDS = 5 + 3 + 1 + 2     # config.SYNC_TIMING_EXTRA_WORDS
MAX_LOCAL_ROWS = 65535                # worker_host: stream headers carry the local row in 16 bits


@dataclass
class MatrixMarket:
    rows: int
    cols: int
    row: np.ndarray          # int32, 0-based, symmetry expanded
    col: np.ndarray          # int32
    data: np.ndarray         # float32
    symmetry: str            # as stored: general | symmetric | skew-symmetric
    field: str               # real | integer | pattern
    stored_entries: int      # entries in the file, before expansion

    @property
    def nnz(self) -> int:
        return len(self.data)


def read_header(path) -> tuple[str, str, str, int, int, int]:
    """(format, field, symmetry, rows, cols, stored entries) from the first lines only."""
    with open(path, "r", encoding="ascii", errors="replace") as f:
        banner = f.readline().split()
        if len(banner) < 5 or banner[0].lower() != "%%matrixmarket" or banner[1].lower() != "matrix":
            raise ValueError(f"{path}: not a MatrixMarket matrix file (first line {' '.join(banner)!r})")
        fmt, field, symmetry = (b.lower() for b in banner[2:5])
        for line in f:
            if line.strip() and not line.startswith("%"):
                dims = line.split()
                break
        else:
            raise ValueError(f"{path}: no size line")
    if fmt == "array":
        raise ValueError(f"{path}: dense 'array' format; SpMV needs a sparse 'coordinate' matrix")
    if fmt != "coordinate":
        raise ValueError(f"{path}: unknown format {fmt!r}")
    if field == "complex":
        raise ValueError(f"{path}: complex values; the kernels are float32")
    if field not in ("real", "integer", "pattern"):
        raise ValueError(f"{path}: unknown field {field!r}")
    if symmetry == "hermitian":
        symmetry = "symmetric"            # real entries only (complex was refused)
    if symmetry not in ("general", "symmetric", "skew-symmetric"):
        raise ValueError(f"{path}: unknown symmetry {symmetry!r}")
    if len(dims) != 3:
        raise ValueError(f"{path}: size line {' '.join(dims)!r} is not 'rows cols entries'")
    return fmt, field, symmetry, int(dims[0]), int(dims[1]), int(dims[2])


def read_matrix_market(path) -> MatrixMarket:
    _, field, symmetry, rows, cols, stored = read_header(path)
    with open(path, "rb") as f:
        text = f.read()
    # skip the banner, the comments and the size line
    pos = text.index(b"\n") + 1                       # past the banner
    while True:
        end = text.find(b"\n", pos)
        end = len(text) if end < 0 else end
        line = text[pos:end].strip()
        pos = end + 1
        if line and not line.startswith(b"%"):
            break                                      # the size line
    per = 2 if field == "pattern" else 3
    body = np.fromstring(text[pos:], dtype=np.float64, sep=" ") if stored else np.empty(0)
    if body.size != stored * per:
        raise ValueError(f"{path}: header says {stored} entries but the body holds "
                         f"{body.size / per:g} ({per} numbers per entry)")
    body = body.reshape(stored, per)
    row = body[:, 0].astype(np.int64) - 1
    col = body[:, 1].astype(np.int64) - 1
    data = np.ones(stored, dtype=np.float64) if field == "pattern" else body[:, 2]
    if stored and (row.min() < 0 or col.min() < 0 or row.max() >= rows or col.max() >= cols):
        raise ValueError(f"{path}: an entry lies outside the declared {rows}x{cols} shape")
    if symmetry != "general":
        if rows != cols:
            raise ValueError(f"{path}: {symmetry} but {rows}x{cols} is not square")
        off = row != col
        sign = -1.0 if symmetry == "skew-symmetric" else 1.0
        row, col, data = (np.concatenate([row, col[off]]), np.concatenate([col, row[off]]),
                          np.concatenate([data, sign * data[off]]))
    return MatrixMarket(rows, cols, row.astype(np.int32), col.astype(np.int32),
                        data.astype(np.float32), symmetry, field, stored)


# --------------------------------------------------------------------------
# sizing on a grid, without tiling
# --------------------------------------------------------------------------

def tile_nnz_counts(row: np.ndarray, col: np.ndarray, rows: int, cols: int, W: int, H: int,
                    mapping: str = "block") -> np.ndarray:
    """(H, W) NNZ per PE.

    ``block``: the tiling of cluster/coo_tiling.py (PE (px, py) holds rows py*KY.. and columns px*KX..).
    ``cyclic``: NOT implemented by the tiler, only sized here: row i on PE row i % H, column j on PE column
    j % W (KX, KY unchanged). A banded or diagonal-dominant matrix piles its NNZ onto the PEs near the
    diagonal under ``block`` and spreads evenly under ``cyclic``; the kernels would not change (they only see
    local indices), the host tiling, x_grid and gather_y would."""
    row, col = np.asarray(row, dtype=np.int64), np.asarray(col, dtype=np.int64)
    if mapping == "block":
        kx, ky = math.ceil(cols / W), math.ceil(rows / H)
        pe = (row // ky) * W + col // kx
    elif mapping == "cyclic":
        pe = (row % H) * W + col % W
    else:
        raise ValueError(f"mapping must be 'block' or 'cyclic', got {mapping!r}")
    return np.bincount(pe, minlength=W * H).reshape(H, W)


def kernel_words(kx: int, ky: int, total_nnz: int, sync_timing: bool = True,
                 encoding: str = "resident") -> int:
    """32-bit words of per-PE data a kernel holds: SpMVConfig's guard. x_vec[KX] + col_indices[N] + row_ptrs[KY+1] +
    y[KY] + ew_buf[KY], plus vals[N] + val_meta for valtask and vals[N] + row_ids[N] for resident."""
    words = kx + total_nnz + (ky + 1) + ky + ky
    if encoding == "resident":
        words += 2 * total_nnz
    elif encoding == "valtask":
        words += total_nnz + 1
    elif encoding != "header":
        raise ValueError(f"unknown encoding {encoding!r}")
    return words + (SYNC_TIMING_WORDS if sync_timing else 0)


def resident_words(kx: int, ky: int, total_nnz: int, sync_timing: bool = True) -> int:
    return kernel_words(kx, ky, total_nnz, sync_timing, "resident")


@dataclass
class FitReport:
    W: int
    H: int
    kx: int
    ky: int
    nnz: int
    mean: float
    max: int
    words: int                 # resident words at TOTAL_NNZ = max
    fits: bool
    reason: str                # why not, or ""
    max_fit_nnz: int           # largest NNZ_max this grid's KX/KY leaves room for (< 0: not even empty)

    def verdict(self) -> str:
        return "fits" if self.fits else f"NO: {self.reason}"


def fit_report(counts: np.ndarray, rows: int, cols: int, sync_timing: bool = True,
               encoding: str = "resident") -> FitReport:
    H, W = counts.shape
    kx, ky = math.ceil(cols / W), math.ceil(rows / H)
    mx = int(counts.max()) if counts.size else 0
    words = kernel_words(kx, ky, mx, sync_timing, encoding)
    fixed = kernel_words(kx, ky, 0, sync_timing, encoding)
    per_nnz = {"header": 1, "valtask": 2, "resident": 3}[encoding]
    max_fit = (PE_SRAM_WORDS - fixed) // per_nnz if fixed <= PE_SRAM_WORDS else -1
    reason = ""
    if ky > MAX_LOCAL_ROWS:
        reason = f"KY={ky} > {MAX_LOCAL_ROWS} (16-bit local row index; use a taller grid)"
    elif words > PE_SRAM_WORDS:
        reason = (f"{words * 4 / 1024:.1f} KB per PE > {PE_SRAM_WORDS * 4 // 1024} KB "
                  f"(KX={kx}, KY={ky}, NNZ_max={mx}; this grid leaves room for {max(max_fit, 0)} NNZ)")
    return FitReport(W, H, kx, ky, int(counts.sum()), float(counts.mean()) if counts.size else 0.0,
                     mx, words, not reason, reason, max_fit)


def smallest_fit(row, col, rows: int, cols: int, grids, sync_timing: bool = True,
                 encoding: str = "resident", mapping: str = "block"):
    """The first (W, H) of ``grids`` (smallest first) on which the matrix fits, with its FitReport; or None."""
    for W, H in grids:
        rep = fit_report(tile_nnz_counts(row, col, rows, cols, W, H, mapping), rows, cols, sync_timing,
                         encoding)
        if rep.fits:
            return (W, H), rep
    return None
