"""Fused COO -> PE-tile kernels for the val/valtask host path, with three backends.

Output is bit-identical to coo_tiling.extract_tile_structure + build_val_stream_buffer
(tests/test_host_tiling_equiv.py), without the global 3-key lexsort or any of its temporaries:

  1. histogram of pe_id over contiguous slices of the input (one slice per thread)
  2. prefix sums -> every slice's write offset into every PE's segment (input order is kept inside
     a segment, which is what makes duplicates stable)
  3. scatter (loc_row*KX + loc_col, value bits) into per-PE segments
  4. per PE (a ~200-element segment that lives in L1): count rows -> row_ptrs, stable counting sort
     by row, insertion sort by column inside each row (rows hold ~1 NNZ), then write col_indices
     and the val buffer straight into their final layouts.

The kernels below are plain serial functions over an index range.  Three ways to run them:

  ``aot``    compiled ahead of time by sdk/build_aot.py into one extension module
             (``spmv_tile_aot``).  No numba and no JIT on the worker; but numba's AOT wrappers hold
             the GIL, so this is single-threaded (~2x slower than threaded numba at 65K).
  ``numba``  ``@njit(nogil=True)`` per kernel, driven by a thread pool (the GIL is released).  Pays a
             compile on first use (~1 s here, ~5-12 s on a loaded worker; ``tile_jit_time``).
  ``numpy``  the lexsort-free radix path in coo_tiling (always available).

``select_backend`` picks one from the problem size; SPMV_TILE_BACKEND=auto|aot|numba|numpy and
SPMV_TILE_NUMBA_MIN_NNZ override it.  Importing this module never imports numba.
"""

import importlib.util
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

AOT_MODULE = "spmv_tile_aot"
NUMBA_MIN_NNZ_DEFAULT = 20_000_000
"""Below this many NNZ the numpy radix path beats a numba JIT: worker numbers (65K PEs, 13M NNZ)
were numpy 4.9 s vs numba 0.06 s + 12.4 s compile, i.e. break-even near 34M NNZ at that compile
cost, ~14M at the ~5 s a lean nogil compile should cost.  20M sits between."""

IMPORT_ERROR = ""
"""Why a requested/preferred backend could not be used (empty if none failed); recorded by the
worker as part of numba_status."""

_aot = None            # the imported extension module, once found
_aot_tried = False
_jit = None            # name -> compiled kernel, once built
_num_threads = os.cpu_count() or 1


# ---------------------------------------------------------------------------
# kernels (serial, over an index range).  Must stay valid numba-nopython code AND
# numba.pycc-exportable: scalars int64, arrays C-contiguous (KERNEL_SIGNATURES below).
# ---------------------------------------------------------------------------

def _hist_chunk(row_idx, col_idx, row_pe, col_pe, h, lo, hi):
    for i in range(lo, hi):
        h[row_pe[row_idx[i]] + col_pe[col_idx[i]]] += 1


def _counts_range(hist, counts, p0, p1):
    nchunk = hist.shape[0]
    for pe in range(p0, p1):
        s = 0
        for c in range(nchunk):
            s += hist[c, pe]
        counts[pe] = s


def _offsets_range(hist, starts, p0, p1):
    """hist[c, pe] -> absolute write offset of chunk c inside PE pe's segment (in place)."""
    nchunk = hist.shape[0]
    for pe in range(p0, p1):
        o = starts[pe]
        for c in range(nchunk):
            h = hist[c, pe]
            hist[c, pe] = o
            o += h


def _scatter_chunk(row_idx, col_idx, vbits, row_pe, col_pe, row_loc, col_loc, o, packed, lo, hi):
    for i in range(lo, hi):
        r = row_idx[i]
        cc = col_idx[i]
        pe = row_pe[r] + col_pe[cc]
        p = o[pe]
        o[pe] = p + 1
        packed[p] = (np.uint64(row_loc[r] + col_loc[cc]) << np.uint64(32)) | np.uint64(vbits[i])


def _build_range(packed, starts, counts, W, H, KX, KY, max_nnz, ci, rp, val_buf, p0, p1):
    """Steps 4 for PEs [p0, p1); returns how many (PE, row) pairs are non-empty."""
    sk = np.empty(max_nnz, np.uint32)
    sv = np.empty(max_nnz, np.uint32)
    cur = np.empty(KY, np.int64)
    ne = 0
    for pe in range(p0, p1):
        n = counts[pe]
        s = starts[pe]
        row = rp[pe]
        row[:] = 0
        if n == 0:
            continue
        for j in range(n):                                  # row histogram -> row_ptrs
            row[np.int64(packed[s + j] >> np.uint64(32)) // KX + 1] += 1
        for r in range(KY):
            row[r + 1] += row[r]
        for r in range(KY):
            cur[r] = row[r]
        for j in range(n):                                  # stable counting sort by row
            x = packed[s + j]
            k = np.uint32(x >> np.uint64(32))
            r = np.int64(k) // KX
            p = cur[r]
            cur[r] = p + 1
            sk[p] = k
            sv[p] = np.uint32(x & np.uint64(0xFFFFFFFF))
        px = pe % W
        py = pe // W
        for r in range(KY):
            lo = row[r]
            hi = row[r + 1]
            if hi == lo:
                continue
            ne += 1
            for a in range(lo + 1, hi):                     # stable insertion sort by column
                k = sk[a]
                v = sv[a]
                t = a
                while t > lo and sk[t - 1] > k:
                    sk[t] = sk[t - 1]
                    sv[t] = sv[t - 1]
                    t -= 1
                sk[t] = k
                sv[t] = v
        for j in range(n):
            ci[pe, j] = np.int64(sk[j]) % KX
            val_buf[px, py, j] = sv[j]
    return ne


KERNELS = {
    "hist_chunk": _hist_chunk,
    "counts_range": _counts_range,
    "offsets_range": _offsets_range,
    "scatter_chunk": _scatter_chunk,
    "build_range": _build_range,
}

KERNEL_SIGNATURES = {
    "hist_chunk": "void(i4[::1], i4[::1], i4[::1], i4[::1], i4[::1], i8, i8)",
    "counts_range": "void(i4[:, ::1], i8[::1], i8, i8)",
    "offsets_range": "void(i4[:, ::1], i8[::1], i8, i8)",
    "scatter_chunk": "void(i4[::1], i4[::1], u4[::1], i4[::1], i4[::1], u4[::1], u4[::1], "
                     "i4[::1], u8[::1], i8, i8)",
    "build_range": "i8(u8[::1], i8[::1], i8[::1], i8, i8, i8, i8, i8, i4[:, ::1], i4[:, ::1], "
                   "u4[:, :, ::1], i8, i8)",
}
"""Fixed types for the AOT build (the JIT infers the same ones from the arrays tile_fused passes)."""


# ---------------------------------------------------------------------------
# backend discovery / selection
# ---------------------------------------------------------------------------

def _load_aot():
    global _aot, _aot_tried, IMPORT_ERROR
    if not _aot_tried:
        _aot_tried = True
        if importlib.util.find_spec(AOT_MODULE) is not None:
            try:
                _aot = importlib.import_module(AOT_MODULE)
            except Exception as e:                       # wrong Python / glibc / corrupt .so
                IMPORT_ERROR = f"{AOT_MODULE}: {type(e).__name__}: {e}"
                logger.warning(f"AOT tiler present but not loadable: {e}")
    return _aot


def _numba_installed() -> bool:
    return importlib.util.find_spec("numba") is not None


def _usable_numba_cache_dir() -> None:
    """Keep compiled kernels where they survive the run (next to the tile cache) instead of in a
    ``__pycache__`` of a staged directory that is gone afterwards; fall back to a temp dir if that
    is not writable (numba refuses ``cache=True`` without a writable cache location)."""
    if os.environ.get("NUMBA_CACHE_DIR"):
        return
    import tempfile
    explicit = os.environ.get("SPMV_TILE_CACHE_DIR")
    cands = [Path(explicit) / "numba"] if explicit else []
    for cand in (*cands, Path(tempfile.gettempdir()) / "spmv-numba-cache"):
        try:
            cand.mkdir(parents=True, exist_ok=True)
            probe = cand / f".probe-{os.getpid()}"
            probe.write_text("x")
            probe.unlink()
            os.environ["NUMBA_CACHE_DIR"] = str(cand)
            return
        except OSError:
            continue


def _load_jit():
    """Import numba and wrap the kernels (lazy: compilation happens on first call)."""
    global _jit, IMPORT_ERROR
    if _jit is None:
        try:
            _usable_numba_cache_dir()
            from numba import njit
            _jit = {n: njit(nogil=True, cache=True)(f) for n, f in KERNELS.items()}
        except Exception as e:
            IMPORT_ERROR = f"numba: {type(e).__name__}: {e}"
            logger.warning(f"numba tiler unavailable ({e})")
            _jit = False
    return _jit or None


AVAILABLE = _numba_installed() or importlib.util.find_spec(AOT_MODULE) is not None
"""A compiled backend may be usable (numba installed or the AOT module importable); the import
itself is attempted lazily by select_backend, which is what can still say no."""


def min_nnz() -> int:
    return int(os.environ.get("SPMV_TILE_NUMBA_MIN_NNZ", NUMBA_MIN_NNZ_DEFAULT))


def supports(rows_per_tile: int, cols_per_tile: int, nnz: int) -> bool:
    """The fused kernels pack (loc_row, loc_col) into u32 and count in i32."""
    return nnz > 0 and nnz < 2**31 - 1 and rows_per_tile * cols_per_tile < 2**32


def select_backend(nnz: int, rows_per_tile: int, cols_per_tile: int) -> tuple[str, str]:
    """``(backend, reason)`` for a problem of this size.

    auto: the AOT module if it loads (no compile, no numba needed); else numba when ``nnz`` is at
    least SPMV_TILE_NUMBA_MIN_NNZ (a JIT must pay for itself); else numpy.  A forced backend that
    is unavailable falls back to numpy with the cause in IMPORT_ERROR.
    """
    want = os.environ.get("SPMV_TILE_BACKEND", "auto").lower()
    if want not in ("auto", "aot", "numba", "numpy"):
        raise ValueError(f"SPMV_TILE_BACKEND={want!r}; expected auto|aot|numba|numpy")
    if want == "numpy":
        return "numpy", "SPMV_TILE_BACKEND=numpy"
    if not supports(rows_per_tile, cols_per_tile, nnz):
        return "numpy", "problem outside the fused kernels' limits"
    if want in ("auto", "aot") and _load_aot() is not None:
        return "aot", "AOT module loaded" if want == "auto" else "SPMV_TILE_BACKEND=aot"
    if want == "aot":
        return "numpy", f"AOT requested but unavailable ({IMPORT_ERROR or AOT_MODULE + ' not found'})"
    if want == "auto" and nnz < min_nnz():
        return "numpy", (f"{nnz:,} NNZ < {min_nnz():,}: a numba compile would cost more than the "
                         f"numpy path saves")
    if _numba_installed() and _load_jit() is not None:
        return "numba", (f"{nnz:,} NNZ >= {min_nnz():,}" if want == "auto" else "SPMV_TILE_BACKEND=numba")
    return "numpy", f"numba unavailable ({IMPORT_ERROR or 'not installed'})"


def set_threads(n: int | None = None) -> int:
    """Threads for the numba backend (default: SPMV_TILE_THREADS, else every usable core)."""
    global _num_threads
    if n is None:
        n = int(os.environ.get("SPMV_TILE_THREADS", "0")) or _usable_cpus()
    _num_threads = max(1, int(n))
    return _num_threads


def _usable_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------

CHUNKS_PER_THREAD = 2    # input slices per thread (histogram memory: chunks x num_pes x 4 B)
MAX_CHUNKS = 64
ROWS_PER_BLOCK = 64      # PEs per work item in step 4 (threaded)


def tile_fused(
    row_idx: np.ndarray, col_idx: np.ndarray, data: np.ndarray,
    W: int, H: int, KX: int, KY: int,
    backend: str = "numba",
):
    """``(ci, rp, tile_nnz, val_buf, nonempty_rows)`` -- see coo_tiling.extract_tile_structure
    and build_val_stream_buffer for the layouts; ``tile_nnz`` is an int64 array.

    ``backend`` is "numba" or "aot" (as chosen by select_backend); output is identical.
    """
    if backend == "aot":
        k, pool, nthreads = _load_aot(), None, 1
    elif backend == "numba":
        k, nthreads = _load_jit(), max(1, _num_threads)
        pool = ThreadPoolExecutor(nthreads) if nthreads > 1 else None
    else:
        raise ValueError(f"tile_fused backend must be 'numba' or 'aot', got {backend!r}")
    if k is None:
        raise RuntimeError(f"backend {backend!r} is not available: {IMPORT_ERROR}")
    try:
        return _tile_fused(k, pool, nthreads, row_idx, col_idx, data, W, H, KX, KY)
    finally:
        if pool is not None:
            pool.shutdown()


def _each(pool, fn, items):
    """Run fn(item) for every item, on the pool if there is one, and return the results."""
    return list(pool.map(fn, items)) if pool is not None else [fn(i) for i in items]


def _tile_fused(k, pool, nthreads, row_idx, col_idx, data, W, H, KX, KY):
    get = (lambda n: getattr(k, n)) if not isinstance(k, dict) else (lambda n: k[n])
    num_pes = W * H
    nnz = len(row_idx)
    row_idx = np.ascontiguousarray(row_idx, dtype=np.int32)
    col_idx = np.ascontiguousarray(col_idx, dtype=np.int32)
    vbits = np.ascontiguousarray(data, dtype=np.float32).view(np.uint32)

    # Division-free pe_id / local-coordinate lookups (the tables stay in L2).
    rows_n, cols_n = int(row_idx.max()) + 1, int(col_idx.max()) + 1
    if rows_n > H * KY or cols_n > W * KX or int(row_idx.min()) < 0 or int(col_idx.min()) < 0:
        raise ValueError(f"COO index outside the {H * KY} x {W * KX} tiled matrix "
                         f"(rows 0..{rows_n - 1}, cols 0..{cols_n - 1})")
    ar = np.arange(rows_n, dtype=np.int64)
    ac = np.arange(cols_n, dtype=np.int64)
    row_pe = ((ar // KY) * W).astype(np.int32)
    row_loc = ((ar % KY) * KX).astype(np.uint32)
    col_pe = (ac // KX).astype(np.int32)
    col_loc = (ac % KX).astype(np.uint32)

    nchunk = 1 if pool is None else min(CHUNKS_PER_THREAD * nthreads, MAX_CHUNKS)
    B = (nnz + nchunk - 1) // nchunk
    P = (num_pes + nchunk - 1) // nchunk
    hist = np.zeros((nchunk, num_pes), dtype=np.int32)
    _each(pool, lambda c: get("hist_chunk")(row_idx, col_idx, row_pe, col_pe, hist[c],
                                            c * B, min(nnz, c * B + B)), range(nchunk))
    counts = np.empty(num_pes, dtype=np.int64)
    _each(pool, lambda c: get("counts_range")(hist, counts, c * P, min(num_pes, c * P + P)),
          range(nchunk))
    starts = np.zeros(num_pes + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    _each(pool, lambda c: get("offsets_range")(hist, starts, c * P, min(num_pes, c * P + P)),
          range(nchunk))                            # hist now holds write offsets
    packed = np.empty(nnz, dtype=np.uint64)         # (loc_row*KX + loc_col) << 32 | value bits
    _each(pool, lambda c: get("scatter_chunk")(row_idx, col_idx, vbits, row_pe, col_pe, row_loc,
                                               col_loc, hist[c], packed, c * B,
                                               min(nnz, c * B + B)), range(nchunk))
    del hist

    max_nnz = int(counts.max())
    ci = np.zeros((num_pes, max(max_nnz, 1)), dtype=np.int32)
    rp = np.empty((num_pes, KY + 1), dtype=np.int32)
    val_buf = np.zeros((W, H, max_nnz), dtype=np.uint32)
    if pool is None:
        spans = [(0, num_pes)]
    else:
        spans = [(p, min(num_pes, p + ROWS_PER_BLOCK)) for p in range(0, num_pes, ROWS_PER_BLOCK)]
    nonempty = sum(_each(pool, lambda s: get("build_range")(packed, starts, counts, W, H, KX, KY,
                                                            max_nnz, ci, rp, val_buf, s[0], s[1]),
                         spans))
    return ci, rp, counts, val_buf, int(nonempty)


def warmup(backend: str) -> float:
    """Load / compile the selected backend's kernels on a tiny problem; returns seconds spent."""
    if backend not in ("numba", "aot"):
        return 0.0
    t = time.perf_counter()
    r = np.array([0, 3, 3, 1], dtype=np.int32)
    c = np.array([1, 0, 0, 3], dtype=np.int32)
    d = np.array([1, 2, 3, 4], dtype=np.float32)
    tile_fused(r, c, d, 2, 2, 2, 2, backend=backend)
    return time.perf_counter() - t
