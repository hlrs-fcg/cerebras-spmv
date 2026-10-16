"""Host-side buffer preparation for one SpMV run, with a timer on every phase.

Everything the worker does on the CPU between "matrix in memory" and "connect to
the wafer" lives here, so that ``bench/host_tiling_bench.py`` times exactly the
code the worker runs.  Pure numpy (no SDK), safe to import anywhere.
"""

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

import tile_cache
import tile_kernels
from coo_tiling import (
    RESIDENT_ENCODINGS,
    VAL_ENCODINGS,
    ValTransferPlan,
    build_resident_vals,
    build_stream_buffer,
    build_val_stream_buffer,
    count_nonempty_rows,
    extract_tile_structure,
    plan_val_transfers,
)

logger = logging.getLogger(__name__)


@dataclass
class HostBuffers:
    """Everything the run needs from the host, plus the time each phase took."""
    ci: np.ndarray                    # (W*H, max_nnz) int32 column indices
    rp: np.ndarray                    # (W*H, KY+1) int32 row pointers
    nnz_counts: list[int]
    x_grid: np.ndarray                # (W*H, KX) float32
    val_only: bool                    # val / valtask: the values-only STREAM
    vals_grid: np.ndarray | None = None     # (W*H, max_nnz) float32, resident: copied once at setup
    val_buf: np.ndarray | None = None       # (W, H, max_nnz) uint32, val/valtask
    val_plan: ValTransferPlan | None = None
    stream_buf: np.ndarray | None = None    # (W, H, max_words) uint32, header
    col_words: list[int] | None = None
    h2d_words_col: int = 0
    h2d_words_full: int = 0
    common_words: int = 0
    total_stream_words: int = 0
    header_words: int = 0             # val-only: what the header stream would carry (log only)
    backend: str = "numpy"            # "numba" = fused parallel tiler, "numpy" = lexsort path,
                                      # "cache" = loaded from the on-disk tile cache
    cache_hit: bool = False
    save_thread: threading.Thread | None = None   # background tile-cache write (non-daemon)
    times: dict[str, float] = field(default_factory=dict)


def build_x_grid(x: np.ndarray, cols: int, W: int, H: int, KX: int) -> np.ndarray:
    """``(W*H, KX)`` float32, PE ``py*W+px`` holding x[px*KX : (px+1)*KX] (zero padded)."""
    xp = np.zeros(W * KX, dtype=np.float32)
    n = min(cols, W * KX)
    xp[:n] = x[:n]
    x_grid = np.empty((H, W, KX), dtype=np.float32)
    x_grid[:] = xp.reshape(1, W, KX)          # the same x slice on every PE row
    return x_grid.reshape(W * H, KX)


def prepare_host_buffers(
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    data: np.ndarray,
    x: np.ndarray,
    cols: int,
    W: int, H: int, KX: int, KY: int,
    encoding: str,
    h2d_mode: str,
    total_nnz_per_pe: int | None = None,
    num_threads: int = 64,
    cache_fingerprint: str | None = None,
) -> HostBuffers:
    """Tile the matrix and build every host buffer; ``times`` has one entry per phase.

    Keys: ``tile_extract`` (col_indices + row_ptrs, and for the fused numba tiler also the val
    buffer; on a tile-cache hit, the time to map the entry), ``x_grid``, ``stream_build`` (transfer plans + the val buffer when the numpy path
    built it, or the header stream buffer), ``jit`` (numba compile/cache load, numba only).
    """
    times: dict[str, float] = {}
    nnz = len(data)

    val_only = encoding in VAL_ENCODINGS
    resident = encoding in RESIDENT_ENCODINGS
    # val, valtask and resident all need the per-PE values in CSR order and nothing from the
    # header stream (no row bounds): they share the cache, the fused tiler and its val buffer.
    val_tiled = val_only or resident
    t = time.perf_counter()
    order = rb = None
    val_buf = None
    nonempty = 0
    key = None
    cached = None
    if val_tiled and cache_fingerprint and not tile_cache.enabled():
        logger.info(f"Tile cache off: {tile_cache.disabled_reason()}")
    if val_tiled and cache_fingerprint and tile_cache.enabled():
        key = tile_cache.make_key(cache_fingerprint, W, H, KX, KY)
        cached = tile_cache.load(key)
        if cached is not None and int(cached[3].sum()) != nnz:
            logger.warning(f"tile cache {key}: entry holds {int(cached[3].sum()):,} NNZ, matrix has "
                           f"{nnz:,}; ignoring it")
            cached = None
        logger.info(f"Tile cache {'HIT' if cached is not None else 'MISS'} ({key}, "
                    f"{tile_cache.cache_dir()})")
    fused = False
    if cached is None and val_tiled:
        backend, why = tile_kernels.select_backend(nnz, KY, KX)
        logger.info(f"Tile backend: {backend} ({why})")
        if backend != "numpy":
            threads = tile_kernels.set_threads()
            try:
                times["jit"] = tile_kernels.warmup(backend)   # compile / load, kept out of tile_extract
                fused = True
            except Exception as e:                            # e.g. an AOT .so for the wrong glibc
                tile_kernels.IMPORT_ERROR = f"{backend}: {type(e).__name__}: {e}"
                logger.warning(f"{backend} tiler failed its warmup ({e}); using the numpy path")
    t = time.perf_counter()                 # tile_extract excludes backend selection and compile
    if cached is not None:
        ci, rp, val_buf, counts, nonempty = cached
        nnz_counts = counts.tolist()
        backend = "cache"
    elif fused:
        # val/valtask: one pass produces col_indices, row_ptrs AND the val buffer (no `order`,
        # no row bounds -- neither is needed by the values-only stream).
        ci, rp, counts, val_buf, nonempty = tile_kernels.tile_fused(
            row_idx, col_idx, data, W, H, KX, KY, backend=backend)
        nnz_counts = counts.tolist()
        logger.info(f"Fused {backend} tiler: {threads if backend == 'numba' else 1} thread(s), "
                    f"jit {times['jit']:.2f}s")
    else:
        # Row bounds exist only for the header stream; val/valtask never read them.
        ci, rp, nnz_counts, order, rb = extract_tile_structure(
            row_idx, col_idx, W, H, KX, KY, want_row_bounds=not val_tiled)
        backend = "numpy"
    times["tile_extract"] = time.perf_counter() - t
    logger.info(f"Structure extraction done in {times['tile_extract']:.3f}s ({backend})")

    max_nnz = max(nnz_counts)
    if total_nnz_per_pe is not None and max_nnz > total_nnz_per_pe:
        raise ValueError(
            f"the busiest PE holds {max_nnz} NNZ but the kernel was compiled with "
            f"TOTAL_NNZ={total_nnz_per_pe} (col_indices is [TOTAL_NNZ]i32). The setup "
            f"memcpy would overrun col_indices into row_ptrs/y_result. Recompile "
            f"with --total-nnz-per-pe >= {max_nnz}, or spread the matrix over "
            f"more PEs."
        )

    t = time.perf_counter()
    x_grid = build_x_grid(x, cols, W, H, KX)
    times["x_grid"] = time.perf_counter() - t

    out = HostBuffers(ci=ci, rp=rp, nnz_counts=nnz_counts, x_grid=x_grid,
                      val_only=val_only, times=times, backend=backend,
                      cache_hit=cached is not None)

    t = time.perf_counter()
    if resident:
        # No stream and no plan: the values are copied once at setup, PE-major like col_indices.
        # The fused tiler / cache already hold them as the (W, H, max_nnz) uint32 val buffer;
        # build_resident_vals is the same values straight from `order` (tests pin the two equal).
        if val_buf is None and key is not None:
            # the numpy path with the tile cache on: build the shared (W, H, max_nnz) buffer so
            # the entry can be written (valtask / val read the same entry)
            val_buf = build_val_stream_buffer(data, order, nnz_counts, W, H)
        if val_buf is not None:
            out.vals_grid = np.ascontiguousarray(
                val_buf.transpose(1, 0, 2)).reshape(W * H, -1).view(np.float32)
        else:
            out.vals_grid = build_resident_vals(data, order, nnz_counts, W, H)
        if key is not None and cached is None and val_buf is not None:
            out.save_thread = tile_cache.save_async(
                key, ci, rp, val_buf, nnz_counts, nonempty,
                {"fingerprint": cache_fingerprint, "W": W, "H": H, "KX": KX, "KY": KY, "nnz": nnz,
                 "backend": backend})
    elif out.val_only:
        out.val_buf = val_buf if val_buf is not None else build_val_stream_buffer(
            data, order, nnz_counts, W, H)
        out.val_plan = plan_val_transfers(nnz_counts, W, H, h2d_mode)
        out.h2d_words_col = plan_val_transfers(nnz_counts, W, H, "column").total_words
        out.h2d_words_full = plan_val_transfers(nnz_counts, W, H, "full").total_words
        out.total_stream_words = out.h2d_words_col // H if H else 0
        out.common_words = int(out.val_buf.shape[2])
        if cached is not None or fused:
            out.header_words = nnz + nonempty + W * H
        else:
            nonempty = count_nonempty_rows(rp)
            out.header_words = nnz + nonempty + W * H
        if key is not None and cached is None:
            # The thread is non-daemon: the interpreter waits for it at exit, and by then it has
            # had the whole program load (minutes) to finish.
            out.save_thread = tile_cache.save_async(
                key, ci, rp, out.val_buf, nnz_counts, nonempty,
                {"fingerprint": cache_fingerprint, "W": W, "H": H, "KX": KX, "KY": KY, "nnz": nnz,
                 "backend": backend})
    else:
        # Every PE stream ends in an explicit 0 sentinel and col_words[px] is the
        # exact number of words column px transfers.
        out.stream_buf, out.col_words = build_stream_buffer(
            data, order, nnz_counts, rb, W, H, num_threads=num_threads,
        )
        out.total_stream_words = int(sum(out.col_words))
        out.common_words = int(out.stream_buf.shape[2])      # == max(col_words)
        out.h2d_words_col = H * out.total_stream_words
        out.h2d_words_full = W * H * out.common_words
    times["stream_build"] = time.perf_counter() - t
    return out
