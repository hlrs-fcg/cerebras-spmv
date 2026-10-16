"""The host half of a worker run, before the device is touched: grid geometry, the matrix, the tiled
host buffers, and the compiled artifact's out.json.  Each stage reads and fills the ``Run`` (see
worker_common.Run); a stage returns an exit code to stop the run, None to go on.  Staged flat next to
worker_runtime.py."""

import json
import logging
import math
import time

import numpy as np

import mm_io
import tile_cache
from coo_tiling import ENCODING_IDS, build_row_ids, generate_random_coo
from host_prep import prepare_host_buffers
from worker_common import (
    artifact_encoding,
    artifact_local_kernel,
    artifact_null_spmv,
    artifact_sync_timing,
)

logger = logging.getLogger(__name__)


def set_geometry(r) -> None:
    args = r.args
    r.W, r.H = args.grid_width, args.grid_height
    r.TOTAL_NNZ = args.total_nnz_per_pe
    r.S = args.num_stripes
    r.KX = math.ceil(args.cols / r.W)
    r.KY = math.ceil(args.rows / r.H)
    r.KX_STRIPE = math.ceil(r.KX / r.S)

    if r.S > 1:
        raise NotImplementedError(
            f"--num-stripes={r.S}: column striping is plumbed but not implemented. "
            f"extract_tile_structure() builds full untiled tiles while "
            f"config.total_nnz_per_pe divides its estimate by the stripe count, "
            f"so S>1 under-sizes col_indices on every PE. Run with --num-stripes 1."
        )
    if r.KY > 65535:
        raise ValueError(
            f"KY={r.KY} exceeds 65535: a stream header carries the local row index "
            f"in 16 bits, so taller tiles alias rows silently. Use a larger "
            f"--grid-height so each PE row owns fewer matrix rows."
        )

    logger.info(
        f"Grid: {r.W}×{r.H} = {args.num_pes:,} PEs  "
        f"tile: KX={r.KX} KY={r.KY}  stripes: S={r.S} KX_STRIPE={r.KX_STRIPE}"
    )


def load_matrix(r):
    """Load or generate A and x.  Returns an exit code on failure."""
    args = r.args
    t = time.perf_counter()
    if args.mtx_file:
        try:
            mm = mm_io.read_matrix_market(args.mtx_file)
        except (ValueError, OSError) as e:
            logger.error(f"cannot use --mtx-file: {e}")
            return 1
        if (mm.rows, mm.cols) != (args.rows, args.cols):
            logger.error(f"{args.mtx_file} is {mm.rows}x{mm.cols} but the run was configured for "
                         f"{args.rows}x{args.cols} (--rows/--cols): the grid and the kernel were sized for those")
            return 1
        logger.info(f"Loaded .mtx: {mm.rows}×{mm.cols}, {mm.stored_entries:,} stored entries "
                    f"({mm.symmetry}), NNZ={mm.nnz:,}")
        r.row_idx, r.col_idx, r.data = mm.row, mm.col, mm.data
    else:
        if args.density is None:
            logger.error("--density required for random matrix generation")
            return 1
        r.row_idx, r.col_idx, r.data = generate_random_coo(
            args.rows, args.cols, args.density, args.seed
        )
    r.matrix_time = time.perf_counter() - t

    r.nnz = len(r.data)
    logger.info(f"NNZ={r.nnz:,}")

    t = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    r.x = rng.integers(0, 100, size=args.cols).astype(np.float32)
    r.vector_time = time.perf_counter() - t
    return None


def prepare_host(r) -> None:
    """One-time structure extraction + host buffers (every phase timed)."""
    args, W, H = r.args, r.W, r.H
    logger.info(f"Extracting tile structure for {W}×{H} grid...")
    t_host = time.perf_counter()
    fingerprint = (tile_cache.file_fingerprint(args.mtx_file) if args.mtx_file else
                   tile_cache.random_fingerprint(args.rows, args.cols, args.density, args.seed))
    hb = prepare_host_buffers(
        r.row_idx, r.col_idx, r.data, r.x, args.cols, W, H, r.KX, r.KY,
        args.encoding, args.h2d_mode, total_nnz_per_pe=r.TOTAL_NNZ, num_threads=64,
        cache_fingerprint=fingerprint,
    )
    r.hb = hb
    r.host_prep_time = time.perf_counter() - t_host
    r.tile_time = hb.times["tile_extract"]
    r.x_grid_time = hb.times["x_grid"]
    r.stream_build_time = hb.times["stream_build"]
    r.ci, r.rp, r.nnz_counts, r.x_grid = hb.ci, hb.rp, hb.nnz_counts, hb.x_grid
    r.val_buf, r.val_plan, r.stream_buf, r.col_words = hb.val_buf, hb.val_plan, hb.stream_buf, hb.col_words
    r.total_stream_words, r.common_words = hb.total_stream_words, hb.common_words
    r.h2d_words_col, r.h2d_words_full = hb.h2d_words_col, hb.h2d_words_full
    r.max_nnz = max(r.nnz_counts)

    nnz_arr          = np.array(r.nnz_counts)
    r.min_tile_nnz   = int(nnz_arr.min())
    r.max_tile_nnz   = int(nnz_arr.max())
    r.avg_tile_nnz   = float(nnz_arr.mean())
    r.load_imbalance = ((r.max_tile_nnz - r.min_tile_nnz) / r.avg_tile_nnz * 100
                        if r.avg_tile_nnz > 0 else 0.0)

    # val and valtask share the stream, the plan and val_meta; only the kernel
    # consuming it differs (armed @fmovs vs per-word data task).  The plan is the
    # single source of truth for BOTH what the transfers send and what each PE is
    # told to expect (val_meta), because the kernel stops by count: any mismatch
    # hangs the @fmovs or strands words in the queue.  Logged for both modes.
    r.val_only = hb.val_only
    logger.info(
        f"Host prep: tile {r.tile_time:.2f}s, x_grid {r.x_grid_time:.2f}s, "
        f"stream build {r.stream_build_time:.2f}s (total {r.host_prep_time:.2f}s)"
    )
    r.vals_grid = hb.vals_grid
    r.row_ids_grid = None
    if r.resident:
        # A lives on the wafer: no stream, no plan, no val_meta.  The values are
        # copied once at setup, PE-major like col_indices, in the same CSR order
        # (host_prep.prepare_host_buffers built them).
        r.total_stream_words = 0
        r.row_ids_grid = build_row_ids(r.rp, r.nnz_counts)            # COO scatter: a row per NNZ
        logger.info(
            f"Resident A: vals {r.vals_grid.nbytes / 1e6:.1f} MB "
            f"({W * H:,} PEs x {r.vals_grid.shape[1]} slots) loaded once at setup; "
            f"nothing streams per pass (--h2d-mode / --memcpy-order ignored)"
        )
    elif r.val_only:
        header_words = hb.header_words
        logger.info(
            f"H2D A-stream size (val-only): column={r.h2d_words_col * 4 / 1e6:.1f} MB "
            f"({r.h2d_words_col:,} words), full={r.h2d_words_full * 4 / 1e6:.1f} MB "
            f"({r.h2d_words_full:,} words, pad to {r.common_words:,}/PE); the header "
            f"encoding would carry {header_words:,} unpadded words "
            f"({100 * (header_words - r.nnz) / max(r.nnz, 1):.0f}% over NNZ)"
        )
    else:
        # 'column' sends each column its exact length; 'full' sends one WxH
        # rectangle padded to the widest column.  Logged for both regardless of
        # which is selected, so the trade is visible without a wafer run.
        h2d_extra_mb   = (r.h2d_words_full - r.h2d_words_col) * 4 / 1e6
        logger.info(
            f"H2D A-stream size: column={r.h2d_words_col * 4 / 1e6:.1f} MB "
            f"({r.h2d_words_col:,} words), full={r.h2d_words_full * 4 / 1e6:.1f} MB "
            f"({r.h2d_words_full:,} words, pad to {r.common_words:,}/PE) "
            f"→ full costs +{h2d_extra_mb:.1f} MB "
            f"(+{100 * (r.h2d_words_full - r.h2d_words_col) / r.h2d_words_col:.1f}%)"
        )


def read_compile_data(r):
    """Read MEMCPYH2D_DATA_1's color ID from the compiled artifact and check the artifact against
    the flags.  Returns an exit code on failure."""
    args = r.args
    out_json_path = f"{args.name}/out.json"
    try:
        with open(out_json_path, encoding="utf-8") as f:
            compile_data = json.load(f)
        r.MEMCPYH2D_DATA_1 = int(compile_data["params"]["MEMCPYH2D_DATA_1_ID"])
        logger.info(f"MEMCPYH2D_DATA_1 color = {r.MEMCPYH2D_DATA_1}")
    except Exception as e:
        logger.error(f"Failed to read MEMCPYH2D_DATA_1_ID from {out_json_path}: {e}")
        return 1

    # The two kernels speak different stream formats; driving one with the
    # other's stream does not fail cleanly, it hangs.  Refuse before touching
    # the device.  An artifact predating the ENCODING param is a header kernel.
    compiled_encoding = artifact_encoding(compile_data)
    if compiled_encoding != args.encoding:
        logger.error(
            f"--encoding {args.encoding} but the artifact was compiled with "
            f"ENCODING={ENCODING_IDS[compiled_encoding]} ({compiled_encoding}). "
            f"Recompile, or pass --encoding {compiled_encoding}."
        )
        return 1

    # f_sync / f_tic / f_toc / sync_pack exist only in a SYNC_TIMING:1 artifact;
    # an artifact that has them but is driven without --sync-timing still runs
    # correctly, it just reports no synced timing.
    old_kernel = artifact_local_kernel(compile_data)
    if old_kernel is not None and old_kernel != 2:
        logger.error(f"the artifact was compiled with LOCAL_KERNEL={old_kernel} (a CSR loop); the resident "
                     f"kernel is now the COO scatter and needs row_ids. Recompile.")
        return 1
    if artifact_null_spmv(compile_data) != args.null_spmv:
        logger.error(f"--null-spmv is {args.null_spmv} but the artifact was compiled with "
                     f"ENCODING={compile_data.get('params', {}).get('ENCODING')}. Recompile.")
        return 1
    if args.sync_timing and not artifact_sync_timing(compile_data):
        logger.error("--sync-timing but the artifact was compiled with SYNC_TIMING:0. "
                     "Recompile with --sync-timing.")
        return 1
    return None
