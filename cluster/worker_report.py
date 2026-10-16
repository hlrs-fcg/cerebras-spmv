"""The end of a worker run: timing summary, gathering y, validating every pass, and the CSV row.
Staged flat next to worker_runtime.py."""

import logging
import time

import numpy as np

import tile_kernels
import worker_env
from benchmark import BenchmarkResult, save_benchmark
from coo_tiling import spmv_reference
from worker_common import validate_results
from worker_layout import gather_y
from worker_timing import WSE_CLOCK_HZ, pass_stats

logger = logging.getLogger(__name__)


def finish(r, mode) -> int:
    """Summarise, validate and record the run; returns the worker's exit code."""
    # The device and host stages left their results on the Run; the body below keeps the names the
    # single-function worker used.
    H = r.H
    KX = r.KX
    KY = r.KY
    S = r.S
    W = r.W
    args = r.args
    avg_tile_nnz = r.avg_tile_nnz
    barrier_d2h_repeat_time = r.barrier_d2h_repeat_time
    barrier_d2h_time = r.barrier_d2h_time
    col_idx = r.col_idx
    cyc_avg = r.cyc_avg
    cyc_max = r.cyc_max
    cyc_min = r.cyc_min
    cyc_std = r.cyc_std
    data = r.data
    device_done_time = r.device_done_time
    env_info = r.env_info
    h2d_words_col = r.h2d_words_col
    h2d_words_full = r.h2d_words_full
    hb = r.hb
    host_prep_time = r.host_prep_time
    k_strips = r.k_strips
    load_imbalance = r.load_imbalance
    matrix_time = r.matrix_time
    max_nnz = r.max_nnz
    max_tile_nnz = r.max_tile_nnz
    memcpy_d2h_time = r.memcpy_d2h_time
    memcpy_h2d_time = r.memcpy_h2d_time
    min_tile_nnz = r.min_tile_nnz
    nnz = r.nnz
    pass_barrier_s = r.pass_barrier_s
    pass_corrected_s = r.pass_corrected_s
    pass_cycles = r.pass_cycles
    pass_host_s = r.pass_host_s
    pass_ratio = r.pass_ratio
    program_load_time = r.program_load_time
    resident = r.resident
    row_idx = r.row_idx
    sdk_init_time = r.sdk_init_time
    serialize_rects = r.serialize_rects
    setup_barrier_d2h_repeat_time = r.setup_barrier_d2h_repeat_time
    setup_barrier_d2h_time = r.setup_barrier_d2h_time
    setup_done_time = r.setup_done_time
    setup_h2d_bytes = r.setup_h2d_bytes
    setup_h2d_time = r.setup_h2d_time
    setup_h2d_time_corrected = r.setup_h2d_time_corrected
    setup_prep_time = r.setup_prep_time
    stream_build_time = r.stream_build_time
    stream_time_corrected = r.stream_time_corrected
    tile_time = r.tile_time
    total_start = r.total_start
    val_only = r.val_only
    val_plan = r.val_plan
    vector_time = r.vector_time
    x = r.x
    x_grid_time = r.x_grid_time
    y_passes = r.y_passes

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    host_stats = pass_stats(pass_host_s)
    corr_stats = pass_stats(pass_corrected_s)
    wafer_cyc  = pass_stats(pass_cycles)
    logger.info(f"Compute+barrier (host-return, a lower bound): {host_stats['avg']*1e3:.3f} ms"
                + (f" (min {host_stats['min']*1e3:.3f}, max {host_stats['max']*1e3:.3f}, "
                   f"std {host_stats['std']*1e3:.3f} over {len(pass_host_s)} passes)"
                   if len(pass_host_s) > 1 else "")
                + f"; stream_time_corrected {stream_time_corrected*1e3:.3f} ms"
                + (f" (min {corr_stats['min']*1e3:.3f}, max {corr_stats['max']*1e3:.3f})"
                   if len(pass_corrected_s) > 1 else ""))
    if pass_cycles:
        logger.info(
            f"On-wafer (synced, max toc - min tic): mean {wafer_cyc['avg']:,.0f} cycles "
            f"= {wafer_cyc['avg'] / WSE_CLOCK_HZ * 1e3:.3f} ms, min {wafer_cyc['min']:,.0f}, "
            f"max {wafer_cyc['max']:,.0f}, std {wafer_cyc['std']:,.0f} over "
            f"{len(pass_cycles)} passes"
        )

    # ------------------------------------------------------------------
    # Gather: assemble y from column-0 D2H data, every pass
    # ------------------------------------------------------------------
    logger.debug("Gather: assembling y from column 0...")
    t = time.perf_counter()
    y_computed_passes = [gather_y(y_flat, H, KY, args.rows) for y_flat in y_passes]
    gather_time = (time.perf_counter() - t) / len(y_passes)

    # ------------------------------------------------------------------
    # Validate: every pass must match, so a pass that leaks state into the next
    # (an uncleared y_result, a stale vals[] slot, a stranded stream word)
    # fails the run.  relative_error is the worst pass.
    # ------------------------------------------------------------------
    logger.debug("Computing numpy reference SpMV...")
    t = time.perf_counter()
    y_reference = spmv_reference(row_idx, col_idx, data, x, args.rows)
    ref_norm  = np.linalg.norm(y_reference)
    passed = True
    rel_error = 0.0
    if args.null_spmv:
        logger.warning("--null-spmv: spmv() computed nothing, y is garbage; validation SKIPPED")
        y_computed_passes = []
    for p_i, y_computed in enumerate(y_computed_passes):
        ok = validate_results(y_computed, y_reference)
        err = (float(np.linalg.norm(y_computed - y_reference) / np.linalg.norm(y_reference))
               if np.linalg.norm(y_reference) > 0 else 0.0)
        if len(y_computed_passes) > 1:
            logger.info(f"Pass {p_i + 1}: {'PASSED' if ok else 'FAILED'} rel_err={err:.2e}")
        passed = passed and ok
        rel_error = max(rel_error, err)
    validation_time = time.perf_counter() - t
    logger.info(f"Validation (numpy reference): {validation_time:.3f}s")

    total_time = time.perf_counter() - total_start

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
    # Memory per PE (streaming design: no a_stream buffer on PE).  pe_valtask.csl
    # adds vals[TOTAL_NNZ] (of which max_nnz slots are live) + val_meta[1].
    max_memory_per_pe = 4 * (KX + max_nnz + (KY + 1) + KY + KY
                             + ((max_nnz + 1) if val_only else 0)
                             + (2 * max_nnz if resident else 0))   # vals[] and row_ids[]
    avg_memory_per_pe = float(max_memory_per_pe)
    # Bytes actually streamed for A values, per the selected --h2d-mode.
    h2d_bytes = 0 if resident else 4 * (
        val_plan.total_words if val_only
        else h2d_words_full if args.h2d_mode == "full" else h2d_words_col)

    if args.save_benchmark:
        try:
            env_info["numba_status"] = worker_env.numba_status(tile_kernels)   # now incl. any import error
            benchmark = BenchmarkResult(
                validation_passed=passed,
                exit_reason=("SUCCESS_NULL_SPMV" if args.null_spmv
                             else "SUCCESS" if passed else "VALIDATION_FAILED"),
                error_message="" if passed else "Results do not match reference",
                relative_error=rel_error,
                matrix_rows=args.rows,
                matrix_cols=args.cols,
                density=args.density or 0.0,
                total_nnz=nnz,
                seed=args.seed,
                num_pes=args.num_pes,
                grid_width=W,
                grid_height=H,
                kx=KX,
                ky=KY,
                num_col_stripes=S,
                h2d_mode=args.h2d_mode,
                memcpy_order=args.memcpy_order,
                encoding=args.encoding,
                setup_order=args.setup_order,
                setup_strips=k_strips,
                # what actually ran: True for every val column-mode row
                serialize_column_rects=serialize_rects and args.h2d_mode == "column",
                architecture=args.arch,
                max_memory_per_pe=max_memory_per_pe,
                avg_memory_per_pe=avg_memory_per_pe,
                compile_time=0.0,
                matrix_generation_time=matrix_time,
                vector_generation_time=vector_time,
                tile_extract_time=tile_time,
                x_grid_time=x_grid_time,
                stream_build_time=stream_build_time,
                setup_prep_time=setup_prep_time,
                host_prep_time=host_prep_time,
                tile_backend=hb.backend,
                tile_cache_hit=hb.cache_hit,
                tile_jit_time=hb.times.get("jit", 0.0),
                sdk_init_time=sdk_init_time,
                **env_info,
                program_load_time=program_load_time,
                h2d_bytes_total=h2d_bytes,
                memcpy_h2d_time=memcpy_h2d_time,
                setup_h2d_bytes=setup_h2d_bytes,
                setup_h2d_time=setup_h2d_time,
                setup_done_time=setup_done_time,
                setup_barrier_d2h_time=setup_barrier_d2h_time,
                setup_barrier_d2h_repeat_time=setup_barrier_d2h_repeat_time,
                setup_h2d_time_corrected=setup_h2d_time_corrected,
                device_done_time=device_done_time,
                barrier_d2h_time=barrier_d2h_time,
                barrier_d2h_repeat_time=barrier_d2h_repeat_time,
                stream_time_corrected=stream_time_corrected,
                # host compute+barrier per pass; loop_count == 1: all equal, std 0
                compute_time_hw_min=host_stats["min"],
                compute_time_hw_avg=host_stats["avg"],
                compute_time_hw_max=host_stats["max"],
                compute_time_hw_std=host_stats["std"],
                loop_count=args.loop_count,
                sync_timing=args.sync_timing,
                # synced on-wafer span per pass, max(toc) - min(tic) on the
                # common reference clock; 0 unless --sync-timing read back
                # every pass.  docs/synced-timing.md
                wafer_cycles_min=int(round(wafer_cyc["min"])),
                wafer_cycles_avg=int(round(wafer_cyc["avg"])),
                wafer_cycles_max=int(round(wafer_cyc["max"])),
                wafer_cycles_std=int(round(wafer_cyc["std"])),
                wafer_time_min=wafer_cyc["min"] / WSE_CLOCK_HZ,
                wafer_time_avg=wafer_cyc["avg"] / WSE_CLOCK_HZ,
                wafer_time_max=wafer_cyc["max"] / WSE_CLOCK_HZ,
                wafer_time_std=wafer_cyc["std"] / WSE_CLOCK_HZ,
                # host time per pass through the completion barrier, the bracket that is a
                # true upper bound on the pass (see wafer_host_ratio)
                host_pass_time_avg=pass_stats(pass_barrier_s)["avg"],
                host_pass_time_std=pass_stats(pass_barrier_s)["std"],
                wafer_host_ratio_max=max(pass_ratio) if pass_cycles else 0.0,
                # useful FLOP rate of the synced window: 2 per NNZ.  0 without sync timing.
                wafer_gflops_avg=(2.0 * nnz / (wafer_cyc["avg"] / WSE_CLOCK_HZ) / 1e9
                                  if pass_cycles and wafer_cyc["avg"] > 0
                                  and not args.null_spmv else 0.0),
                null_spmv=args.null_spmv,
                # compute_hw_cycles_*: per-PE tsc span, first streamed word ->
                # EW-chain-done.  This INCLUDES per-word H2D stalls and the skew
                # from streaming grid columns one after another; it is not
                # on-wafer compute in isolation.  Read the spread (max vs min),
                # not the absolute value.  See unpack_tsc_cycles() and
                # memory/csl-cycle-timing.md.
                compute_hw_cycles_min=cyc_min,
                compute_hw_cycles_avg=int(round(cyc_avg)),
                compute_hw_cycles_max=cyc_max,
                compute_hw_cycles_std=int(round(cyc_std)),
                memcpy_d2h_time=memcpy_d2h_time,
                gather_results_time=gather_time,
                validation_time=validation_time,
                total_time=total_time,
                min_nnz_per_pe=min_tile_nnz,
                max_nnz_per_pe=max_tile_nnz,
                avg_nnz_per_pe=avg_tile_nnz,
                load_imbalance_pct=load_imbalance,
            )
            save_benchmark(benchmark, csv_path=args.save_benchmark)
            logger.info(f"Benchmark saved to {args.save_benchmark}")
        except Exception as e:
            logger.warning(f"Failed to record benchmark: {e}")

    logger.info("=" * 60)
    logger.info(f"Total: {total_time:.2f}s — {'PASSED' if passed else 'FAILED'}")
    logger.info("=" * 60)
    return 0 if passed else 1
