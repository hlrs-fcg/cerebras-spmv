"""The device half of a worker run: connect, load, the setup copy, the timed passes, and the per-PE
cycle readback.  Everything that differs between streaming and resident A is behind the ``Mode``
hooks (worker_modes.py).  Results land on the ``Run``.  Staged flat next to worker_runtime.py."""

import logging
import time
import traceback
from types import SimpleNamespace

import numpy as np

from worker_common import SdkRuntimeWrapper
from worker_layout import setup_strip_bounds, setup_strip_calls
from worker_timing import (
    WSE_CLOCK_HZ,
    corrected_time,
    summarise_tsc_cycles,
    synced_pass_cycles,
    timed_d2h,
    unpack_sync_pack,
    unpack_tsc_cycles,
    wafer_host_ratio,
)

logger = logging.getLogger(__name__)


def run_device(r, mode, sdk):
    """Run the device phase.  ``sdk`` is (SdkRuntime, MemcpyDataType, MemcpyOrder).  Returns an
    exit code on failure, None on success."""
    args, W, H, KX, KY = r.args, r.W, r.H, r.KX, r.KY
    progress = r.progress
    SdkRuntime, MemcpyDataType, MemcpyOrder = sdk
    resident = r.resident
    max_nnz = r.max_nnz

    # Per-PE on-wafer cycle counts, filled from the tsc D2H below.  Defaults hold
    # only if the run throws before the timestamps are read back.
    r.cyc_min = r.cyc_max = 0
    r.cyc_avg = r.cyc_std = 0.0
    r.cyc_recorded_pes = 0

    progress("host matrix and tiles ready")
    logger.info("Connecting to Cerebras system...")
    sdk_init_start = time.perf_counter()
    try:
        runtime_kwargs = {"cmaddr": args.cmaddr}
        if args.sim_trace:
            runtime_kwargs["suppress_simfab_trace"] = False
        with SdkRuntimeWrapper(SdkRuntime, args.name, **runtime_kwargs) as runner:
            r.sdk_init_time = time.perf_counter() - sdk_init_start

            syms = {}
            for name in ("x_vec", "col_indices", "row_ptrs", "y_result", "tsc_pack"):
                syms[name] = runner.get_id(name)
            for name in mode.extra_symbol_names(r):
                syms[name] = runner.get_id(name)
            # Exported only under SYNC_TIMING:1 (layout.csl gates the @export_name).
            sym_sync_pack = runner.get_id("sync_pack") if args.sync_timing else None
            sym_x_vec, sym_col_indices = syms["x_vec"], syms["col_indices"]
            sym_row_ptrs, sym_y_result, sym_tsc_pack = syms["row_ptrs"], syms["y_result"], syms["tsc_pack"]

            t = time.perf_counter()
            progress("runner.load / run")
            runner.load()
            runner.run()
            r.program_load_time = time.perf_counter() - t

            # ── Setup phase: one-time permanent H2D ──
            #
            # These go streaming=False -- copy mode, DMA'd straight into PE
            # memory with no compute-engine involvement.  That makes this block the
            # only COPY-MODE transfer in the run, and therefore the only measurement
            # that sees the link without paying our per-word kernel task (header
            # decode + FMAC in stream_word).  Cerebras' own bandwidth benchmark
            # measures exclusively in this mode
            # (csl-examples/benchmarks/bandwidth-test, streaming=False throughout),
            # which is why timing it is what makes the streaming rate interpretable:
            # setup rate ~= link, compute rate = link + kernel, so the gap is ours.
            #
            # The last call is blocking, so the bracket covers all of them.
            setup_h2d_bytes = (4 * W * H * (KX + max_nnz + (KY + 1))
                               + mode.setup_extra_bytes(r))
            r.setup_h2d_bytes = setup_h2d_bytes
            logger.info(
                f"Setup: loading x_vec, col_indices, {'vals, ' if resident else ''}row_ptrs "
                f"(one-time, copy mode, "
                f"{setup_h2d_bytes / 1e6:.1f} MB)..."
            )
            # Flattened before the bracket: "col" needs a host-side transpose,
            # which must not be counted as transfer time.
            setup_enum = (MemcpyOrder.COL_MAJOR if args.setup_order == "col"
                          else MemcpyOrder.ROW_MAJOR)
            d = SimpleNamespace(runner=runner, MemcpyDataType=MemcpyDataType,
                                MemcpyOrder=MemcpyOrder, setup_enum=setup_enum, syms=syms)
            # Each array is sent as args.setup_strips vertical strips, one after another
            # (ADR-008 rule 2: narrow strips reach the H2D ceiling, one full rectangle
            # about half).  All strips are flattened here, outside the bracket.
            k_strips = len(setup_strip_bounds(W, args.setup_strips))
            r.k_strips = k_strips
            setup_arrays = [(sym_x_vec, r.x_grid, KX), (sym_col_indices, r.ci, max_nnz)]
            setup_arrays += mode.setup_extra_arrays(r, d)
            setup_arrays.append((sym_row_ptrs, r.rp, KY + 1))
            t_prep = time.perf_counter()
            setup_calls = [
                (sym, l, setup_strip_calls(arr, W, H, args.setup_order, k_strips,
                                           last_array=(i == len(setup_arrays) - 1)))
                for i, (sym, arr, l) in enumerate(setup_arrays)
            ]
            r.setup_prep_time = time.perf_counter() - t_prep
            progress("setup copy start")
            t_setup = time.perf_counter()
            for sym, l, calls in setup_calls:
                for x0, w, flat, nb in calls:
                    runner.memcpy_h2d(sym, flat, x0, 0, w, H, l,
                                      streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                                      order=setup_enum, nonblock=nb)
            r.setup_h2d_time = time.perf_counter() - t_setup   # host return
            # Completion barrier: one word of row_ptrs (the LAST setup array)
            # from EVERY PE.  Each PE answers only after its own copy-mode
            # writes ahead of it in the command order, so this cannot finish
            # before all of setup has landed.  Then the same D2H again, on an
            # idle device, prices the barrier itself.
            r.setup_barrier_d2h_time = timed_d2h(runner, sym_row_ptrs, W, H,
                                                 MemcpyDataType.MEMCPY_32BIT,
                                                 setup_enum)
            r.setup_done_time = time.perf_counter() - t_setup
            r.setup_barrier_d2h_repeat_time = timed_d2h(runner, sym_row_ptrs, W, H,
                                                        MemcpyDataType.MEMCPY_32BIT,
                                                        setup_enum)
            progress("setup copy and its barrier done")
            r.setup_h2d_time_corrected = corrected_time(r.setup_done_time,
                                                        r.setup_barrier_d2h_repeat_time)
            logger.info(
                f"Setup H2D (copy mode, {args.setup_order}-major, {k_strips} strip(s)): "
                f"{setup_h2d_bytes / 1e6:.1f} MB; "
                f"host-return {r.setup_h2d_time * 1e3:.1f} ms, device-done "
                f"{r.setup_done_time * 1e3:.1f} ms (barrier {r.setup_barrier_d2h_time * 1e3:.1f} ms, "
                f"idle repeat {r.setup_barrier_d2h_repeat_time * 1e3:.1f} ms) -> corrected "
                f"{r.setup_h2d_time_corrected * 1e3:.1f} ms = "
                f"{setup_h2d_bytes / max(r.setup_h2d_time_corrected, 1e-12) / 1e9:.2f} GB/s"
            )

            if mode.drops_setup_arrays:
                setup_calls = setup_arrays = None
                mode.after_setup(r)

            # ── Compute phase ──
            mode.log_spmv(r)
            mode.prepare_passes(r, d)

            if args.sync_timing:
                progress("f_sync")
                # Once per run: every PE samples the reference clock broadcast
                # from P0.0.  f_sync releases each PE's command stream itself
                # when that PE is synced.  The tsc free-runs from here on, so
                # the reference holds for every pass.
                runner.launch("f_sync", nonblock=True)

            pass_host_s, pass_h2d_s, pass_d2h_s = [], [], []
            pass_done_s, pass_barrier_d2h_s, pass_repeat_s, pass_corrected_s = [], [], [], []
            pass_barrier_s = []         # t_comp -> completion barrier returned, per pass
            pass_ratio = []             # synced wafer time / pass_barrier_s
            pass_cycles = []            # synced on-wafer cycles, one per pass
            y_passes = []
            r.pass_h2d_s = pass_h2d_s   # StreamMode.memcpy_h2d_time reads it

            wafer_reason = "sync timing off" if not args.sync_timing else "no pass read back"
            for p_i in range(args.loop_count):
                progress(f"pass {p_i + 1}/{args.loop_count} start")

                # Started BEFORE f_tic is issued, so every device stamp of the
                # pass falls inside [t_comp, return of the barrier] (see
                # wafer_host_ratio).  Unsynced, nothing precedes it: unchanged.
                t_comp = time.perf_counter()
                progress(f"pass {p_i + 1} launches start")
                t_h2d = mode.launch_pass(r, d, t_comp)

                pass_h2d_s.append(time.perf_counter() - t_h2d)
                pass_host_s.append(time.perf_counter() - t_comp)

                # Completion barrier, INSIDE the pass, so every per-pass time
                # is a device-done time.  wait_ew / f_toc can return before the
                # device has finished; y_result[0] from each column-0 PE (W 1 x
                # H) cannot come back until the column-0 PE's command stream is
                # released, i.e. after its row's EW chain, so this D2H cannot
                # complete before every PE has finished the pass.  Re-issued on
                # the now-idle device to price the barrier itself.
                progress(f"pass {p_i + 1} launches returned, barrier D2H start")
                barrier_t = timed_d2h(runner, sym_y_result, 1, H,
                                      MemcpyDataType.MEMCPY_32BIT,
                                      MemcpyOrder.ROW_MAJOR)
                t_done = time.perf_counter()
                pass_done_s.append(t_done - t_h2d)
                # t_comp (before f_tic) -> barrier returned: every device stamp
                # of the pass falls inside it, and the barrier cannot return
                # before the pass is done, so it is the host upper bound on the
                # pass (wafer_host_ratio).  Taken before the repeat D2H.
                pass_barrier_s.append(t_done - t_comp)
                pass_barrier_d2h_s.append(barrier_t)
                pass_repeat_s.append(timed_d2h(runner, sym_y_result, 1, H,
                                               MemcpyDataType.MEMCPY_32BIT,
                                               MemcpyOrder.ROW_MAJOR))
                pass_corrected_s.append(corrected_time(pass_done_s[-1],
                                                       pass_repeat_s[-1]))

                # ── D2H: read y_result from column 0 (EW-reduced), every pass ──
                t = time.perf_counter()
                y_flat = np.zeros(H * KY, dtype=np.float32)
                runner.memcpy_d2h(y_flat, sym_y_result, 0, 0, 1, H, KY,
                                  streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                                  order=MemcpyOrder.ROW_MAJOR, nonblock=False)
                pass_d2h_s.append(time.perf_counter() - t)
                y_passes.append(y_flat)
                progress(f"pass {p_i + 1} y read back, sync_pack next")

                wafer_msg = ""
                if args.sync_timing:
                    try:
                        sync_pack = np.zeros(W * H * 5, dtype=np.float32)
                        runner.memcpy_d2h(sync_pack, sym_sync_pack, 0, 0, W, H, 5,
                                          streaming=False,
                                          data_type=MemcpyDataType.MEMCPY_32BIT,
                                          order=MemcpyOrder.ROW_MAJOR, nonblock=False)
                        cycles = synced_pass_cycles(*unpack_sync_pack(sync_pack, H, W))
                        pass_cycles.append(cycles)
                        ratio = wafer_host_ratio(cycles, pass_barrier_s[-1])
                        pass_ratio.append(ratio)
                        wafer_msg = (f", on-wafer {cycles:,} cycles = "
                                     f"{cycles / WSE_CLOCK_HZ * 1e3:.3f} ms "
                                     f"({ratio:.2f} of host-through-barrier "
                                     f"{pass_barrier_s[-1] * 1e3:.3f} ms)")
                        if ratio > 1.0:
                            logger.warning(
                                f"pass {p_i + 1}: synced on-wafer time EXCEEDS the "
                                f"host time that must contain it ({ratio:.2f}x) -- "
                                f"tsc rate or stamp placement is wrong; treat "
                                f"wafer_* as untrustworthy")
                    except Exception:
                        wafer_reason = f"pass {p_i + 1}: sync_pack readback failed"
                        logger.warning(f"{wafer_reason}\n{traceback.format_exc()}")
                progress(f"pass {p_i + 1}/{args.loop_count} end")
                logger.info(f"Pass {p_i + 1}/{args.loop_count}: host-return "
                            f"{pass_host_s[-1] * 1e3:.3f} ms, device-done "
                            f"{pass_done_s[-1] * 1e3:.3f} ms (corrected "
                            f"{pass_corrected_s[-1] * 1e3:.3f} ms){wafer_msg}")

            if args.sync_timing and len(pass_cycles) == args.loop_count:
                wafer_reason = "ok"
            if args.sync_timing and wafer_reason != "ok":
                logger.warning(f"wafer_cycles_* / wafer_time_* RECORDED AS 0 — "
                               f"reason: {wafer_reason}")
                pass_cycles = []

            r.pass_host_s, r.pass_done_s = pass_host_s, pass_done_s
            r.pass_barrier_d2h_s, r.pass_repeat_s = pass_barrier_d2h_s, pass_repeat_s
            r.pass_corrected_s, r.pass_barrier_s = pass_corrected_s, pass_barrier_s
            r.pass_ratio, r.pass_cycles, r.y_passes = pass_ratio, pass_cycles, y_passes

            # loop_count == 1 reproduces the single-pass figures exactly.
            # memcpy_h2d_time / compute_time_s stay the host-return times (a
            # lower bound, comparable with old rows); the device-done times
            # are the means of the per-pass barrier-closed brackets.
            r.memcpy_h2d_time = mode.memcpy_h2d_time(r)
            r.compute_time_s = float(np.mean(pass_host_s))
            r.memcpy_d2h_time = float(np.mean(pass_d2h_s))
            r.device_done_time = float(np.mean(pass_done_s))
            r.barrier_d2h_time = float(np.mean(pass_barrier_d2h_s))
            r.barrier_d2h_repeat_time = float(np.mean(pass_repeat_s))
            r.stream_time_corrected = float(np.mean(pass_corrected_s))
            mode.log_pass_summary(r)
            logger.info(f"D2H: {r.memcpy_d2h_time:.3f}s")

            read_cycle_stats(r, runner, sym_tsc_pack, MemcpyDataType, MemcpyOrder)

    except Exception as e:
        logger.error(f"Runtime execution failed: {e}")
        logger.error(traceback.format_exc())
        return 1
    return None


def read_cycle_stats(r, runner, sym_tsc_pack, MemcpyDataType, MemcpyOrder) -> None:
    """D2H: per-PE cycle timestamps (all PEs), into r.cyc_*.

    Each PE's chain_done packed two 48-bit tsc counts into 3 f32:
    `start` stamped on the FIRST streamed A word to reach that PE,
    `end` when its EW chain finished.  The span end-start therefore
    spans stream consumption + the EW reduction and INCLUDES per-word
    H2D stalls and the skew from streaming columns one after another —
    it is NOT a pure on-wafer compute figure.  Its value is the
    per-PE *distribution*: max-vs-min is the EW pipeline-vs-serialise
    signal that the single host "compute+barrier" number cannot show.

    A readback problem must never sink an otherwise good wafer run, but
    it must never be silent either: every path that leaves
    compute_hw_cycles_* at 0 logs WARNING with a concrete reason.  (At
    65,536 PEs this section produced no output at all and recorded
    zeros — the launcher buffers the whole worker log, so a lone
    warning is easy to miss; hence the loud, greppable final line.)
    """
    W, H = r.W, r.H
    compute_time_s = r.compute_time_s
    stream_time_corrected = r.stream_time_corrected
    pass_barrier_s = r.pass_barrier_s
    cyc_reason = "cycle readback not reached"
    try:
        tsc_pack = None
        n_words_ts = W * H * 3
        logger.info(
            f"D2H: reading per-PE cycle timestamps "
            f"(tsc_pack, {W}x{H} rectangle, {n_words_ts:,} f32 = "
            f"{n_words_ts * 4 / 1e6:.1f} MB)..."
        )
        try:
            tsc_pack = np.zeros(n_words_ts, dtype=np.float32)
            runner.memcpy_d2h(tsc_pack, sym_tsc_pack, 0, 0, W, H, 3,
                              streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                              order=MemcpyOrder.ROW_MAJOR, nonblock=False)
        except Exception:
            tsc_pack = None
            cyc_reason = f"tsc_pack D2H raised for the {W}x{H} rectangle"
            logger.warning(f"{cyc_reason}\n{traceback.format_exc()}")

        if tsc_pack is not None:
            pe_cycles, start48 = unpack_tsc_cycles(tsc_pack, H, W)
            summ = summarise_tsc_cycles(pe_cycles, start48)
            cyc_reason = summ["reason"]
            r.cyc_recorded_pes = summ["n_stamped"]
            if summ["n_zero"]:
                logger.warning(
                    f"{summ['n_zero']:,}/{summ['n_total']:,} PEs report "
                    f"start48==0 (tsc never stamped) — excluded from "
                    f"compute_hw_cycles_*"
                )
            if cyc_reason == "ok":
                r.cyc_min, r.cyc_max = summ["min"], summ["max"]
                r.cyc_avg, r.cyc_std = summ["mean"], summ["std"]
                logger.info(
                    f"Per-PE tsc span (first streamed word -> EW chain done; "
                    f"includes H2D stalls + column-streaming skew, NOT pure "
                    f"compute): {r.cyc_min:,}-{r.cyc_max:,} cycles "
                    f"(mean {r.cyc_avg:,.0f}, std {r.cyc_std:,.0f}) over "
                    f"{r.cyc_recorded_pes:,} PEs; host-return compute+barrier "
                    f"{compute_time_s*1e3:.3f} ms, stream_time_corrected "
                    f"{stream_time_corrected*1e3:.3f} ms"
                )
                # The span (last pass) must fit in that pass's host time
                # THROUGH the completion barrier; it may legitimately exceed the
                # compute+barrier figure, which can stop early.
                span_ratio = wafer_host_ratio(r.cyc_max, pass_barrier_s[-1])
                if span_ratio > 1.0:
                    logger.warning(
                        f"per-PE tsc span max exceeds the host time through "
                        f"the barrier ({span_ratio:.2f}x): tsc rate or stamp "
                        f"placement is wrong")
                elif r.cyc_max / WSE_CLOCK_HZ > compute_time_s:
                    logger.warning(
                        f"per-PE tsc span max ({r.cyc_max / WSE_CLOCK_HZ * 1e3:.1f} ms) "
                        f"exceeds host compute+barrier ({compute_time_s * 1e3:.1f} ms) "
                        f"but fits the host time through the barrier "
                        f"({pass_barrier_s[-1] * 1e3:.1f} ms): wait_ew "
                        f"returned before the device finished, so "
                        f"compute_time_hw_* UNDER-measures this run")
            else:
                logger.warning(
                    f"per-PE cycle stats unavailable: {cyc_reason} "
                    f"(start48[:8]={summ['sample_start48']}, "
                    f"pe_cycles[:8]={summ['sample_cycles']})"
                )
    except Exception:
        cyc_reason = "unexpected exception in cycle readback"
        logger.warning(f"{cyc_reason}\n{traceback.format_exc()}")

    if cyc_reason != "ok":
        logger.warning(
            f"compute_hw_cycles_min/avg/max/std RECORDED AS 0 for this row "
            f"— reason: {cyc_reason}"
        )
