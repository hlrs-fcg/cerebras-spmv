"""Streaming mode (header / val / valtask): A is streamed over MEMCPYH2D_DATA_1 on every pass.
Staged flat next to worker_runtime.py."""

import logging
import time

import numpy as np

from worker_layout import _flatten_whl, rect_nonblock_flags
from worker_modes import Mode

logger = logging.getLogger(__name__)


class StreamMode(Mode):
    def extra_symbol_names(self, r):
        # Exported by pe_valtask.csl only (layout.csl gates the @export_name).
        return ["val_meta"] if r.val_only else []

    def setup_extra_bytes(self, r):
        return 4 * r.W * r.H * (1 if r.val_only else 0)

    def setup_extra_arrays(self, r, d):
        if r.val_only:
            # val_meta[0] = words this PE receives per pass.  pe_words is indexed
            # pe = py*W + px like ci / rp, so it takes the same flatten (l = 1).  A wrong
            # layout here hangs the PE: it waits for exactly that many words.
            return [(d.syms["val_meta"], r.val_plan.pe_words, 1)]
        return []

    def log_spmv(self, r):
        args = r.args
        logger.info(
            f"SpMV: launch → stream [{args.encoding} encoding, {args.h2d_mode} mode, "
            f"{args.memcpy_order}-major] "
            f"({r.W} cols × {r.H} rows, {r.total_stream_words:,} words/row-of-grid) "
            f"→ wait_ew, {args.loop_count} pass(es)"
            f"{', synced timing' if args.sync_timing else ''}"
        )

    def prepare_passes(self, r, d):
        # Stream A values via MEMCPYH2D_DATA_1.  Both modes rely on the same
        # property: every PE's stream is 0-terminated and the kernel's
        # `draining` flag swallows anything after the sentinel, so a column
        # padded to a longer common length delivers only harmless trailing
        # zeros.
        self.order_enum = (d.MemcpyOrder.COL_MAJOR if r.args.memcpy_order == "col"
                           else d.MemcpyOrder.ROW_MAJOR)

    def send_stream(self, r, d):
        args, W, H, runner, MemcpyDataType = r.args, r.W, r.H, d.runner, d.MemcpyDataType
        order_enum = self.order_enum
        MEMCPYH2D_DATA_1 = r.MEMCPYH2D_DATA_1
        if r.val_only:
            # Exactly the plan, nothing else: each rect delivers n_words to
            # every PE in columns x0..x0+width-1, and val_meta was built by
            # summing these same rects.  Zero-length columns are already
            # absent from the plan, and their PEs were told 0.  In column
            # mode serialize_rects is always on (see above); full mode is
            # one rect, so there it changes nothing.
            nonblock = rect_nonblock_flags(len(r.val_plan.rects), r.serialize_rects)
            for (x0, width, n_words), nb in zip(r.val_plan.rects, nonblock):
                rect_data = _flatten_whl(r.val_buf[x0:x0 + width, :, :n_words],
                                         args.memcpy_order)
                runner.memcpy_h2d(MEMCPYH2D_DATA_1, rect_data,
                                  x0, 0, width, H, n_words,
                                  streaming=True, data_type=MemcpyDataType.MEMCPY_32BIT,
                                  order=order_enum, nonblock=nb)
        elif args.h2d_mode == "column":
            # W separate 1-wide x H-tall rectangles, each its exact length.
            # Order still matters: with w == 1, ROW_MAJOR runs l fastest while
            # COL_MAJOR runs h fastest, so the two need different host layouts.
            # Overlapping (all nonblock=True) unless --serialize-column-rects:
            # the header kernel never hung this way, so its default is the
            # shape every recorded header result was measured under.
            nonblock = rect_nonblock_flags(W, r.serialize_rects)
            for px, nb in zip(range(W), nonblock):
                n_words  = r.col_words[px]
                col_data = _flatten_whl(r.stream_buf[px:px + 1, :, :n_words],
                                        args.memcpy_order)
                runner.memcpy_h2d(MEMCPYH2D_DATA_1, col_data,
                                  px, 0, 1, H, n_words,
                                  streaming=True, data_type=MemcpyDataType.MEMCPY_32BIT,
                                  order=order_enum, nonblock=nb)
        else:
            # One WxH rectangle, every column padded to common_words.
            full_data = _flatten_whl(r.stream_buf, args.memcpy_order)
            logger.info(
                f"H2D full mode: one {W}x{H} rectangle, {r.common_words:,} words/PE "
                f"({full_data.nbytes / 1e6:.1f} MB), order={args.memcpy_order}"
            )
            runner.memcpy_h2d(MEMCPYH2D_DATA_1, full_data,
                              0, 0, W, H, r.common_words,
                              streaming=True, data_type=MemcpyDataType.MEMCPY_32BIT,
                              order=order_enum, nonblock=True)

    def launch_pass(self, r, d, t_comp):
        runner, args = d.runner, r.args
        if args.sync_timing:
            runner.launch("f_tic", nonblock=True)

        # Per-pass gate: clears y_result, resets the state machine, unblocks
        # the data task, and releases the command stream straight away — the
        # streaming H2D below carries the sentinel this pass waits for, so it
        # must not be queued behind a blocked command stream.
        runner.launch("spmv", nonblock=True)

        t_h2d = time.perf_counter()
        self.send_stream(r, d)

        # Barrier: wait_ew blocks each PE's command stream until its EW
        # chain has completed.  Unsynced, the host blocks on it.  Synced,
        # it is queued nonblocking and the host blocks on f_toc instead,
        # so each PE stamps toc the moment its own wait_ew releases --
        # blocking on wait_ew first would put a host round trip inside
        # the bracket (bandwidth-test: transfers nonblocking, toc blocking).
        if args.sync_timing:
            runner.launch("wait_ew", nonblock=True)
            runner.launch("f_toc", nonblock=False)
        else:
            runner.launch("wait_ew", nonblock=False)
        return t_h2d

    def memcpy_h2d_time(self, r):
        return float(np.mean(r.pass_h2d_s))

    def log_pass_summary(self, r):
        logger.info(
            f"Stream+compute: host-return {r.memcpy_h2d_time * 1e3:.1f} ms "
            f"(memcpy_h2d_time), device-done {r.device_done_time * 1e3:.1f} ms "
            f"(barrier {r.barrier_d2h_time * 1e3:.1f} ms, idle repeat "
            f"{r.barrier_d2h_repeat_time * 1e3:.1f} ms) -> stream_time_corrected "
            f"{r.stream_time_corrected * 1e3:.1f} ms"
            f"{' (mean of %d passes)' % r.args.loop_count if r.args.loop_count > 1 else ''}"
        )
        if r.stream_time_corrected > 1.5 * r.memcpy_h2d_time:
            logger.warning(
                f"host-return time understates the pass "
                f"{r.stream_time_corrected / max(r.memcpy_h2d_time, 1e-12):.1f}x: use "
                f"stream_time_corrected for streaming rates")
