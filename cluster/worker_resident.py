"""Resident-A mode (pe_resident.csl, docs/resident-spmv.md): A is copied to the wafer once, a pass is
f_tic / spmv / f_toc launches and streams nothing.  Staged flat next to worker_runtime.py."""

import logging

from worker_modes import Mode

logger = logging.getLogger(__name__)


def resident_pass_commands(sync_timing: bool) -> list[tuple[str, bool]]:
    """(exported function, nonblock) launched, in order, for ONE resident SpMV.

    spmv() holds the command stream until this PE's pass is done (there is no wait_ew),
    so whatever is queued behind it runs after the PE finished.  Synced: f_tic and spmv
    are queued nonblocking and the host blocks on f_toc, which stamps right behind the
    PE's own pass (the streaming kernels' pattern).  Unsynced: one blocking launch.
    """
    if sync_timing:
        return [("f_tic", True), ("spmv", True), ("f_toc", False)]
    return [("spmv", False)]


class ResidentMode(Mode):
    drops_setup_arrays = True

    def extra_symbol_names(self, r):
        # vals and row_ids: exported by pe_resident.csl only; the setup copy fills them.
        return ["vals", "row_ids"]

    def setup_extra_bytes(self, r):
        return 2 * 4 * r.W * r.H * r.max_nnz        # vals[] and row_ids[], copied once like col_indices

    def setup_extra_arrays(self, r, d):
        # vals[k] pairs with col_indices[k]: same PE-major (W*H, max_nnz) layout,
        # so the same flatten.  row_ptrs stays last: the setup barrier reads it.
        return [(d.syms["vals"], r.vals_grid, r.max_nnz),
                (d.syms["row_ids"], r.row_ids_grid, r.max_nnz)]

    def after_setup(self, r):
        # The setup copies are on the wafer now.  At the full wafer they are ~8 GB
        # of host arrays (x replicated H times, col_indices, vals, row_ptrs and
        # their flattened strips), so drop them.
        r.ci = r.rp = r.vals_grid = r.row_ids_grid = r.x_grid = r.val_buf = None
        # hb holds the same arrays: drop its references too, or nothing is freed
        r.hb.ci = r.hb.rp = r.hb.x_grid = r.hb.vals_grid = r.hb.val_buf = None

    def log_spmv(self, r):
        args = r.args
        logger.info(
            f"SpMV: A resident, per pass "
            f"{' -> '.join(n for n, _ in resident_pass_commands(args.sync_timing))}; "
            f"{args.loop_count} pass(es)"
            f"{', synced timing' if args.sync_timing else ''}"
        )

    def launch_pass(self, r, d, t_comp):
        # No stream: the whole pass is the launches.  spmv() holds each PE's
        # command stream until that PE's pass is done, f_toc stamps behind it.
        t_h2d = t_comp
        for fn_name, nb in resident_pass_commands(r.args.sync_timing):
            d.runner.launch(fn_name, nonblock=nb)
        return t_h2d

    def memcpy_h2d_time(self, r):
        # Nothing is streamed, so there is no stream H2D time (pass_h2d_s would be the launch
        # time, which compute_time_hw_* already is).
        return 0.0

    def log_pass_summary(self, r):
        logger.info(
            f"Resident SpMV (A not streamed): host-return {r.compute_time_s * 1e6:.1f} us, "
            f"device-done {r.device_done_time * 1e6:.1f} us (barrier "
            f"{r.barrier_d2h_time * 1e6:.1f} us, idle repeat "
            f"{r.barrier_d2h_repeat_time * 1e6:.1f} us) -> barrier-closed "
            f"{r.stream_time_corrected * 1e6:.1f} us per pass"
            f"{' (mean of %d passes)' % r.args.loop_count if r.args.loop_count > 1 else ''}"
        )
