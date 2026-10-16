"""The worker's command line (staged flat next to worker_runtime.py).

Every flag here is passed by sdk/launcher.py:build_worker_command; tests/test_worker_staging.py pins
the two lists against each other."""

import argparse

from coo_tiling import ENCODINGS, RESIDENT_ENCODINGS


def parse_args(argv=None):
    """The parsed worker arguments, with the cross-flag checks applied (exits via parser.error)."""
    parser = argparse.ArgumentParser(
        description="SpMV Worker Runtime (runs on worker node via SdkLauncher)"
    )
    parser.add_argument("--name",     required=True)
    # Omitted on simulator runs: SdkLauncher(simulator=True) refuses a command
    # containing %CMADDR%, and SdkRuntime(cmaddr=None) selects the simulator.
    parser.add_argument("--cmaddr",   default=None)
    parser.add_argument("--rows",     type=int, required=True)
    parser.add_argument("--cols",     type=int, required=True)
    parser.add_argument("--density",  type=float)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--mtx-file", type=str)
    parser.add_argument("--num-pes",       type=int, required=True)
    parser.add_argument("--grid-width",    type=int, required=True)
    parser.add_argument("--grid-height",   type=int, required=True)
    parser.add_argument("--arch",          type=str, default="wse3", choices=["wse2", "wse3"])
    parser.add_argument("--total-nnz-per-pe", type=int, required=True)
    parser.add_argument("--num-stripes",   type=int, default=1)
    parser.add_argument(
        "--h2d-mode", choices=["column", "full"], default="column",
        help="Shape of the streaming A-values H2D transfer. 'column' (default, "
             "unchanged): W separate memcpy_h2d calls, one per PE column, each a "
             "1-wide x H-tall rectangle of that column's exact length. 'full': a "
             "single memcpy_h2d over the whole WxH rectangle, every column padded "
             "to max(col_words) — a few percent more bytes, but a shape the "
             "hardware can split across I/O channels. Padding is safe because "
             "each PE stream carries an explicit 0 sentinel and the kernel drains "
             "trailing padding (src/csl/pe.csl: stream_word 'draining').",
    )
    parser.add_argument(
        "--memcpy-order", choices=["row", "col"], default="row",
        help="Host-side memory order of the streaming A transfer. 'row' (default, "
             "unchanged) is MemcpyOrder.ROW_MAJOR; 'col' is COL_MAJOR, which the SDK "
             "guide documents as the faster order and which Cerebras' own bandwidth "
             "benchmark uses for the transfer it measures. See _flatten_whl() for the "
             "host layout each one needs -- the guide's prose about which axis is "
             "fastest is wrong; the shipped example is authoritative.",
    )
    parser.add_argument(
        "--encoding", choices=list(ENCODINGS), default="valtask",
        help="A-values stream encoding; MUST match the ENCODING the artifact was "
             "compiled with (checked against out.json). 'resident': nothing is "
             "streamed per pass; setup also copies the values and a pass is one "
             "launch (pe_resident.csl), so --h2d-mode / --memcpy-order are ignored. "
             "'header': row-header stream, pe.csl. 'valtask' (default): float32 values only, "
             "stopped by count (val_meta, preloaded) rather than on a sentinel, so every "
             "transfer must deliver exactly the planned word count "
             "(coo_tiling.plan_val_transfers); consumed by a per-word data task "
             "(pe_valtask.csl). valtask serialises its column rects.",
    )
    parser.add_argument(
        "--setup-order", choices=["row", "col"], default="col",
        help="Memory order of the three copy-mode setup transfers (x_vec, "
             "col_indices, row_ptrs). Independent of --memcpy-order, so col-major "
             "copy mode can be measured while streaming stays row-major.",
    )
    parser.add_argument(
        "--setup-strips", type=int, default=1,
        help="Split each copy-mode setup transfer (x_vec, col_indices, val_meta, row_ptrs) "
             "into this many vertical strips of the grid, issued one after another (all but "
             "the last nonblock=False). 1 (default here; sdk/launcher.py always passes the "
             "resolved value, 2 x channels unless set) is the old single rectangle. Narrow "
             "strips reach the H2D ceiling, one full rectangle about half (ADR-008 rule 2). "
             "Clamped to the grid width.",
    )
    parser.add_argument(
        "--serialize-column-rects", action="store_true",
        help="Issue every streaming column rect but the last with nonblock=False, "
             "so at most one streaming memcpy is in flight; the last stays "
             "nonblock=True so wait_ew still follows it. Automatic for --encoding "
             "val/valtask with --h2d-mode column (coo_tiling.column_rects_serialized); "
             "this opts the header kernel in.",
    )
    parser.add_argument(
        "--sync-timing", action="store_true",
        help="Bracket every pass with f_tic/f_toc against the sync module's "
             "common reference clock (needs an artifact compiled with "
             "SYNC_TIMING:1; checked against out.json). docs/synced-timing.md",
    )
    parser.add_argument(
        "--loop-count", type=int, default=1,
        help="Passes over the same A and x (default 1; run.py makes it 20 for "
             "--encoding resident). Every pass is validated.",
    )
    parser.add_argument(
        "--null-spmv", action="store_true",
        help="--encoding resident only; MUST match the artifact's NULL_SPMV. spmv() is a "
             "no-op that releases the command stream at once, so the synced window is the "
             "f_tic/f_toc launch machinery alone (the floor under every resident window). "
             "y is never written: validation is skipped. docs/resident-spmv.md",
    )
    parser.add_argument(
        "--progress-file", type=str, default="progress.log",
        help="Phase log, one fsynced line per phase (setup, each pass's refresh / launch / "
             "barrier / readback). sdk/launcher.py downloads it even when the run raises, "
             "so a hang names its phase. '' disables it.",
    )
    parser.add_argument("--save-benchmark",   type=str, default="benchmark_results.csv")
    parser.add_argument("--vendor", type=str, default=None,
                        help="staged zip of numba + llvmlite for this Python (already applied "
                             "at import by worker_env.activate_vendor; accepted here so the flag "
                             "is valid)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--sim-trace", action="store_true",
        help="Simulator only: produce simfab traces (suppress_simfab_trace=False) "
             "so a hang can be decoded per PE. Never passed on hardware.",
    )
    args = parser.parse_args(argv)
    if args.loop_count < 1:
        parser.error(f"--loop-count must be >= 1, got {args.loop_count}")
    resident = args.encoding in RESIDENT_ENCODINGS
    if args.null_spmv and not resident:
        parser.error("--null-spmv needs --encoding resident")
    return args
