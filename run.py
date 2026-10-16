#!/usr/bin/env python3
"""SpMV entry point: single run or parameter sweep."""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

from cluster import mm_io
from cluster.sweep_config import SweepConfig
from config import (
    DEFAULT_CHANNELS,
    DEFAULT_ENCODING,
    DEFAULT_H2D_MODE,
    DEFAULT_MEMCPY_ORDER,
    DEFAULT_SETUP_ORDER,
    DEFAULT_WIDTH_EAST_BUF,
    DEFAULT_WIDTH_WEST_BUF,
    ENCODINGS,
    H2D_MODES,
    MEMCPY_ORDERS,
    RESIDENT_ENCODINGS,
    RESIDENT_LOOP_COUNT,
    SpMVConfig,
)
from sdk.build_info import resolve as resolve_build, sweep_refusal
from sdk.compiler import compile as compile_csl
from sdk.launcher import run as run_sdk

logging.basicConfig(format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def io_overrides(args) -> dict:
    """The per-run settings the user actually named on the command line.

    Covers the compile-time memcpy I/O flags and --encoding, and the runtime
    --h2d-mode / --memcpy-order.
    Anything the user did not name is left out entirely, so precedence stays
    CLI > TOML > config default.
    """
    return {
        name: getattr(args, name)
        for name in ("channels", "width_west_buf", "width_east_buf", "h2d_mode",
                     "memcpy_order", "encoding", "setup_order", "serialize_column_rects",
                     "sync_timing", "loop_count", "setup_strips", "null_spmv")
        if getattr(args, name, None) is not None
    }


def single_run_overrides(args) -> dict:
    """io_overrides for ``run``: --encoding resident without --loop-count runs
    RESIDENT_LOOP_COUNT passes (a resident SpMV takes microseconds, so one pass says
    nothing about the spread).  Sweeps keep their TOML's loop_count."""
    overrides = io_overrides(args)
    if overrides.get("encoding") in RESIDENT_ENCODINGS:
        overrides.setdefault("loop_count", RESIDENT_LOOP_COUNT)
    return overrides


def default_results_path(cfg: SpMVConfig) -> Path:
    """Auto-generated CSV path for a single run; simulator results are kept apart."""
    prefix = "sim_single" if cfg.simulator else "single"
    if cfg.mtx_file is not None:
        return Path(f"cluster/results/data/{prefix}_{Path(cfg.mtx_file).stem}.csv")
    return Path(f"cluster/results/data/{prefix}_{cfg.rows}x{cfg.cols}.csv")


def grid_options(grid_height, simulator: bool):
    """Candidate grids for the "smallest grid that fits" hint: 16 PEs up to the full wafer, near-square."""
    pes = [16 * 4 ** k for k in range(0, 9) if 16 * 4 ** k < 857_010] + [857_010]
    out = []
    for p in pes:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = SpMVConfig(rows=64, cols=64, density=0.1, num_pes_target=p, grid_height_target=grid_height,
                               simulator=simulator)
        except ValueError:
            continue
        out.append((c.width, c.height))
    return out


def size_from_mtx(args, overrides: dict):
    """Rows, cols, density and the exact TOTAL_NNZ from the matrix file, or None after logging why not.

    Reads the whole file on the login side (numpy only), counts the NNZ of every PE on the grid this run
    would use, and refuses BEFORE anything is compiled when the busiest PE does not fit its SRAM, saying what
    it needs and the smallest grid that would hold it."""
    path = Path(args.mtx_file)
    try:
        mm = mm_io.read_matrix_market(path)
    except (ValueError, OSError) as e:
        logger.error(f"--mtx-file: {e}")
        return None
    for name, given, have in (("--rows", args.rows, mm.rows), ("--cols", args.cols, mm.cols)):
        if given is not None and given != have:
            logger.error(f"{name} {given} contradicts {path.name} ({mm.rows}x{mm.cols}); drop it, the file decides")
            return None
    if args.density is not None:
        logger.warning("--density is ignored with --mtx-file (the file's NNZ decides)")

    encoding = overrides.get("encoding", DEFAULT_ENCODING)
    sync = bool(overrides.get("sync_timing", False))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe = SpMVConfig(rows=64, cols=64, density=0.1, num_pes_target=args.num_pes,
                               grid_height_target=args.grid_height, simulator=args.simulator, sync_timing=sync)
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return None
    W, H = probe.width, probe.height
    counts = mm_io.tile_nnz_counts(mm.row, mm.col, mm.rows, mm.cols, W, H)
    rep = mm_io.fit_report(counts, mm.rows, mm.cols, sync, encoding)
    logger.info(f"{path.name}: {mm.rows}x{mm.cols}, {mm.stored_entries:,} stored entries ({mm.symmetry}), "
                f"NNZ={mm.nnz:,}; grid {W}x{H}: NNZ per PE mean {rep.mean:.1f}, max {rep.max} "
                f"({rep.max / rep.mean if rep.mean else 0:.1f}x), {rep.words * 4 / 1024:.1f} KB per PE")
    if not rep.fits:
        better = mm_io.smallest_fit(mm.row, mm.col, mm.rows, mm.cols, grid_options(args.grid_height, args.simulator),
                                    sync, encoding)
        hint = (f"; the smallest grid that fits is {better[0][0]}x{better[0][1]} = {better[0][0] * better[0][1]:,} PEs "
                f"(--num-pes {better[0][0] * better[0][1]})" if better else
                "; no grid up to the full wafer holds it with the block tiling")
        logger.error(f"{path.name} does not fit on {W}x{H} = {W * H:,} PEs with --encoding {encoding}: "
                     f"{rep.reason}{hint}. Nothing was compiled.")
        return None
    return dict(rows=mm.rows, cols=mm.cols, density=mm.nnz / (mm.rows * mm.cols),
                total_nnz_per_pe=max(rep.max, 1), mtx_file=path)


def run_one(cfg: SpMVConfig, results_path: Path) -> bool:
    logger.info(f"Grid: {cfg.width}x{cfg.height} = {cfg.num_pes:,} PEs  "
                f"matrix: {cfg.rows}x{cfg.cols} density={cfg.density}")
    try:
        compile_csl(cfg)
    except Exception:
        logger.exception("Compilation failed")
        return False
    try:
        run_sdk(cfg, download_path=results_path)
    except Exception:
        logger.exception("Execution failed")
        return False
    return True


def cmd_run(args) -> int:
    overrides = single_run_overrides(args)
    if args.mtx_file:
        sized = size_from_mtx(args, overrides)
        if sized is None:
            return 1
    elif None in (args.rows, args.cols, args.density):
        logger.error("give --rows, --cols and --density, or --mtx-file")
        return 1
    else:
        sized = dict(rows=args.rows, cols=args.cols, density=args.density)
    try:
        cfg = SpMVConfig(
            **sized,
            seed=args.seed,
            num_pes_target=args.num_pes,
            grid_height_target=args.grid_height,
            out_path=Path(args.work_dir),
            comment=args.comment,
            simulator=args.simulator,
            **overrides,
        )
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return 1

    results = Path(args.output) if args.output else default_results_path(cfg)
    results.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_path.mkdir(parents=True, exist_ok=True)
    return 0 if run_one(cfg, results) else 1


def cmd_sweep(
    toml_paths: list[Path],
    start_index: int = 0,
    max_runs: Optional[int] = None,
    simulator: bool = False,
    io_override: Optional[dict] = None,
    allow_dirty: bool = False,
) -> int:
    if not simulator:
        # Hardware sweep rows are the data of record: each must name a commit and a clean tree.
        refusal = sweep_refusal(resolve_build(), allow_dirty=allow_dirty)
        if refusal:
            logger.error(f"Refusing to start a hardware sweep: {refusal}")
            return 1
    configs: list[tuple[SpMVConfig, Path]] = []
    for toml_path in toml_paths:
        try:
            sweep = SweepConfig.from_toml(toml_path, cli_override_simulator=simulator,
                                          cli_override_io=io_override)
        except ValueError as e:
            logger.error(f"{toml_path}: {e}")
            return 1
        results = sweep.results_path()
        logger.info(f"Loaded {toml_path.name} -> {results}")
        try:
            for cfg in sweep.expand():
                configs.append((cfg, results))
        except ValueError as e:
            logger.error(f"{toml_path}: {e}")
            return 1

    configs = configs[start_index:]
    if max_runs is not None:
        configs = configs[:max_runs]

    if start_index > 0:
        logger.info(f"Resuming from index {start_index}, {len(configs)} runs remaining")

    total = len(configs)
    succeeded = failed = 0
    t0 = time.time()

    for i, (cfg, results_path) in enumerate(configs, start=start_index):
        logger.info("=" * 72)
        logger.info(f"Run {i + 1}/{start_index + total}: {cfg.comment}")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.out_path.mkdir(parents=True, exist_ok=True)

        if run_one(cfg, results_path):
            succeeded += 1
        else:
            failed += 1
            logger.error(f"Run {i + 1} failed")

        done = i - start_index + 1
        elapsed = time.time() - t0
        eta = (elapsed / done) * (total - done)
        logger.info(f"Progress: {done}/{total} | elapsed={elapsed/60:.1f}m | ETA={eta/60:.1f}m")

    logger.info("=" * 72)
    logger.info(f"DONE: {succeeded}/{total} succeeded in {(time.time()-t0)/60:.1f} min")
    return 0 if failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser, built separately so tests can exercise it without running."""
    # -v is accepted both before and after the subcommand.  The subparser copies
    # default to SUPPRESS so that omitting it there does not overwrite a -v
    # given at the top level.
    verbose_parent = argparse.ArgumentParser(add_help=False)
    verbose_parent.add_argument("-v", "--verbose", action="store_true",
                                default=argparse.SUPPRESS)

    # memcpy I/O tuning, shared by both subcommands.  Defaults reproduce the
    # previously hardcoded flags, so omitting them changes nothing.
    # Defaults are None so "not given" is distinguishable from an explicit
    # value: precedence is CLI > TOML > the config default.
    io_parent = argparse.ArgumentParser(add_help=False)
    io_parent.add_argument("--channels", type=int, default=None,
                           help=f"memcpy I/O channels (default: {DEFAULT_CHANNELS}). "
                                "0 compiles the legacy single-I/O-channel version; "
                                "any N >= 1 compiles the multi-channel version with "
                                "N channels. No maximum is documented; the compiler "
                                "enforces it")
    io_parent.add_argument("--width-west-buf", type=int, dest="width_west_buf", default=None,
                           help=f"width of the west buffer (default: {DEFAULT_WIDTH_WEST_BUF}, "
                                "i.e. no buffer to mitigate slow input)")
    io_parent.add_argument("--width-east-buf", type=int, dest="width_east_buf", default=None,
                           help=f"width of the east buffer (default: {DEFAULT_WIDTH_EAST_BUF}, "
                                "i.e. no buffer to mitigate slow output)")
    # Runtime, not compile-time: this one reaches the worker, never cslc.
    io_parent.add_argument("--h2d-mode", choices=list(H2D_MODES), dest="h2d_mode",
                           default=None,
                           help=f"shape of the streaming A-values H2D transfer "
                                f"(default: {DEFAULT_H2D_MODE}). 'column' sends W "
                                "one-column rectangles of exact length; 'full' sends "
                                "a single padded WxH rectangle, which is the shape "
                                "the hardware can split across I/O channels. Cross "
                                "with --channels to tell a link ceiling from a "
                                "transfer-shape artifact")
    io_parent.add_argument("--memcpy-order", choices=list(MEMCPY_ORDERS),
                           dest="memcpy_order", default=None,
                           help=f"host-side memory order of the streaming A transfer "
                                f"(default: {DEFAULT_MEMCPY_ORDER}). The SDK documents "
                                "column-major as the faster order, and Cerebras' own "
                                "bandwidth benchmark uses it for the measured transfer")
    # Compile-time: selects the kernel (pe.csl, pe_valtask.csl, pe_resident.csl) via cslc params.
    io_parent.add_argument("--encoding", choices=list(ENCODINGS), dest="encoding",
                           default=None,
                           help=f"A-values stream encoding (default: {DEFAULT_ENCODING}). "
                                "'header' is ADR-003's row-header stream consumed by a "
                                "per-word task, a comparison mode. Recompiles. "
                                "'valtask' streams 4 B/NNZ (float32 values only; "
                                "the structure is preloaded) into a per-word data "
                                "task and serialises its column transfers "
                                "automatically. "
                                "'resident' keeps A on the wafer (setup copies the "
                                "values too) and streams nothing per call; "
                                "--h2d-mode / --memcpy-order then do nothing. See "
                                "docs/val-only-encoding.md, docs/resident-spmv.md")
    io_parent.add_argument("--serialize-column-rects", action="store_true",
                           dest="serialize_column_rects", default=None,
                           help="send every streaming column rectangle but the last "
                                "with nonblock=False, so only one streaming memcpy is "
                                "in flight at a time. Automatic for --encoding valtask "
                                "with --h2d-mode column (it hangs otherwise); this "
                                "opts the header kernel in. Runtime-only (no "
                                "recompile); a no-op in full mode (one rectangle)")
    # Synced timing, docs/synced-timing.md.  --sync-timing is compile-time.
    io_parent.add_argument("--sync-timing", action="store_true", dest="sync_timing",
                           default=None,
                           help="compile in the csl-examples sync module and bracket "
                                "every pass with f_tic/f_toc against a common reference "
                                "clock, reporting per-pass on-wafer time and its spread "
                                "(wafer_cycles_*, wafer_time_*). Recompiles. Needs a grid "
                                "of at least 2x2")
    io_parent.add_argument("--loop-count", type=int, dest="loop_count", default=None,
                           help="passes over the same A and x per run (default: 1; "
                                f"{RESIDENT_LOOP_COUNT} for --encoding resident). "
                                "Every pass is validated; compute_time_hw_* become "
                                "per-pass statistics")
    io_parent.add_argument("--null-spmv", action="store_true", dest="null_spmv",
                           default=None,
                           help="--encoding resident only: compile spmv() as a no-op that "
                                "releases the command stream at once, so the synced window "
                                "is the f_tic/f_toc launch machinery alone. y is never "
                                "written, so validation is skipped. Recompiles")
    io_parent.add_argument("--setup-order", choices=list(MEMCPY_ORDERS),
                           dest="setup_order", default=None,
                           help=f"memory order of the copy-mode setup transfers "
                                f"(default: {DEFAULT_SETUP_ORDER}), independent of "
                                "--memcpy-order")
    io_parent.add_argument("--setup-strips", type=int, dest="setup_strips", default=None,
                           help="split each copy-mode setup transfer into this many vertical "
                                "strips of the grid, issued one after another (default: "
                                "2 x channels; 1 = the old single rectangle). Runtime only: "
                                "no recompile. Narrow strips reach the H2D ceiling, one full "
                                "rectangle gets about half (docs/adr/ADR-008)")

    parser = argparse.ArgumentParser(
        description="SpMV on Cerebras WSE-3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: run.py sweep bench/configs/weak-scaling.toml",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- run subcommand ---
    p_run = sub.add_parser("run", parents=[verbose_parent, io_parent],
                           help="Compile and run a single configuration")
    p_run.add_argument("--rows", type=int, default=None, help="required unless --mtx-file")
    p_run.add_argument("--cols", type=int, default=None, help="required unless --mtx-file")
    p_run.add_argument("--density", type=float, default=None, help="required unless --mtx-file")
    p_run.add_argument("--mtx-file", type=Path, default=None, dest="mtx_file",
                       help="a MatrixMarket coordinate file (real/integer/pattern; symmetric files are expanded). "
                            "Rows, cols and TOTAL_NNZ come from the file: the busiest PE's NNZ is counted on the "
                            "login side, and the run is refused BEFORE the compile if it does not fit the PE's "
                            "SRAM (it says what it needs and the smallest grid that fits)")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--num-pes", type=int, default=None, dest="num_pes",
                       help="Target PE count (default: full fabric)")
    p_run.add_argument("--grid-height", type=int, default=None, dest="grid_height",
                       help="PE grid height H (default: near-square shape for --num-pes)")
    p_run.add_argument("--work-dir", default="cluster_work/out")
    p_run.add_argument("--output", default="", help="Results CSV path (default: auto)")
    p_run.add_argument("--comment", default="")
    p_run.add_argument("--simulator", action="store_true", help="Run on simulator instead of hardware")

    # --- sweep subcommand ---
    p_sweep = sub.add_parser("sweep", parents=[verbose_parent, io_parent],
                             help="Run one or more TOML sweep configs")
    p_sweep.add_argument("configs", nargs="+", type=Path, metavar="config.toml")
    p_sweep.add_argument("--start-index", type=int, default=0,
                          help="Resume from this run index (0-based across all configs)")
    p_sweep.add_argument("--max-runs", type=int, default=None)
    p_sweep.add_argument("--allow-dirty", action="store_true",
                         help="start a hardware sweep although the tree differs from its commit "
                              "(rows are then recorded git_dirty=True); an unknown commit is "
                              "always refused")
    p_sweep.add_argument("--simulator", action="store_true", help="Run on simulator instead of hardware")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)

    if args.cmd == "run":
        return cmd_run(args)

    if args.cmd == "sweep":
        for p in args.configs:
            if not p.exists():
                logger.error(f"Config not found: {p}")
                return 1
        return cmd_sweep(args.configs, start_index=args.start_index,
                         max_runs=args.max_runs, simulator=args.simulator,
                         io_override=io_overrides(args), allow_dirty=args.allow_dirty)

    return 0


if __name__ == "__main__":
    sys.exit(main())
