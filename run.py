#!/usr/bin/env python3
"""SpMV entry point: single run or parameter sweep."""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from cluster.sweep_config import SweepConfig
from config import SpMVConfig
from sdk.compiler import compile as compile_csl
from sdk.launcher import run as run_sdk

logging.basicConfig(format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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
    cfg = SpMVConfig(
        rows=args.rows,
        cols=args.cols,
        density=args.density,
        seed=args.seed,
        num_pes_target=args.num_pes,
        out_path=Path(args.work_dir),
        comment=args.comment,
    )
    results = Path(args.output) if args.output else Path(f"cluster/results/data/single_{cfg.rows}x{cfg.cols}.csv")
    results.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_path.mkdir(parents=True, exist_ok=True)
    return 0 if run_one(cfg, results) else 1


def cmd_sweep(
    toml_paths: list[Path],
    start_index: int = 0,
    max_runs: Optional[int] = None,
) -> int:
    configs: list[tuple[SpMVConfig, Path]] = []
    for toml_path in toml_paths:
        sweep = SweepConfig.from_toml(toml_path)
        results = sweep.results_path()
        logger.info(f"Loaded {toml_path.name} -> {results}")
        for cfg in sweep.expand():
            configs.append((cfg, results))

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SpMV on Cerebras WSE-3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: run.py sweep bench/configs/weak-scaling.toml",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- run subcommand ---
    p_run = sub.add_parser("run", help="Compile and run a single configuration")
    p_run.add_argument("--rows", type=int, required=True)
    p_run.add_argument("--cols", type=int, required=True)
    p_run.add_argument("--density", type=float, required=True)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--num-pes", type=int, default=None, dest="num_pes",
                       help="Target PE count (default: full fabric)")
    p_run.add_argument("--work-dir", default="cluster_work/out")
    p_run.add_argument("--output", default="", help="Results CSV path (default: auto)")
    p_run.add_argument("--comment", default="")

    # --- sweep subcommand ---
    p_sweep = sub.add_parser("sweep", help="Run one or more TOML sweep configs")
    p_sweep.add_argument("configs", nargs="+", type=Path, metavar="config.toml")
    p_sweep.add_argument("--start-index", type=int, default=0,
                         help="Resume from this run index (0-based across all configs)")
    p_sweep.add_argument("--max-runs", type=int, default=None)

    args = parser.parse_args()
    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)

    if args.cmd == "run":
        return cmd_run(args)

    if args.cmd == "sweep":
        for p in args.configs:
            if not p.exists():
                logger.error(f"Config not found: {p}")
                return 1
        return cmd_sweep(args.configs, start_index=args.start_index, max_runs=args.max_runs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
