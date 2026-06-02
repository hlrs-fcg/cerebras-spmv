#!/usr/bin/env python3
"""
Cluster Sweep Runner

Runs benchmark sweeps by calling compile + run directly (no subprocesses).
"""

import argparse
import csv
import datetime
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from utils.logging import logger, setup_logging
from config import SpMVConfig
from sdk.compiler import compile as compile_csl
from sdk.launcher import run as run_sdk


def read_sweep_config(config_path: Path) -> List[Dict]:
    """Read sweep configuration from CSV."""
    configs = []
    with open(config_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('rows', '').startswith('#'):
                continue

            num_pes_str = row.get('num_pes', 'max').strip()
            try:
                num_pes_target = int(num_pes_str)
            except ValueError:
                num_pes_target = None  # 'max' or 'min' → full fabric
                if num_pes_str not in ('max', 'min', ''):
                    logger.warning(f"Unknown num_pes value '{num_pes_str}', using full fabric")

            configs.append({
                'rows': int(row['rows']),
                'cols': int(row['cols']),
                'density': float(row['density']),
                'seed': int(row.get('seed', 42)),
                'comment': row.get('comment', ''),
                'num_pes_target': num_pes_target,
            })
    return configs


def run_single(config: Dict, out_path: Path, output_csv: Path) -> bool:
    """Compile and run one configuration. Returns True on success."""
    cfg = SpMVConfig(
        rows=config['rows'],
        cols=config['cols'],
        density=config['density'],
        seed=config['seed'],
        out_path=out_path,
        num_pes_target=config['num_pes_target'],
        comment=config['comment'],
    )
    logger.info(f"Grid: {cfg.width}×{cfg.height} = {cfg.num_pes:,} PEs")

    try:
        compile_csl(cfg)
    except Exception:
        logger.exception("Compilation failed")
        return False

    try:
        run_sdk(cfg, download_path=output_csv)
    except Exception:
        logger.exception("Execution failed")
        return False

    return True


def run_sweep(
    config_path: Path,
    output_csv: Path,
    work_dir: Path = Path('cluster_work/sweeps'),
    start_index: int = 0,
    max_runs: Optional[int] = None,
):
    configs = read_sweep_config(config_path)
    logger.info(f"Loaded {len(configs)} configurations from {config_path}")

    configs = configs[start_index:]
    if max_runs is not None:
        configs = configs[:max_runs]
    if start_index > 0:
        logger.info(f"Resuming from index {start_index}: {len(configs)} configs remaining")

    work_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    total = len(configs)
    succeeded = failed = 0
    sweep_start = time.time()

    for i, config in enumerate(configs, start=start_index):
        logger.info("=" * 80)
        logger.info(f"Run {i + 1}/{start_index + total}: {config['comment']}")
        logger.info(f"  {config['rows']}×{config['cols']}, density={config['density']}, "
                    f"num_pes_target={config['num_pes_target'] or 'full fabric'}")

        if run_single(config, out_path=work_dir / f"out_{i:04d}", output_csv=output_csv):
            succeeded += 1
            logger.info(f"Run {i + 1} succeeded")
        else:
            failed += 1
            logger.error(f"Run {i + 1} failed")

        done = i - start_index + 1
        elapsed = time.time() - sweep_start
        avg = elapsed / done
        eta = avg * (total - done)
        logger.info(f"Progress: {done}/{total} | elapsed={elapsed/60:.1f}m | ETA={eta/60:.1f}m")

    total_time = time.time() - sweep_start
    logger.info("=" * 80)
    logger.info(f"SWEEP COMPLETE — {succeeded}/{total} succeeded in {total_time/60:.1f} min")
    logger.info(f"Results: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description='Run benchmark sweep on cluster')
    parser.add_argument('config', help='Path to sweep configuration CSV')
    parser.add_argument('--output', type=str, default='',
                        help='Output benchmark CSV (default: auto-generated from date + config name)')
    parser.add_argument('--work-dir', type=str, default='cluster_work/sweeps',
                        help='Working directory for artifacts')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Resume from this configuration index')
    parser.add_argument('--max-runs', type=int, default=None,
                        help='Stop after this many runs (for testing)')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    setup_logging(level="DEBUG" if args.verbose else "INFO")

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return 1

    if not args.output:
        date = datetime.date.today().isoformat()
        args.output = f"cluster/results/data/{date}_{config_path.stem}.csv"
        logger.info(f"Output: {args.output}")

    run_sweep(
        config_path=config_path,
        output_csv=Path(args.output),
        work_dir=Path(args.work_dir),
        start_index=args.start_index,
        max_runs=args.max_runs,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
