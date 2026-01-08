#!/usr/bin/env python3
"""
Cluster Sweep Runner

Runs benchmark sweeps on the cluster using cluster_compile.py + cluster_run.py workflow.
Handles batching, result aggregation, and error recovery.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def read_sweep_config(config_path: str) -> List[Dict]:
    """Read sweep configuration from CSV."""
    configs = []
    with open(config_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip comment lines
            if row.get('rows', '').startswith('#'):
                continue

            # Convert types
            config = {
                'rows': int(row['rows']),
                'cols': int(row['cols']),
                'density': float(row['density']),
                'num_pes': row.get('num_pes', 'min'),  # Can be int or string ('min', 'max')
                'seed': int(row.get('seed', 42)),
                'comment': row.get('comment', '')
            }

            # Convert num_pes to int if it's a number
            try:
                config['num_pes'] = int(config['num_pes'])
            except ValueError:
                pass  # Keep as string ('min' or 'max')

            configs.append(config)

    return configs


def run_cluster_compile(config: Dict, artifact_path: str, out_path: str) -> bool:
    """Run cluster/compile.py for given configuration."""
    logger.debug(f"DEBUG: Compiling with artifact_path={artifact_path}, out_path={out_path}")
    cmd = [
        sys.executable,
        'cluster/compile.py',
        '--random',
        '--rows', str(config['rows']),
        '--cols', str(config['cols']),
        '--density', str(config['density']),
        '--num-pes', str(config['num_pes']),
        '--seed', str(config['seed']),
        '--artifact-path', artifact_path,
        '--out-path', out_path
    ]

    logger.info(f"Compiling: {config['comment']}")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute compile timeout
        )

        if result.returncode != 0:
            logger.error(f"Compilation failed!")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False

        logger.info("✓ Compilation successful")
        return True

    except subprocess.TimeoutExpired:
        logger.error("Compilation timed out (10 minutes)")
        return False
    except Exception as e:
        logger.error(f"Compilation error: {e}")
        return False


def run_cluster_execution(
    artifact_path: str,
    benchmark_csv: str,
    simulator: bool = False
) -> bool:
    """Run cluster/run.py to execute on cluster."""
    cmd = [
        sys.executable,
        'cluster/run.py',
        '--artifact-path', artifact_path,
        '--download-benchmark', benchmark_csv
    ]

    if simulator:
        cmd.append('--simulator')

    logger.info("Launching cluster execution...")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute execution timeout
        )

        if result.returncode != 0:
            logger.error(f"Execution failed!")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False

        logger.info("✓ Execution successful")
        return True

    except subprocess.TimeoutExpired:
        logger.error("Execution timed out (30 minutes)")
        return False
    except Exception as e:
        logger.error(f"Execution error: {e}")
        return False


def count_benchmark_entries(csv_path: str) -> int:
    """Count number of entries in benchmark CSV."""
    if not Path(csv_path).exists():
        return 0

    with open(csv_path, 'r') as f:
        return sum(1 for line in f) - 1  # Subtract header


def run_sweep(
    config_path: str,
    output_csv: str,
    work_dir: str = 'cluster_work/sweeps',
    simulator: bool = False,
    start_index: int = 0,
    max_runs: Optional[int] = None
):
    """Run complete sweep from configuration file."""

    # Read configurations
    configs = read_sweep_config(config_path)
    logger.info(f"Loaded {len(configs)} configurations from {config_path}")

    if max_runs:
        configs = configs[start_index:start_index + max_runs]
        logger.info(f"Running subset: {len(configs)} configs starting from index {start_index}")
    elif start_index > 0:
        configs = configs[start_index:]
        logger.info(f"Resuming from index {start_index}: {len(configs)} configs remaining")

    # Setup work directory
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    # Setup output CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Track statistics
    total = len(configs)
    succeeded = 0
    failed = 0
    start_time = time.time()

    # Initial benchmark count
    initial_benchmarks = count_benchmark_entries(output_csv)
    logger.info(f"Initial benchmark entries: {initial_benchmarks}")

    for i, config in enumerate(configs, start=start_index):
        logger.info("="*80)
        logger.info(f"Configuration {i+1}/{start_index + total}")
        logger.info(f"  Matrix: {config['rows']}×{config['cols']}, density={config['density']}")
        logger.info(f"  PEs: {config['num_pes']}")
        logger.info(f"  Comment: {config['comment']}")
        logger.info("="*80)

        # Setup paths for this run
        run_artifact = str(work_path / f"artifact_{i:04d}.json")
        run_out = str(work_path / f"out_{i:04d}")

        try:
            # Step 1: Compile
            compile_success = run_cluster_compile(config, run_artifact, run_out)
            if not compile_success:
                logger.error(f"❌ Run {i+1} failed at compilation")
                failed += 1
                continue

            # Step 2: Execute
            execution_success = run_cluster_execution(
                run_artifact,
                output_csv,
                simulator=simulator
            )

            if execution_success:
                succeeded += 1
                logger.info(f"✅ Run {i+1} completed successfully")

                # Verify benchmark was written
                new_count = count_benchmark_entries(output_csv)
                if new_count > initial_benchmarks + succeeded:
                    logger.warning(f"Multiple benchmarks written? Count: {new_count}")

            else:
                failed += 1
                logger.error(f"❌ Run {i+1} failed at execution")

        except Exception as e:
            failed += 1
            logger.error(f"❌ Run {i+1} failed with exception: {e}")
            import traceback
            logger.error(traceback.format_exc())

        # Progress summary
        elapsed = time.time() - start_time
        avg_time = elapsed / (i - start_index + 1)
        remaining = total - (i - start_index + 1)
        eta = avg_time * remaining

        logger.info("")
        logger.info(f"Progress: {i - start_index + 1}/{total} completed")
        logger.info(f"  Succeeded: {succeeded}, Failed: {failed}")
        logger.info(f"  Elapsed: {elapsed/60:.1f} min, Avg: {avg_time/60:.1f} min/run")
        logger.info(f"  ETA: {eta/60:.1f} min ({eta/3600:.1f} hours)")
        logger.info("")

    # Final summary
    total_time = time.time() - start_time
    logger.info("="*80)
    logger.info("SWEEP COMPLETE")
    logger.info("="*80)
    logger.info(f"Total configurations: {total}")
    logger.info(f"Succeeded: {succeeded}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {succeeded/total*100:.1f}%")
    logger.info(f"Total time: {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    logger.info(f"Average time per run: {total_time/total/60:.1f} min")
    logger.info(f"Results saved to: {output_csv}")
    logger.info("="*80)

    final_benchmarks = count_benchmark_entries(output_csv)
    logger.info(f"Final benchmark entries: {final_benchmarks} (added {final_benchmarks - initial_benchmarks})")


def main():
    parser = argparse.ArgumentParser(
        description='Run benchmark sweep on cluster using SdkLauncher workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete sweep
  python cluster/run_sweep.py bench/configs/cluster/density-sweep.csv

  # Resume from specific index
  python cluster/run_sweep.py bench/configs/cluster/pe-scaling.csv --start-index 10

  # Run only 5 configurations
  python cluster/run_sweep.py bench/configs/cluster/test.csv --max-runs 5

  # Test in simulator mode
  python cluster/run_sweep.py bench/configs/cluster/small-test.csv --simulator

  # Use custom output location
  python cluster/run_sweep.py bench/configs/cluster/my-sweep.csv \\
      --output results/experiment1.csv \\
      --work-dir work/experiment1
        """
    )

    parser.add_argument('--config',
                       help='Path to sweep configuration CSV')
    parser.add_argument('--output', type=str, default='results/data/benchmark_results.csv',
                       help='Output benchmark CSV (default: results/data/benchmark_results.csv)')
    parser.add_argument('--work-dir', type=str, default='sweep_work',
                       help='Working directory for artifacts (default: sweep_work)')
    parser.add_argument('--simulator', action='store_true',
                       help='Run in simulator mode (for testing)')
    parser.add_argument('--start-index', type=int, default=0,
                       help='Start from specific configuration index (for resuming)')
    parser.add_argument('--max-runs', type=int, default=None,
                       help='Maximum number of runs (for testing subset)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose debug output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Validate config file exists
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        return 1

    run_sweep(
        args.config,
        args.output,
        args.work_dir,
        args.simulator,
        args.start_index,
        args.max_runs
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
