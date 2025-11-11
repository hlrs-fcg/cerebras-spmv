#!/usr/bin/env python3
"""
Parameter Sweep Runner

Runs SpMV benchmarks with multiple parameter configurations from a CSV file.
Reports success/failure for each run with detailed exit reason tracking.
"""

import argparse
import csv
import subprocess
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def load_sweep_config(csv_path: str) -> List[Dict[str, str]]:
    """
    Load parameter configurations from CSV file.

    Expected columns (all optional except those marked required):
    - rows (required): Number of matrix rows
    - cols (required): Number of matrix columns
    - density (required): Sparsity density
    - num_pes: Number of PEs ('min', 'max', or integer)
    - seed: Random seed
    - arch: Architecture ('wse2' or 'wse3')

    Returns:
        List of configuration dictionaries
    """
    configs = []
    csv_file = Path(csv_path)

    if not csv_file.exists():
        logger.error(f"Config file not found: {csv_path}")
        return []

    with open(csv_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader, 1):
            # Validate required fields
            required = ['rows', 'cols', 'density']
            missing = [field for field in required if field not in row or not row[field]]

            if missing:
                logger.warning(f"Skipping row {i}: missing required fields {missing}")
                continue

            configs.append(row)

    logger.info(f"Loaded {len(configs)} configurations from {csv_path}")
    return configs


def get_exit_reason_from_csv(result_csv: str) -> str:
    """
    Read the most recent exit_reason from the benchmark CSV.

    Args:
        result_csv: Path to benchmark results CSV

    Returns:
        Exit reason string or "UNKNOWN" if not found
    """
    csv_file = Path(result_csv)
    if not csv_file.exists():
        return "UNKNOWN"

    try:
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows and 'exit_reason' in rows[-1]:
                return rows[-1]['exit_reason']
    except Exception:
        pass

    return "UNKNOWN"


def run_benchmark(config: Dict[str, str], run_script: str, result_csv: str, verbose: bool = False) -> Tuple[bool, str, str]:
    """
    Run a single benchmark configuration.

    Args:
        config: Parameter dictionary
        run_script: Path to run.py script
        result_csv: Path to save benchmark results
        verbose: Whether to show detailed output

    Returns:
        Tuple of (success, exit_reason, error_output)
    """
    # Build command
    cmd = [sys.executable, run_script]

    # Add required parameters
    cmd.extend(['--rows', config['rows']])
    cmd.extend(['--cols', config['cols']])
    cmd.extend(['--density', config['density']])

    # Add optional parameters
    if 'num_pes' in config and config['num_pes']:
        cmd.extend(['--num-pes', config['num_pes']])

    if 'seed' in config and config['seed']:
        cmd.extend(['--seed', config['seed']])

    if 'arch' in config and config['arch']:
        cmd.extend(['--arch', config['arch']])

    # Always save benchmark results
    cmd.extend(['--save-benchmark', result_csv])

    # Run the command
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        success = result.returncode == 0

        # Get exit reason from CSV
        exit_reason = get_exit_reason_from_csv(result_csv)

        # Capture error output for debugging
        error_output = ""
        if not success:
            if result.stderr:
                error_output = result.stderr
            elif result.stdout:
                # Sometimes errors are in stdout
                error_output = result.stdout

        if verbose and error_output:
            logger.info(f"\n  Output:\n{error_output}")

        return success, exit_reason, error_output

    except subprocess.TimeoutExpired as e:
        error_output = e.stderr if hasattr(e, 'stderr') and e.stderr else "Process timed out after 600 seconds"
        if verbose:
            logger.info(f"\n  Output:\n{error_output}")
        return False, "TIMEOUT", error_output

    except Exception as e:
        error_output = str(e)
        if verbose:
            logger.info(f"\n  Output:\n{error_output}")
        return False, "RUNTIME_ERROR", error_output


def main():
    parser = argparse.ArgumentParser(
        description='Run parameter sweep from CSV configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example CSV format (sweep_config.csv):
    rows,cols,density,num_pes,seed,arch
    100,100,0.01,1,42,wse3
    1000,1000,0.01,min,42,wse3
    1000,1000,0.05,max,43,wse3
    5000,5000,0.001,16,44,wse3

Usage:
    python tests/sweep.py sweep_config.csv
    python tests/sweep.py sweep_config.csv --results sweep_results.csv
        """
    )

    parser.add_argument('config', type=str,
                       help='Path to CSV file with parameter configurations')
    parser.add_argument('--results', type=str, default='tests/benchmark_results.csv',
                       help='Path to save benchmark results CSV')
    parser.add_argument('--run-script', type=str, default='run.py',
                       help='Path to run.py script')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue sweep even if individual runs fail')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output from each run')

    args = parser.parse_args()

    # Load configurations
    configs = load_sweep_config(args.config)

    if not configs:
        logger.error("No valid configurations found")
        return 1

    # Check run script exists
    run_script = Path(args.run_script)
    if not run_script.exists():
        logger.error(f"Run script not found: {args.run_script}")
        return 1

    # Run sweep
    logger.info("="*80)
    logger.info(f"Starting parameter sweep: {len(configs)} configurations")
    logger.info(f"Results will be saved to: {args.results}")
    logger.info("="*80)

    successes = 0
    failures = 0
    exit_reason_counts = defaultdict(int)
    failed_configs = []

    for i, config in enumerate(configs, 1):
        # Format configuration for display
        config_str = f"rows={config['rows']}, cols={config['cols']}, density={config['density']}"
        if 'num_pes' in config and config['num_pes']:
            config_str += f", num_pes={config['num_pes']}"

        logger.info(f"\n[{i}/{len(configs)}] Running: {config_str}")

        success, exit_reason, error_output = run_benchmark(config, str(run_script), args.results, args.verbose)

        # Track exit reason
        exit_reason_counts[exit_reason] += 1

        if success:
            logger.info("  ✓ SUCCESS")
            successes += 1
        else:
            logger.info(f"  ✗ FAILED - {exit_reason}")
            failures += 1
            failed_configs.append((config_str, exit_reason, error_output))

            if not args.continue_on_error:
                logger.error("\nStopping sweep due to failure (use --continue-on-error to continue)")
                break

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Parameter Sweep Complete")
    logger.info("="*80)
    total_runs = successes + failures
    logger.info(f"Total runs:    {total_runs}")
    logger.info(f"Successful:    {successes}")
    logger.info(f"Failed:        {failures}")
    logger.info(f"Success rate:  {successes/total_runs*100:.1f}%")
    logger.info("")

    # Print exit reason breakdown
    logger.info("Exit Reason Breakdown:")
    logger.info("-" * 80)
    logger.info(f"{'Exit Reason':<25} {'Count':<10} {'Percentage':<15}")
    logger.info("-" * 80)

    # Sort by count (descending)
    for exit_reason, count in sorted(exit_reason_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_runs) * 100
        logger.info(f"{exit_reason:<25} {count:<10} {percentage:>6.1f}%")

    logger.info("-" * 80)
    logger.info("")

    # Print failed configurations details
    if failed_configs and not args.verbose:
        logger.info("Failed Configurations:")
        logger.info("-" * 80)
        for config_str, exit_reason, error_output in failed_configs[:10]:  # Show first 10
            logger.info(f"  {config_str}")
            logger.info(f"    Exit Reason: {exit_reason}")
            # Show first line of error if available
            if error_output:
                first_error_line = error_output.split('\n')[0][:120]
                logger.info(f"    Error: {first_error_line}")
        if len(failed_configs) > 10:
            logger.info(f"  ... and {len(failed_configs) - 10} more failures")
        logger.info("-" * 80)
        logger.info("")

    logger.info(f"Results saved: {args.results}")
    logger.info("="*80)

    return 0 if failures == 0 or args.continue_on_error else 1


if __name__ == "__main__":
    exit(main())
