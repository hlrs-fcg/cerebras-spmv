#!/usr/bin/env python3
"""
Benchmark Viewer and Analyzer

View and analyze saved SpMV benchmark results.
"""

import argparse
import logging
from pathlib import Path
from benchmark import load_benchmarks, BenchmarkResult

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def print_summary(results, num_recent=10):
    """Print summary of benchmark results."""
    if not results:
        logger.info("No benchmark results found.")
        return

    logger.info("="*100)
    logger.info(f"Benchmark Results Summary ({len(results)} total runs)")
    logger.info("="*100)

    recent = results[-num_recent:] if len(results) > num_recent else results

    # Table header
    logger.info(f"{'#':<4} {'Timestamp':<20} {'Matrix':<15} {'PEs':<8} {'Grid':<8} "
                f"{'Compute(s)':<12} {'Total(s)':<10} {'Memory':<12} {'Status':<8}")
    logger.info("-"*100)

    for i, result in enumerate(reversed(recent), 1):
        timestamp_short = result.timestamp.split('T')[0] + ' ' + result.timestamp.split('T')[1][:8]
        matrix_str = f"{result.matrix_rows}×{result.matrix_cols}"
        grid_str = f"{result.grid_width}×{result.grid_height}"
        memory_str = f"{result.max_memory_per_pe/1024:.1f}KB"
        status = "PASS" if result.validation_passed else "FAIL"

        logger.info(f"{i:<4} {timestamp_short:<20} {matrix_str:<15} {result.num_pes:<8} {grid_str:<8} "
                   f"{result.compute_time:<12.4f} {result.total_time:<10.2f} "
                   f"{memory_str:<12} {status:<8}")

    logger.info("="*100)


def print_detailed(results, index):
    """Print detailed information about a specific result."""
    if index < 0 or index >= len(results):
        logger.error(f"Invalid index {index}. Valid range: 0-{len(results)-1}")
        return

    result = results[index]
    logger.info(result.summary())


def print_statistics(results):
    """Print aggregate statistics across all results."""
    if not results:
        logger.info("No results to analyze.")
        return

    logger.info("="*80)
    logger.info("Aggregate Statistics")
    logger.info("="*80)

    # Filter successful runs
    successful = [r for r in results if r.validation_passed]
    logger.info(f"Total runs: {len(results)}")
    logger.info(f"Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")

    if not successful:
        logger.info("No successful runs to analyze.")
        return

    # Compute statistics
    avg_compile = sum(r.compile_time for r in successful) / len(successful)
    avg_compute = sum(r.compute_time for r in successful) / len(successful)
    avg_total = sum(r.total_time for r in successful) / len(successful)

    min_compute = min(r.compute_time for r in successful)
    max_compute = max(r.compute_time for r in successful)

    avg_memory_util = sum(r.memory_utilization_pct for r in successful) / len(successful)
    max_memory_util = max(r.memory_utilization_pct for r in successful)

    avg_load_imbalance = sum(r.load_imbalance_pct for r in successful) / len(successful)

    logger.info(f"\nTiming:")
    logger.info(f"  Avg compile time: {avg_compile:.2f}s")
    logger.info(f"  Avg compute time: {avg_compute:.4f}s")
    logger.info(f"  Min compute time: {min_compute:.4f}s")
    logger.info(f"  Max compute time: {max_compute:.4f}s")
    logger.info(f"  Avg total time:   {avg_total:.2f}s")

    logger.info(f"\nMemory:")
    logger.info(f"  Avg utilization: {avg_memory_util:.1f}%")
    logger.info(f"  Max utilization: {max_memory_util:.1f}%")

    logger.info(f"\nLoad Balance:")
    logger.info(f"  Avg imbalance: {avg_load_imbalance:.1f}%")

    # PE count distribution
    pe_counts = {}
    for r in successful:
        pe_counts[r.num_pes] = pe_counts.get(r.num_pes, 0) + 1

    logger.info(f"\nPE Count Distribution:")
    for pe_count in sorted(pe_counts.keys()):
        logger.info(f"  {pe_count} PEs: {pe_counts[pe_count]} runs")

    logger.info("="*80)


def export_csv_subset(results, output_path, filter_func=None):
    """Export a subset of results to a new CSV file."""
    from benchmark import save_benchmark

    filtered = [r for r in results if filter_func is None or filter_func(r)]

    if not filtered:
        logger.info("No results match the filter criteria.")
        return

    # Save first result with append=False to create new file
    save_benchmark(filtered[0], csv_path=output_path, append=False)

    # Append remaining results
    for result in filtered[1:]:
        save_benchmark(result, csv_path=output_path, append=True)

    logger.info(f"Exported {len(filtered)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='View and analyze SpMV benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show recent results
  python tests/view_benchmarks.py

  # Show more results
  python tests/view_benchmarks.py --num-recent 20

  # Show detailed info for a specific run
  python tests/view_benchmarks.py --detail 5

  # Show aggregate statistics
  python tests/view_benchmarks.py --stats

  # Export successful runs to a new CSV
  python tests/view_benchmarks.py --export-passed successful_runs.csv
        """
    )

    parser.add_argument('--csv', type=str, default='tests/benchmark_results.csv',
                       help='Path to benchmark CSV file')
    parser.add_argument('--num-recent', type=int, default=10,
                       help='Number of recent results to show')
    parser.add_argument('--detail', type=int, metavar='INDEX',
                       help='Show detailed info for result at INDEX')
    parser.add_argument('--stats', action='store_true',
                       help='Show aggregate statistics')
    parser.add_argument('--export-passed', type=str, metavar='OUTPUT',
                       help='Export only passed results to OUTPUT CSV')
    parser.add_argument('--export-failed', type=str, metavar='OUTPUT',
                       help='Export only failed results to OUTPUT CSV')

    args = parser.parse_args()

    # Load results
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"Benchmark file not found: {csv_path}")
        logger.info(f"Run SpMV with --save-benchmark to create benchmark results.")
        return 1

    results = load_benchmarks(str(csv_path))

    if not results:
        logger.info("No benchmark results found in file.")
        return 0

    # Handle different modes
    if args.detail is not None:
        print_detailed(results, args.detail)
    elif args.stats:
        print_statistics(results)
    elif args.export_passed:
        export_csv_subset(results, args.export_passed,
                         filter_func=lambda r: r.validation_passed)
    elif args.export_failed:
        export_csv_subset(results, args.export_failed,
                         filter_func=lambda r: not r.validation_passed)
    else:
        print_summary(results, args.num_recent)

    return 0


if __name__ == "__main__":
    exit(main())
