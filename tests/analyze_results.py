#!/usr/bin/env python3
"""
Analyze benchmark results from sweep runs.

Provides summary statistics, performance trends, and failure analysis.
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import Optional


def load_results(csv_path: str) -> pd.DataFrame:
    """Load benchmark results CSV."""
    df = pd.read_csv(csv_path)
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print high-level summary of results."""
    total = len(df)
    successful = len(df[df['exit_reason'] == 'SUCCESS'])
    failed = total - successful

    print("=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total runs:        {total}")
    print(f"Successful:        {successful} ({successful/total*100:.1f}%)")
    print(f"Failed:            {failed} ({failed/total*100:.1f}%)")
    print()


def print_exit_reason_breakdown(df: pd.DataFrame) -> None:
    """Print breakdown of exit reasons."""
    print("EXIT REASON BREAKDOWN")
    print("-" * 80)
    print(f"{'Exit Reason':<25} {'Count':<10} {'Percentage':<15}")
    print("-" * 80)

    exit_counts = df['exit_reason'].value_counts()
    total = len(df)

    for reason, count in exit_counts.items():
        percentage = (count / total) * 100
        print(f"{reason:<25} {count:<10} {percentage:>6.1f}%")

    print("-" * 80)
    print()


def print_performance_stats(df: pd.DataFrame) -> None:
    """Print performance statistics for successful runs."""
    successful = df[df['exit_reason'] == 'SUCCESS']

    if len(successful) == 0:
        print("No successful runs to analyze.")
        print()
        return

    print("PERFORMANCE STATISTICS (Successful Runs Only)")
    print("-" * 80)
    print(f"Number of successful runs: {len(successful)}")
    print()

    # GFLOPS
    if 'gflops_hw' in successful.columns:
        gflops = successful['gflops_hw']
        print(f"Hardware GFLOPS:")
        print(f"  Min:    {gflops.min():.3f}")
        print(f"  Max:    {gflops.max():.3f}")
        print(f"  Mean:   {gflops.mean():.3f}")
        print(f"  Median: {gflops.median():.3f}")
        print()

    # Compute time
    if 'compute_time_hw_avg' in successful.columns:
        compute_time = successful['compute_time_hw_avg']
        print(f"Compute Time (seconds):")
        print(f"  Min:    {compute_time.min():.6f}")
        print(f"  Max:    {compute_time.max():.6f}")
        print(f"  Mean:   {compute_time.mean():.6f}")
        print(f"  Median: {compute_time.median():.6f}")
        print()

    # Memory utilization
    if 'memory_utilization_pct' in successful.columns:
        mem_util = successful['memory_utilization_pct']
        print(f"Memory Utilization (%):")
        print(f"  Min:    {mem_util.min():.1f}%")
        print(f"  Max:    {mem_util.max():.1f}%")
        print(f"  Mean:   {mem_util.mean():.1f}%")
        print(f"  Median: {mem_util.median():.1f}%")
        print()

    # Load imbalance
    if 'load_imbalance_pct' in successful.columns:
        load_imb = successful['load_imbalance_pct']
        print(f"Load Imbalance (%):")
        print(f"  Min:    {load_imb.min():.1f}%")
        print(f"  Max:    {load_imb.max():.1f}%")
        print(f"  Mean:   {load_imb.mean():.1f}%")
        print(f"  Median: {load_imb.median():.1f}%")
        print()

    print("-" * 80)
    print()


def print_best_worst_configs(df: pd.DataFrame, metric: str = 'gflops_hw', n: int = 5) -> None:
    """Print best and worst performing configurations."""
    successful = df[df['exit_reason'] == 'SUCCESS']

    if len(successful) == 0:
        return

    if metric not in successful.columns:
        print(f"Metric '{metric}' not found in results.")
        return

    print(f"TOP {n} CONFIGURATIONS (by {metric})")
    print("-" * 80)

    top = successful.nlargest(n, metric)
    for i, (_, row) in enumerate(top.iterrows(), 1):
        print(f"{i}. {row['matrix_rows']}×{row['matrix_cols']}, density={row['density']}, "
              f"num_pes={row['num_pes']}")
        print(f"   {metric}: {row[metric]:.6f}")
        if 'compute_time_hw_avg' in row:
            print(f"   Compute time: {row['compute_time_hw_avg']:.6f}s")
        print()

    print("-" * 80)
    print()


def compare_pe_configs(df: pd.DataFrame) -> None:
    """Compare min vs max PE configurations."""
    successful = df[df['exit_reason'] == 'SUCCESS']

    if len(successful) == 0:
        return

    # Find configurations that have both min and max
    min_pe = successful[successful['num_pes'].astype(str).str.contains('min', case=False, na=False)]
    max_pe = successful[successful['num_pes'].astype(str).str.contains('max', case=False, na=False)]

    if len(min_pe) == 0 or len(max_pe) == 0:
        print("Not enough data to compare min vs max PE configurations.")
        print()
        return

    print("MIN vs MAX PE CONFIGURATION COMPARISON")
    print("-" * 80)
    print(f"{'Metric':<30} {'Min PE':<15} {'Max PE':<15} {'Speedup':<15}")
    print("-" * 80)

    metrics = ['gflops_hw', 'compute_time_hw_avg', 'memory_utilization_pct', 'load_imbalance_pct']

    for metric in metrics:
        if metric not in successful.columns:
            continue

        min_val = min_pe[metric].mean()
        max_val = max_pe[metric].mean()

        # For compute time, lower is better, so invert speedup
        if 'time' in metric:
            speedup = min_val / max_val if max_val != 0 else 0
        else:
            speedup = max_val / min_val if min_val != 0 else 0

        print(f"{metric:<30} {min_val:<15.6f} {max_val:<15.6f} {speedup:<15.2f}x")

    print("-" * 80)
    print()


def print_failure_analysis(df: pd.DataFrame) -> None:
    """Analyze failure patterns."""
    failed = df[df['exit_reason'] != 'SUCCESS']

    if len(failed) == 0:
        print("No failures to analyze.")
        print()
        return

    print("FAILURE ANALYSIS")
    print("-" * 80)

    # Group by exit reason
    by_reason = failed.groupby('exit_reason')

    for reason, group in by_reason:
        print(f"\n{reason} ({len(group)} failures):")

        # Show common characteristics
        if 'matrix_rows' in group.columns:
            print(f"  Matrix sizes: {group['matrix_rows'].min()}-{group['matrix_rows'].max()}")

        if 'density' in group.columns:
            print(f"  Densities: {group['density'].min():.6f}-{group['density'].max():.6f}")

        if 'total_nnz' in group.columns and group['total_nnz'].max() > 0:
            print(f"  NNZ range: {int(group['total_nnz'].min())}-{int(group['total_nnz'].max())}")

        # Show a sample error message
        if 'error_message' in group.columns:
            sample_error = group['error_message'].dropna().head(1)
            if len(sample_error) > 0:
                error_msg = str(sample_error.iloc[0])[:200]
                print(f"  Sample error: {error_msg}")

    print()
    print("-" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('results', type=str,
                       help='Path to benchmark results CSV')
    parser.add_argument('--metric', type=str, default='gflops_hw',
                       help='Metric to use for best/worst analysis')
    parser.add_argument('--top-n', type=int, default=5,
                       help='Number of top configurations to show')
    parser.add_argument('--export-successful', type=str,
                       help='Export successful runs to separate CSV')

    args = parser.parse_args()

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {args.results}")
        return 1

    df = load_results(args.results)

    if len(df) == 0:
        print("No results found in CSV.")
        return 1

    # Print analysis
    print_summary(df)
    print_exit_reason_breakdown(df)
    print_performance_stats(df)
    print_best_worst_configs(df, metric=args.metric, n=args.top_n)
    compare_pe_configs(df)
    print_failure_analysis(df)

    # Export successful runs if requested
    if args.export_successful:
        successful = df[df['exit_reason'] == 'SUCCESS']
        export_path = Path(args.export_successful)
        successful.to_csv(export_path, index=False)
        print(f"Exported {len(successful)} successful runs to: {export_path}")
        print()

    return 0


if __name__ == "__main__":
    exit(main())
