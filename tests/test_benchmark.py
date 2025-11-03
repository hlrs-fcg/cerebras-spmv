#!/usr/bin/env python3
"""
Quick test of benchmark save/load functionality
"""

import sys
from pathlib import Path
from benchmark import BenchmarkResult, save_benchmark, load_benchmarks

def test_save_load():
    """Test saving and loading benchmark results"""

    test_file = "tests/test_benchmark_data.csv"

    # Create test result
    result1 = BenchmarkResult(
        matrix_rows=1000,
        matrix_cols=1000,
        density=0.01,
        total_nnz=10000,
        num_pes=4,
        grid_width=2,
        grid_height=2,
        compile_time=5.5,
        compute_time=0.123,
        total_time=6.0,
        validation_passed=True
    )

    result2 = BenchmarkResult(
        matrix_rows=2000,
        matrix_cols=2000,
        density=0.02,
        total_nnz=40000,
        num_pes=16,
        grid_width=4,
        grid_height=4,
        compile_time=7.2,
        compute_time=0.456,
        total_time=8.0,
        validation_passed=True
    )

    # Save results
    print("Saving results...")
    save_benchmark(result1, csv_path=test_file, append=False)
    save_benchmark(result2, csv_path=test_file, append=True)

    # Load results
    print("Loading results...")
    loaded = load_benchmarks(test_file)

    assert len(loaded) == 2, f"Expected 2 results, got {len(loaded)}"
    assert loaded[0].matrix_rows == 1000
    assert loaded[1].matrix_rows == 2000
    assert loaded[0].num_pes == 4
    assert loaded[1].num_pes == 16

    print("✓ Save/load works")

    # Test summary
    print("\nSummary of first result:")
    print(loaded[0].summary())

    # Cleanup
    Path(test_file).unlink()
    print(f"\n✓ Test passed - cleaned up {test_file}")

if __name__ == "__main__":
    test_save_load()
