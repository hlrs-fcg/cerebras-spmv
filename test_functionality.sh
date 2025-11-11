#!/bin/bash
# Quick functionality test suite for SpMV

set -e

echo "=================================="
echo "SpMV Functionality Tests"
echo "=================================="

# 1. Basic single PE test
echo -e "\n[1/7] Testing single PE..."
python run.py --rows 100 --cols 100 --density 0.1 --num-pes 1 --save-benchmark ""

# 2. Multi-PE test
echo -e "\n[2/7] Testing 4 PEs..."
python run.py --rows 200 --cols 200 --density 0.05 --num-pes 4 --save-benchmark ""

# 3. Auto-min PE selection
echo -e "\n[3/7] Testing auto-min PE selection..."
python run.py --rows 500 --cols 500 --density 0.01 --num-pes min --save-benchmark ""

# 4. Auto-max PE selection
echo -e "\n[4/7] Testing auto-max PE selection..."
python run.py --rows 100 --cols 100 --density 0.1 --num-pes max --save-benchmark ""

# 5. PE suggestion
echo -e "\n[5/7] Testing PE suggestions..."
python run.py --rows 1000 --cols 1000 --density 0.01 --suggest-pes

# 6. Compile-only mode
echo -e "\n[6/7] Testing compile-only mode..."
python run.py --rows 50 --cols 50 --density 0.1 --num-pes 2 --compile-only --save-benchmark ""

# 7. Benchmark and results saving
echo -e "\n[7/7] Testing benchmark and results saving..."
python run.py --rows 200 --cols 200 --density 0.05 --num-pes 4 \
    --save-benchmark "test_benchmark.csv" \
    --save-results "test_results.npz"

if [ -f "test_benchmark.csv" ]; then
    echo "  ✓ Benchmark CSV created"
    rm test_benchmark.csv
fi

if [ -f "test_results.npz" ]; then
    echo "  ✓ Results NPZ created"
    rm test_results.npz
fi

echo -e "\n=================================="
echo "All tests passed! ✓"
echo "=================================="
