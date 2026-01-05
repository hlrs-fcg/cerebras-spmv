# SpMV Benchmark Suite

Modular benchmarking system with separation between data collection, batch execution, and analysis.

## Structure

### Core Modules

- **`benchmark.py`** - Data structures and CSV I/O (no statistics/analysis)
- **`view_benchmarks.py`** - Analysis and visualization of results
- **`sweep.py`** - Parameter sweep runner for batch benchmarking

## Usage

### Single benchmark run

```bash
# Normal run - auto-saves to tests/benchmark_results.csv
python run.py --rows 1000 --cols 1000 --density 0.01 --num-pes min

# Save to different file
python run.py --rows 1000 --cols 1000 --density 0.01 --num-pes 4 --save-benchmark my_results.csv

# Disable saving
python run.py --rows 1000 --cols 1000 --density 0.01 --num-pes 4 --save-benchmark ""
```

### Parameter sweep from CSV

Create a CSV with parameter configurations:

```csv
rows,cols,density,num_pes,seed,arch
100,100,0.01,1,42,wse3
1000,1000,0.01,min,42,wse3
1000,1000,0.05,max,43,wse3
5000,5000,0.001,16,44,wse3
```

Run the sweep:

```bash
python tests/sweep.py config.csv --results sweep_results.csv
python tests/sweep.py config.csv --continue-on-error  # Keep going if runs fail
```

### View and analyze results

```bash
# Show recent runs
python tests/view_benchmarks.py

# Show more
python tests/view_benchmarks.py --num-recent 20

# Aggregate statistics
python tests/view_benchmarks.py --stats

# Details for run #5
python tests/view_benchmarks.py --detail 5

# Export successful runs
python tests/view_benchmarks.py --export-passed successful.csv
```

## What's saved

- Matrix params: rows, cols, density, nnz
- PE config: num_pes, grid layout, architecture
- Memory: max/avg per PE, utilization percentage
- Timing: compile, H2D, compute (min/avg/max/std), D2H, total
- Performance: GFLOPS (HW and Python reference)
- Load balance: min/max/avg nnz per PE, imbalance percentage
- Validation: pass/fail status

## Design Philosophy

**Separation of concerns:**

1. **Data Collection** - `run.py` + `benchmark.py` collect raw metrics, save to CSV
2. **Batch Execution** - `sweep.py` runs many configs, reports pass/fail only
3. **Analysis** - `view_benchmarks.py` computes statistics and generates reports

This allows fast batch execution and flexible post-hoc analysis.

## Custom Analysis

Open the CSV in Excel, pandas, or any tool. Standard CSV format with headers.

```python
from tests.benchmark import load_benchmarks

results = load_benchmarks("tests/benchmark_results.csv")
for r in results:
    print(f"{r.num_pes} PEs: {r.compute_time_hw_avg:.6f}s, {r.gflops_hw:.2f} GFLOPS")
```
