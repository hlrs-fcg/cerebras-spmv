# Benchmarking

Saves all run results to CSV for later analysis.

## Usage

Runs automatically save to `tests/benchmark_results.csv`:

```bash
# Normal run - auto-saves
python run.py --rows 1000 --cols 1000 --density 0.01 --num-pes min

# Save to different file
python run.py --rows 1000 --cols 1000 --density 0.01 --num-pes 4 --save-benchmark my_results.csv

# Disable saving
python run.py --rows 1000 --cols 1000 --density 0.01 --num-pes 4 --save-benchmark ""
```

## View results

```bash
# Show recent runs
python tests/view_benchmarks.py

# Show more
python tests/view_benchmarks.py --num-recent 20

# Stats
python tests/view_benchmarks.py --stats

# Details for run #5
python tests/view_benchmarks.py --detail 5
```

## What's saved

- Matrix params: rows, cols, density, nnz
- PE config: num_pes, grid layout
- Memory: max/avg per PE, utilization
- Timing: compile, H2D, compute, D2H, total
- Load balance: min/max/avg nnz per PE
- Validation: pass/fail

## Analysis

Open the CSV in Excel, pandas, or whatever. Standard format with headers.

```python
from tests.benchmark import load_benchmarks

results = load_benchmarks("tests/benchmark_results.csv")
for r in results:
    print(f"{r.num_pes} PEs: {r.compute_time:.4f}s")
```
