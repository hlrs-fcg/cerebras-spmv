# SpMV Benchmarking System

Comprehensive benchmarking system for SpMV with automatic configuration generation, exit reason tracking, and detailed failure analysis.

## Quick Start

```bash
# 1. Generate test configurations
python tests/generate_sweep_config.py -o tests/sweep_config.csv

# 2. Run the benchmark sweep
python tests/sweep.py tests/sweep_config.csv --continue-on-error

# 3. View results
# Results are saved to tests/benchmark_results.csv
```

## Generated Test Configurations

The default comprehensive configuration includes **38 test cases** organized into three categories:

### 1. Same NNZ, Different Dimensions (16 tests)
Tests how performance scales with matrix size when NNZ is held constant (~10,000 NNZ).

| Matrix Size | Density  | Expected NNZ |
|-------------|----------|--------------|
| 1000×1000   | 0.01     | ~10,000      |
| 1414×1414   | 0.005002 | ~10,000      |
| 2000×2000   | 0.0025   | ~10,000      |
| 2828×2828   | 0.00125  | ~10,000      |
| 4000×4000   | 0.000625 | ~10,000      |
| 5657×5657   | 0.000312 | ~10,000      |
| 8000×8000   | 0.000156 | ~10,000      |
| 10000×10000 | 0.0001   | ~10,000      |

**Purpose**: Understand how matrix dimension affects performance when computational workload (NNZ) is constant.

### 2. Density Sweep (10 tests)
Tests different densities on a fixed 5000×5000 matrix with min/max PE configurations.

| Density | Expected NNZ |
|---------|--------------|
| 0.0001  | ~2,500       |
| 0.0005  | ~12,500      |
| 0.001   | ~25,000      |
| 0.005   | ~125,000     |
| 0.01    | ~250,000     |
| 0.05    | ~1,250,000   |

**Purpose**: Understand how sparsity affects performance for a fixed matrix size.

### 3. Size Sweep (12 tests)
Tests different matrix sizes at constant density (0.001) with min/max PE configurations.

| Matrix Size | Expected NNZ |
|-------------|--------------|
| 1000×1000   | ~1,000       |
| 2000×2000   | ~4,000       |
| 5000×5000   | ~25,000      |
| 10000×10000 | ~100,000     |
| 20000×20000 | ~400,000     |

**Purpose**: Understand how matrix size affects performance at constant sparsity.

## Custom Configurations

### Generate Same-NNZ Tests Only
```bash
python tests/generate_sweep_config.py --mode same-nnz --target-nnz 50000 -o tests/same_nnz_50k.csv
```

### Generate Density Sweep Only
```bash
python tests/generate_sweep_config.py --mode density-sweep -o tests/density_only.csv
```

### Generate Size Sweep Only
```bash
python tests/generate_sweep_config.py --mode size-sweep -o tests/size_only.csv
```

## Understanding Exit Reasons

The improved error tracking system categorizes all failures:

| Exit Reason          | Meaning                                           |
|---------------------|---------------------------------------------------|
| `SUCCESS`           | Benchmark completed and validation passed         |
| `MEMORY_EXCEEDED`   | PE memory limit (48KB) exceeded                   |
| `COMPILATION_FAILED`| CSL compilation failed                            |
| `RUNTIME_ERROR`     | Device execution or general runtime error         |
| `VALIDATION_FAILED` | Computation completed but results don't match     |
| `TIMEOUT`           | Execution exceeded 10 minute timeout              |

## Sweep Output Example

```
================================================================================
Starting parameter sweep: 38 configurations
Results will be saved to: tests/benchmark_results.csv
================================================================================

[1/38] Running: rows=1000, cols=1000, density=0.01, num_pes=min
  ✓ SUCCESS

[2/38] Running: rows=1000, cols=1000, density=0.01, num_pes=max
  ✗ FAILED - COMPILATION_FAILED

...

================================================================================
Parameter Sweep Complete
================================================================================
Total runs:    38
Successful:    15
Failed:        23
Success rate:  39.5%

Exit Reason Breakdown:
--------------------------------------------------------------------------------
Exit Reason               Count      Percentage
--------------------------------------------------------------------------------
COMPILATION_FAILED        18              47.4%
SUCCESS                   15              39.5%
MEMORY_EXCEEDED           3               7.9%
RUNTIME_ERROR             2               5.3%
--------------------------------------------------------------------------------

Failed Configurations:
--------------------------------------------------------------------------------
  rows=1000, cols=1000, density=0.01, num_pes=max
    Exit Reason: COMPILATION_FAILED
    Error: RuntimeError: CSL compilation failed...
  ...
================================================================================
```

## Benchmark Results CSV

All runs (successful AND failed) are saved to `tests/benchmark_results.csv` with:

### Key Fields
- `exit_reason`: Why the run succeeded/failed
- `error_message`: Detailed error description
- `validation_passed`: Whether results matched reference
- `matrix_rows`, `matrix_cols`, `density`, `total_nnz`: Matrix configuration
- `num_pes`, `grid_width`, `grid_height`: PE configuration
- `max_memory_per_pe`, `memory_utilization_pct`: Memory usage
- `compile_time`, `compute_time_hw_avg`, `total_time`: Timing metrics
- `gflops_hw`: Performance metric
- `load_imbalance_pct`: Load distribution quality

### Analyzing Results

```python
import pandas as pd

# Load results
df = pd.read_csv('tests/benchmark_results.csv')

# Filter successful runs only
successful = df[df['exit_reason'] == 'SUCCESS']

# Group by matrix size
by_size = successful.groupby('matrix_rows').agg({
    'gflops_hw': 'mean',
    'compute_time_hw_avg': 'mean',
    'load_imbalance_pct': 'mean'
})

# Compare min vs max PE performance
min_pe = successful[successful['num_pes'] == 'min']
max_pe = successful[successful['num_pes'] == 'max']
```

## Verbose Mode

For detailed debugging, use `--verbose` to see full error output:

```bash
python tests/sweep.py tests/sweep_config.csv --continue-on-error --verbose
```

## Tips for Meaningful Results

1. **Start Small**: Run a few tests first to ensure your setup works
   ```bash
   python tests/sweep.py tests/test.csv --continue-on-error
   ```

2. **Check Exit Reasons**: If many tests fail with `COMPILATION_FAILED`, check your CSL SDK setup

3. **Memory Issues**: If seeing `MEMORY_EXCEEDED`, the configurations may be too large for your hardware

4. **Compare Min vs Max PEs**: Each configuration tests both `min` and `max` PE counts to understand scalability

5. **Fixed NNZ Tests**: These are especially useful for understanding how matrix structure (not just size) affects performance

## Advanced: Creating Custom Test Sets

Edit `generate_sweep_config.py` to add your own test patterns:

```python
# Add to main() function
if args.mode == 'my-custom-test':
    # Your custom logic here
    sizes = [1000, 5000, 10000]
    densities = [0.001, 0.01]
    for size in sizes:
        for density in densities:
            configs.append({
                'rows': str(size),
                'cols': str(size),
                'density': str(density),
                'num_pes': 'min',
                'seed': '42',
                'arch': 'wse3'
            })
```

## File Overview

- `generate_sweep_config.py` - Generates test configuration CSVs
- `sweep.py` - Runs parameter sweeps with error tracking
- `benchmark.py` - BenchmarkResult dataclass and CSV utilities
- `test.csv` - Small test configuration (4 tests)
- `sweep_config.csv` - Comprehensive configuration (38 tests)
- `benchmark_results.csv` - Output results with exit reasons and timing

## Next Steps

After collecting results:

1. **Filter successful runs** for performance analysis
2. **Analyze failure patterns** to identify system limitations
3. **Compare PE configurations** (min vs max) for scaling insights
4. **Plot results** by matrix size, density, or NNZ
5. **Identify optimal configurations** for your use case
