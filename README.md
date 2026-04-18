# SpMV on Cerebras WSE

Sparse matrix-vector multiplication (SpMV) on the Cerebras WSE-3 using a matched pairs approach.

## How it works

Vector values are pre-gathered on the host using column indices, sending
`(matrix_val, vector_val)` pairs to each PE. This eliminates vector broadcast
and index lookup on-device.

**Memory per PE**: ~8 bytes/nonzero + ~8 bytes/row

## Project structure

```
.
├── src/
│   ├── csl/
│   │   ├── layout.csl              # PE grid layout
│   │   └── pe.csl                  # Per-PE SpMV kernel
│   ├── spmv_runner/
│   │   ├── core/
│   │   │   ├── compiler.py         # CSL compilation wrapper
│   │   │   └── grid.py             # PE grid calculation
│   │   └── utils/
│   │       ├── memory.py           # Per-PE memory estimation
│   │       └── pe_selection.py     # PE count auto-selection (min/max)
│   └── spmv_prototype.py           # Host-side data distribution & validation
├── cluster/
│   ├── compile.py                  # Compile on login node
│   ├── run.py                      # Launch on worker via SdkLauncher
│   ├── run_sweep.py                # Sweep orchestration
│   └── worker_runtime.py           # Runs on worker node
├── bench/
│   ├── benchmark.py                # BenchmarkResult dataclass + CSV I/O
│   ├── generate_cluster_sweeps.py  # Generate sweep config CSVs
│   └── configs/cluster/            # Sweep configurations
├── results/
│   ├── data/                       # Benchmark CSVs
│   └── notebooks/                  # Analysis notebooks
└── Makefile
```

## Usage

The workflow is always: **compile on login node → run on worker**.

```bash
# Test compilation with a tiny matrix
make cluster-test

# Compile from a random matrix
make cluster-compile ROWS=10000 COLS=10000 DENSITY=0.01

# Compile from a .mtx file
make cluster-compile-mtx MTX=bench/matrices/HB_bcsstk14/HB_bcsstk14.mtx

# Fetch a matrix from SuiteSparse then compile
make fetch-matrix ID=HB/bcsstk14

# Run the compiled artifact on the cluster
make cluster-run

# Run a full parameter sweep
make cluster-sweep CONFIG=bench/configs/cluster/strong-scaling-large.csv
```

`--num-pes` accepts an integer, `min` (auto memory-constrained), or `max` (auto parallelism).

## Generating sweep configs

```bash
uv run bench/generate_cluster_sweeps.py comprehensive
uv run bench/generate_cluster_sweeps.py pe-scaling --rows 100000 --cols 100000 --density 0.01
```
