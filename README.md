# SpMV on Cerebras WSE

Sparse matrix-vector multiplication (SpMV) on the Cerebras WSE-3.

## Published work

The initial implementation and scaling results were presented at ISC26 (June 2026).
Source code at that point is tagged **v1.0**. See [docs/adr/ADR-000](docs/adr/ADR-000-v1-baseline-isc26.md) for a description of the v1.0 architecture and its known limitations.

Architecture decisions going forward are documented in [docs/adr/](docs/adr/).

## Structure

```
.
├── bench/                  # BenchmarkResult dataclass and CSV I/O
├── cluster/
│   ├── results/data/       # Benchmark CSVs from cluster runs
│   ├── run_sweep.py        # Sweep orchestration (compile + run per config)
│   └── worker_runtime.py   # Runs on the worker node via SdkLauncher
├── docs/adr/               # Architecture decision records
├── sdk/
│   ├── compiler.py         # CSL compilation wrapper (login node)
│   └── launcher.py         # SdkLauncher wrapper (worker node)
├── src/csl/
│   ├── layout.csl          # PE grid layout
│   └── pe.csl              # Per-PE SpMV kernel
├── utils/
│   ├── matrix_generation.py
│   └── validation.py
├── compile.py              # Compile a single configuration on the login node
├── config.py               # SpMVConfig dataclass
└── run.py                  # Run a compiled artifact on the worker node
```

## Usage

The workflow is: **compile on login node, run on worker node.**

```bash
# Compile a single configuration
uv run python compile.py --rows 10000 --cols 10000 --density 0.01

# Run the compiled artifact
uv run python run.py

# Run a parameter sweep from a CSV config
uv run python cluster/run_sweep.py bench/configs/cluster/strong-scaling-large.csv
```
