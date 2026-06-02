# SpMV on Cerebras WSE

Sparse matrix-vector multiplication (SpMV) on the Cerebras WSE-3.

## Published work

The initial implementation and scaling results were presented at ISC26 (June 2026).
Source code at that point is tagged **v1.0**. See [docs/adr/ADR-000](docs/adr/ADR-000-v1-baseline-isc26.md) for a description of the v1.0 architecture and its known limitations.

Architecture decisions going forward are documented in [docs/adr/](docs/adr/).

## Structure

```
.
├── bench/
│   ├── configs/            # TOML sweep configs (copy and edit to start a new sweep)
│   └── benchmark.py        # BenchmarkResult dataclass and CSV I/O
├── cluster/
│   ├── results/data/       # Benchmark CSVs from cluster runs
│   ├── sweep_config.py     # TOML loading and sweep expansion
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
├── config.py               # SpMVConfig dataclass
└── run.py                  # Entry point: single run or sweep
```

## Usage

```bash
# Single run
uv run python run.py run --rows 10000 --cols 10000 --density 0.01

# Parameter sweep from a TOML config
uv run python run.py sweep bench/configs/weak-scaling.toml

# Resume a sweep from run index N
uv run python run.py sweep bench/configs/weak-scaling.toml --start-index N

# Run multiple sweep configs back to back
uv run python run.py sweep bench/configs/weak-scaling.toml bench/configs/strong-scaling.toml
```

Example sweep configs are in `bench/configs/`; copy one and edit to define a new sweep.
