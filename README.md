# SpMV on Cerebras WSE

Sparse matrix-vector multiplication (SpMV) on the Cerebras WSE-3.

## Published work

The initial implementation and scaling results were presented at ISC26 (June 2026).
Source code at that point is tagged **v1.0**. See [docs/adr/ADR-000](docs/adr/ADR-000-v1-baseline-isc26.md) for a description of the v1.0 architecture and its known limitations.

Architecture decisions going forward are documented in [docs/adr/](docs/adr/).

## Modes

| `--encoding` | What it does | Use |
|---|---|---|
| `resident` | A is copied to the wafer once; a call is one launch (local CSR over the resident arrays, then the east-west reduction). Nothing streams per call. | **The v2.0 headline.** The case for the wafer, and the base for CG (v3.0). `docs/resident-spmv.md` |
| `valtask` (default) | A streamed every call as 4 B/NNZ (values only; the structure is preloaded), consumed by a per-word data task | Streaming SpMV when A does not fit; the default streaming mode |
| `header` | A streamed with a row header per non-empty row (~4.6 B/NNZ) | A comparison mode (ADR-003) |

Streaming has two transfer shapes, `--h2d-mode column` (default, W narrow transfers) and `full` (one
padded rectangle, a comparison mode). Timings are barrier-closed only (`docs/host-timing-barrier.md`);
`--sync-timing` adds an on-wafer window (`docs/synced-timing.md`). The measured behaviour of the link that
the design rules rest on is in ADR-008.

## Structure

```
.
├── analysis/               # result notebooks and their helpers (microbenchmark notebooks: cs3-microbench)
├── archive/                # superseded notes and scripts, kept for the record (archive/README.md)
├── bench/
│   ├── configs/            # TOML sweep configs (copy and edit to start a new sweep)
│   ├── benchmark.py        # BenchmarkResult dataclass and CSV I/O
│   ├── host_spmv_baseline.py  # scipy / numba CSR on the host, the comparison for the wafer numbers
│   └── host_tiling_bench.py   # host tiling backends, timed
├── cluster/                # host-side code and the worker (cluster/README.md)
│   └── results/            # v1.0-data (ISC26), v2.0-data (release candidate runs, host baselines), data/ for new runs
├── docs/                   # design notes; docs/adr/ has the architecture decision records
├── sdk/
│   ├── build_info.py       # which commit a run came from
│   ├── compiler.py         # SdkCompiler wrapper (login node)
│   ├── launcher.py         # SdkLauncher wrapper: stages and runs the worker
│   ├── build_aot.py        # ahead-of-time tiler module for the worker
│   └── vendor_pack.py      # numba vendor zip for the worker
├── src/csl/
│   ├── layout.csl          # PE grid layout, picks the kernel by ENCODING
│   ├── pe.csl              # header kernel
│   ├── pe_valtask.csl      # values-only streaming kernel
│   ├── pe_resident.csl     # resident-A kernel
│   └── sync/               # the csl-examples sync module (vendored, see NOTICE)
├── tests/
├── utils/
├── config.py               # SpMVConfig dataclass
└── run.py                  # Entry point: single run or sweep
```

## Microbenchmarks

The host↔wafer microbenchmarks (M1–M10: copy/streaming H2D and D2H, overlap, per-call overhead, sync
latency, on-wafer collectives, edge buffers, noise) moved to their own repository,
[hlrs-fcg/cs3-microbench](https://github.com/hlrs-fcg/cs3-microbench) (private), with the raw results and
the analysis notebooks. `docs/microbenchmarks.md` here is a pointer; the rules they established for this
project are in `docs/adr/ADR-008-host-wafer-transfer-and-timing.md`.

## Usage

```bash
# Single run (default: full fabric)
uv run python run.py run --rows 10000 --cols 10000 --density 0.01

# Resident A with the synced on-wafer window, 20 passes (run.py defaults --loop-count to 20 for resident)
uv run python run.py run --rows 1000 --cols 1000 --density 0.01 --num-pes 16 --encoding resident --sync-timing

# Single run on a fixed PE count; the grid shape is near-square by default
uv run python run.py run --rows 1000 --cols 1000 --density 0.01 --num-pes 64

# Override the grid height (H); width becomes ceil(num_pes / H)
uv run python run.py run --rows 1000 --cols 1000 --density 0.01 --num-pes 16 --grid-height 1

# Run against the simulator instead of hardware, with verbose logging
uv run python run.py run --rows 1000 --cols 1000 --density 0.01 --num-pes 64 --simulator -v

# Parameter sweep from a TOML config
uv run python run.py sweep bench/configs/weak-scaling.toml

# Resume a sweep from a run index
uv run python run.py sweep bench/configs/weak-scaling.toml --start-index 12

# Run multiple sweep configs back to back
uv run python run.py sweep bench/configs/weak-scaling.toml bench/configs/strong-scaling.toml
```

`-v/--verbose` is accepted either before or after the subcommand.

On the cluster the tree has no `.git`: run `python -m sdk.build_info write` before syncing it (a `git archive`
of a tag needs nothing), or a hardware sweep refuses to start. See `cluster/README.md`.

Example sweep configs are in `bench/configs/`; copy one and edit to define a new sweep.

A real matrix: `run.py run --mtx-file path.mtx --num-pes N --encoding resident` sizes the run from the file and refuses
before compiling if the busiest PE does not fit. Which matrices fit where: `docs/results.md`, "Supported matrices".
A sweep `[[run]]` may name `mtx_file = "path.mtx"` the same way (rows, cols, density and `total_nnz_per_pe` come from the file; a run that
does not fit makes `run.py sweep` stop before any run starts).

## Licence

The code is licensed under the [Apache License 2.0](LICENSE); see [NOTICE](NOTICE) for the vendored
Cerebras sync module (`src/csl/sync/pe.csl`, Apache-2.0, Copyright 2025 Cerebras Systems). The results,
notebooks and documentation are licensed under [CC-BY-4.0](cluster/results/LICENSE-DATA.md).

## Results and data

The v2.0 numbers are in `docs/results.md`; the raw rows are in `cluster/results/v2.0-data/` (README there): the hardware benchmarks ran
on commit `4739dc4` (release candidate 1), a simulator smoke from a fresh archive on `580e41f` (no change to the kernel or the worker
between the two; only the sweep's matrix check was added), and the host-CPU baselines on the same user node, including the four real
matrices (`bench/host_spmv_baseline.py --mtx`). Result files have the appliance job, user-node and account names redacted
(`<job id>`, `<user node>`, ...). `analysis/v1_vs_v2.ipynb` walks through what changed against v1.0 from these files.

## Citing

If you use this software, please cite it using [CITATION.cff](CITATION.cff) (GitHub's "Cite this
repository" reads it). The v1.0 release accompanies the ISC High Performance 2026 research poster
"SpMV for the Cerebras Wafer-Scale Engine" (Renschler, Schäfer, Gebert, Parsons), listed under `references`
in that file.
