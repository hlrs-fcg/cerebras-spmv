# cluster/: host-side code and the worker

Everything that runs on the CS-3 user node ("login side") or on the appliance worker.

## A run, end to end

```
run.py                       parse flags / TOML sweep, build an SpMVConfig, resolve the build (sdk/build_info.py)
 └ sdk/compiler.py           cslc on the user node: src/csl/* -> an artifact (a staged copy of the CSL only)
 └ sdk/launcher.py           stage the worker files + options, SdkLauncher.run(), download the CSV
     └ cluster/worker_runtime.py  (on the appliance) tile A on the host, copy it to the wafer, run, validate, write one CSV row
```

`sdk/launcher.py` stages `STAGED_FILES` **flat** into the worker's working directory, so the worker
modules import each other by bare name and a module that is not in that list does not exist on the
appliance. `tests/test_worker_staging.py` checks the list against the imports.

## Modules

| Module | Role |
|---|---|
| `sweep_config.py` | TOML sweep configs (`bench/configs/`): loading and expansion into `SpMVConfig`s (ADR-007) |
| `coo_tiling.py` | COO to per-PE tiles; the header, values-only and resident encoders; the transfer plans; the encoding ids |
| `host_prep.py` | every host buffer the worker needs, one timer per phase |
| `tile_kernels.py` | the fused tiler (numba, an ahead-of-time module, or a numpy fallback) and the backend choice; `docs/worker-numba.md` |
| `tile_cache.py` | on-disk tile cache, off unless `SPMV_TILE_CACHE_DIR` is set |
| `worker_env.py` | what the worker's Python looks like; unpacks the vendored numba zip before numba is imported |
| `worker_runtime.py` | the worker entry point and the order of its stages |
| `worker_cli.py`, `worker_host.py`, `worker_device.py`, `worker_report.py` | the worker's stages: flags, geometry/matrix/tiling, device phase, summary/validation/CSV row |
| `worker_modes.py`, `worker_stream.py`, `worker_resident.py` | what differs per mode: streaming (header, valtask) and resident A. A CG mode goes beside them |
| `worker_common.py`, `worker_layout.py`, `worker_timing.py` | shared helpers: validation and progress log; memcpy layouts and setup strips; cycle/barrier timing |

## Usage

From the repo root on the user node (`run.py` is the only entry point):

```sh
uv run python run.py run --rows 1000 --cols 1000 --density 0.01 --num-pes 16 --encoding resident --sync-timing
uv run python run.py sweep bench/configs/weak-scaling.toml
```

`--simulator` runs the same worker on the simulator. See the top-level `README.md` for the flags that matter
and `docs/` for the design notes.

## Environment

| Variable | Effect |
|---|---|
| `SPMV_VENDOR_ZIP`, `SPMV_VENDOR_DIR` | stage a zip of numba (or the AOT tiler) for the worker (`docs/worker-numba.md`) |
| `SPMV_TILE_BACKEND`, `SPMV_TILE_NUMBA_MIN_NNZ`, `SPMV_TILE_THREADS` | tiler backend and threads |
| `SPMV_TILE_CACHE_DIR`, `SPMV_TILE_CACHE`, `SPMV_TILE_CACHE_MAX_GB` | the tile cache |
| `SPMV_GIT_COMMIT`, `SPMV_GIT_DIRTY`, `SPMV_GIT_DESCRIBE` | override the recorded build (`sdk/build_info.py`) |
| `SPMV_DATA` | where the analysis notebooks read result CSVs |

## Provenance and results

Every CSV row records `git_commit`, `git_commit_full`, `git_dirty` and `repo_describe`. The cluster copy has
no `.git`, so the build is read from `BUILD_INFO.json` (written by `python -m sdk.build_info write` before a
sync) or from `BUILD_INFO.archive.json` (filled in by `git archive`). A hardware sweep refuses to start when
the commit is unknown, and when the tree is dirty unless `--allow-dirty` is given.

Results are written to `cluster/results/data/` by default. `cluster/results/v1.0-data/` holds the ISC26
poster data, with its own README.
