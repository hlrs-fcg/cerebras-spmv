# numba / AOT tiler on the SdkLauncher worker

## Backends and how one is chosen

`tile_backend` in the CSV: `aot` | `numba` | `numpy` | `cache`.

| backend | needs on the worker | cost | speed |
|---|---|---|---|
| `aot` | `spmv_tile_aot*.so` from `sdk/build_aot.py` (numpy only, ~0.3 MB) | none: no JIT, no numba import | 1 thread (numba's AOT wrappers hold the GIL): ~2x slower than threaded numba |
| `numba` | the numba vendor zip (below) | numba import + compile of 5 kernels (`tile_jit_time`) | all cores, nogil kernels on a thread pool |
| `numpy` | nothing | none | LSD radix sort; ~10x slower than numba at 65K |

`auto` (default): `aot` if the module loads; else `numba` when NNZ >= `SPMV_TILE_NUMBA_MIN_NNZ`
(default 20,000,000, below that the JIT costs more than the numpy path saves); else `numpy`.
`SPMV_TILE_BACKEND=auto|aot|numba|numpy` forces one (a forced unavailable backend falls back to
numpy and `numba_status` carries the reason).

## AOT (preferred when a C compiler is on the user node)

```
uv run --no-project --python 3.11 --with numpy==1.25.0 --with numba==0.60.0 --with setuptools \
    python -m sdk.build_aot --out vendor/aot_vendor.zip
SPMV_VENDOR_ZIP=vendor/aot_vendor.zip uv run python3 run.py ...
```
Use the worker's Python minor (`worker_python` in the CSV) and the user node's architecture. The code
is numba's default generic target, not tuned to the build host's CPU.

## numba

The worker runs the appliance's own Python, which has no numba, so the host tiler falls back to
numpy (radix sort; ~10x slower than numba at 65K). Every worker log and CSV row says what it
saw: `worker_python`, `worker_platform`, `worker_glibc`, `worker_cpus`, `numba_status` (the numba
version, or the import error text), `tile_backend` (`numba` | `numpy` | `cache`).

## Shipping numba (opt-in)

1. Read `worker_python` / `worker_glibc` from a previous row's CSV (e.g. `CPython 3.11.9 numpy 1.25.0`,
   `glibc 2.28`). The pin table is `sdk/vendor_pack.py:VERSIONS` (3.9-3.12: numba 0.60.0 +
   llvmlite 0.43.0, supports numpy 1.22-2.0; 3.8: 0.58.1 + 0.41.1).
2. On the user node, build one zip. It holds numba + llvmlite only, never numpy:

   ```
   uv run --no-project --python 3.11 python -m sdk.vendor_pack build \
       --python-version <worker minor, e.g. 3.11> --out vendor/numba_vendor.zip
   ```

   which runs

   ```
   uv pip install --target <tmp> --python-version <3.x> --python-platform x86_64-manylinux_2_28 \
       --only-binary :all: --no-deps numba==0.60.0 llvmlite==0.43.0
   ```

   (~145 MB unpacked, ~48 MB zipped. Use `--platform x86_64-manylinux_2_17` for glibc < 2.28.)
3. Run with `SPMV_VENDOR_ZIP=vendor/numba_vendor.zip uv run python3 run.py ...` (or
   `SPMV_VENDOR_DIR=<unpacked dir>`; it is zipped on the fly). The launcher stages the zip and adds
   `--vendor numba_vendor.zip`; the worker unpacks it next to the job and appends it to `sys.path`
   before importing the tiler. One file is staged, not a directory: zipimport cannot load the
   compiled extensions, so the worker has to unpack anyway.

First use costs a numba compile (~2-4 s, `tile_jit_time`) because the compiled-kernel cache lives in a
throwaway directory unless `SPMV_TILE_CACHE_DIR` points somewhere that outlives the job.

## Tile cache

Off unless `SPMV_TILE_CACHE_DIR` is set (the staged directory is deleted after the job). Set it to a
shared path that persists; the worker logs `Tile cache off: ...` otherwise.
