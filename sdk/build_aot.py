"""Compile the tiler kernels ahead of time into one extension module and zip it for the worker.

    uv run --no-project --python 3.11 --with numpy==1.25.0 --with numba==0.60.0 --with setuptools \
        python -m sdk.build_aot --out vendor/aot_vendor.zip
    SPMV_VENDOR_ZIP=vendor/aot_vendor.zip uv run python3 run.py ...

What it buys: no numba on the worker at all and no JIT (the compile cost, 5-12 s there), for a
~0.3 MB file.  What it costs: it must be built on a machine with the worker's CPU architecture,
CPython minor version and a C compiler (numba.pycc links with the system compiler), i.e. the user
node; and the compiled functions hold the GIL, so the tiler runs on one thread.  The code is
generic x86-64 (numba's default target), so it does not depend on the build host's CPU model.

Run with the same Python minor version as the worker (``worker_python`` in the CSV).
"""

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build(out_dir: Path) -> Path:
    """Compile spmv_tile_aot into ``out_dir``; returns the extension file."""
    sys.path.insert(0, str(ROOT / "cluster"))
    import tile_kernels
    from numba.pycc import CC

    cc = CC(tile_kernels.AOT_MODULE)
    cc.output_dir = str(out_dir)
    cc.verbose = False
    for name, fn in tile_kernels.KERNELS.items():
        sig = tile_kernels.KERNEL_SIGNATURES[name]
        # the kernels are plain functions; pycc compiles them with njit semantics
        cc.export(name, sig)(fn)
    cc.compile()
    found = sorted(out_dir.glob(f"{tile_kernels.AOT_MODULE}*.so"))
    if not found:
        raise RuntimeError(f"pycc produced no {tile_kernels.AOT_MODULE}*.so in {out_dir}")
    return found[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default="vendor/aot_vendor.zip")
    a = ap.parse_args(argv)
    from sdk.vendor_pack import pack
    with tempfile.TemporaryDirectory() as tmp:
        so = build(Path(tmp))
        out = pack(tmp, a.out)
    print(f"{out}  ({out.stat().st_size / 1e6:.2f} MB; {so.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
