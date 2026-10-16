"""Ship numba to a worker whose Python has none: build, pack and validate the vendor zip.

The SdkLauncher worker runs the appliance's own Python (no numba, no pip).  numba and llvmlite are
binary wheels, so they must match that interpreter and its glibc exactly; the worker log / CSV
columns ``worker_python``, ``worker_glibc`` say which.  Build on the user node, stage the zip, and
the worker unpacks it (cluster/worker_env.py).  Stdlib only.

    uv run --no-project --python 3.11 python -m sdk.vendor_pack build \\
        --python-version 3.11 --out vendor/numba_vendor.zip
    SPMV_VENDOR_ZIP=vendor/numba_vendor.zip uv run python3 run.py ...

The build is this uv command (no dependencies: numpy must stay the worker's own):

    uv pip install --target <dir> --python-version 3.11 --python-platform x86_64-manylinux_2_28 \\
        --only-binary :all: --no-deps numba==0.60.0 llvmlite==0.43.0
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Python (major, minor) -> (numba, llvmlite).  0.58.1 is the last numba for 3.8; 0.60.0 covers
# 3.9-3.12 and numpy 1.22-2.0, which includes the appliance's numpy 1.25.  (The repo's code
# itself needs Python >= 3.10.)
VERSIONS = {
    (3, 8): ("0.58.1", "0.41.1"),
    (3, 9): ("0.60.0", "0.43.0"),
    (3, 10): ("0.60.0", "0.43.0"),
    (3, 11): ("0.60.0", "0.43.0"),
    (3, 12): ("0.60.0", "0.43.0"),
}
DEFAULT_PLATFORM = "x86_64-manylinux_2_28"


def uv_command(python_version: str, out_dir: str, platform: str = DEFAULT_PLATFORM) -> list[str]:
    key = tuple(int(p) for p in python_version.split(".")[:2])
    if key not in VERSIONS:
        raise ValueError(f"no numba pin for Python {python_version}; known: "
                         f"{', '.join('%d.%d' % k for k in VERSIONS)}")
    numba, llvmlite = VERSIONS[key]
    return ["uv", "pip", "install", "--target", out_dir, "--python-version", python_version,
            "--python-platform", platform, "--only-binary", ":all:", "--no-deps",
            f"numba=={numba}", f"llvmlite=={llvmlite}"]


def validate(src: Path) -> None:
    """The directory must hold numba + llvmlite, or the AOT tiler module (spmv_tile_aot*.so), and
    must not carry numpy (it would shadow the worker's, whose C ABI the rest of the appliance
    stack was built against)."""
    names = {p.name for p in src.iterdir()}
    has_aot = any(n.startswith("spmv_tile_aot") and n.endswith(".so") for n in names)
    if not has_aot:
        for need in ("numba", "llvmlite"):
            if need not in names:
                raise RuntimeError(f"{src} has no {need}/ and no spmv_tile_aot*.so -- not a "
                                   f"vendor directory")
    if "numpy" in names:
        raise RuntimeError(f"{src} contains numpy; build with --no-deps so the worker's stays in use")


def pack(src: str | Path, out_zip: str | Path) -> Path:
    """Zip the unpacked packages (top-level entries at the archive root), dropping uv's bin/."""
    src, out_zip = Path(src), Path(out_zip)
    validate(src)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(src.rglob("*")):
            rel = f.relative_to(src)
            if f.is_file() and rel.parts[0] != "bin" and "__pycache__" not in rel.parts:
                z.write(f, rel.as_posix())
    return out_zip


def build(python_version: str, out_zip: str | Path, platform: str = DEFAULT_PLATFORM) -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(uv_command(python_version, tmp, platform), check=True)
        return pack(tmp, out_zip)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="download the wheels with uv and pack them")
    b.add_argument("--python-version", required=True, help="the worker's Python, e.g. 3.11")
    b.add_argument("--platform", default=DEFAULT_PLATFORM, help="uv --python-platform (default %(default)s)")
    b.add_argument("--out", default="vendor/numba_vendor.zip")
    p = sub.add_parser("pack", help="pack an already unpacked directory")
    p.add_argument("src")
    p.add_argument("--out", default="vendor/numba_vendor.zip")
    a = ap.parse_args(argv)
    out = (build(a.python_version, a.out, a.platform) if a.cmd == "build" else pack(a.src, a.out))
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
