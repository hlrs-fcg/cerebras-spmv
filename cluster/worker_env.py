"""What the worker's Python looks like, and how to give it numba when it has none.

Stdlib only; imported by worker_runtime.py *before* any module that imports numba, because the
SdkLauncher worker runs the appliance's own Python, which has no numba.  ``activate_vendor``
unpacks a staged zip of numba + llvmlite (built by sdk/vendor_pack.py on the user node for this
exact Python / manylinux) and appends it to sys.path.
"""

import logging
import os
import platform
import sys
import tempfile
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

VENDOR_STATUS = "not requested"
"""One line for the log and the CSV: what happened to the vendored packages."""


def vendor_arg(argv: list[str]) -> str | None:
    """Value of ``--vendor`` in argv, without argparse: this runs at import."""
    for i, tok in enumerate(argv):
        if tok == "--vendor" and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith("--vendor="):
            return tok.split("=", 1)[1]
    return None


def activate_vendor(argv: list[str]) -> str | None:
    """Unpack the staged vendor zip and put it at the END of sys.path (the worker's own packages
    win).  Returns the directory, or None; never raises -- the tiler falls back to numpy."""
    global VENDOR_STATUS
    name = vendor_arg(argv)
    if not name:
        return None
    try:
        zpath = Path(name)
        if not zpath.is_file():
            VENDOR_STATUS = f"failed: {name} was not staged"
            return None
        with zipfile.ZipFile(zpath) as z:
            if any(n.split("/", 1)[0] == "numpy" for n in z.namelist()):
                VENDOR_STATUS = "refused: archive contains numpy (would shadow the worker's)"
                return None
            # cwd first, /tmp is often mounted noexec and dlopen of the .so files would fail
            for base in (Path.cwd(), Path(tempfile.gettempdir())):
                try:
                    dest = Path(tempfile.mkdtemp(prefix=".spmv-vendor-", dir=base))
                    z.extractall(dest)
                    break
                except OSError as e:
                    err = e
            else:
                VENDOR_STATUS = f"failed: cannot extract anywhere ({err})"
                return None
        sys.path.append(str(dest))
        VENDOR_STATUS = f"active: {name} -> {dest}"
        return str(dest)
    except Exception as e:  # corrupt zip, disk full, ...
        VENDOR_STATUS = f"failed: {type(e).__name__}: {e}"
        return None


def glibc() -> str:
    try:
        v = os.confstr("CS_GNU_LIBC_VERSION")          # "glibc 2.28"
        if v:
            return v
    except (ValueError, OSError, AttributeError):
        pass
    lib, ver = platform.libc_ver()
    return f"{lib} {ver}".strip() or "unknown"


def cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))             # what this job may use, not the node
    except AttributeError:
        return os.cpu_count() or 0


def _dist_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def numba_status(tile_kernels) -> str:
    """Which compiled tiler backends exist on this worker, without importing numba (which costs
    seconds): numba / llvmlite versions and whether the AOT module is present.  If an import was
    attempted and failed, its error text is appended."""
    import importlib.util
    nb, ll = _dist_version("numba"), _dist_version("llvmlite")
    parts = [f"numba {nb} llvmlite {ll}" if nb else "numba not installed"]
    parts.append("aot module present" if importlib.util.find_spec(tile_kernels.AOT_MODULE)
                 else "no aot module")
    if tile_kernels.IMPORT_ERROR:
        parts.append(f"error: {tile_kernels.IMPORT_ERROR}")
    return f"{'; '.join(parts)} ({VENDOR_STATUS})"[:400]


def diagnostics(tile_kernels) -> dict:
    """The fields recorded in the worker log and CSV."""
    import numpy
    return {
        "worker_python": f"{platform.python_implementation()} {platform.python_version()} "
                         f"numpy {numpy.__version__}",
        "worker_platform": f"{platform.platform()} {platform.machine()}",
        "worker_glibc": glibc(),
        "worker_cpus": cpu_count(),
        "numba_status": numba_status(tile_kernels),
    }
