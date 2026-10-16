"""Worker-side numba plumbing: diagnostics, the opt-in vendor zip, and the cache-off default.

None of this can run on the appliance here; the tests pin the logic that decides what the worker
does when its Python has no numba (it does not, see docs/worker-numba.md) and when a zip of numba is staged.
"""

import logging
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cluster"))

import tile_cache  # noqa: E402
import tile_kernels  # noqa: E402
import worker_env  # noqa: E402

from config import SpMVConfig  # noqa: E402
from sdk import launcher, vendor_pack  # noqa: E402


# --------------------------------------------------------------------------
# tile cache is opt-in
# --------------------------------------------------------------------------

def test_cache_is_off_without_an_explicit_directory(monkeypatch):
    monkeypatch.delenv("SPMV_TILE_CACHE_DIR", raising=False)
    assert not tile_cache.enabled()
    assert "SPMV_TILE_CACHE_DIR is not set" in tile_cache.disabled_reason()


def test_cache_needs_both_a_directory_and_not_being_switched_off(monkeypatch, tmp_path):
    monkeypatch.setenv("SPMV_TILE_CACHE_DIR", str(tmp_path))
    assert tile_cache.enabled() and tile_cache.cache_dir() == tmp_path
    monkeypatch.setenv("SPMV_TILE_CACHE", "0")
    assert not tile_cache.enabled() and tile_cache.disabled_reason() == "SPMV_TILE_CACHE=0"


def test_a_run_without_cache_dir_logs_one_line_and_stores_nothing(monkeypatch, tmp_path, caplog):
    import numpy as np
    import host_prep
    monkeypatch.delenv("SPMV_TILE_CACHE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    r = np.array([0, 3, 5], dtype=np.int32)
    with caplog.at_level(logging.INFO, logger="host_prep"):
        hb = host_prep.prepare_host_buffers(r, r, np.ones(3, np.float32), np.zeros(8, np.float32),
                                            8, 2, 2, 4, 4, "valtask", "column",
                                            cache_fingerprint="random:x")
    assert hb.save_thread is None and not hb.cache_hit
    assert sum("Tile cache off" in m for m in caplog.messages) == 1
    assert not (tmp_path / "matrices").exists()


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------

def test_diagnostics_carry_everything_needed_to_pick_wheels():
    info = worker_env.diagnostics(tile_kernels)
    assert set(info) == {"worker_python", "worker_platform", "worker_glibc", "worker_cpus", "numba_status"}
    assert info["worker_python"].startswith(("CPython", "PyPy")) and "numpy" in info["worker_python"]
    assert info["worker_cpus"] >= 1 and info["worker_glibc"]
    assert info["numba_status"].startswith(("numba ", "numba not installed"))
    assert "aot module" in info["numba_status"]


def test_numba_status_reports_what_exists_and_the_import_error_text():
    class Fake:
        AOT_MODULE = "spmv_tile_aot_surely_absent"
        IMPORT_ERROR = "numba: ImportError: libllvmlite.so: version `GLIBC_2.38' not found"
    status = worker_env.numba_status(Fake)
    assert "no aot module" in status and "GLIBC_2.38" in status and "error:" in status


def test_importing_the_tiler_does_not_import_numba():
    """numba costs seconds to import; the backend choice decides whether it is ever needed."""
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); import tile_kernels, host_prep; "
            "assert 'numba' not in sys.modules; print('ok')" % str(Path(tile_kernels.__file__).parent))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


def test_benchmark_row_has_the_diagnostic_columns():
    from bench.benchmark import BenchmarkResult
    row = BenchmarkResult(worker_python="CPython 3.11.9 numpy 1.25.0", worker_cpus=32).to_dict()
    assert row["worker_python"].startswith("CPython") and row["worker_cpus"] == 32


# --------------------------------------------------------------------------
# vendor zip: worker side
# --------------------------------------------------------------------------

@pytest.fixture
def clean_state(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(worker_env, "VENDOR_STATUS", "not requested")
    return tmp_path


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return path


def test_vendor_arg_parsing():
    assert worker_env.vendor_arg(["w.py", "--vendor", "v.zip", "--rows", "3"]) == "v.zip"
    assert worker_env.vendor_arg(["w.py", "--vendor=v.zip"]) == "v.zip"
    assert worker_env.vendor_arg(["w.py", "--rows", "3"]) is None
    assert worker_env.vendor_arg(["w.py", "--vendor"]) is None


def test_no_vendor_flag_changes_nothing(clean_state):
    before = list(sys.path)
    assert worker_env.activate_vendor(["w.py"]) is None
    assert sys.path == before and worker_env.VENDOR_STATUS == "not requested"


def test_staged_zip_is_unpacked_and_importable_at_the_end_of_sys_path(clean_state):
    z = _zip(clean_state / "v.zip", {"spmv_fake_pkg/__init__.py": "VALUE = 42\n"})
    dest = worker_env.activate_vendor(["w.py", "--vendor", str(z)])
    assert dest and sys.path[-1] == dest                 # appended: the worker's own packages win
    assert Path(dest).parent == clean_state              # next to the job, not /tmp (may be noexec)
    assert worker_env.VENDOR_STATUS.startswith("active")
    import spmv_fake_pkg
    assert spmv_fake_pkg.VALUE == 42
    sys.modules.pop("spmv_fake_pkg")


def test_zip_carrying_numpy_is_refused(clean_state):
    z = _zip(clean_state / "v.zip", {"numpy/__init__.py": "", "numba/__init__.py": ""})
    before = list(sys.path)
    assert worker_env.activate_vendor(["w.py", "--vendor", str(z)]) is None
    assert sys.path == before and "numpy" in worker_env.VENDOR_STATUS


@pytest.mark.parametrize("make", ["missing", "corrupt"])
def test_bad_zip_never_raises(clean_state, make):
    p = clean_state / "v.zip"
    if make == "corrupt":
        p.write_bytes(b"this is not a zip")
    assert worker_env.activate_vendor(["w.py", "--vendor", str(p)]) is None
    assert worker_env.VENDOR_STATUS.startswith("failed")


# --------------------------------------------------------------------------
# vendor zip: user-node side
# --------------------------------------------------------------------------

def _fake_target(root: Path, numpy=False):
    for d in ("numba", "llvmlite", "llvmlite.libs", "numba-0.60.0.dist-info", "bin"):
        (root / d).mkdir(parents=True)
        (root / d / "f.txt").write_text(d)
    (root / "numba" / "__pycache__").mkdir()
    (root / "numba" / "__pycache__" / "x.pyc").write_text("x")
    if numpy:
        (root / "numpy").mkdir()
    return root


def test_pack_keeps_packages_at_the_archive_root_and_drops_bin_and_pycache(tmp_path):
    out = vendor_pack.pack(_fake_target(tmp_path / "t"), tmp_path / "o" / "v.zip")
    names = zipfile.ZipFile(out).namelist()
    assert "numba/f.txt" in names and "llvmlite.libs/f.txt" in names
    assert not any(n.startswith("bin/") or "__pycache__" in n for n in names)


def test_pack_rejects_numpy_and_incomplete_directories(tmp_path):
    with pytest.raises(RuntimeError, match="numpy"):
        vendor_pack.pack(_fake_target(tmp_path / "a", numpy=True), tmp_path / "a.zip")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "numba").mkdir()
    with pytest.raises(RuntimeError, match="llvmlite"):
        vendor_pack.pack(tmp_path / "b", tmp_path / "b.zip")


def test_uv_command_is_the_documented_one():
    cmd = vendor_pack.uv_command("3.11", "OUT")
    assert cmd[:4] == ["uv", "pip", "install", "--target"] and cmd[4] == "OUT"
    assert "--no-deps" in cmd and "--only-binary" in cmd and cmd[cmd.index("--only-binary") + 1] == ":all:"
    assert cmd[cmd.index("--python-version") + 1] == "3.11"
    assert cmd[cmd.index("--python-platform") + 1] == "x86_64-manylinux_2_28"
    assert "numba==0.60.0" in cmd and "llvmlite==0.43.0" in cmd
    assert "numba==0.58.1" in vendor_pack.uv_command("3.8", "OUT")
    with pytest.raises(ValueError, match="no numba pin"):
        vendor_pack.uv_command("3.7", "OUT")


def test_launcher_stages_and_passes_the_vendor_zip(monkeypatch, tmp_path):
    cfg = SpMVConfig(rows=10_000, cols=10_000, density=0.001, num_pes_target=256, simulator=True)
    assert "--vendor" not in launcher.build_worker_command(cfg)
    assert launcher.build_worker_command(cfg, vendor="numba_vendor.zip").endswith(
        "--vendor numba_vendor.zip") or " --vendor numba_vendor.zip" in \
        launcher.build_worker_command(cfg, vendor="numba_vendor.zip")
    monkeypatch.delenv("SPMV_VENDOR_ZIP", raising=False)
    monkeypatch.delenv("SPMV_VENDOR_DIR", raising=False)
    assert launcher.resolve_vendor_zip() is None
    z = _zip(tmp_path / "mine.zip", {"numba/__init__.py": ""})
    monkeypatch.setenv("SPMV_VENDOR_ZIP", str(z))
    assert launcher.resolve_vendor_zip() == z
    monkeypatch.delenv("SPMV_VENDOR_ZIP")
    monkeypatch.setenv("SPMV_VENDOR_DIR", str(_fake_target(tmp_path / "t")))
    packed = launcher.resolve_vendor_zip()
    assert packed.name == "numba_vendor.zip" and "numba/f.txt" in zipfile.ZipFile(packed).namelist()
    monkeypatch.setenv("SPMV_VENDOR_ZIP", str(tmp_path / "nope.zip"))
    with pytest.raises(RuntimeError, match="does not exist"):
        launcher.resolve_vendor_zip()
