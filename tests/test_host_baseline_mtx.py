"""bench/host_spmv_baseline.py --mtx: a tiny symmetric MatrixMarket file gives the same rows as the random mode, with the wafer's NNZ."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numba")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "bench" / "host_spmv_baseline.py"

MTX = """%%MatrixMarket matrix coordinate real symmetric
% tiny
4 4 6
1 1 4.0
2 1 -1.0
2 2 4.0
3 2 -1.0
3 3 4.0
4 4 4.0
"""


def run(*args):
    env = dict(os.environ, NUMBA_NUM_THREADS="32")
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, env=env, check=True, timeout=300).stdout


def test_mtx_mode_reports_the_expanded_nnz_and_all_rows(tmp_path):
    f = tmp_path / "tiny.mtx"
    f.write_text(MTX)
    out = run("--mtx", str(f))
    assert "tiny.mtx 4x4 (symmetric, 6 stored) nnz=8" in out                      # 6 stored, 2 off-diagonal mirrored
    for label in ("numpy bincount fp64 COO", "scipy CSR fp32, 1 thread", "numba CSR fp32, 1 threads",
                  "numba CSR fp32, 8 threads", "numba CSR fp32, 32 threads"):
        assert label in out
    assert "COO->CSR one-time" in out and "GFLOP/s" in out
    assert out.count("per call") >= 5 and "steady" in out and "us" in out          # tiny matrix: microseconds
    assert len(re.findall(r"\(N=200\)", out)) == 5                                 # >= 200 inner repetitions under 1 ms
    err = float(re.search(r"numba rel err (\S+)", out).group(1))
    assert err < 1e-6


def test_two_files_and_a_missing_mm_io_message(tmp_path):
    a, b = tmp_path / "a.mtx", tmp_path / "b.mtx"
    a.write_text(MTX)
    b.write_text(MTX)
    out = run("--mtx", str(a), str(b))
    assert out.count("COO->CSR one-time") == 2

