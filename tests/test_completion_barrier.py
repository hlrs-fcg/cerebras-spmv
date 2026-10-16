"""Completion barriers: host brackets end when the DEVICE is done.

A blocking launch("wait_ew") / memcpy_h2d can return before the device has
finished, so the next blocking call soaks up the rest: the 0.46 MB y D2H took
1.0 ms in the 10-01 column/8 row but 50 ms after full/row/8, 27 ms after
full/row/16 and 193 / 371 ms at channels 1.  The worker now closes each
bracket with a tiny copy-mode D2H that cannot complete before the device has,
prices that D2H by repeating it on the idle device, and records corrected
times.  See docs/host-timing-barrier.md.
"""

import csv
import sys
from pathlib import Path

import pytest

from bench.benchmark import BenchmarkResult, load_benchmarks, save_benchmark

from tests.worker_src import worker_source  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKER = PROJECT_ROOT / "cluster" / "worker_runtime.py"

NEW_COLUMNS = ("device_done_time", "barrier_d2h_time", "barrier_d2h_repeat_time",
               "stream_time_corrected", "setup_done_time", "setup_barrier_d2h_time",
               "setup_barrier_d2h_repeat_time", "setup_h2d_time_corrected")


def _worker():
    sys.path.insert(0, str(PROJECT_ROOT / "cluster"))
    sys.path.insert(0, str(PROJECT_ROOT / "bench"))
    import worker_runtime
    return worker_runtime


def _bench(name):
    sys.path.insert(0, str(PROJECT_ROOT / "bench"))
    import importlib
    return importlib.import_module(name)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class FakeRunner:
    def __init__(self):
        self.calls = []

    def memcpy_d2h(self, buf, sym, x0, y0, w, h, l, **kw):
        self.calls.append(dict(buf=buf, sym=sym, x0=x0, y0=y0, w=w, h=h, l=l, **kw))


def test_timed_d2h_is_one_blocking_copy_mode_word_per_pe():
    r = FakeRunner()
    t = _worker().timed_d2h(r, "SYM", 1, 256, "DT", "ORD")
    assert t >= 0
    (c,) = r.calls
    assert (c["sym"], c["x0"], c["y0"], c["w"], c["h"], c["l"]) == ("SYM", 0, 0, 1, 256, 1)
    assert c["streaming"] is False and c["nonblock"] is False
    assert c["data_type"] == "DT" and c["order"] == "ORD"
    assert c["buf"].size == 256


def test_corrected_time():
    w = _worker()
    assert w.corrected_time(0.200, 0.001) == pytest.approx(0.199)
    assert w.corrected_time(0.020, 0.0) == pytest.approx(0.020)


# --------------------------------------------------------------------------
# the worker's sequence
# --------------------------------------------------------------------------

def _src():
    return worker_source()


def test_pass_barrier_follows_wait_ew_and_precedes_the_y_read():
    src = _src()
    # Synced timing: the barrier sits INSIDE the pass loop, so every pass is closed
    # by one and the per-pass times are device-done times.
    seq = ['runner.launch("wait_ew", nonblock=False)',
           "pass_h2d_s.append(time.perf_counter() - t_h2d)",
           "barrier_t = timed_d2h(runner, sym_y_result, 1, H,",
           "pass_done_s.append(t_done - t_h2d)",
           "pass_repeat_s.append(timed_d2h(runner, sym_y_result, 1, H,",
           "pass_corrected_s.append(corrected_time(pass_done_s[-1],",
           "runner.memcpy_d2h(y_flat, sym_y_result"]
    pos = [src.index(s) for s in seq]
    assert pos == sorted(pos)


def test_setup_barrier_follows_the_last_setup_h2d_and_covers_every_pe():
    src = _src()
    seq = ["for sym, l, calls in setup_calls:",
           "r.setup_h2d_time = time.perf_counter() - t_setup",
           "r.setup_barrier_d2h_time = timed_d2h(runner, sym_row_ptrs, W, H,",
           "r.setup_done_time = time.perf_counter() - t_setup",
           "r.setup_barrier_d2h_repeat_time = timed_d2h(runner, sym_row_ptrs, W, H,",
           "r.setup_h2d_time_corrected = corrected_time(r.setup_done_time,",
           "t_h2d = mode.launch_pass(r, d, t_comp)"]
    pos = [src.index(s) for s in seq]
    assert pos == sorted(pos)


def test_barrier_happens_before_any_other_d2h_of_the_pass():
    """The y read and tsc readback must not close the bracket instead."""
    src = _src()
    barrier = src.index("barrier_t = timed_d2h(runner, sym_y_result")
    assert barrier < src.index("runner.memcpy_d2h(y_flat")
    assert barrier < src.index("runner.memcpy_d2h(tsc_pack")
    # ... and inside the pass loop, before the loop's own y read and the
    # sync_pack read
    assert src.index("for p_i in range(args.loop_count):") < barrier
    assert barrier < src.index("runner.memcpy_d2h(sync_pack")
    # the scalar columns are the means of the per-pass brackets
    assert "r.stream_time_corrected = float(np.mean(pass_corrected_s))" in src


def test_host_return_columns_kept_and_new_ones_recorded():
    src = _src()
    for col in ("memcpy_h2d_time=memcpy_h2d_time", "setup_h2d_time=setup_h2d_time",
                "compute_time_hw_avg=host_stats[\"avg\"]"):
        assert col in src, col
    for col in NEW_COLUMNS:
        assert f"{col}={col}," in src, col
    assert "host-return" in src


def test_no_kernel_change():
    """Host-only: the artifact hash must not move.  Checked on the barrier
    commit itself (080e3ca vs its parent e640653); later branches, e.g. synced timing,
    change kernels on purpose."""
    import subprocess
    try:
        out = subprocess.run(["git", "diff", "--name-only", "e640653", "080e3ca", "--", "src/"],
                             capture_output=True, text=True, check=True,
                             cwd=PROJECT_ROOT).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("e640653 not available")
    assert out.strip() == ""


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------

def test_new_columns_default_to_zero_and_round_trip(tmp_path):
    path = tmp_path / "r.csv"
    save_benchmark(BenchmarkResult(), csv_path=str(path))
    save_benchmark(BenchmarkResult(device_done_time=0.215, barrier_d2h_time=0.19,
                                   barrier_d2h_repeat_time=0.001,
                                   stream_time_corrected=0.214,
                                   setup_h2d_time_corrected=0.6), csv_path=str(path))
    old, new = load_benchmarks(str(path))
    for col in NEW_COLUMNS:
        assert getattr(old, col) == 0.0
    assert (new.stream_time_corrected, new.barrier_d2h_repeat_time) == (0.214, 0.001)


def test_old_narrow_csv_backfills(tmp_path):
    path = tmp_path / "old.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_pes", "compute_time_hw_avg"])
        w.writerow([16, 0.001])
    save_benchmark(BenchmarkResult(stream_time_corrected=0.002), csv_path=str(path))
    rows = load_benchmarks(str(path))
    assert rows[0].stream_time_corrected == 0.0
    assert rows[1].stream_time_corrected == 0.002


# --------------------------------------------------------------------------
# analyses use the corrected columns
# --------------------------------------------------------------------------

