"""Setup copy in narrow strips (--setup-strips) and the 2026-10-08 defaults.

ADR-008 rule 2: narrow strips issued serially reach the H2D ceiling (10.3 GB/s at 16 channels),
one full rectangle about half.  Each setup array is therefore split into k vertical strips of the
grid.  A wrong sub-rectangle layout is silent apart from a bad relative error on the device, so
every PE's data is checked here by simulating what memcpy_h2d does with each call.
"""

import shlex
import sys
from pathlib import Path

import numpy as np
import pytest

from bench.benchmark import BenchmarkResult
from cluster.sweep_config import OVERRIDE_KEYS, SETUP_KEYS, SweepConfig
from config import (DEFAULT_CHANNELS, DEFAULT_ENCODING, DEFAULT_H2D_MODE, DEFAULT_MEMCPY_ORDER,
                    DEFAULT_SETUP_ORDER, SETUP_STRIPS_PER_CHANNEL, SpMVConfig)
from run import build_parser, io_overrides
from sdk.compiler import compile_options
from sdk.launcher import build_worker_command

from tests.worker_src import worker_source  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKER = PROJECT_ROOT / "cluster" / "worker_runtime.py"


def _worker():
    sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "bench"), str(PROJECT_ROOT / "cluster")]
    import worker_runtime
    return worker_runtime


def _arr(W, H, l):
    """(W*H, l) float32 whose entry is unique per (pe, k)."""
    return (np.arange(W * H * l, dtype=np.float32) + 1).reshape(W * H, l)


def _land(calls, W, H, l, order):
    """What the device ends up holding after the calls: memcpy_h2d's reading of each flat
    host array (ROW_MAJOR [h][w][l], COL_MAJOR [l][w][h]) into the PEs of its rectangle."""
    mem = np.zeros((H, W, l), dtype=np.float32)
    seen = np.zeros((H, W), dtype=int)
    for x0, w, flat, _nb in calls:
        assert flat.size == w * H * l, "a strip's buffer must hold exactly its w x H x l words"
        block = (flat.reshape(H, w, l) if order == "row"
                 else flat.reshape(l, w, H).transpose(2, 1, 0))
        mem[:, x0:x0 + w, :] = block
        seen[:, x0:x0 + w] += 1
    assert (seen == 1).all(), "every PE is written exactly once"
    return mem.reshape(W * H, l)


# --------------------------------------------------------------------------
# strip bounds and per-PE correctness
# --------------------------------------------------------------------------

@pytest.mark.parametrize("W,k", [(8, 1), (8, 2), (8, 3), (8, 8), (7, 2), (7, 3), (7, 5),
                                 (256, 32), (742, 32), (5, 32), (1, 4)])
def test_strip_bounds_cover_the_columns_exactly(W, k):
    wr = _worker()
    bounds = wr.setup_strip_bounds(W, k)
    assert len(bounds) == max(1, min(k, W))
    cols = [x for x0, w in bounds for x in range(x0, x0 + w)]
    assert cols == list(range(W))
    widths = [w for _x0, w in bounds]
    assert max(widths) - min(widths) <= 1 and widths == sorted(widths, reverse=True)


@pytest.mark.parametrize("order", ["row", "col"])
@pytest.mark.parametrize("W,H", [(8, 5), (7, 4), (5, 3), (1, 6), (6, 1)])
@pytest.mark.parametrize("l", [1, 3, 7])
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 8, 32])
def test_every_pe_receives_exactly_its_own_data(order, W, H, l, k):
    wr = _worker()
    arr = _arr(W, H, l)
    calls = wr.setup_strip_calls(arr, W, H, order, k, last_array=False)
    assert np.array_equal(_land(calls, W, H, l, order), arr)


@pytest.mark.parametrize("order", ["row", "col"])
def test_k_equal_one_is_the_old_single_full_rectangle(order):
    wr = _worker()
    W, H, l = 7, 4, 3
    arr = _arr(W, H, l)
    (x0, w, flat, nb), = wr.setup_strip_calls(arr, W, H, order, 1, last_array=False)
    assert (x0, w, nb) == (0, W, True)                   # the old calls were nonblocking ...
    assert np.array_equal(flat, wr._flatten_pe_major(arr, W, H, order))   # ... same buffer
    (_, _, _, nb_last), = wr.setup_strip_calls(arr, W, H, order, 1, last_array=True)
    assert nb_last is False                              # ... except the final one (row_ptrs)


@pytest.mark.parametrize("k", [2, 5, 32])
def test_strips_are_issued_serially_and_the_final_one_blocks(k):
    wr = _worker()
    W = 40
    arr = _arr(W, 3, 2)
    mid = wr.setup_strip_calls(arr, W, 3, "col", k, last_array=False)
    assert [c[3] for c in mid] == [False] * (len(mid) - 1) + [True]
    last = wr.setup_strip_calls(arr, W, 3, "col", k, last_array=True)
    assert [c[3] for c in last] == [False] * len(last), "the whole last array blocks, then the barrier"


def test_strip_buffers_are_independent_contiguous_copies():
    wr = _worker()
    arr = _arr(6, 4, 3)
    for order in ("row", "col"):
        for _x0, _w, flat, _nb in wr.setup_strip_calls(arr, 6, 4, order, 3, last_array=False):
            assert flat.flags["C_CONTIGUOUS"] and flat.ndim == 1


# --------------------------------------------------------------------------
# the worker wiring
# --------------------------------------------------------------------------

def test_worker_issues_all_strips_in_one_loop_and_keeps_the_barrier_after_the_last():
    src = worker_source()
    setup = src[src.index("Setup phase"):src.index("Compute phase")]
    assert "k_strips = len(setup_strip_bounds(W, args.setup_strips))" in setup
    assert "for sym, l, calls in setup_calls:" in setup and "for x0, w, flat, nb in calls:" in setup
    assert "runner.memcpy_h2d(sym, flat, x0, 0, w, H, l," in setup
    # arrays: x_vec, col_indices, (val_meta), row_ptrs LAST, and only that one is last_array
    assert setup.index("sym_x_vec, r.x_grid") < setup.index("sym_col_indices, r.ci")
    assert setup.index("mode.setup_extra_arrays(r, d)") < setup.index("sym_row_ptrs, r.rp")
    assert "last_array=(i == len(setup_arrays) - 1)" in setup
    # all strips are built before the clock starts, and the barrier reads the last array
    assert setup.index("setup_calls = [") < setup.index("t_setup = time.perf_counter()")
    assert setup.index("for sym, l, calls in setup_calls:") < setup.index("setup_h2d_time = ")
    assert setup.index("setup_h2d_time = ") < setup.index("timed_d2h(runner, sym_row_ptrs")


def test_the_strip_count_is_recorded_in_the_csv_row():
    src = worker_source()
    assert "setup_strips=k_strips," in src
    assert BenchmarkResult().setup_strips == 1, "rows from before the flag are one rectangle"


def test_worker_flag_default_is_one_strip_and_launcher_always_passes_the_resolved_value():
    wr = _worker()
    src = worker_source()
    assert '"--setup-strips", type=int, default=1,' in src
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.01, num_pes_target=64, simulator=True)
    cmd = build_worker_command(cfg)
    assert f"--setup-strips {cfg.resolved_setup_strips}" in cmd
    assert wr  # imported fine (flat) with the new helpers


# --------------------------------------------------------------------------
# config, CLI, TOML
# --------------------------------------------------------------------------

def test_default_strips_are_two_per_channel_clamped_to_the_grid_width():
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.01, num_pes_target=64 * 64)
    assert cfg.setup_strips is None and SETUP_STRIPS_PER_CHANNEL == 2
    assert cfg.channels == 16 and cfg.resolved_setup_strips == 32
    assert SpMVConfig(rows=1000, cols=1000, density=0.01, num_pes_target=64 * 64,
                      channels=1).resolved_setup_strips == 2
    assert SpMVConfig(rows=1000, cols=1000, density=0.01, num_pes_target=16,
                      simulator=True).resolved_setup_strips == 4        # 4 columns
    assert SpMVConfig(rows=1000, cols=1000, density=0.01, num_pes_target=64 * 64,
                      setup_strips=1).resolved_setup_strips == 1
    assert SpMVConfig(rows=1000, cols=1000, density=0.01, num_pes_target=64 * 64,
                      setup_strips=7).resolved_setup_strips == 7


@pytest.mark.parametrize("bad", [0, -3, 2.5, True, "4"])
def test_bad_strip_counts_are_rejected(bad):
    with pytest.raises(ValueError, match="setup_strips"):
        SpMVConfig(rows=100, cols=100, density=0.1, num_pes_target=16, simulator=True,
                   setup_strips=bad)


def test_cli_and_toml_carry_the_flag(tmp_path):
    args = build_parser().parse_args(shlex.split(
        "run --rows 64 --cols 64 --density 0.5 --setup-strips 1"))
    assert io_overrides(args)["setup_strips"] == 1
    args = build_parser().parse_args(shlex.split("run --rows 64 --cols 64 --density 0.5"))
    assert "setup_strips" not in io_overrides(args)
    assert "setup_strips" in OVERRIDE_KEYS and "setup_strips" in SETUP_KEYS
    toml = tmp_path / "s.toml"
    toml.write_text('[sweep]\ntype = "fixed"\n')
    assert SweepConfig.from_toml(toml).setup_strips is None
    toml.write_text('[sweep]\ntype = "fixed"\nsetup_strips = 8\n')
    assert SweepConfig.from_toml(toml).setup_strips == 8
    assert SweepConfig.from_toml(toml, cli_override_io={"setup_strips": 1}).setup_strips == 1


def test_no_recompile_for_the_strip_count():
    """Runtime only: it must not appear in the cslc option string."""
    a = SpMVConfig(rows=1000, cols=1000, density=0.01, num_pes_target=64, simulator=True,
                   setup_strips=1)
    b = SpMVConfig(rows=1000, cols=1000, density=0.01, num_pes_target=64, simulator=True,
                   setup_strips=9)
    assert compile_options(a) == compile_options(b)


# --------------------------------------------------------------------------
# the 2026-10-08 defaults, and the old options still reachable by name
# --------------------------------------------------------------------------

def test_the_new_defaults():
    assert (DEFAULT_ENCODING, DEFAULT_H2D_MODE, DEFAULT_SETUP_ORDER, DEFAULT_CHANNELS,
            DEFAULT_MEMCPY_ORDER) == ("valtask", "column", "col", 16, "row")
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.01, num_pes_target=256, simulator=True)
    assert (cfg.encoding, cfg.h2d_mode, cfg.setup_order, cfg.channels, cfg.memcpy_order) == (
        "valtask", "column", "col", 16, "row")


def test_the_worker_parser_mirrors_the_defaults():
    src = worker_source()
    assert '"--encoding", choices=list(ENCODINGS), default="valtask",' in src
    assert '"--setup-order", choices=["row", "col"], default="col",' in src
    assert '"--h2d-mode", choices=["column", "full"], default="column",' in src
    assert '"--memcpy-order", choices=["row", "col"], default="row",' in src


def test_rows_from_before_the_change_still_read_back_as_they_were():
    """BenchmarkResult defaults backfill history: they are NOT the config defaults."""
    r = BenchmarkResult()
    assert (r.encoding, r.setup_order, r.channels, r.setup_strips) == ("header", "row", 1, 1)


OLD = dict(encoding="header", h2d_mode="full", setup_order="row", channels=1, memcpy_order="row")


def test_the_old_options_still_produce_the_old_command_and_option_string():
    cfg = SpMVConfig(rows=10_000, cols=10_000, density=0.001, num_pes_target=256,
                     simulator=True, setup_strips=1, **OLD)
    opts = compile_options(cfg)
    assert "--channels=1 " in opts and "ENCODING:0" in opts
    cmd = build_worker_command(cfg)
    for flag in ("--encoding header", "--h2d-mode full", "--setup-order row",
                 "--memcpy-order row", "--setup-strips 1"):
        assert flag in cmd, flag
    # the artifact differs from the new default's only in ENCODING and --channels
    new = SpMVConfig(rows=10_000, cols=10_000, density=0.001, num_pes_target=256, simulator=True)
    assert (compile_options(new).replace("--channels=16", "--channels=1")
            .replace("ENCODING:2", "ENCODING:0")) == opts


def test_the_old_options_on_the_command_line():
    args = build_parser().parse_args(shlex.split(
        "run --rows 64 --cols 64 --density 0.5 --encoding header --h2d-mode full "
        "--setup-order row --channels 1 --setup-strips 1"))
    assert io_overrides(args) == {"encoding": "header", "h2d_mode": "full", "setup_order": "row",
                                  "channels": 1, "setup_strips": 1}


def test_shipped_strong_scaling_sweeps_keep_the_header_kernel_they_fit_with():
    """At 64 PEs valtask adds vals[TOTAL_NNZ] and does not fit the 38 KB budget."""
    for name in ("strong-scaling", "sim-strong-scaling"):
        sweep = SweepConfig.from_toml(PROJECT_ROOT / "bench" / "configs" / f"{name}.toml")
        assert sweep.encoding == "header", name
        assert list(sweep.expand()), name       # and every point still fits
