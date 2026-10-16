"""Tests for --memcpy-order, and for the copy-mode setup H2D measurement.

Two separate things land together here because they answer one question.

`--memcpy-order` exists because https://sdk.cerebras.ai/tensor-streaming states
"Column-major delivers better bandwidth than row-major", and Cerebras' own
bandwidth benchmark uses COL_MAJOR for the transfer it measures.  We had
ROW_MAJOR hardcoded everywhere.

The guide's prose about WHICH axis is fastest is wrong, so the layout tests
below check our implementation against the shipped example's own reference loop
(`hwl_2_oned_colmajor`, bandwidth-test/run.appliance.py:178) rather than against
the documentation.  A transposed layout is silent apart from a bad relative
error on device, so this is the cheap place to catch it.

`setup_h2d_{bytes,time}` exist because the setup memcpys are the only
`streaming=False` transfer in a run.  Copy mode has no per-word kernel cost, so
it measures the link alone -- which is what makes the streaming rate
interpretable, and is the mode Cerebras' benchmark measures exclusively.
"""

import csv
import sys
import textwrap
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from bench.benchmark import BenchmarkResult, load_benchmarks, save_benchmark
from cluster.sweep_config import IO_KEYS, OVERRIDE_KEYS, RUNTIME_KEYS, SweepConfig
from config import DEFAULT_MEMCPY_ORDER, MEMCPY_ORDERS, SpMVConfig
from run import build_parser, io_overrides
from sdk.launcher import build_worker_command, compile_io_settings

from tests.worker_src import worker_source  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKER = PROJECT_ROOT / "cluster" / "worker_runtime.py"


def _flatten_whl():
    """Import the worker's helper the way the appliance does: flat, no package."""
    sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "bench"),
                    str(PROJECT_ROOT / "cluster")]
    from worker_runtime import _flatten_whl as fn
    return fn


@pytest.fixture
def cfg():
    return SpMVConfig(rows=10_000, cols=10_000, density=0.001,
                      num_pes_target=256, simulator=True)


# --------------------------------------------------------------------------
# the layout, checked against Cerebras' reference rather than the prose
# --------------------------------------------------------------------------

def test_col_major_matches_the_shipped_reference_loop():
    """`for l / for w / for h` with h innermost -- bandwidth-test:178.

    The guide claims column-major makes `w` fastest-varying. It does not.
    """
    fn = _flatten_whl()
    h, w, length = 3, 4, 5
    block = np.arange(w * h * length, dtype=np.float32).reshape(w, h, length)
    hwl = block.transpose(1, 0, 2)

    expected = [hwl[i, j, k]
                for k in range(length) for j in range(w) for i in range(h)]
    assert list(fn(block, "col")) == expected


def test_row_major_is_unchanged_from_what_we_already_shipped():
    """Guards against a regression in the path every existing result used."""
    fn = _flatten_whl()
    h, w, length = 3, 4, 5
    block = np.arange(w * h * length, dtype=np.float32).reshape(w, h, length)
    # What the code did before --memcpy-order existed.
    assert list(fn(block, "row")) == list(block.transpose(1, 0, 2).reshape(-1))


def test_the_two_orders_actually_differ():
    fn = _flatten_whl()
    block = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    assert list(fn(block, "col")) != list(fn(block, "row"))


def test_orders_differ_even_for_a_one_column_rectangle():
    """column mode sends w == 1 slices; order still changes the layout there."""
    fn = _flatten_whl()
    block = np.arange(1 * 4 * 3, dtype=np.float32).reshape(1, 4, 3)
    assert list(fn(block, "col")) != list(fn(block, "row"))


def test_both_orders_are_permutations_of_the_same_data():
    fn = _flatten_whl()
    block = np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7)
    for order in MEMCPY_ORDERS:
        out = fn(block, order)
        assert out.size == block.size
        assert sorted(out.tolist()) == sorted(block.reshape(-1).tolist())


# --------------------------------------------------------------------------
# the knob
# --------------------------------------------------------------------------

def test_default_is_the_historically_measured_order(cfg):
    assert DEFAULT_MEMCPY_ORDER == "row"
    assert cfg.memcpy_order == "row"


def test_worker_choices_match_config(cfg):
    assert '"--memcpy-order", choices=["row", "col"]' in worker_source()
    assert set(MEMCPY_ORDERS) == {"row", "col"}


@pytest.mark.parametrize("bad", ["ROW", "column", "col_major", "", None, 1])
def test_unknown_order_raises_before_the_compile(bad):
    with pytest.raises(ValueError, match="memcpy_order"):
        SpMVConfig(rows=1000, cols=1000, density=0.001, num_pes_target=256,
                   simulator=True, memcpy_order=bad)


def test_order_is_runtime_not_compile_time(cfg):
    assert "memcpy_order" in RUNTIME_KEYS
    assert "memcpy_order" not in IO_KEYS
    assert "memcpy_order" in OVERRIDE_KEYS
    # Must not be stamped post hoc: the worker knows its own order.
    assert "memcpy_order" not in compile_io_settings(cfg)


def test_order_does_not_trigger_the_fabric_geometry_warning():
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.001, num_pes_target=256,
                     simulator=True, memcpy_order="col")
    assert cfg.uses_non_default_io is False


@pytest.mark.parametrize("order", MEMCPY_ORDERS)
def test_worker_command_carries_the_order(order):
    cfg = SpMVConfig(rows=10_000, cols=10_000, density=0.001, num_pes_target=256,
                     simulator=True, memcpy_order=order)
    assert f"--memcpy-order {order}" in build_worker_command(cfg)


def test_cli_exposes_the_flag_on_both_subcommands():
    parser = build_parser()
    for argv in (["run", "--rows", "10", "--cols", "10", "--density", "0.1",
                  "--memcpy-order", "col"],
                 ["sweep", "x.toml", "--memcpy-order", "col"]):
        assert parser.parse_args(argv).memcpy_order == "col"


def test_cli_rejects_an_unknown_order():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["sweep", "x.toml", "--memcpy-order", "diagonal"])


def test_omitted_flag_is_left_out_of_the_overrides():
    parser = build_parser()
    args = parser.parse_args(["sweep", "x.toml"])
    assert args.memcpy_order is None
    assert "memcpy_order" not in io_overrides(args)


def test_order_and_mode_are_independently_settable():
    parser = build_parser()
    args = parser.parse_args(["run", "--rows", "1000", "--cols", "1000",
                              "--density", "0.001", "--memcpy-order", "col",
                              "--h2d-mode", "full", "--channels", "16"])
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.001, num_pes_target=256,
                     simulator=True, **io_overrides(args))
    assert (cfg.memcpy_order, cfg.h2d_mode, cfg.channels) == ("col", "full", 16)


# --------------------------------------------------------------------------
# TOML
# --------------------------------------------------------------------------

def _toml(tmp_path: Path, body: str = "") -> Path:
    path = tmp_path / "order-probe.toml"
    path.write_text(textwrap.dedent(f"""\
        [meta]
        name = "order-probe"

        [output]
        results = "cluster/results/data/{{date}}_{{name}}.csv"

        [sweep]
        type = "weak_scaling"
        nnz_per_pe = 200
        density = 0.001
        simulator = true
        {body}

        [sweep.num_pes]
        pow2_up_to = 256
        """))
    return path


def test_toml_sets_the_order_for_every_run(tmp_path):
    sweep = SweepConfig.from_toml(_toml(tmp_path, 'memcpy_order = "col"'))
    assert sweep.memcpy_order == "col"
    expanded = list(sweep.expand())
    assert expanded and all(c.memcpy_order == "col" for c in expanded)


def test_cli_beats_toml(tmp_path):
    sweep = SweepConfig.from_toml(_toml(tmp_path, 'memcpy_order = "row"'),
                                  cli_override_io={"memcpy_order": "col"})
    assert sweep.memcpy_order == "col"


def test_bad_toml_order_raises(tmp_path):
    with pytest.raises(ValueError, match="memcpy_order"):
        SweepConfig.from_toml(_toml(tmp_path, 'memcpy_order = "sideways"'))


def test_order_is_not_mistaken_for_a_sweep_parameter(tmp_path):
    sweep = SweepConfig.from_toml(_toml(tmp_path, 'memcpy_order = "col"'))
    assert "memcpy_order" not in sweep.sweep_params


# --------------------------------------------------------------------------
# results CSV
# --------------------------------------------------------------------------

def test_results_record_order_and_the_copy_mode_setup_transfer():
    names = {f.name for f in fields(BenchmarkResult)}
    assert {"memcpy_order", "setup_h2d_bytes", "setup_h2d_time"} <= names
    r = BenchmarkResult()
    assert r.memcpy_order == "row"
    assert (r.setup_h2d_bytes, r.setup_h2d_time) == (0, 0.0)


def test_new_columns_round_trip_through_the_csv(tmp_path):
    path = tmp_path / "order.csv"
    for order in MEMCPY_ORDERS:
        save_benchmark(BenchmarkResult(memcpy_order=order, h2d_mode="full",
                                       setup_h2d_bytes=316_000_000,
                                       setup_h2d_time=0.05), str(path))
    with open(path) as f:
        cols = csv.DictReader(f).fieldnames or []
    assert {"memcpy_order", "setup_h2d_bytes", "setup_h2d_time"} <= set(cols)

    rows = load_benchmarks(str(path))
    assert [r.memcpy_order for r in rows] == list(MEMCPY_ORDERS)
    assert all(r.setup_h2d_bytes == 316_000_000 for r in rows)


def test_worker_populates_the_new_fields():
    """Statically pinned: the appliance worker cannot be exercised from here."""
    src = worker_source()
    assert "memcpy_order=args.memcpy_order," in src
    assert "setup_h2d_bytes=setup_h2d_bytes," in src
    assert "setup_h2d_time=setup_h2d_time," in src


def test_setup_transfer_is_still_copy_mode():
    """The whole point of the setup measurement is streaming=False.

    If someone flips these to streaming=True the column silently stops being a
    link measurement and starts including per-word kernel cost.
    """
    src = worker_source()
    setup = src[src.index("Setup phase"):src.index("Compute phase")]
    # Count code only -- the explanatory comment names the flag too.
    code = [ln for ln in setup.splitlines()
            if not ln.lstrip().startswith("#")]
    # One call site in the strip loop serves x_vec, col_indices, row_ptrs and (valtask/val) val_meta.
    assert sum(ln.count("streaming=False") for ln in code) == 1
    assert not any("streaming=True" in ln for ln in code)


def test_setup_bytes_count_all_three_arrays():
    src = worker_source()
    # The mode adds its own arrays: val_meta[1] (stream, val encodings), vals / row_ids (resident).
    assert ("setup_h2d_bytes = (4 * W * H * (KX + max_nnz + (KY + 1))\n"
            "                               + mode.setup_extra_bytes(r))") in src
    assert "return 4 * r.W * r.H * (1 if r.val_only else 0)" in src
