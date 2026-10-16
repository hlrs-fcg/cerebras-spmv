"""Tests for --h2d-mode: the shape of the streaming A-values H2D transfer.

cluster/worker_runtime.py has implemented both shapes since the sentinel fix,
but the flag was unreachable: sdk/launcher.build_worker_command never emitted
it, so the worker always took its argparse default and the mode was never
recorded in the results CSV.  That is the same failure that cost the first
channels sweep its independent variable, so the mode is pinned here at every
hop it has to survive: CLI -> SpMVConfig -> worker command -> BenchmarkResult.

--h2d-mode is deliberately NOT one of IO_KEYS.  Those three reach cslc and
change fabric geometry; this one reaches only the worker, so it must not drag
in the fabric-geometry warning.
"""

import csv
import textwrap
from dataclasses import fields
from pathlib import Path

import pytest

from tests.worker_src import worker_source

from bench.benchmark import BenchmarkResult, load_benchmarks, save_benchmark
from cluster.sweep_config import (
    BOOL_KEYS, IO_KEYS, KERNEL_KEYS, LOOP_KEYS, OVERRIDE_KEYS, RUNTIME_KEYS, SETUP_KEYS,
    SweepConfig,
)
from config import DEFAULT_H2D_MODE, H2D_MODES, SpMVConfig
from run import build_parser, io_overrides
from sdk.launcher import build_worker_command, compile_io_settings


@pytest.fixture
def cfg():
    return SpMVConfig(rows=10_000, cols=10_000, density=0.001,
                      num_pes_target=256, simulator=True)


def _sweep_toml(tmp_path: Path, body: str = "") -> Path:
    path = tmp_path / "h2d-probe.toml"
    path.write_text(textwrap.dedent(f"""\
        [meta]
        name = "h2d-probe"

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


# --------------------------------------------------------------------------
# the knob itself
# --------------------------------------------------------------------------

def test_modes_are_what_the_worker_accepts():
    """H2D_MODES must match cluster/worker_runtime.py's choices= exactly."""
    worker = worker_source()
    assert '"--h2d-mode", choices=["column", "full"]' in worker
    assert set(H2D_MODES) == {"column", "full"}


def test_default_is_the_historically_measured_shape(cfg):
    assert DEFAULT_H2D_MODE == "column"
    assert cfg.h2d_mode == "column"


@pytest.mark.parametrize("mode", H2D_MODES)
def test_valid_modes_are_accepted(mode):
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.001,
                     num_pes_target=256, simulator=True, h2d_mode=mode)
    assert cfg.h2d_mode == mode


@pytest.mark.parametrize("bad", ["COLUMN", "rectangle", "", None, 1])
def test_unknown_mode_raises_before_the_compile(bad):
    """Caught on the login node, not after a ~70 s appliance compile."""
    with pytest.raises(ValueError, match="h2d_mode"):
        SpMVConfig(rows=1000, cols=1000, density=0.001,
                   num_pes_target=256, simulator=True, h2d_mode=bad)


def test_h2d_mode_does_not_trigger_the_fabric_geometry_warning():
    """It changes no fabric columns, so it is not 'non-default I/O'."""
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.001,
                     num_pes_target=256, simulator=True, h2d_mode="full")
    assert cfg.uses_non_default_io is False


def test_runtime_and_compile_time_keys_stay_disjoint():
    assert "h2d_mode" in RUNTIME_KEYS
    assert "h2d_mode" not in IO_KEYS
    assert set(OVERRIDE_KEYS) == (set(IO_KEYS) | set(RUNTIME_KEYS) | set(KERNEL_KEYS)
                                  | set(BOOL_KEYS) | set(LOOP_KEYS) | set(SETUP_KEYS))
    assert not set(IO_KEYS) & set(RUNTIME_KEYS)
    assert not set(KERNEL_KEYS) & (set(IO_KEYS) | set(RUNTIME_KEYS))
    assert not set(BOOL_KEYS) & (set(IO_KEYS) | set(RUNTIME_KEYS) | set(KERNEL_KEYS))
    assert not set(LOOP_KEYS) & (set(IO_KEYS) | set(RUNTIME_KEYS) | set(KERNEL_KEYS)
                                 | set(BOOL_KEYS))


def test_compile_io_settings_excludes_the_runtime_knob(cfg):
    """The worker knows its own mode, so it must not be stamped post hoc."""
    assert "h2d_mode" not in compile_io_settings(cfg)


# --------------------------------------------------------------------------
# CLI -> SpMVConfig
# --------------------------------------------------------------------------

def test_cli_exposes_the_flag_on_both_subcommands():
    parser = build_parser()
    for argv in (["run", "--rows", "10", "--cols", "10", "--density", "0.1",
                  "--h2d-mode", "full"],
                 ["sweep", "x.toml", "--h2d-mode", "full"]):
        assert parser.parse_args(argv).h2d_mode == "full"


def test_cli_rejects_an_unknown_mode():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["sweep", "x.toml", "--h2d-mode", "diagonal"])


def test_omitted_flag_is_left_out_of_the_overrides():
    """None means 'not given', so TOML and config defaults still win."""
    parser = build_parser()
    args = parser.parse_args(["sweep", "x.toml"])
    assert args.h2d_mode is None
    assert "h2d_mode" not in io_overrides(args)


def test_named_flag_reaches_the_overrides():
    parser = build_parser()
    args = parser.parse_args(["sweep", "x.toml", "--h2d-mode", "full"])
    assert io_overrides(args)["h2d_mode"] == "full"


def test_overrides_are_accepted_by_spmvconfig():
    """io_overrides() is splatted straight into SpMVConfig by run.py."""
    parser = build_parser()
    args = parser.parse_args(["run", "--rows", "1000", "--cols", "1000",
                              "--density", "0.001", "--h2d-mode", "full",
                              "--channels", "8"])
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.001, num_pes_target=256,
                     simulator=True, **io_overrides(args))
    assert (cfg.h2d_mode, cfg.channels) == ("full", 8)


# --------------------------------------------------------------------------
# worker command
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", H2D_MODES)
def test_worker_command_carries_the_mode(mode):
    cfg = SpMVConfig(rows=10_000, cols=10_000, density=0.001,
                     num_pes_target=256, simulator=True, h2d_mode=mode)
    assert f"--h2d-mode {mode}" in build_worker_command(cfg)


def test_worker_command_always_states_the_mode(cfg):
    """Explicit even at the default, so the appliance log records the shape."""
    assert "--h2d-mode column" in build_worker_command(cfg)


# --------------------------------------------------------------------------
# TOML sweeps
# --------------------------------------------------------------------------

def test_toml_sets_the_mode_for_every_run(tmp_path):
    sweep = SweepConfig.from_toml(_sweep_toml(tmp_path, 'h2d_mode = "full"'))
    assert sweep.h2d_mode == "full"
    expanded = list(sweep.expand())
    assert expanded and all(c.h2d_mode == "full" for c in expanded)


def test_toml_default_is_column(tmp_path):
    sweep = SweepConfig.from_toml(_sweep_toml(tmp_path))
    assert sweep.h2d_mode == "column"
    assert all(c.h2d_mode == "column" for c in sweep.expand())


def test_cli_beats_toml(tmp_path):
    sweep = SweepConfig.from_toml(_sweep_toml(tmp_path, 'h2d_mode = "column"'),
                                  cli_override_io={"h2d_mode": "full"})
    assert sweep.h2d_mode == "full"
    assert all(c.h2d_mode == "full" for c in sweep.expand())


def test_toml_mode_is_not_mistaken_for_a_sweep_parameter(tmp_path):
    """h2d_mode is reserved; leaking it into sweep_params breaks expansion."""
    sweep = SweepConfig.from_toml(_sweep_toml(tmp_path, 'h2d_mode = "full"'))
    assert "h2d_mode" not in sweep.sweep_params


def test_bad_toml_mode_raises(tmp_path):
    with pytest.raises(ValueError, match="h2d_mode"):
        SweepConfig.from_toml(_sweep_toml(tmp_path, 'h2d_mode = "sideways"'))


def test_unknown_override_key_still_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown per-run override"):
        SweepConfig.from_toml(_sweep_toml(tmp_path),
                              cli_override_io={"h2d_modes": "full"})


# --------------------------------------------------------------------------
# results CSV — the independent variable has to survive to disk
# --------------------------------------------------------------------------

def test_benchmark_result_records_the_mode():
    assert "h2d_mode" in {f.name for f in fields(BenchmarkResult)}
    assert BenchmarkResult().h2d_mode == "column"


def test_mode_round_trips_through_the_csv(tmp_path):
    path = tmp_path / "h2d.csv"
    for mode in H2D_MODES:
        save_benchmark(BenchmarkResult(h2d_mode=mode, channels=8), str(path))

    with open(path) as f:
        assert "h2d_mode" in (csv.DictReader(f).fieldnames or [])

    rows = load_benchmarks(str(path))
    assert [r.h2d_mode for r in rows] == list(H2D_MODES)


def test_worker_populates_the_field_from_its_argument():
    """Statically pinned: the worker node cannot be exercised from here."""
    worker = worker_source()
    assert "h2d_mode=args.h2d_mode," in worker


# --------------------------------------------------------------------------
# explicit [[run]] entries
# --------------------------------------------------------------------------

EXPERIMENT = Path(__file__).resolve().parent.parent / \
    "bench" / "configs" / "resident.toml"


def test_explicit_runs_get_real_paths_not_strings():
    """Regression: TOML [[run]] scalars reached SpMVConfig uncoerced.

    cmd_sweep does cfg.out_path.mkdir(), which raised AttributeError on a str.
    Latent until this experiment, because it is the first shipped config to use
    explicit [[run]] entries rather than a [sweep] block.
    """
    for cfg in SweepConfig.from_toml(EXPERIMENT).expand():
        assert isinstance(cfg.out_path, Path), f"out_path is {type(cfg.out_path)}"


def test_mtx_file_is_coerced_too():
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.001, num_pes_target=256,
                     simulator=True, mtx_file="matrices/example.mtx")
    assert isinstance(cfg.mtx_file, Path)
