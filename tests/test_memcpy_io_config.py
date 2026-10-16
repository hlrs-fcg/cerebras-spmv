"""Tests for the memcpy I/O configuration: --channels and the west/east buffers.

sdk/compiler.py used to hardcode `--memcpy --channels=1`.  Per the SDK's own
argument help (default=0), absent or 0 compiles the legacy single-I/O-channel
version while any N >= 1 compiles the multi-channel version with N channels, so
the value is a real performance knob and must be recorded alongside every
result.  Defaults here reproduce the previously hardcoded flags exactly.
"""

import csv
import shlex
import textwrap
import warnings
from dataclasses import fields
from pathlib import Path

import pytest

from bench.benchmark import BenchmarkResult, load_benchmarks, save_benchmark
from cluster.sweep_config import IO_KEYS, SweepConfig
from config import (
    DEFAULT_CHANNELS,
    DEFAULT_WIDTH_EAST_BUF,
    DEFAULT_WIDTH_WEST_BUF,
    SpMVConfig,
)
from run import build_parser, io_overrides
from sdk.compiler import compile_options
from sdk.launcher import compile_io_settings

IO_FIELDS = ("channels", "width_west_buf", "width_east_buf")
# 16 channels since 2026-10-08 (was 1); the buffers never changed
IO_DEFAULTS = {"channels": 16, "width_west_buf": 0, "width_east_buf": 0}


@pytest.fixture
def cfg():
    return SpMVConfig(rows=10_000, cols=10_000, density=0.001,
                      num_pes_target=256, simulator=True)


def _sweep_toml(tmp_path: Path, body: str = "") -> Path:
    path = tmp_path / "io-probe.toml"
    path.write_text(textwrap.dedent(f"""\
        [meta]
        name = "io-probe"

        [output]
        results = "cluster/results/data/{{date}}_{{name}}.csv"

        [sweep]
        type    = "strong_scaling"
        rows    = 10_000
        cols    = 10_000
        density = 0.001
        {body}

        [sweep.num_pes]
        pow2_up_to        = 256
        points_per_octave = 1
    """))
    return path


# --------------------------------------------------------------------------
# defaults preserve the previous hardcoded behaviour
# --------------------------------------------------------------------------

def test_config_defaults_are_16_channels_and_no_buffers():
    assert (DEFAULT_CHANNELS, DEFAULT_WIDTH_WEST_BUF, DEFAULT_WIDTH_EAST_BUF) == (16, 0, 0)


def test_config_fields_default_to_those_values(cfg):
    for name, expected in IO_DEFAULTS.items():
        assert getattr(cfg, name) == expected


@pytest.mark.parametrize("channels", [1, 16])
def test_option_string_flag_section_is_pinned_literally(channels):
    """Pinned against the literal format, not rebuilt from the function under test, so a change
    to the flag section fails here.  channels=1 is the pre-2026-10-08 default, still reachable
    by name and byte-identical."""
    cfg = SpMVConfig(rows=10_000, cols=10_000, density=0.001, num_pes_target=256,
                     simulator=True, channels=channels)
    head, _, params = compile_options(cfg).partition("--params=")
    assert head == (
        f"--fabric-dims={cfg.fabric_dims[0]},{cfg.fabric_dims[1]} "
        f"--fabric-offsets={cfg.fabric_offsets[0]},{cfg.fabric_offsets[1]} "
        f"--memcpy --channels={channels} "
        f"--arch={cfg.arch} "
    )
    assert params, "no --params section emitted"


def test_default_option_string_carries_no_width_buf_flags(cfg):
    options = compile_options(cfg)
    assert "--channels=1" in options
    assert "--width-west-buf" not in options
    assert "--width-east-buf" not in options


def test_default_config_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SpMVConfig(rows=1000, cols=1000, density=0.001)  # full fabric, defaults


# --------------------------------------------------------------------------
# plumbing into the compile options
# --------------------------------------------------------------------------

def test_channels_reaches_the_option_string():
    cfg = SpMVConfig(rows=10_000, cols=10_000, density=0.001,
                     num_pes_target=256, simulator=True, channels=16)
    assert "--channels=16" in compile_options(cfg)


def test_zero_channels_is_emitted_explicitly():
    """0 is meaningful (legacy path), so it must not be silently dropped."""
    cfg = SpMVConfig(rows=10_000, cols=10_000, density=0.001,
                     num_pes_target=256, simulator=True, channels=0)
    assert "--channels=0" in compile_options(cfg)


def test_width_bufs_are_emitted_only_when_non_zero():
    cfg = SpMVConfig(rows=10_000, cols=10_000, density=0.001,
                     num_pes_target=256, simulator=True,
                     width_west_buf=3, width_east_buf=5)
    options = compile_options(cfg)
    assert "--width-west-buf=3" in options
    assert "--width-east-buf=5" in options


def test_width_bufs_are_independent():
    cfg = SpMVConfig(rows=10_000, cols=10_000, density=0.001,
                     num_pes_target=256, simulator=True, width_west_buf=2)
    options = compile_options(cfg)
    assert "--width-west-buf=2" in options
    assert "--width-east-buf" not in options


def test_io_flags_precede_arch_and_params(cfg):
    """cslc flags must not land inside the --params value."""
    options = compile_options(cfg)
    assert options.index("--channels=") < options.index("--params=")


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", IO_FIELDS)
def test_negative_values_are_rejected(name):
    with pytest.raises(ValueError, match="non-negative int"):
        SpMVConfig(rows=1000, cols=1000, density=0.01,
                   num_pes_target=64, **{name: -1})


@pytest.mark.parametrize("name", IO_FIELDS)
@pytest.mark.parametrize("value", [1.5, "4", True])
def test_non_int_values_are_rejected(name, value):
    with pytest.raises(ValueError, match="non-negative int"):
        SpMVConfig(rows=1000, cols=1000, density=0.01,
                   num_pes_target=64, **{name: value})


def test_non_default_io_on_a_full_width_grid_warns():
    """The geometry rule is undocumented, so a full-width grid must not go quietly."""
    with pytest.warns(UserWarning, match="memcpy halo"):
        SpMVConfig(rows=1000, cols=1000, density=0.001, channels=8)


def test_non_default_io_on_a_narrow_grid_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SpMVConfig(rows=1000, cols=1000, density=0.01,
                   num_pes_target=64, simulator=True, channels=8)


@pytest.mark.parametrize("kwargs, expected", [
    ({}, False),
    ({"channels": 16}, False),
    ({"channels": 1}, True),
    ({"channels": 0}, True),
    ({"channels": 4}, True),  # 8 is the clamped default on this 8x8 grid
    ({"width_west_buf": 1}, True),
    ({"width_east_buf": 1}, True),
])
def test_uses_non_default_io(kwargs, expected):
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.01,
                     num_pes_target=64, simulator=True, **kwargs)
    assert cfg.uses_non_default_io is expected


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sub", ["run --rows 64 --cols 64 --density 0.5",
                                 "sweep bench/configs/sim-weak-scaling.toml"])
def test_both_subcommands_accept_the_io_flags(sub):
    args = build_parser().parse_args(
        shlex.split(f"{sub} --channels 12 --width-west-buf 3 --width-east-buf 4")
    )
    assert (args.channels, args.width_west_buf, args.width_east_buf) == (12, 3, 4)


def test_unset_cli_flags_are_none_not_defaults():
    """None is what makes CLI > TOML > default precedence expressible."""
    args = build_parser().parse_args(shlex.split("run --rows 64 --cols 64 --density 0.5"))
    assert args.channels is None
    assert args.width_west_buf is None
    assert args.width_east_buf is None


def test_io_overrides_collects_only_what_was_given():
    parser = build_parser()
    bare = parser.parse_args(shlex.split("run --rows 64 --cols 64 --density 0.5"))
    assert io_overrides(bare) == {}

    partial = parser.parse_args(
        shlex.split("run --rows 64 --cols 64 --density 0.5 --channels 4")
    )
    assert io_overrides(partial) == {"channels": 4}


def test_zero_channels_survives_the_override_collection():
    """0 is falsy but meaningful; it must not be dropped as 'not given'."""
    args = build_parser().parse_args(
        shlex.split("run --rows 64 --cols 64 --density 0.5 --channels 0")
    )
    assert io_overrides(args) == {"channels": 0}


# --------------------------------------------------------------------------
# TOML schema and CLI > TOML > default precedence
# --------------------------------------------------------------------------

def test_sweep_defaults_when_toml_is_silent(tmp_path):
    sweep = SweepConfig.from_toml(_sweep_toml(tmp_path))
    assert sweep.io_settings == IO_DEFAULTS
    for cfg in sweep.expand():
        assert cfg.channels == min(16, cfg.height) and cfg.width_west_buf == 0 and cfg.width_east_buf == 0


def test_toml_values_reach_every_expanded_config(tmp_path):
    toml = _sweep_toml(tmp_path, "channels = 6\n        width_west_buf = 2\n        width_east_buf = 3")
    sweep = SweepConfig.from_toml(toml)
    assert sweep.io_settings == {"channels": 6, "width_west_buf": 2, "width_east_buf": 3}

    configs = list(sweep.expand())
    assert configs
    for cfg in configs:
        assert (cfg.channels, cfg.width_west_buf, cfg.width_east_buf) == (6, 2, 3)


def test_cli_overrides_the_toml(tmp_path):
    toml = _sweep_toml(tmp_path, "channels = 6\n        width_west_buf = 2")
    sweep = SweepConfig.from_toml(toml, cli_override_io={"channels": 99})
    assert sweep.channels == 99          # CLI wins
    assert sweep.width_west_buf == 2     # TOML survives where CLI is silent
    assert sweep.width_east_buf == 0     # default survives where both are silent


def test_io_keys_do_not_leak_into_sweep_params(tmp_path):
    toml = _sweep_toml(tmp_path, "channels = 6")
    sweep = SweepConfig.from_toml(toml)
    for key in IO_KEYS:
        assert key not in sweep.sweep_params


def test_unknown_override_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown per-run override"):
        SweepConfig.from_toml(_sweep_toml(tmp_path), cli_override_io={"chanels": 4})


def test_shipped_configs_keep_the_defaults():
    for name in ("sim-strong-scaling", "sim-weak-scaling",
                 "strong-scaling", "weak-scaling"):
        sweep = SweepConfig.from_toml(Path(f"bench/configs/{name}.toml"))
        assert sweep.io_settings == IO_DEFAULTS, f"{name} changed the I/O defaults"


# --------------------------------------------------------------------------
# BenchmarkResult: recording and CSV round-trip
# --------------------------------------------------------------------------

def test_benchmark_fields_exist_with_historical_defaults():
    """A row written before these columns existed is a 1-channel, no-buffer row: the
    BenchmarkResult defaults backfill history and stay 1 even though the config default is 16."""
    defaults = {f.name: f.default for f in fields(BenchmarkResult)}
    for name, expected in {"channels": 1, "width_west_buf": 0, "width_east_buf": 0}.items():
        assert name in defaults, f"BenchmarkResult is missing {name!r}"
        assert defaults[name] == expected


def test_csv_round_trip_preserves_io_columns(tmp_path):
    csv_path = tmp_path / "bench.csv"
    save_benchmark(BenchmarkResult(channels=12, width_west_buf=3, width_east_buf=4),
                   csv_path=str(csv_path))

    loaded = load_benchmarks(str(csv_path))
    assert len(loaded) == 1
    assert (loaded[0].channels, loaded[0].width_west_buf,
            loaded[0].width_east_buf) == (12, 3, 4)


def test_older_csv_reads_back_as_the_configuration_it_was_measured_under(tmp_path):
    """Rows written before these columns existed were all taken at channels=1."""
    csv_path = tmp_path / "old.csv"
    old_fieldnames = [f.name for f in fields(BenchmarkResult) if f.name not in IO_FIELDS]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=old_fieldnames)
        writer.writeheader()
        writer.writerow({"num_pes": 857010, "compute_time_hw_avg": 0.0066})

    loaded = load_benchmarks(str(csv_path))
    assert len(loaded) == 1
    assert loaded[0].channels == 1
    assert loaded[0].width_west_buf == 0
    assert loaded[0].width_east_buf == 0


@pytest.mark.parametrize("name", IO_FIELDS)
def test_io_columns_appear_in_the_csv_header(tmp_path, name):
    csv_path = tmp_path / "bench.csv"
    save_benchmark(BenchmarkResult(), csv_path=str(csv_path))
    with open(csv_path, newline="", encoding="utf-8") as f:
        assert name in (csv.DictReader(f).fieldnames or [])


# --------------------------------------------------------------------------
# launcher stamping — the worker cannot know the compile flags
# --------------------------------------------------------------------------

def test_compile_io_settings_reads_the_config():
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.01, num_pes_target=64,
                     simulator=True, channels=7, width_west_buf=1, width_east_buf=2)
    assert compile_io_settings(cfg) == {"channels": 7, "width_west_buf": 1,
                                        "width_east_buf": 2}


def test_downloaded_rows_are_stamped_with_the_compiled_io_settings(tmp_path):
    """A non-default-channels run must not record the default in its CSV."""
    from sdk.launcher import _download_benchmark

    worker_csv = tmp_path / "worker.csv"
    save_benchmark(BenchmarkResult(num_pes=64), csv_path=str(worker_csv))

    class _Launcher:
        def download_artifact(self, _name, dest):
            Path(dest).write_text(worker_csv.read_text())

    out = tmp_path / "results.csv"
    _download_benchmark(_Launcher(), out, git_commit="abc1234",
                        io_settings={"channels": 16, "width_west_buf": 3,
                                     "width_east_buf": 4})

    row = load_benchmarks(str(out))[0]
    assert row.git_commit == "abc1234"
    assert (row.channels, row.width_west_buf, row.width_east_buf) == (16, 3, 4)


def test_stamping_is_optional_and_leaves_git_commit_working(tmp_path):
    from sdk.launcher import _download_benchmark

    worker_csv = tmp_path / "worker.csv"
    save_benchmark(BenchmarkResult(num_pes=64), csv_path=str(worker_csv))

    class _Launcher:
        def download_artifact(self, _name, dest):
            Path(dest).write_text(worker_csv.read_text())

    out = tmp_path / "results.csv"
    _download_benchmark(_Launcher(), out, git_commit="deadbee")

    row = load_benchmarks(str(out))[0]
    assert row.git_commit == "deadbee"
    assert row.channels == 1


# --------------------------------------------------------------------------
# cslc: "Can not have more memcpy channels than kernel rows" (4x4 smoke, 2026-10-09)
# --------------------------------------------------------------------------

def test_default_channels_clamp_silently_to_the_grid_height():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = SpMVConfig(rows=1000, cols=1000, density=0.01, num_pes_target=16)
    assert (cfg.height, cfg.channels) == (4, 4)
    assert not cfg.uses_non_default_io


def test_explicit_channels_above_the_height_clamp_with_a_warning():
    with pytest.warns(UserWarning, match="clamped to 4"):
        cfg = SpMVConfig(rows=1000, cols=1000, density=0.01, num_pes_target=16, channels=8)
    assert cfg.channels == 4


def test_tall_enough_grids_keep_16_channels():
    cfg = SpMVConfig(rows=14300, cols=14300, density=0.001, num_pes_target=1024)
    assert (cfg.height, cfg.channels) == (32, 16)
