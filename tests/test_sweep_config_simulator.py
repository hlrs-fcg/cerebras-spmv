"""Tests for SweepConfig simulator flag propagation."""

import textwrap
from pathlib import Path

import pytest

from cluster.sweep_config import SweepConfig

SIM_STRONG = Path("bench/configs/sim-strong-scaling.toml")


def _write_toml(tmp_path: Path, *, simulator: bool) -> Path:
    """A minimal, feasible non-simulator sweep used to test flag plumbing.

    Purpose-built rather than borrowed from bench/configs so these tests fail
    only when the simulator flag misbehaves, not when a production campaign's
    matrix size changes.
    """
    flag = "simulator = true\n" if simulator else ""
    path = tmp_path / "flag-probe.toml"
    path.write_text(textwrap.dedent(f"""\
        [meta]
        name = "flag-probe"

        [output]
        results = "cluster/results/data/{{date}}_{{name}}.csv"

        [sweep]
        type    = "strong_scaling"
        {flag}rows    = 10_000
        cols    = 10_000
        density = 0.001
        seed    = 42

        [sweep.num_pes]
        pow2_up_to        = 256
        points_per_octave = 1
    """))
    return path


def test_sweepconfig_passes_simulator_true_to_expanded_configs():
    """A shipped simulator config propagates simulator=True to every run."""
    if not SIM_STRONG.exists():
        pytest.skip(f"{SIM_STRONG} not present")

    sweep = SweepConfig.from_toml(SIM_STRONG)
    assert sweep.simulator is True

    configs = list(sweep.expand())
    assert configs, "sweep expanded to zero configs"
    for cfg in configs:
        assert cfg.simulator is True


def test_sweepconfig_defaults_to_simulator_false(tmp_path):
    """A sweep with no simulator key stays on the hardware path."""
    sweep = SweepConfig.from_toml(_write_toml(tmp_path, simulator=False))
    assert sweep.simulator is False

    configs = list(sweep.expand())
    assert configs, "sweep expanded to zero configs"
    for cfg in configs:
        assert cfg.simulator is False


def test_cli_override_simulator(tmp_path):
    """--simulator on the CLI forces simulator=True over the TOML's default."""
    toml_path = _write_toml(tmp_path, simulator=False)

    sweep = SweepConfig.from_toml(toml_path, cli_override_simulator=True)
    assert sweep.simulator is True

    configs = list(sweep.expand())
    assert configs, "sweep expanded to zero configs"
    for cfg in configs:
        assert cfg.simulator is True


def test_cli_override_does_not_disable_simulator(tmp_path):
    """Omitting --simulator must not clear a TOML that asked for the simulator."""
    toml_path = _write_toml(tmp_path, simulator=True)

    sweep = SweepConfig.from_toml(toml_path, cli_override_simulator=False)
    assert sweep.simulator is True
