"""Tests for simulator result naming convention."""

from config import SpMVConfig
from run import default_results_path


def test_simulator_results_path_includes_sim_marker():
    """Simulator runs land in a sim_-prefixed CSV, hardware runs do not."""
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.01, simulator=True)
    assert cfg.simulator is True
    assert default_results_path(cfg).name == "sim_single_1000x1000.csv"


def test_hardware_results_path_has_no_sim_marker():
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.01, simulator=False)
    assert default_results_path(cfg).name == "single_1000x1000.csv"


def test_simulator_and_hardware_paths_do_not_collide():
    """The two modes must not append into the same CSV for the same matrix."""
    sim = SpMVConfig(rows=2048, cols=4096, density=0.01, simulator=True)
    hw = SpMVConfig(rows=2048, cols=4096, density=0.01, simulator=False)
    assert default_results_path(sim) != default_results_path(hw)
    assert default_results_path(sim).parent == default_results_path(hw).parent
