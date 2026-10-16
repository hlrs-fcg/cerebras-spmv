"""Tests for the shipped TOML sweep configs."""

from pathlib import Path

import pytest

from cluster.sweep_config import SweepConfig
from config import PE_SRAM_BUDGET_BYTES, SIM_MAX_GRID_DIM

CONFIG_DIR = Path("bench/configs")


def test_sim_strong_scaling_toml_loads_and_expands():
    """Verify sim-strong-scaling.toml loads and expands to 7 configs."""
    config_path = Path("bench/configs/sim-strong-scaling.toml")
    assert config_path.exists(), f"Config not found: {config_path}"
    
    sweep = SweepConfig.from_toml(config_path)
    configs = list(sweep.expand())
    
    assert len(configs) == 7, f"Expected 7 configs, got {len(configs)}"
    
    # Verify PE counts
    pe_counts = [cfg.num_pes for cfg in configs]
    expected = [64, 128, 256, 512, 1024, 2048, 4096]
    assert pe_counts == expected, f"PE counts mismatch: {pe_counts} != {expected}"


def test_sim_weak_scaling_toml_loads_and_expands():
    """Verify sim-weak-scaling.toml loads and expands to 7 configs."""
    config_path = Path("bench/configs/sim-weak-scaling.toml")
    assert config_path.exists(), f"Config not found: {config_path}"
    
    sweep = SweepConfig.from_toml(config_path)
    configs = list(sweep.expand())
    
    assert len(configs) == 7, f"Expected 7 configs, got {len(configs)}"
    
    # Verify PE counts
    pe_counts = [cfg.num_pes for cfg in configs]
    expected = [64, 128, 256, 512, 1024, 2048, 4096]
    assert pe_counts == expected, f"PE counts mismatch: {pe_counts} != {expected}"


def test_all_expanded_configs_have_simulator_true():
    """Verify all expanded configs have simulator=True."""
    for config_name in ["sim-strong-scaling.toml", "sim-weak-scaling.toml"]:
        config_path = Path(f"bench/configs/{config_name}")
        sweep = SweepConfig.from_toml(config_path)
        configs = list(sweep.expand())
        
        for cfg in configs:
            assert cfg.simulator is True, f"Config {config_name} has simulator=False"


def test_max_pe_count_is_4096():
    """Verify max PE count is 4096."""
    for config_name in ["sim-strong-scaling.toml", "sim-weak-scaling.toml"]:
        config_path = Path(f"bench/configs/{config_name}")
        sweep = SweepConfig.from_toml(config_path)
        configs = list(sweep.expand())
        
        max_pe = max(cfg.num_pes for cfg in configs)
        assert max_pe == 4096, f"Max PE count is {max_pe}, expected 4096"


@pytest.mark.parametrize("config_path", sorted(CONFIG_DIR.glob("*.toml")),
                         ids=lambda p: p.stem)
def test_shipped_config_expands_within_the_pe_sram_budget(config_path):
    """Every shipped sweep must produce runs that actually fit on a PE.

    SpMVConfig raises when a run does not fit, so a config whose matrix is too
    large for its smallest PE count fails here rather than mid-campaign.
    """
    sweep = SweepConfig.from_toml(config_path)
    configs = list(sweep.expand())
    assert configs, f"{config_path.name} expanded to zero runs"

    for cfg in configs:
        words = cfg.kx + cfg.total_nnz_per_pe + (cfg.ky + 1) + cfg.ky + cfg.ky
        assert words * 4 <= PE_SRAM_BUDGET_BYTES, (
            f"{config_path.name} at {cfg.num_pes} PEs needs {words * 4 / 1024:.1f} KB"
        )


@pytest.mark.parametrize("config_path", sorted(CONFIG_DIR.glob("sim-*.toml")),
                         ids=lambda p: p.stem)
def test_simulator_config_grids_stay_within_the_simulator_cap(config_path):
    sweep = SweepConfig.from_toml(config_path)
    for cfg in sweep.expand():
        assert cfg.width <= SIM_MAX_GRID_DIM
        assert cfg.height <= SIM_MAX_GRID_DIM


@pytest.mark.parametrize("config_path", sorted(CONFIG_DIR.glob("sim-*.toml")),
                         ids=lambda p: p.stem)
def test_simulator_config_grids_are_near_square(config_path):
    """A 1-high grid makes KY the full row count, which is what broke before."""
    sweep = SweepConfig.from_toml(config_path)
    for cfg in sweep.expand():
        assert cfg.height > 1
        assert max(cfg.width, cfg.height) / min(cfg.width, cfg.height) <= 2
