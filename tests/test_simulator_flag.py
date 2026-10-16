"""Tests for --simulator flag in run.py CLI."""

import subprocess
import sys


def test_run_subcommand_accepts_simulator_flag():
    """Verify 'run' subcommand accepts --simulator flag."""
    result = subprocess.run(
        [sys.executable, "run.py", "run", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--simulator" in result.stdout


def test_sweep_subcommand_accepts_simulator_flag():
    """Verify 'sweep' subcommand accepts --simulator flag."""
    result = subprocess.run(
        [sys.executable, "run.py", "sweep", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--simulator" in result.stdout


def test_simulator_flag_propagates_to_spMVConfig():
    """Verify simulator flag propagates to SpMVConfig."""
    from config import SpMVConfig
    
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.01, simulator=True)
    assert cfg.simulator is True
    
    cfg2 = SpMVConfig(rows=1000, cols=1000, density=0.01, simulator=False)
    assert cfg2.simulator is False
    
    cfg3 = SpMVConfig(rows=1000, cols=1000, density=0.01)
    assert cfg3.simulator is False  # default
