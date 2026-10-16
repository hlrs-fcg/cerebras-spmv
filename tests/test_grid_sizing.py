"""Tests for SpMVConfig grid shaping and the per-PE SRAM guard."""

import math
import re

import pytest

from config import (
    PE_SRAM_BUDGET_BYTES,
    SIM_MAX_GRID_DIM,
    SIM_MAX_PES,
    SpMVConfig,
    grid_shape,
    total_nnz_estimate,
)

HW_MAX_W = 762 - 4 - 16   # 742
HW_MAX_H = 1172 - 1 - 16  # 1155


def _sram_words(cfg: SpMVConfig) -> int:
    """x_vec[KX] + col_indices[TOTAL_NNZ] + row_ptrs[KY+1] + y_result[KY] + ew_buf[KY]."""
    return cfg.kx + cfg.total_nnz_per_pe + (cfg.ky + 1) + cfg.ky + cfg.ky


# --------------------------------------------------------------------------
# grid_shape
# --------------------------------------------------------------------------

@pytest.mark.parametrize("num_pes", [64, 128, 256, 512, 1024, 2048, 4096])
def test_powers_of_two_get_an_exact_near_square_grid(num_pes):
    w, h = grid_shape(num_pes, SIM_MAX_GRID_DIM, SIM_MAX_GRID_DIM)
    assert w * h == num_pes
    assert w <= SIM_MAX_GRID_DIM and h <= SIM_MAX_GRID_DIM
    assert w / h <= 2, f"{w}x{h} is not near-square"


def test_square_counts_are_square():
    assert grid_shape(64, 100, 100) == (8, 8)
    assert grid_shape(4096, 100, 100) == (64, 64)


def test_prime_target_does_not_collapse_to_a_one_high_strip():
    """181x1 would put every matrix row on every PE; round the grid up instead."""
    w, h = grid_shape(181, HW_MAX_W, HW_MAX_H)
    assert h > 1
    assert w * h >= 181
    assert w / h <= 2


def test_grid_covers_the_target_even_when_not_an_exact_factorisation():
    for n in range(64, 400):
        w, h = grid_shape(n, HW_MAX_W, HW_MAX_H)
        assert w * h >= n, f"{n}: grid {w}x{h} is smaller than the target"


def test_target_larger_than_the_box_raises():
    with pytest.raises(ValueError, match="exceeds the largest usable grid"):
        grid_shape(SIM_MAX_PES + 1, SIM_MAX_GRID_DIM, SIM_MAX_GRID_DIM)


def test_zero_or_negative_target_raises():
    with pytest.raises(ValueError, match="must be >= 1"):
        grid_shape(0, 100, 100)


def test_full_hardware_fabric_is_reachable():
    w, h = grid_shape(HW_MAX_W * HW_MAX_H, HW_MAX_W, HW_MAX_H)
    assert (w, h) == (HW_MAX_W, HW_MAX_H)


# --------------------------------------------------------------------------
# SpMVConfig: simulator sizing
# --------------------------------------------------------------------------

def test_simulator_grid_is_two_dimensional():
    """The old sizing forced H=1, which made KY equal the full row count."""
    cfg = SpMVConfig(rows=10_000, cols=10_000, density=0.001,
                     num_pes_target=4096, simulator=True)
    assert (cfg.width, cfg.height) == (64, 64)
    assert cfg.num_pes == 4096
    assert cfg.ky < cfg.rows


def test_simulator_fabric_wraps_the_grid():
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.01,
                     num_pes_target=256, simulator=True)
    assert cfg.fabric_dims == (cfg.width + 4 + 16, cfg.height + 1 + 16)


def test_simulator_honours_the_requested_pe_count_exactly():
    for target in [64, 128, 256, 512, 1024, 2048, 4096]:
        cfg = SpMVConfig(rows=10_000, cols=10_000, density=0.001,
                         num_pes_target=target, simulator=True)
        assert cfg.num_pes == target, f"{target} silently became {cfg.num_pes}"


def test_simulator_rejects_an_unsatisfiable_pe_count():
    """Previously clamped to 100 PEs and reported success."""
    with pytest.raises(ValueError, match="exceeds the largest usable grid"):
        SpMVConfig(rows=1000, cols=1000, density=0.01,
                   num_pes_target=SIM_MAX_PES + 1, simulator=True)


def test_simulator_without_a_target_stays_within_the_cap():
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.01, simulator=True)
    assert cfg.width <= SIM_MAX_GRID_DIM and cfg.height <= SIM_MAX_GRID_DIM


# --------------------------------------------------------------------------
# SpMVConfig: hardware sizing
# --------------------------------------------------------------------------

def test_hardware_default_is_the_full_fabric():
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.001)
    assert (cfg.width, cfg.height) == (HW_MAX_W, HW_MAX_H)


def test_hardware_rejects_a_target_beyond_the_fabric():
    with pytest.raises(ValueError, match="exceeds the largest usable grid"):
        SpMVConfig(rows=1000, cols=1000, density=0.001,
                   num_pes_target=HW_MAX_W * HW_MAX_H + 1)


def test_explicit_grid_height_is_honoured():
    cfg = SpMVConfig(rows=4096, cols=4096, density=0.001,
                     num_pes_target=1024, grid_height_target=16)
    assert cfg.height == 16
    assert cfg.width == math.ceil(1024 / 16)


def test_explicit_grid_height_beyond_the_fabric_raises():
    with pytest.raises(ValueError, match="exceeds the largest usable height"):
        SpMVConfig(rows=1000, cols=1000, density=0.001,
                   grid_height_target=HW_MAX_H + 1)


def test_explicit_grid_height_that_needs_too_much_width_raises():
    with pytest.raises(ValueError, match="exceeds the largest usable width"):
        SpMVConfig(rows=1000, cols=1000, density=0.001,
                   num_pes_target=HW_MAX_W * 4, grid_height_target=2)


# --------------------------------------------------------------------------
# SRAM guard
# --------------------------------------------------------------------------

def test_guard_counts_col_indices_once_not_twice():
    """a_stream is gone from pe.csl; the old formula double-counted TOTAL_NNZ.

    Pinned against the figure the guard reports for itself, so the coefficient
    is checked directly rather than via a budget-boundary coincidence.
    """
    rows = cols = 200_000
    with pytest.raises(ValueError, match="usable SRAM per PE") as exc:
        SpMVConfig(rows=rows, cols=cols, density=0.001, num_pes_target=1024,
                   encoding="header")    # header sizing; valtask adds vals[TOTAL_NNZ]
    reported_kb = float(re.search(r"usage ([\d.]+) KB", str(exc.value)).group(1))

    w, h = grid_shape(1024, HW_MAX_W, HW_MAX_H)
    kx, ky = -(-cols // w), -(-rows // h)
    total_nnz = total_nnz_estimate(int(rows * cols * 0.001 / (w * h)), w * h)

    once = (kx + total_nnz + (ky + 1) + ky + ky) * 4 / 1024
    twice = (kx + 2 * total_nnz + (ky + 1) + ky + ky) * 4 / 1024

    assert reported_kb == pytest.approx(once, abs=0.1)
    assert reported_kb != pytest.approx(twice, abs=0.1)


def test_guard_uses_three_ky_terms_not_four():
    """row_ptrs[KY+1] + y_result[KY] + ew_buf[KY]; TOTAL_NNZ no longer adds a KY."""
    rows = cols = 200_000
    with pytest.raises(ValueError, match="usable SRAM per PE") as exc:
        SpMVConfig(rows=rows, cols=cols, density=0.001, num_pes_target=1024)
    reported_kb = float(re.search(r"usage ([\d.]+) KB", str(exc.value)).group(1))

    w, h = grid_shape(1024, HW_MAX_W, HW_MAX_H)
    kx, ky = -(-cols // w), -(-rows // h)
    total_nnz = total_nnz_estimate(int(rows * cols * 0.001 / (w * h)), w * h)

    four_ky = (kx + total_nnz + (ky + 1) + ky + ky + ky) * 4 / 1024
    assert reported_kb != pytest.approx(four_ky, abs=0.1)


def test_guard_rejects_a_matrix_that_cannot_fit():
    with pytest.raises(ValueError, match="usable SRAM per PE"):
        SpMVConfig(rows=1_000_000, cols=1_000_000, density=0.001,
                   num_pes_target=64, simulator=True)


def test_guard_reports_the_shape_that_overflowed():
    with pytest.raises(ValueError, match=r"KX=\d+ KY=\d+ TOTAL_NNZ=\d+"):
        SpMVConfig(rows=1_000_000, cols=1_000_000, density=0.001,
                   num_pes_target=64, simulator=True)
