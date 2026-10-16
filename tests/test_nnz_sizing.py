"""Tests for the per-PE NNZ sizing that feeds the TOTAL_NNZ compile param.

TOTAL_NNZ sizes col_indices[TOTAL_NNZ] in src/csl/pe.csl — one i32 per non-zero
in that PE's tile.  It used to carry a spurious `+ KY` term borrowed from
max_stream_words = max_nnz + KY (the length of the value *stream*, which the PE
never buffers), plus a fixed +100 margin that did not grow with PE count.
"""

import csv
import glob
import math
from pathlib import Path

import pytest

from config import (
    NNZ_ABS_FLOOR,
    NNZ_TAIL_SAFETY,
    PE_SRAM_BUDGET_BYTES,
    SpMVConfig,
    total_nnz_estimate,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FABRIC_W, FABRIC_H = 742, 1155
FULL_WAFER = FABRIC_W * FABRIC_H  # 857,010

# Measured per-tile load from cluster/results/data/ under the current block
# tiling: (num_pes, avg_nnz_per_pe, max_nnz_per_pe).
MEASURED = [
    (1,           2048.00, 2048),
    (4,           2500.00, 2551),
    (16,           625.00,  674),
    (1024,         191.41,  235),
    (65536,        198.30,  259),
    (428505,       198.98,  269),
]


def _old_total_nnz(avg: int, ky: int) -> int:
    """The formula this change replaced, for ceiling comparisons."""
    return max(avg + 100 + ky, 10)


def _guard_words(kx: int, ky: int, total_nnz: int) -> int:
    """x_vec[KX] + col_indices[TOTAL_NNZ] + row_ptrs[KY+1] + y_result[KY] + ew_buf[KY]."""
    return kx + total_nnz + (ky + 1) + ky + ky


# --------------------------------------------------------------------------
# the margin covers measured hardware
# --------------------------------------------------------------------------

@pytest.mark.parametrize("num_pes, avg, observed_max", MEASURED)
def test_margin_exceeds_every_measured_tile_maximum(num_pes, avg, observed_max):
    sized = total_nnz_estimate(int(avg), num_pes)
    assert sized >= observed_max, (
        f"P={num_pes} avg={avg}: sized {sized} < measured max {observed_max}"
    )


@pytest.mark.parametrize("num_pes, avg, observed_max", MEASURED)
def test_margin_has_room_to_spare_over_measured(num_pes, avg, observed_max):
    """Not merely covering the observation — covering it with headroom."""
    sized = total_nnz_estimate(int(avg), num_pes)
    needed = observed_max - avg
    if needed <= 0:
        assert sized > observed_max
        return
    have = sized - avg
    assert have >= 1.25 * needed, (
        f"P={num_pes} avg={avg}: margin {have:.0f} is under 1.25x the "
        f"observed excess {needed:.0f}"
    )


@pytest.mark.parametrize("num_pes, avg, observed_max", MEASURED)
def test_measured_spread_stays_under_the_sqrt_2_ln_p_model(num_pes, avg, observed_max):
    """The calibration claim itself: k = (max-avg)/sqrt(avg) <= sqrt(2 ln P)."""
    k = (observed_max - avg) / math.sqrt(avg)
    assert k <= math.sqrt(2 * math.log(max(num_pes, 2)))


def test_margin_covers_the_live_results_csvs():
    """Re-check against the CSVs themselves, so new runs re-validate the bound."""
    rows = []
    for path in sorted(glob.glob(str(PROJECT_ROOT / "cluster/results/data/*.csv"))):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    p = int(float(row["num_pes"]))
                    avg = float(row["avg_nnz_per_pe"])
                    mx = int(float(row["max_nnz_per_pe"]))
                except (KeyError, ValueError):
                    continue
                if avg > 0 and mx > 0:
                    rows.append((Path(path).name, p, avg, mx))

    if not rows:
        pytest.skip("no measured rows in cluster/results/data/")

    for src, p, avg, mx in rows:
        sized = total_nnz_estimate(int(avg), p)
        assert sized >= mx, f"{src}: P={p} avg={avg:.1f} max={mx} but sized {sized}"


# --------------------------------------------------------------------------
# the bug: no KY term
# --------------------------------------------------------------------------

def test_total_nnz_does_not_depend_on_ky():
    """Same matrix and PE count, different grid shape: TOTAL_NNZ must not move.

    Only grid_height_target differs, so avg NNZ per PE is identical by
    construction while KY changes by ~8x.  The old formula added KY here.
    """
    short = SpMVConfig(rows=20_000, cols=2_000, density=0.001,
                       num_pes_target=256, grid_height_target=128)
    tall = SpMVConfig(rows=20_000, cols=2_000, density=0.001,
                      num_pes_target=256, grid_height_target=16)

    assert tall.ky > 5 * short.ky
    assert short.num_pes == tall.num_pes
    assert short.total_nnz_per_pe == tall.total_nnz_per_pe

    # The replaced formula would have differed by exactly the KY gap.
    assert (_old_total_nnz(156, tall.ky) - _old_total_nnz(156, short.ky)
            == tall.ky - short.ky)


def test_total_nnz_is_close_to_avg_for_large_ky():
    """The old formula added KY here; a tall thin tile must not inflate the array."""
    cfg = SpMVConfig(rows=1_990_000, cols=1_990_000, density=7 / 1_990_000)
    avg = int(cfg.rows * cfg.cols * cfg.density / cfg.num_pes)
    assert cfg.ky > 1500
    assert cfg.total_nnz_per_pe < avg + 100
    assert _old_total_nnz(avg, cfg.ky) > 20 * cfg.total_nnz_per_pe


# --------------------------------------------------------------------------
# degenerate inputs
# --------------------------------------------------------------------------

@pytest.mark.parametrize("num_pes", [1, 2, 64, 857_010])
def test_small_avg_never_produces_a_degenerate_size(num_pes):
    for avg in (0, 1, 2, 5, 10):
        sized = total_nnz_estimate(avg, num_pes)
        assert sized >= NNZ_ABS_FLOOR
        assert sized > avg


def test_single_pe_is_sane():
    """num_pes=1 has no spread at all, but log(1) = 0 must not zero the margin."""
    sized = total_nnz_estimate(2048, 1)
    assert sized > 2048
    assert sized < 2 * 2048


def test_zero_avg_is_the_floor():
    assert total_nnz_estimate(0, 1) == NNZ_ABS_FLOOR
    assert total_nnz_estimate(0, 857_010) == NNZ_ABS_FLOOR


def test_estimate_is_monotonic_in_avg_and_in_pe_count():
    assert total_nnz_estimate(100, 1024) < total_nnz_estimate(200, 1024)
    assert total_nnz_estimate(100, 64) < total_nnz_estimate(100, 857_010)


def test_safety_factor_and_floor_are_the_documented_values():
    assert NNZ_TAIL_SAFETY == 1.5
    assert NNZ_ABS_FLOOR == 16


# --------------------------------------------------------------------------
# dimension ceiling
# --------------------------------------------------------------------------

def _max_square_n(nnz_per_row: int, encoding: str = "header") -> int:
    """Largest square N that SpMVConfig accepts at full wafer."""
    lo, hi = 1000, 20_000_000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            SpMVConfig(rows=mid, cols=mid, density=nnz_per_row / mid, encoding=encoding)
            lo = mid
        except ValueError:
            hi = mid - 1
    return lo


@pytest.mark.parametrize("nnz_per_row, at_least", [(7, 2_400_000),
                                                   (27, 2_400_000),
                                                   (100, 2_300_000)])
def test_full_wafer_ceiling_is_lifted(nnz_per_row, at_least):
    """Header sizing (the valtask default adds vals[TOTAL_NNZ] and has a lower ceiling)."""
    assert _max_square_n(nnz_per_row) >= at_least


def test_old_formula_rejected_what_the_new_one_admits():
    """N=2.4M at full wafer: accepted now, over budget under the old sizing."""
    n, r = 2_400_000, 7
    cfg = SpMVConfig(rows=n, cols=n, density=r / n, encoding="header")  # must not raise

    avg = int(n * n * (r / n) / cfg.num_pes)
    old_words = _guard_words(cfg.kx, cfg.ky, _old_total_nnz(avg, cfg.ky))
    new_words = _guard_words(cfg.kx, cfg.ky, cfg.total_nnz_per_pe)

    assert old_words * 4 > PE_SRAM_BUDGET_BYTES
    assert new_words * 4 <= PE_SRAM_BUDGET_BYTES


def test_ceiling_is_a_real_edge():
    """One unknown past the ceiling must be rejected, not silently accepted."""
    n = _max_square_n(7)
    with pytest.raises(ValueError, match="usable SRAM per PE"):
        SpMVConfig(rows=n + 1, cols=n + 1, density=7 / (n + 1))


# --------------------------------------------------------------------------
# the guard still guards
# --------------------------------------------------------------------------

def test_guard_still_rejects_a_genuinely_oversized_config():
    with pytest.raises(ValueError, match="usable SRAM per PE"):
        SpMVConfig(rows=1_000_000, cols=1_000_000, density=0.001,
                   num_pes_target=64, simulator=True)


def test_guard_rejects_a_dense_tile_that_the_new_sizing_cannot_shrink():
    """Removing the KY term must not make an over-dense tile pass."""
    with pytest.raises(ValueError, match="usable SRAM per PE"):
        SpMVConfig(rows=20_000, cols=20_000, density=0.5, num_pes_target=4096)


def test_guard_accounts_for_the_sized_nnz_array():
    cfg = SpMVConfig(rows=10_000, cols=10_000, density=0.001,
                     num_pes_target=64, simulator=True)
    words = _guard_words(cfg.kx, cfg.ky, cfg.total_nnz_per_pe)
    assert words * 4 <= PE_SRAM_BUDGET_BYTES
    # ...and the NNZ array is genuinely part of what is being counted.
    assert cfg.total_nnz_per_pe > 0.1 * words
