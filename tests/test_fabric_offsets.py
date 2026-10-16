"""--width-west-buf needs --fabric-offsets=4+width_west_buf,1 on the SpMV path too.

cslc's memcpy generator, SDK 2.10.0: "The core must have --fabric-offsets=4+width_west_buf,1".
The default option string must not move, or every existing artifact hash does.
"""

import re

import pytest

from config import DEFAULT_FABRIC_OFFSETS, SpMVConfig
from sdk.compiler import compile_options


def _cfg(**kw):
    return SpMVConfig(rows=1000, cols=1000, density=0.01, num_pes_target=16, **kw)


def _off(cfg):
    return tuple(int(v) for v in re.search(r"--fabric-offsets=(\d+),(\d+)",
                                           compile_options(cfg)).groups())


def test_old_explicit_options_give_the_old_option_string_byte_for_byte():
    """Pinned literally: the hardware string the pre-2026-10-08 defaults produced (1 channel,
    header kernel), reached now by naming them.  If this changes, every artifact hash of those
    runs changes with it."""
    cfg = _cfg(channels=1, encoding="header")
    assert cfg.fabric_offsets == (4, 1) == DEFAULT_FABRIC_OFFSETS
    assert compile_options(cfg) == (
        f"--fabric-dims={cfg.fabric_dims[0]},{cfg.fabric_dims[1]} "
        f"--fabric-offsets=4,1 "
        f"--memcpy --channels=1 "
        f"--arch=wse3 "
        f"--params=width:{cfg.width},height:{cfg.height},KX:{cfg.kx},KY:{cfg.ky},"
        f"TOTAL_NNZ:{cfg.total_nnz_per_pe},EW_ENABLED:1,MEMCPYH2D_DATA_1_ID:0,"
        f"ENCODING:0,SYNC_TIMING:0")


def test_new_default_option_string_is_pinned_literally():
    """The valtask kernel (ENCODING:2); 16 channels clamped to the 4-row grid; nothing else moved."""
    cfg = _cfg()
    assert compile_options(cfg) == (
        f"--fabric-dims={cfg.fabric_dims[0]},{cfg.fabric_dims[1]} "
        f"--fabric-offsets=4,1 "
        f"--memcpy --channels=4 "
        f"--arch=wse3 "
        f"--params=width:{cfg.width},height:{cfg.height},KX:{cfg.kx},KY:{cfg.ky},"
        f"TOTAL_NNZ:{cfg.total_nnz_per_pe},EW_ENABLED:1,MEMCPYH2D_DATA_1_ID:0,"
        f"ENCODING:2,SYNC_TIMING:0")


@pytest.mark.parametrize("sim", [False, True])
@pytest.mark.parametrize("channels", [1, 16])
def test_default_buffers_never_move_the_offsets(sim, channels):
    cfg = _cfg(simulator=sim, channels=channels)
    assert _off(cfg) == (4, 1) and "width-" not in compile_options(cfg)
    # an east buffer alone does not move the program either
    assert _off(_cfg(simulator=sim, width_east_buf=4)) == (4, 1)


@pytest.mark.parametrize("west", [1, 2, 4])
@pytest.mark.parametrize("east", [0, 1, 4])
def test_west_buffer_moves_the_x_offset(west, east):
    cfg = _cfg(width_west_buf=west, width_east_buf=east)
    assert cfg.fabric_offsets == (4 + west, 1)
    opts = compile_options(cfg)
    assert _off(cfg) == (4 + west, 1) and f"--width-west-buf={west}" in opts


def test_the_grid_is_fitted_into_what_is_left_of_the_fabric():
    cfg = SpMVConfig(rows=114000, cols=114000, density=0.001, num_pes_target=65536,
                     width_west_buf=4, channels=16)
    assert (cfg.width, cfg.height) == (256, 256) and cfg.fabric_offsets == (8, 1)
    assert cfg.fabric_offsets[0] + cfg.width + cfg.padding[0] <= cfg.fabric_dims[0]
    full = SpMVConfig(rows=10, cols=10, density=0.5)             # the whole usable fabric
    assert full.width == 762 - 4 - 16
    wider = SpMVConfig(rows=10, cols=10, density=0.5, width_west_buf=2)
    assert wider.width == 762 - 6 - 16, "max usable width shrinks by the west buffer"


def test_simulator_fabric_wraps_the_new_offset():
    cfg = _cfg(simulator=True, width_west_buf=2)
    assert cfg.fabric_dims[0] == cfg.width + 6 + cfg.padding[0]


def test_explicit_offsets_are_respected_and_a_round_trip_is_stable():
    cfg = _cfg(width_west_buf=2, fabric_offsets=(10, 1))
    assert cfg.fabric_offsets == (10, 1)
    derived = _cfg(width_west_buf=2)
    again = _cfg(width_west_buf=2, fabric_offsets=derived.fabric_offsets)
    assert again.fabric_offsets == derived.fabric_offsets == (6, 1)
    listy = _cfg(width_west_buf=3, fabric_offsets=[4, 1])        # a TOML list
    assert listy.fabric_offsets == (7, 1)
