"""Tests for --setup-order: memory order of the copy-mode setup transfers.

The setup memcpys (x_vec, col_indices, row_ptrs) are the only streaming=False
transfers in a run, and they were hardcoded ROW_MAJOR.  Cerebras documents
column-major as the faster order and benchmarks it in copy mode only, so this
knob tests that claim where it was actually made.  It is independent of
--memcpy-order because col-major is 2.2x worse for the streaming transfer.

The setup arrays are (W*H, l) indexed `py * W + px`, not the [w][h][l] blocks
the streaming path flattens, so they get their own helper.  A wrong layout is
silent apart from a bad relative error on device.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from cluster.sweep_config import IO_KEYS, OVERRIDE_KEYS, RUNTIME_KEYS, SweepConfig
from config import DEFAULT_SETUP_ORDER, MEMCPY_ORDERS, SpMVConfig
from bench.benchmark import BenchmarkResult
from run import build_parser, io_overrides
from sdk.launcher import build_worker_command, compile_io_settings

from tests.worker_src import worker_source  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKER = PROJECT_ROOT / "cluster" / "worker_runtime.py"


def _worker():
    """Import the worker the way the appliance does: flat, no package."""
    sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "bench"),
                    str(PROJECT_ROOT / "cluster")]
    import worker_runtime
    return worker_runtime


def _pe_major(W, H, length):
    """A (W*H, l) array whose entry encodes (py, px, k), as the worker builds."""
    arr = np.zeros((W * H, length), dtype=np.float32)
    for py in range(H):
        for px in range(W):
            for k in range(length):
                arr[py * W + px, k] = py * 10_000 + px * 100 + k
    return arr


def test_row_is_the_plain_ravel_every_existing_result_used():
    wr = _worker()
    arr = _pe_major(4, 3, 5)
    out = wr._flatten_pe_major(arr, 4, 3, "row")
    assert np.array_equal(out, arr.ravel())
    # A view, so the row path adds no host copy (it is outside the bracket
    # anyway, but the historical path had none).
    assert np.shares_memory(out, arr)


def test_col_matches_the_shipped_reference_loop():
    """`for l / for w / for h`, h innermost -- bandwidth-test:178."""
    wr = _worker()
    W, H, length = 4, 3, 5
    arr = _pe_major(W, H, length)
    expected = [arr[py * W + px, k]
                for k in range(length) for px in range(W) for py in range(H)]
    assert list(wr._flatten_pe_major(arr, W, H, "col")) == expected


def test_col_agrees_with_the_streaming_helper_on_the_same_data():
    wr = _worker()
    W, H, length = 3, 5, 2
    arr = _pe_major(W, H, length)
    whl = arr.reshape(H, W, length).transpose(1, 0, 2)
    for order in MEMCPY_ORDERS:
        assert np.array_equal(wr._flatten_pe_major(arr, W, H, order),
                              wr._flatten_whl(whl, order))


def test_setup_order_is_runtime_and_independent_of_memcpy_order():
    assert "setup_order" in RUNTIME_KEYS
    assert "setup_order" not in IO_KEYS
    assert "setup_order" in OVERRIDE_KEYS
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.001, num_pes_target=256,
                     simulator=True, memcpy_order="row", setup_order="col")
    assert (cfg.memcpy_order, cfg.setup_order) == ("row", "col")
    assert "setup_order" not in compile_io_settings(cfg)
    assert cfg.uses_non_default_io is False


def test_default_is_col_and_old_rows_are_row():
    """col since 2026-10-08; a row written before the flag existed (BenchmarkResult default)
    and an explicit setup_order="row" are still row-major."""
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.001, num_pes_target=256,
                     simulator=True)
    assert cfg.setup_order == "col" == DEFAULT_SETUP_ORDER
    assert SpMVConfig(rows=1000, cols=1000, density=0.001, num_pes_target=256,
                      simulator=True, setup_order="row").setup_order == "row"
    assert BenchmarkResult().setup_order == "row"


def test_unknown_order_raises_before_the_compile():
    with pytest.raises(ValueError, match="setup_order"):
        SpMVConfig(rows=1000, cols=1000, density=0.001, num_pes_target=256,
                   simulator=True, setup_order="diagonal")


@pytest.mark.parametrize("order", MEMCPY_ORDERS)
def test_worker_command_carries_the_order(order):
    cfg = SpMVConfig(rows=10_000, cols=10_000, density=0.001, num_pes_target=256,
                     simulator=True, setup_order=order)
    assert f"--setup-order {order}" in build_worker_command(cfg)


def test_cli_flag_on_both_subcommands_and_omitted_when_unset():
    parser = build_parser()
    for argv in (["run", "--rows", "10", "--cols", "10", "--density", "0.1",
                  "--setup-order", "col"],
                 ["sweep", "x.toml", "--setup-order", "col"]):
        args = parser.parse_args(argv)
        assert io_overrides(args)["setup_order"] == "col"
    assert "setup_order" not in io_overrides(parser.parse_args(["sweep", "x.toml"]))


def test_sweep_accepts_the_cli_override_and_rejects_bad_toml(tmp_path):
    toml = tmp_path / "s.toml"
    toml.write_text('[sweep]\ntype = "fixed"\nsetup_order = "col"\n')
    assert SweepConfig.from_toml(toml).setup_order == "col"
    assert SweepConfig.from_toml(
        toml, cli_override_io={"setup_order": "row"}).setup_order == "row"
    toml.write_text('[sweep]\ntype = "fixed"\nsetup_order = "diagonal"\n')
    with pytest.raises(ValueError, match="setup_order"):
        SweepConfig.from_toml(toml)


def test_worker_wires_the_order_into_all_three_setup_calls_and_the_result():
    src = worker_source()
    setup = src[src.index("Setup phase"):src.index("Compute phase")]
    # One call site inside the strip loop serves x_vec, col_indices, row_ptrs and (valtask/val)
    # val_meta; each array is flattened by setup_strip_calls with args.setup_order.
    assert setup.count("order=setup_enum") == 1
    for arr in ("(sym_x_vec, r.x_grid, KX)", "(sym_col_indices, r.ci, max_nnz)",
                "(sym_row_ptrs, r.rp, KY + 1)"):
        assert arr in setup
    assert 'return [(d.syms["val_meta"], r.val_plan.pe_words, 1)]' in src   # StreamMode, val encodings
    assert "setup_strip_calls(arr, W, H, args.setup_order," in setup
    assert "MemcpyOrder.ROW_MAJOR" not in setup.replace(
        "else MemcpyOrder.ROW_MAJOR", "")
    # Flattening must happen before the timing bracket starts.
    assert setup.index("setup_strip_calls(arr") < setup.index("t_setup = ")
    assert "setup_order=args.setup_order," in src


# --------------------------------------------------------------------------
# end to end on the real tiler: what each PE would receive
# --------------------------------------------------------------------------

def _device_receive(flat, W, H, length, order):
    """Model the device: PE (px, py) offset k reads this host index.

    COL_MAJOR is `for l / for w / for h` (bandwidth-test:178), ROW_MAJOR is
    [h][w][l].  Returns recv[px][py] -> array of `length`.
    """
    recv = {}
    for px in range(W):
        for py in range(H):
            if order == "col":
                idx = [k * W * H + px * H + py for k in range(length)]
            else:
                idx = [(py * W + px) * length + k for k in range(length)]
            recv[px, py] = flat[idx]
    return recv


@pytest.mark.parametrize("order", MEMCPY_ORDERS)
def test_each_pe_receives_its_own_tile_structure(order):
    """ci and rp from extract_tile_structure, checked against the dense tile.

    W != H so a transposed layout cannot pass by symmetry.  A wrong col_indices
    layout would index the wrong x silently, so ground truth comes from the
    matrix itself, not from the tiler's py * W + px convention.
    """
    from cluster.coo_tiling import extract_tile_structure
    wr = _worker()
    W, H, KX, KY = 3, 2, 4, 5
    rng = np.random.default_rng(0)
    dense = rng.random((H * KY, W * KX)) < 0.3
    row_idx, col_idx = (a.astype(np.int32) for a in np.nonzero(dense))
    ci, rp, nnz_counts, _, _ = extract_tile_structure(row_idx, col_idx, W, H, KX, KY)

    ci_recv = _device_receive(wr._flatten_pe_major(ci, W, H, order),
                              W, H, ci.shape[1], order)
    rp_recv = _device_receive(wr._flatten_pe_major(rp, W, H, order),
                              W, H, KY + 1, order)
    for px in range(W):
        for py in range(H):
            tile = dense[py * KY:(py + 1) * KY, px * KX:(px + 1) * KX]
            r, c = np.nonzero(tile)            # row-major order, as the tiler sorts
            n = len(c)
            assert list(ci_recv[px, py][:n]) == list(c), (px, py)
            assert list(rp_recv[px, py]) == [0, *np.cumsum(tile.sum(axis=1))], (px, py)


@pytest.mark.parametrize("order", MEMCPY_ORDERS)
def test_each_pe_receives_its_own_x_slice(order):
    """x_grid built exactly as the worker builds it, KX != H != W."""
    wr = _worker()
    W, H, KX, cols = 3, 2, 4, 11                  # last column short: 11 < 12
    x = np.arange(cols, dtype=np.float32) + 1
    x_grid = np.zeros((W * H, KX), dtype=np.float32)
    for px in range(W):
        cs, ce = px * KX, min((px + 1) * KX, cols)
        for py in range(H):
            x_grid[py * W + px, :ce - cs] = x[cs:ce]
    recv = _device_receive(wr._flatten_pe_major(x_grid, W, H, order),
                           W, H, KX, order)
    for px in range(W):
        want = np.zeros(KX, dtype=np.float32)
        seg = x[px * KX:(px + 1) * KX]
        want[:len(seg)] = seg
        for py in range(H):
            assert np.array_equal(recv[px, py], want), (px, py)


@pytest.mark.parametrize("order", MEMCPY_ORDERS)
def test_each_pe_receives_its_own_val_meta(order):
    """val_meta is one word per PE (1-D, l = 1).  Each PE must receive its own
    planned word count: the val kernel waits for exactly that many words, so
    a misplaced count hangs the PE instead of failing."""
    wr = _worker()
    W, H = 3, 2
    pe_words = (np.arange(W * H, dtype=np.int32) + 1) * 10   # pe = py * W + px
    recv = _device_receive(wr._flatten_pe_major(pe_words, W, H, order),
                           W, H, 1, order)
    for px in range(W):
        for py in range(H):
            assert list(recv[px, py]) == [pe_words[py * W + px]], (px, py)
