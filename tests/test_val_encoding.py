"""Tests for --encoding val: ADR-003's val-only stream and src/csl/pe_val.csl.

The CSL cannot be compiled here (the SDK is Linux-only), so the kernel is
checked through a numpy model of exactly what pe_val.csl does per pass:

    1. receive exactly val_meta[0] words (one async @fmovs, extent = that count)
    2. local CSR loop over row_ptrs / col_indices / vals, writing all KY rows
    3. the unchanged east->west EW reduction to column 0

and the model is fed through the REAL host path -- extract_tile_structure,
build_val_stream_buffer, plan_val_transfers and the worker's _flatten_whl --
with each memcpy unflattened the way the device would receive it.  The one
property the hardware enforces that a model would otherwise gloss over is
termination by count: a PE that is sent fewer words than val_meta says hangs,
more strands words in the queue.  The model raises on either.

Then the flag is pinned at every hop it has to survive, as for --h2d-mode and
--memcpy-order: CLI -> SpMVConfig -> cslc params -> worker command -> worker
cross-check -> BenchmarkResult/CSV, plus the TOML sweep and the bench config.
Unlike those two, encoding is COMPILE-TIME (it selects the kernel).
"""

import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from bench.benchmark import BenchmarkResult, load_benchmarks, save_benchmark
from cluster.coo_tiling import (
    MAX_VAL_STREAM_WORDS,
    ValTransferPlan,
    build_stream_buffer,
    build_val_stream_buffer,
    extract_tile_structure,
    plan_val_transfers,
    spmv_reference,
)
from cluster.sweep_config import IO_KEYS, KERNEL_KEYS, RUNTIME_KEYS, SweepConfig
from config import DEFAULT_ENCODING, ENCODING_IDS, ENCODINGS, SpMVConfig
from run import build_parser, io_overrides
from sdk.compiler import compile_options
from sdk.launcher import build_worker_command

from tests.worker_src import symbols_read, worker_source  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKER = PROJECT_ROOT / "cluster" / "worker_runtime.py"
CSL = PROJECT_ROOT / "src" / "csl"


def _worker():
    """Import the worker the way the appliance does: flat, no package."""
    for sub in ("", "bench", "cluster"):
        p = str(PROJECT_ROOT / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    import worker_runtime
    return worker_runtime


# --------------------------------------------------------------------------
# numpy model of the device
# --------------------------------------------------------------------------

def _deliver(plan, buf, W, H, order):
    """Per-PE word lists, as the device would receive the plan's memcpys.

    Each rect is flattened with the worker's own _flatten_whl and unflattened
    with the layout the memcpy order implies -- ROW_MAJOR (h, w, l), COL_MAJOR
    (l, w, h) -- so a transposed host layout shows up as wrong values here.
    """
    flatten = _worker()._flatten_whl
    got = [[np.empty(0, np.uint32) for _ in range(H)] for _ in range(W)]
    for x0, width, n in plan.rects:
        flat = flatten(buf[x0:x0 + width, :, :n], order)
        assert flat.size == width * H * n
        if order == "col":
            per_pe = flat.reshape(n, width, H).transpose(1, 2, 0)
        else:
            per_pe = flat.reshape(H, width, n).transpose(1, 0, 2)
        for i in range(width):
            for py in range(H):
                got[x0 + i][py] = np.concatenate([got[x0 + i][py], per_pe[i, py]])
    return got


def _pe_val_pass(ci, rp, x_grid, val_meta, delivered, W, H, KY):
    """pe_val.csl, one pass: count-terminated receive, local CSR, EW reduce."""
    y = np.zeros((H, W, KY), dtype=np.float32)
    for py in range(H):
        for px in range(W):
            pe = py * W + px
            words = delivered[px][py]
            expect = int(val_meta[pe])
            if len(words) < expect:
                raise RuntimeError(f"PE ({px},{py}) hangs: sent {len(words)}, expects {expect}")
            if len(words) > expect:
                raise RuntimeError(f"PE ({px},{py}) strands {len(words) - expect} words")
            vals = words.view(np.float32)
            for r in range(KY):
                acc = np.float32(0.0)
                for k in range(rp[pe, r], rp[pe, r + 1]):
                    acc = np.float32(acc + np.float32(vals[k] * x_grid[pe, ci[pe, k]]))
                y[py, px, r] = acc
    return y


def _pe_header_local(ci, rp, x_grid, buf, col_words, W, H, KY):
    """pe.csl stream_word over a column-mode header stream (pre-EW y)."""
    y = np.zeros((H, W, KY), dtype=np.float32)
    for py in range(H):
        for px in range(W):
            pe = py * W + px
            state, cur, rem, k = 0, 0, 0, 0
            for word in buf[px, py, :col_words[px]]:
                word = int(word)
                if state == 0:
                    if word == 0:
                        break
                    cur, rem = word & 0xFFFF, (word >> 16) & 0xFFFF
                    k, state = int(rp[pe, cur]), 1
                else:
                    v = np.uint32(word).view(np.float32)
                    y[py, px, cur] = np.float32(
                        y[py, px, cur] + np.float32(v * x_grid[pe, ci[pe, k]]))
                    k += 1
                    rem -= 1
                    if rem == 0:
                        state = 0
    return y


def _ew_reduce(y_local):
    """East->west chain: column 0 ends with sum over px, added east to west."""
    H, W, KY = y_local.shape
    carry = y_local[:, W - 1, :].copy()
    for px in range(W - 2, -1, -1):
        carry = (y_local[:, px, :] + carry).astype(np.float32)
    return carry.reshape(-1)


def _problem(rows, cols, nnz, W, H, seed=0, dup=False):
    rng = np.random.default_rng(seed)
    r = rng.integers(0, rows, nnz, dtype=np.int32)
    c = rng.integers(0, cols, nnz, dtype=np.int32)
    if dup and nnz >= 4:
        r[1], c[1] = r[0], c[0]          # duplicate COO entries must sum
        r[3], c[3] = r[2], c[2]
    v = rng.random(nnz).astype(np.float32)
    x = rng.integers(0, 100, cols).astype(np.float32)
    KX, KY = -(-cols // W), -(-rows // H)
    ci, rp, counts, order, rb = extract_tile_structure(r, c, W, H, KX, KY)
    x_grid = np.zeros((W * H, KX), dtype=np.float32)
    for px in range(W):
        cs, ce = px * KX, min((px + 1) * KX, cols)
        if cs < cols:
            for py in range(H):
                x_grid[py * W + px, :ce - cs] = x[cs:ce]
    return dict(r=r, c=c, v=v, x=x, KX=KX, KY=KY, ci=ci, rp=rp,
                counts=counts, order=order, rb=rb, x_grid=x_grid)


def _run_val(p, W, H, rows, mode, order):
    buf = build_val_stream_buffer(p["v"], p["order"], p["counts"], W, H)
    plan = plan_val_transfers(p["counts"], W, H, mode)
    got = _deliver(plan, buf, W, H, order)
    y_local = _pe_val_pass(p["ci"], p["rp"], p["x_grid"], plan.pe_words, got, W, H, p["KY"])
    return _worker().gather_y(_ew_reduce(y_local), H, p["KY"], rows), y_local, plan


SHAPES = [
    # rows, cols, nnz, W, H -- covers W==1 (no EW), 2x2, deep chains, ragged
    # edges (rows/cols not multiples of the grid), and non-square grids.
    (64, 64, 2048, 1, 1),
    (64, 64, 300, 2, 2),
    (200, 200, 2000, 8, 8),
    (97, 131, 900, 4, 3),
    (512, 512, 4000, 16, 4),
]


@pytest.mark.parametrize("mode", ["column", "full"])
@pytest.mark.parametrize("order", ["row", "col"])
@pytest.mark.parametrize("rows,cols,nnz,W,H", SHAPES)
def test_val_round_trip_matches_reference(rows, cols, nnz, W, H, mode, order):
    p = _problem(rows, cols, nnz, W, H, dup=True)
    y, _, _ = _run_val(p, W, H, rows, mode, order)
    ref = spmv_reference(p["r"], p["c"], p["v"], p["x"], rows)
    rel = np.linalg.norm(y - ref) / np.linalg.norm(ref)
    assert rel < 1e-6


@pytest.mark.parametrize("mode", ["column", "full"])
def test_zero_nnz_pes_and_empty_columns_terminate(mode):
    """The val-only analogue of the missing-sentinel deadlock.

    Very sparse: most PEs and whole PE columns get no NNZ.  Empty columns must
    send nothing (no zero-length memcpy) and their PEs must be told 0, so the
    kernel's n_words == 0 branch activates ew_start without waiting.
    """
    W, H, rows = 8, 8, 400
    p = _problem(rows, rows, 12, W, H, seed=3)
    counts = np.asarray(p["counts"]).reshape(H, W)
    assert (counts == 0).any()
    empty_cols = np.where(counts.sum(axis=0) == 0)[0]
    assert empty_cols.size, "fixture should include a fully empty PE column"

    y, _, plan = _run_val(p, W, H, rows, mode, "row")
    ref = spmv_reference(p["r"], p["c"], p["v"], p["x"], rows)
    assert np.allclose(y, ref, rtol=1e-6)
    assert all(n > 0 for _, _, n in plan.rects)
    if mode == "column":
        sent_cols = {x0 for x0, _, _ in plan.rects}
        assert not sent_cols & set(empty_cols.tolist())
        assert (plan.pe_words.reshape(H, W)[:, empty_cols] == 0).all()


def test_exact_fill_needs_no_padding_and_still_terminates():
    """Every PE equally full: the shape that once left no room for a sentinel.

    Val-only has no sentinel to lose -- it stops on its count -- and with no
    padding pe_words equals each PE's NNZ exactly.
    """
    W = H = 2
    r, c = np.meshgrid(np.arange(8, dtype=np.int32), np.arange(8, dtype=np.int32))
    r, c = r.ravel(), c.ravel()
    v = np.arange(1, 65, dtype=np.float32)
    x = np.arange(8, dtype=np.float32) + 1
    ci, rp, counts, order, rb = extract_tile_structure(r, c, W, H, 4, 4)
    assert len(set(counts)) == 1
    for mode in ("column", "full"):
        plan = plan_val_transfers(counts, W, H, mode)
        assert plan.pe_words.tolist() == counts


def test_all_zero_matrix_sends_nothing():
    plan = plan_val_transfers([0] * 6, 3, 2, "column")
    assert plan.rects == [] and plan.total_words == 0
    assert plan_val_transfers([0] * 6, 3, 2, "full").rects == []
    assert build_val_stream_buffer(np.zeros(0, np.float32), np.zeros(0, np.int64),
                                   [0] * 6, 3, 2).shape == (3, 2, 0)


def test_val_and_header_kernels_sum_each_row_identically():
    """Same accumulation order (row by row, stream order, from 0.0).

    In the model this is bitwise.  On hardware it is expected but NOT proven:
    the two kernels' codegen may contract `acc += v*x` and `y[r] += v*x`
    differently (FMA vs mul+add), which would show as ~1e-7 relative noise.
    """
    W, H, rows = 4, 4, 300
    p = _problem(rows, rows, 3000, W, H, seed=7)
    _, y_val, _ = _run_val(p, W, H, rows, "column", "row")
    hbuf, col_words = build_stream_buffer(p["v"], p["order"], p["counts"], p["rb"],
                                          W, H, num_threads=4)
    y_hdr = _pe_header_local(p["ci"], p["rp"], p["x_grid"], hbuf, col_words, W, H, p["KY"])
    assert np.array_equal(y_val.view(np.uint32), y_hdr.view(np.uint32))


def test_model_detects_a_plan_mismatch():
    """Guard the model itself: if val_meta and the transfers disagree it must
    not silently pass, because on hardware that disagreement is a hang."""
    W, H, rows = 2, 2, 64
    p = _problem(rows, rows, 200, W, H)
    buf = build_val_stream_buffer(p["v"], p["order"], p["counts"], W, H)
    plan = plan_val_transfers(p["counts"], W, H, "full")
    got = _deliver(plan, buf, W, H, "row")

    over = plan.pe_words.copy()
    over[0] += 1                      # PE told to expect one word too many
    with pytest.raises(RuntimeError, match="hangs"):
        _pe_val_pass(p["ci"], p["rp"], p["x_grid"], over, got, W, H, p["KY"])

    under = plan.pe_words.copy()
    under[0] -= 1                     # one word left behind in the queue
    with pytest.raises(RuntimeError, match="strands"):
        _pe_val_pass(p["ci"], p["rp"], p["x_grid"], under, got, W, H, p["KY"])


# --------------------------------------------------------------------------
# multiple passes, on the geometries that broke the header kernel
# --------------------------------------------------------------------------

class PeValDevice:
    """Stateful model of pe_val.csl across passes (a CG loop, loop_count > 1).

    What persists on hardware persists here: each PE's input queue (words a
    pass does not consume are still there next pass), vals[TOTAL_NNZ] and
    y_result[KY].  vals and y_result start as garbage, so a pass that read a
    stale slot or skipped writing a row would show up as a wrong y.
    """

    GARBAGE = np.float32(12345.0)

    def __init__(self, ci, rp, x_grid, val_meta, W, H, KY, total_nnz):
        self.ci, self.rp, self.x_grid = ci, rp, x_grid.copy()
        self.val_meta = np.asarray(val_meta).copy()
        self.W, self.H, self.KY = W, H, KY
        self.queue = [[np.empty(0, np.uint32) for _ in range(H)] for _ in range(W)]
        self.vals = np.full((W, H, max(total_nnz, 1)), self.GARBAGE, np.float32)
        self.y = np.full((H, W, KY), self.GARBAGE, np.float32)

    def stream(self, delivered):
        for px in range(self.W):
            for py in range(self.H):
                self.queue[px][py] = np.concatenate([self.queue[px][py], delivered[px][py]])

    def spmv_pass(self):
        W, H = self.W, self.H
        for py in range(H):
            for px in range(W):
                pe, q = py * W + px, self.queue[px][py]
                n = int(self.val_meta[pe])
                if len(q) < n:
                    raise RuntimeError(f"PE ({px},{py}) hangs: {len(q)} queued, expects {n}")
                if n > self.vals.shape[2]:
                    raise RuntimeError(f"PE ({px},{py}) overruns vals[]: {n} words")
                self.vals[px, py, :n] = q[:n].view(np.float32)
                self.queue[px][py] = q[n:]
                for r in range(self.KY):
                    acc = np.float32(0.0)
                    for k in range(self.rp[pe, r], self.rp[pe, r + 1]):
                        acc = np.float32(acc + np.float32(
                            self.vals[px, py, k] * self.x_grid[pe, self.ci[pe, k]]))
                    self.y[py, px, r] = acc
        # wait_ew returns and the host moves on: anything still queued would be
        # read as this PE's first values next pass (or keep the H2D from
        # retiring at all).
        left = [(px, py) for px in range(W) for py in range(H) if len(self.queue[px][py])]
        if left:
            raise RuntimeError(f"words stranded after the pass on PEs {left[:4]}")
        return _ew_reduce(self.y)


def _coo(rows, cols, density, seed, tmp_path):
    from cluster.coo_tiling import generate_random_coo
    return generate_random_coo(rows, cols, density, seed, cache_dir=str(tmp_path))


def _dense_8x8():
    r, c = np.meshgrid(np.arange(8, dtype=np.int32), np.arange(8, dtype=np.int32),
                       indexing="ij")
    return r.ravel(), c.ravel(), np.arange(1, 65, dtype=np.float32) / 7


# The three shapes that deadlocked the header kernel before the sentinel fix
# (archive/docs/streaming-rewrite-status.md section 2b): the 64x64 d=0.5 crash config
# on its 1x1 grid (cluster/results/data/single_64x64.csv: 1 PE, 2,048 NNZ),
# 8x8 dense on 2x2 (every PE exactly full), and 512x512 d=0.4 on 16x16.  The
# random ones use the real generator at seed 42, as the hardware runs did.
HEADER_KILLERS = {
    "64x64_d0.5_1x1":    (64, 64, 0.5, 1, 1),
    "8x8_dense_2x2":     (8, 8, None, 2, 2),
    "512x512_d0.4_16x16": (512, 512, 0.4, 16, 16),
}


@pytest.mark.parametrize("mode", ["column", "full"])
@pytest.mark.parametrize("name", list(HEADER_KILLERS))
def test_multi_pass_on_the_geometries_that_broke_the_header_kernel(name, mode, tmp_path):
    rows, cols, density, W, H = HEADER_KILLERS[name]
    r, c, v = _dense_8x8() if density is None else _coo(rows, cols, density, 42, tmp_path)
    if name == "64x64_d0.5_1x1":
        assert len(v) == 2048                      # the recorded crash config
    KX, KY = -(-cols // W), -(-rows // H)
    ci, rp, counts, order, _ = extract_tile_structure(r, c, W, H, KX, KY)
    if name == "8x8_dense_2x2":
        assert len(set(counts)) == 1               # exact fill, no padding

    rng = np.random.default_rng(1)
    x = rng.integers(0, 100, cols).astype(np.float32)

    def x_grid_for(xv):
        g = np.zeros((W * H, KX), dtype=np.float32)
        for px in range(W):
            cs, ce = px * KX, min((px + 1) * KX, cols)
            for py in range(H):
                g[py * W + px, :ce - cs] = xv[cs:ce]
        return g

    plan = plan_val_transfers(counts, W, H, mode)      # setup: once
    dev = PeValDevice(ci, rp, x_grid_for(x), plan.pe_words, W, H, KY, max(counts))
    gather = _worker().gather_y

    v2 = rng.random(len(v)).astype(np.float32)         # same sparsity, new values
    x2 = rng.integers(0, 100, cols).astype(np.float32)
    passes = [(v, x), (v, x), (v2, x), (v2, x2)]       # repeat, new A, new x
    ys = []
    for vals_in, x_in in passes:
        dev.x_grid = x_grid_for(x_in)                  # a CG loop re-sends x
        buf = build_val_stream_buffer(vals_in, order, counts, W, H)
        dev.stream(_deliver(plan, buf, W, H, "row"))   # same plan every pass
        y = gather(dev.spmv_pass(), H, KY, rows)
        ref = spmv_reference(r, c, vals_in, x_in, rows)
        assert np.linalg.norm(y - ref) / np.linalg.norm(ref) < 1e-6
        ys.append(y)
    # a repeated pass is bit-identical: nothing carries over between passes
    assert np.array_equal(ys[0].view(np.uint32), ys[1].view(np.uint32))


def test_multi_pass_catches_a_transfer_sent_twice():
    """Guard the stateful model: one extra transfer must surface as stranded
    words, not be absorbed silently by the next pass."""
    W = H = 2
    r, c, v = _dense_8x8()
    ci, rp, counts, order, _ = extract_tile_structure(r, c, W, H, 4, 4)
    plan = plan_val_transfers(counts, W, H, "full")
    buf = build_val_stream_buffer(v, order, counts, W, H)
    dev = PeValDevice(ci, rp, np.ones((4, 4), np.float32), plan.pe_words, W, H, 4, max(counts))
    dev.stream(_deliver(plan, buf, W, H, "row"))
    dev.stream(_deliver(plan, buf, W, H, "row"))
    with pytest.raises(RuntimeError, match="stranded"):
        dev.spmv_pass()


# --------------------------------------------------------------------------
# encoder and plan
# --------------------------------------------------------------------------

def test_val_buffer_holds_each_pes_values_in_csr_order():
    W, H = 4, 2
    p = _problem(50, 70, 400, W, H, seed=5)
    buf = build_val_stream_buffer(p["v"], p["order"], p["counts"], W, H)
    sorted_vals = p["v"][p["order"]].view(np.uint32)
    start = 0
    for pe, n in enumerate(p["counts"]):
        px, py = pe % W, pe // W
        assert np.array_equal(buf[px, py, :n], sorted_vals[start:start + n])
        assert not buf[px, py, n:].any()      # padding is zero, never read
        assert p["rp"][pe, -1] == n           # row_ptrs[KY] == this PE's NNZ
        start += n


def test_plan_shapes_per_mode():
    W, H = 3, 2
    counts = [5, 0, 2,
              1, 0, 7]
    col = plan_val_transfers(counts, W, H, "column")
    assert col.rects == [(0, 1, 5), (2, 1, 7)]
    assert col.pe_words.tolist() == [5, 0, 7, 5, 0, 7]
    assert col.total_words == H * (5 + 7)

    full = plan_val_transfers(counts, W, H, "full")
    assert full.rects == [(0, 3, 7)]
    assert full.pe_words.tolist() == [7] * 6
    assert full.total_words == W * H * 7

    for plan in (col, full):
        assert (plan.pe_words >= np.asarray(counts)).all()


def test_plan_rejects_more_words_than_the_per_pe_limit():
    with pytest.raises(ValueError, match="limit"):
        ValTransferPlan([(0, 1, MAX_VAL_STREAM_WORDS + 1)], 1, 1)
    ValTransferPlan([(0, 1, MAX_VAL_STREAM_WORDS)], 1, 1)


def test_plan_rejects_unknown_mode():
    with pytest.raises(ValueError):
        plan_val_transfers([1], 1, 1, "diagonal")


def test_val_removes_the_header_overhead_at_the_benchmark_tile_shape():
    """At the A/B's tile shape most NNZ sit alone in their row slice.

    Same per-PE geometry as 114,000^2 @ d=1e-3 on 256x256 (KX=KY=446, ~198
    NNZ/PE), on a 4x4 grid so it runs in milliseconds.  Header words are
    NNZ + non-empty rows + one sentinel per PE; val-only is NNZ exactly.

    At the real 65,536-PE point the same arithmetic gives 23.5M header words
    for 13.0M NNZ (+81% unpadded) and 119.5 MB as sent in full mode (+130%,
    matching the hardware log), against 67.9 MB val-only full.
    """
    W = H = 4
    side = 446 * 4
    p = _problem(side, side, 198 * 16, W, H, seed=42)
    nnz = sum(p["counts"])
    header_words = nnz + len(p["rb"].row) + W * H
    assert 1.7 < header_words / nnz < 1.9    # E[non-empty rows]/NNZ ~ 0.81 here
    for mode in ("column", "full"):
        plan = plan_val_transfers(p["counts"], W, H, mode)
        # no headers, no sentinels: only padding to the busiest PE separates
        # what is sent from NNZ, and that padding is never read
        assert plan.total_words >= nnz
        assert plan.total_words == int(plan.pe_words.sum())


# --------------------------------------------------------------------------
# flag threading
# --------------------------------------------------------------------------

@pytest.fixture
def cfg():
    """The header kernel, named explicitly (the default is valtask since 2026-10-08)."""
    return SpMVConfig(rows=10_000, cols=10_000, density=0.001,
                      num_pes_target=256, simulator=True, encoding="header")


def test_default_is_valtask_and_header_stays_selectable(cfg):
    """valtask since 2026-10-08 (443 vs 670 ms at 857K PEs); an old row (BenchmarkResult) and an
    explicit --encoding header are still header."""
    assert DEFAULT_ENCODING == "valtask"
    assert ENCODING_IDS == {"header": 0, "valtask": 2, "resident": 3}   # ids are compile params; 1 retired
    assert SpMVConfig(rows=10_000, cols=10_000, density=0.001, num_pes_target=256,
                      simulator=True).encoding == "valtask"
    assert cfg.encoding == "header"
    assert BenchmarkResult().encoding == "header"


def test_worker_and_config_agree_on_the_encoding_ids():
    assert _worker().ENCODINGS == ENCODINGS
    assert _worker().ENCODING_IDS == ENCODING_IDS


def test_config_rejects_an_unknown_encoding():
    with pytest.raises(ValueError, match="encoding"):
        SpMVConfig(rows=1000, cols=1000, density=0.01, num_pes_target=16,
                   simulator=True, encoding="csr")


def test_cli_flag_reaches_the_config():
    args = build_parser().parse_args(
        ["run", "--rows", "8", "--cols", "8", "--density", "0.5", "--encoding", "valtask"])
    assert io_overrides(args)["encoding"] == "valtask"
    args = build_parser().parse_args(["run", "--rows", "8", "--cols", "8", "--density", "0.5"])
    assert "encoding" not in io_overrides(args)    # not given: the config default (valtask) applies
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["run", "--rows", "8", "--cols", "8", "--density", "0.5", "--encoding", "x"])


def test_encoding_is_a_compile_param(cfg):
    assert "ENCODING:0" in compile_options(cfg)
    val = SpMVConfig(rows=10_000, cols=10_000, density=0.001,
                     num_pes_target=256, simulator=True, encoding="valtask",
                     h2d_mode="full")
    assert "ENCODING:2" in compile_options(val)
    # It is the ONLY difference, so the two kernels never share an artifact
    # and nothing else about the compile moves.
    assert (compile_options(val).replace("ENCODING:2", "ENCODING:0")
            == compile_options(cfg))
    assert "ENCODING:1" not in compile_options(val)     # id 1 (the bulk val kernel) is retired


def test_encoding_is_a_kernel_key_not_io_or_runtime():
    assert "encoding" in KERNEL_KEYS
    assert "encoding" not in IO_KEYS and "encoding" not in RUNTIME_KEYS
    # changes no fabric columns, so no halo warning
    val = SpMVConfig(rows=1000, cols=1000, density=0.001,
                     num_pes_target=256, simulator=True, encoding="valtask",
                     h2d_mode="full")
    assert val.uses_non_default_io is False


def test_worker_command_carries_the_encoding(cfg):
    assert "--encoding header" in build_worker_command(cfg)
    cfg.encoding = "valtask"
    assert "--encoding valtask" in build_worker_command(cfg)


def test_artifact_encoding_reads_out_json_params():
    w = _worker()
    assert w.artifact_encoding({"params": {"MEMCPYH2D_DATA_1_ID": "0"}}) == "header"
    assert w.artifact_encoding({"params": {"ENCODING": "0"}}) == "header"
    assert w.artifact_encoding({"params": {"ENCODING": "2"}}) == "valtask"
    assert w.artifact_encoding({"params": {"ENCODING": "3"}}) == "resident"
    assert w.artifact_encoding({"params": {"ENCODING": "4"}}) == "resident"   # --null-spmv
    with pytest.raises(ValueError, match="recompile"):
        w.artifact_encoding({"params": {"ENCODING": "1"}})            # the removed bulk val kernel


def test_worker_refuses_a_mismatched_artifact_before_the_device():
    """Statically pinned: driving one kernel with the other's stream hangs."""
    src = worker_source()
    check = src.index("compiled_encoding != args.encoding")
    assert check < src.index("SdkRuntimeWrapper(SdkRuntime")
    assert "encoding=args.encoding," in src


def _sweep_toml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "enc-probe.toml"
    path.write_text(textwrap.dedent(f"""\
        [meta]
        name = "enc-probe"

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


def test_sweep_block_sets_the_encoding(tmp_path):
    sweep = SweepConfig.from_toml(_sweep_toml(tmp_path, 'encoding = "valtask"\nh2d_mode = "full"'))
    assert {c.encoding for c in sweep.expand()} == {"valtask"}
    assert "encoding" not in sweep.sweep_params


def test_cli_beats_toml(tmp_path):
    sweep = SweepConfig.from_toml(_sweep_toml(tmp_path, 'encoding = "valtask"'),
                                  cli_override_io={"encoding": "header"})
    assert {c.encoding for c in sweep.expand()} == {"header"}


def test_sweep_rejects_an_unknown_encoding(tmp_path):
    with pytest.raises(ValueError, match="encoding"):
        SweepConfig.from_toml(_sweep_toml(tmp_path, 'encoding = "csr"'))


def test_csv_round_trips_the_column(tmp_path):
    path = tmp_path / "r.csv"
    for enc in ENCODINGS:
        save_benchmark(BenchmarkResult(encoding=enc), csv_path=str(path))
    assert [r.encoding for r in load_benchmarks(str(path))] == list(ENCODINGS)


# --------------------------------------------------------------------------
# SRAM budget
# --------------------------------------------------------------------------

def test_val_guard_adds_vals_and_val_meta():
    """pe_valtask.csl holds vals[TOTAL_NNZ] + val_meta[1] on top of pe.csl."""
    kw = dict(rows=200_000, cols=200_000, density=0.001, num_pes_target=1024)

    def reported(**extra):
        with pytest.raises(ValueError, match="usable SRAM per PE") as exc:
            SpMVConfig(**kw, **{"encoding": "header", **extra})
        return float(re.search(r"usage ([\d.]+) KB", str(exc.value)).group(1))

    with pytest.raises(ValueError) as exc:
        SpMVConfig(**kw, encoding="header")
    total_nnz = int(re.search(r"TOTAL_NNZ=(\d+)", str(exc.value)).group(1))
    assert reported(encoding="valtask", h2d_mode="full") - reported() == pytest.approx(
        (total_nnz + 1) * 4 / 1024, abs=0.1)


def test_val_extra_words_are_zero_for_header(cfg):
    assert cfg.val_only_extra_words == 0
    val = SpMVConfig(rows=10_000, cols=10_000, density=0.001,
                     num_pes_target=256, simulator=True, encoding="valtask",
                     h2d_mode="full")
    assert val.val_only_extra_words == val.total_nnz_per_pe + 1


def test_ab_geometry_fits_under_valtask():
    """The 65K benchmark point must not trip the guard once vals[] is added."""
    c = SpMVConfig(rows=114_000, cols=114_000, density=0.001,
                   num_pes_target=65_536, encoding="valtask", channels=8,
                   h2d_mode="full")
    assert c.val_only_extra_words < 1024


# --------------------------------------------------------------------------
# static CSL contract
# --------------------------------------------------------------------------

def _exports(path: Path) -> set[str]:
    return set(re.findall(r'@export_symbol\(\s*\w+\s*,\s*"(\w+)"', path.read_text()))


def test_header_kernel_knows_nothing_of_val_only():
    """pe.csl is the hardware-validated kernel; the val path lives elsewhere."""
    src = (CSL / "pe.csl").read_text()
    assert "val_meta" not in src and "ENCODING" not in src


def test_layout_selects_the_kernel_and_gates_the_export():
    src = (CSL / "layout.csl").read_text()
    assert re.search(r"param ENCODING\s*:", src)
    assert '"pe_valtask.csl"' in src and '"pe.csl"' in src
    assert '"pe_val.csl"' not in src                   # the bulk kernel is gone (id 1 retired)
    gated = re.search(r'if \(ENCODING == 2\) \{\s*@export_name\("val_meta"', src)
    assert gated, "val_meta must only be declared when pe_valtask.csl exports it"


