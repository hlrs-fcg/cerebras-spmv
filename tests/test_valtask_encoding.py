"""--encoding valtask: val's 4 B/NNZ stream, consumed by a per-word data task.

pe_val.csl's single armed @fmovs hung on hardware whenever streams into the PEs
overlap and a PE gets more words than the fabric buffers (column mode at 4x4
~650 words/PE; full mode at 65,536 PEs, channels 8).  pe_valtask.csl keeps the
stream, the plan and val_meta, but consumes word by word like pe.csl's
stream_word: vals[n] = word, n += 1, ew_start at n == val_meta[0], anything
after the count swallowed.

Checked through a stateful numpy model of exactly that (queues, vals[] and
y_result persist across passes and start as garbage), on the same problems and
geometries as the val tests, plus the static CSL contract and flag threading.
See docs/val-only-encoding.md, "valtask".
"""

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from cluster.coo_tiling import (
    VAL_ENCODINGS,
    build_val_stream_buffer,
    extract_tile_structure,
    plan_val_transfers,
    spmv_reference,
)
from cluster.sweep_config import SweepConfig
from config import ENCODINGS, SpMVConfig
from config import VAL_ENCODINGS as CONFIG_VAL_ENCODINGS
from run import build_parser, io_overrides
from sdk.compiler import compile_options
from sdk.launcher import build_worker_command
from tests.test_val_encoding import (
    HEADER_KILLERS,
    PeValDevice,
    _coo,
    _deliver,
    _dense_8x8,
    _ew_reduce,
    _problem,
)

from tests.worker_src import symbols_read, worker_source  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSL = PROJECT_ROOT / "src" / "csl"
WORKER = PROJECT_ROOT / "cluster" / "worker_runtime.py"


def _worker():
    sys.path.insert(0, str(PROJECT_ROOT / "cluster"))
    sys.path.insert(0, str(PROJECT_ROOT / "bench"))
    import worker_runtime
    return worker_runtime


def _code(path: Path) -> str:
    return "\n".join(ln.split("//")[0] for ln in path.read_text().splitlines())


# --------------------------------------------------------------------------
# stateful model of pe_valtask.csl
# --------------------------------------------------------------------------

class PeValTaskDevice:
    """pe_valtask.csl across passes.

    Per PE: an input queue (persists), vals[] and y_result (persist, start as
    garbage), and the stream state spmv() resets.  The data task runs whenever
    words are queued and the task is unblocked: below the count it stores, at
    the count it activates ew_start, past it (draining) it swallows.  Words can
    arrive in any number of chunks -- the per-word task makes progress on each,
    which is what lets it tolerate overlapping, backpressured streams.
    """

    GARBAGE = np.float32(12345.0)

    def __init__(self, ci, rp, x_grid, val_meta, W, H, KY, total_nnz):
        self.ci, self.rp, self.x_grid = ci, rp, x_grid.copy()
        self.val_meta = np.asarray(val_meta).copy()
        self.W, self.H, self.KY = W, H, KY
        self.queue = [[[] for _ in range(W)] for _ in range(H)]
        self.vals = np.full((H, W, max(total_nnz, 1)), self.GARBAGE, np.float32)
        self.y = np.full((H, W, KY), self.GARBAGE, np.float32)
        self.unblocked = np.zeros((H, W), bool)       # @block at comptime
        self.draining = np.ones((H, W), bool)
        self.n_recv = np.zeros((H, W), np.int64)
        self.n_expect = np.zeros((H, W), np.int64)
        self.done = np.zeros((H, W), bool)
        self.swallowed = 0

    def spmv(self):
        for py in range(self.H):
            for px in range(self.W):
                pe = py * self.W + px
                self.n_recv[py, px] = 0
                self.n_expect[py, px] = int(self.val_meta[pe])
                self.done[py, px] = False
                if self.n_expect[py, px] == 0:
                    self.draining[py, px] = True
                    self._ew_start(px, py)
                else:
                    self.draining[py, px] = False
                self.unblocked[py, px] = True
                self._run(px, py)

    def stream(self, delivered):
        """Append words to each PE's queue (one chunk) and let the task run."""
        for px in range(self.W):
            for py in range(self.H):
                self.queue[py][px].extend(int(w) for w in delivered[px][py])
                self._run(px, py)

    def _run(self, px, py):
        if not self.unblocked[py, px]:
            return
        q = self.queue[py][px]
        while q:
            word = q.pop(0)
            if self.draining[py, px]:
                self.swallowed += 1
                continue
            n = self.n_recv[py, px]
            if n >= self.vals.shape[2]:
                raise RuntimeError(f"PE ({px},{py}) overruns vals[]")
            self.vals[py, px, n] = np.uint32(word).view(np.float32)
            self.n_recv[py, px] = n + 1
            if n + 1 == self.n_expect[py, px]:
                self.draining[py, px] = True
                self._ew_start(px, py)

    def _ew_start(self, px, py):
        pe = py * self.W + px
        for r in range(self.KY):
            acc = np.float32(0.0)
            for k in range(self.rp[pe, r], self.rp[pe, r + 1]):
                acc = np.float32(acc + np.float32(
                    self.vals[py, px, k] * self.x_grid[pe, self.ci[pe, k]]))
            self.y[py, px, r] = acc
        self.done[py, px] = True

    def wait_ew(self):
        stuck = [(px, py) for py in range(self.H) for px in range(self.W)
                 if not self.done[py, px]]
        if stuck:
            raise RuntimeError(f"PEs never reached their count, hang: {stuck[:4]}")
        return _ew_reduce(self.y)


def _chunks(delivered, n_chunks, rng):
    """Split every PE's words into n_chunks arrival chunks at random cut points."""
    W, H = len(delivered), len(delivered[0])
    out = [[[None] * H for _ in range(W)] for _ in range(n_chunks)]
    for px in range(W):
        for py in range(H):
            words = delivered[px][py]
            cuts = np.sort(rng.integers(0, len(words) + 1, n_chunks - 1))
            parts = np.split(words, cuts)
            for c in range(n_chunks):
                out[c][px][py] = parts[c]
    return out


def _run_valtask(p, W, H, rows, mode, order, n_chunks=1, seed=0):
    buf = build_val_stream_buffer(p["v"], p["order"], p["counts"], W, H)
    plan = plan_val_transfers(p["counts"], W, H, mode)
    dev = PeValTaskDevice(p["ci"], p["rp"], p["x_grid"], plan.pe_words, W, H,
                          p["KY"], max(max(p["counts"]), 1))
    dev.spmv()
    for chunk in _chunks(_deliver(plan, buf, W, H, order), n_chunks,
                         np.random.default_rng(seed)):
        dev.stream(chunk)
    return _worker().gather_y(dev.wait_ew(), H, p["KY"], rows), dev


# --------------------------------------------------------------------------
# one pass
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rows, cols, nnz, W, H", [
    (16, 16, 40, 2, 2),
    (100, 70, 900, 4, 3),     # ragged edges, non-square grid
    (64, 64, 2048, 1, 1),     # one PE, the recorded crash size
    (37, 53, 500, 5, 4),
])
@pytest.mark.parametrize("mode", ["column", "full"])
@pytest.mark.parametrize("order", ["row", "col"])
def test_valtask_round_trip_matches_reference(rows, cols, nnz, W, H, mode, order):
    p = _problem(rows, cols, nnz, W, H, seed=rows + W, dup=True)
    y, dev = _run_valtask(p, W, H, rows, mode, order)
    ref = spmv_reference(p["r"], p["c"], p["v"], p["x"], rows)
    assert np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1e-30) < 1e-6
    assert dev.swallowed == 0           # the plan sends exactly the count


@pytest.mark.parametrize("n_chunks", [2, 7, 50])
def test_arrival_in_chunks_changes_nothing(n_chunks):
    """Overlapping, backpressured streams reach a PE in pieces; a per-word task
    is indifferent to how the words are split in time."""
    p = _problem(100, 70, 900, 4, 3, seed=5)
    y1, _ = _run_valtask(p, 4, 3, 100, "column", "row")
    yn, _ = _run_valtask(p, 4, 3, 100, "column", "row", n_chunks=n_chunks, seed=n_chunks)
    assert np.array_equal(y1.view(np.uint32), yn.view(np.uint32))


def test_words_arriving_before_spmv_wait_in_the_queue():
    """The data task starts @block-ed: a stream that beats launch("spmv") to a
    PE sits in the queue until spmv() unblocks it, rather than being drained."""
    p = _problem(16, 16, 40, 2, 2, seed=1)
    buf = build_val_stream_buffer(p["v"], p["order"], p["counts"], 2, 2)
    plan = plan_val_transfers(p["counts"], 2, 2, "full")
    dev = PeValTaskDevice(p["ci"], p["rp"], p["x_grid"], plan.pe_words, 2, 2,
                          p["KY"], max(p["counts"]))
    dev.stream(_deliver(plan, buf, 2, 2, "row"))   # before spmv()
    assert dev.swallowed == 0 and any(dev.queue[0][0] or dev.queue[1][1])
    dev.spmv()
    y = _worker().gather_y(dev.wait_ew(), 2, p["KY"], 16)
    ref = spmv_reference(p["r"], p["c"], p["v"], p["x"], 16)
    assert np.linalg.norm(y - ref) / np.linalg.norm(ref) < 1e-6


@pytest.mark.parametrize("mode", ["column", "full"])
def test_zero_nnz_pes_and_empty_columns_terminate(mode):
    """10 NNZ on 8x8.  Column mode: PEs in empty columns get val_meta == 0
    and go straight to ew_start.  Full mode pads every PE to the busiest, so
    there it is the zero-NNZ PEs consuming pure padding."""
    p = _problem(64, 64, 10, 8, 8, seed=3)
    plan = plan_val_transfers(p["counts"], 8, 8, mode)
    if mode == "column":
        assert (plan.pe_words == 0).any()
    else:
        assert (np.asarray(p["counts"]) == 0).any() and (plan.pe_words > 0).all()
    y, _ = _run_valtask(p, 8, 8, 64, mode, "row")
    ref = spmv_reference(p["r"], p["c"], p["v"], p["x"], 64)
    assert np.linalg.norm(y - ref) / np.linalg.norm(ref) < 1e-6


def test_all_zero_matrix_sends_nothing_and_finishes():
    p = _problem(16, 16, 0, 2, 2)
    y, dev = _run_valtask(p, 2, 2, 16, "full", "row")
    assert not y.any() and dev.swallowed == 0


def test_full_mode_padding_is_counted_and_never_read():
    """Full mode pads every PE to the busiest PE; val_meta counts the padding,
    which lands in vals[] past row_ptrs[KY] and is never read."""
    p = _problem(100, 70, 900, 4, 3, seed=9)
    plan = plan_val_transfers(p["counts"], 4, 3, "full")
    assert (plan.pe_words > np.asarray(p["counts"])).any()     # padding exists
    y, _ = _run_valtask(p, 4, 3, 100, "full", "row")
    ref = spmv_reference(p["r"], p["c"], p["v"], p["x"], 100)
    assert np.linalg.norm(y - ref) / np.linalg.norm(ref) < 1e-6


def test_stray_words_past_the_count_are_swallowed_not_queued():
    """Unlike pe_val.csl (stranded words), extra words drain harmlessly and the
    queue is empty for the next pass."""
    p = _problem(16, 16, 40, 2, 2, seed=2)
    buf = build_val_stream_buffer(p["v"], p["order"], p["counts"], 2, 2)
    plan = plan_val_transfers(p["counts"], 2, 2, "full")
    dev = PeValTaskDevice(p["ci"], p["rp"], p["x_grid"], plan.pe_words, 2, 2,
                          p["KY"], max(p["counts"]))
    dev.spmv()
    got = _deliver(plan, buf, 2, 2, "row")
    dev.stream(got)
    dev.stream([[np.full(5, 0x7F800001, np.uint32) for _ in range(2)] for _ in range(2)])
    y = _worker().gather_y(dev.wait_ew(), 2, p["KY"], 16)
    ref = spmv_reference(p["r"], p["c"], p["v"], p["x"], 16)
    assert np.linalg.norm(y - ref) / np.linalg.norm(ref) < 1e-6
    assert dev.swallowed == 4 * 5
    assert not any(dev.queue[py][px] for py in range(2) for px in range(2))


def test_too_few_words_is_a_hang():
    """The count contract is still exact from below."""
    p = _problem(16, 16, 40, 2, 2, seed=2)
    buf = build_val_stream_buffer(p["v"], p["order"], p["counts"], 2, 2)
    plan = plan_val_transfers(p["counts"], 2, 2, "full")
    dev = PeValTaskDevice(p["ci"], p["rp"], p["x_grid"], plan.pe_words + 1, 2, 2,
                          p["KY"], max(p["counts"]) + 1)
    dev.spmv()
    dev.stream(_deliver(plan, buf, 2, 2, "row"))
    with pytest.raises(RuntimeError, match="hang"):
        dev.wait_ew()


def test_valtask_and_val_agree_bitwise():
    p = _problem(100, 70, 900, 4, 3, seed=4)
    for mode in ("column", "full"):
        y_task, _ = _run_valtask(p, 4, 3, 100, mode, "row")
        plan = plan_val_transfers(p["counts"], 4, 3, mode)
        buf = build_val_stream_buffer(p["v"], p["order"], p["counts"], 4, 3)
        dev = PeValDevice(p["ci"], p["rp"], p["x_grid"], plan.pe_words, 4, 3,
                          p["KY"], max(p["counts"]))
        dev.stream(_deliver(plan, buf, 4, 3, "row"))
        y_val = _worker().gather_y(dev.spmv_pass(), 3, p["KY"], 100)
        assert np.array_equal(y_task.view(np.uint32), y_val.view(np.uint32))


# --------------------------------------------------------------------------
# multiple passes, on the geometries that broke the header kernel
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["column", "full"])
@pytest.mark.parametrize("name", list(HEADER_KILLERS))
def test_multi_pass_on_the_geometries_that_broke_the_header_kernel(name, mode, tmp_path):
    rows, cols, density, W, H = HEADER_KILLERS[name]
    r, c, v = _dense_8x8() if density is None else _coo(rows, cols, density, 42, tmp_path)
    KX, KY = -(-cols // W), -(-rows // H)
    ci, rp, counts, order, _ = extract_tile_structure(r, c, W, H, KX, KY)
    rng = np.random.default_rng(1)

    def x_grid_for(xv):
        g = np.zeros((W * H, KX), dtype=np.float32)
        for px in range(W):
            cs, ce = px * KX, min((px + 1) * KX, cols)
            for py in range(H):
                g[py * W + px, :ce - cs] = xv[cs:ce]
        return g

    x = rng.integers(0, 100, cols).astype(np.float32)
    v2 = rng.random(len(v)).astype(np.float32)
    x2 = rng.integers(0, 100, cols).astype(np.float32)
    plan = plan_val_transfers(counts, W, H, mode)
    dev = PeValTaskDevice(ci, rp, x_grid_for(x), plan.pe_words, W, H, KY, max(counts))
    gather = _worker().gather_y
    ys = []
    for i, (vals_in, x_in) in enumerate([(v, x), (v, x), (v2, x), (v2, x2), (v, x)]):
        dev.x_grid = x_grid_for(x_in)
        dev.spmv()
        buf = build_val_stream_buffer(vals_in, order, counts, W, H)
        for chunk in _chunks(_deliver(plan, buf, W, H, "row"), 3,
                             np.random.default_rng(i)):
            dev.stream(chunk)
        y = gather(dev.wait_ew(), H, KY, rows)
        ref = spmv_reference(r, c, vals_in, x_in, rows)
        assert np.linalg.norm(y - ref) / np.linalg.norm(ref) < 1e-6
        ys.append(y)
        assert not any(dev.queue[py][px] for py in range(H) for px in range(W))
    assert dev.swallowed == 0
    # nothing carries over: the last pass repeats the first bit for bit
    assert np.array_equal(ys[0].view(np.uint32), ys[-1].view(np.uint32))
    assert np.array_equal(ys[0].view(np.uint32), ys[1].view(np.uint32))


# --------------------------------------------------------------------------
# flag threading
# --------------------------------------------------------------------------

HW = dict(rows=1000, cols=1000, density=0.01, num_pes_target=16)


def test_encoding_ids_line_up_everywhere():
    assert ENCODINGS == ("header", "valtask", "resident")
    assert _worker().ENCODINGS == ENCODINGS
    assert VAL_ENCODINGS == CONFIG_VAL_ENCODINGS == ("valtask",)


def test_valtask_is_compile_param_2():
    base = SpMVConfig(**HW, h2d_mode="full", encoding="header")
    task = SpMVConfig(**HW, h2d_mode="full", encoding="valtask")
    assert "ENCODING:2" in compile_options(task)
    assert (compile_options(task).replace("ENCODING:2", "ENCODING:0")
            == compile_options(base))


def test_valtask_costs_vals_and_val_meta_of_sram():
    task = SpMVConfig(**HW, h2d_mode="full", encoding="valtask")
    assert task.val_only_extra_words == task.total_nnz_per_pe + 1


def test_valtask_column_mode_is_not_refused_on_hardware(recwarn):
    """No refusal guard: valtask runs column mode, serialised (the worker decides)."""
    cfg = SpMVConfig(**HW, encoding="valtask")
    assert (cfg.h2d_mode, cfg.simulator) == ("column", False)
    assert not recwarn.list


def test_valtask_needs_no_flag_on_the_worker_command():
    """The worker serialises valtask + column by itself, as for val."""
    cmd = build_worker_command(SpMVConfig(**HW, encoding="valtask"))
    assert "--encoding valtask" in cmd
    assert "--serialize-column-rects" not in cmd


def test_cli_and_sweep_accept_valtask(tmp_path):
    args = build_parser().parse_args(
        ["run", "--rows", "8", "--cols", "8", "--density", "0.5", "--encoding", "valtask"])
    assert io_overrides(args)["encoding"] == "valtask"
    path = tmp_path / "t.toml"
    path.write_text(textwrap.dedent("""\
        [meta]
        name = "t"

        [sweep]
        type = "weak_scaling"
        nnz_per_pe = 200
        density = 0.001
        encoding = "valtask"

        [sweep.num_pes]
        pow2_up_to = 256
        """))
    assert {c.encoding for c in SweepConfig.from_toml(path).expand()} == {"valtask"}


def test_worker_treats_valtask_as_values_only():
    w = _worker()
    assert w.artifact_encoding({"params": {"ENCODING": "2"}}) == "valtask"
    src = worker_source()
    # the decision moved into host_prep.prepare_host_buffers; the worker reads it back
    assert "val_only = hb.val_only" in src
    assert "val_only = encoding in VAL_ENCODINGS" in (WORKER.parent / "host_prep.py").read_text()
    # valtask hung on overlapping column rects on hardware (4x4, d=0.01,
    # cancelled at 2m54s), so it is auto-serialised like val; full mode is
    # one rect, so nothing changes there
    from cluster.coo_tiling import column_rects_serialized
    assert column_rects_serialized("valtask", "column", False) is True
    assert column_rects_serialized("valtask", "column", True) is True
    assert column_rects_serialized("valtask", "full", False) is False


# --------------------------------------------------------------------------
# static CSL contract
# --------------------------------------------------------------------------

def _body(code, sig):
    i = code.index(sig)
    return code[i:code.index("\n}\n", i)]


def test_layout_selects_valtask_and_exports_val_meta():
    src = (CSL / "layout.csl").read_text()
    assert re.search(r'if \(ENCODING == 2\) \{\s*@set_tile_code\(px, py, "pe_valtask.csl"', src)
    assert re.search(r'if \(ENCODING == 2\) \{\s*@export_name\("val_meta"', src)


def test_valtask_binds_a_data_task_on_the_h2d_queue():
    code = _code(CSL / "pe_valtask.csl")
    assert "const h2d_iq: input_queue = @get_input_queue(2);" in code
    assert "@get_data_task_id(h2d_iq)" in code
    assert "@bind_data_task(stream_word, stream_task_id);" in code
    assert "@block(stream_task_id);" in code
    assert "@unblock(stream_task_id);" in _body(code, "fn spmv() void {")
    assert "@fmovs(val_mem_dsd" not in code and "fabin_dsd, .{\n    .extent = TOTAL_NNZ" not in code


def test_valtask_stream_word_stores_counts_and_drains():
    task = _body(_code(CSL / "pe_valtask.csl"), "task stream_word(word: u32) void {")
    assert task.index("if (draining) { return; }") < task.index("vals[n_recv]")
    assert "vals[n_recv] = @bitcast(f32, word);" in task
    assert "n_recv += 1;" in task
    m = re.search(r"if \(n_recv == n_expect\) \{(.*?)\}", task, re.S)
    assert m and "draining = true;" in m.group(1)
    assert "@activate(ew_start_task_id);" in m.group(1)


def test_valtask_spmv_resets_and_handles_zero_words():
    spmv = _body(_code(CSL / "pe_valtask.csl"), "fn spmv() void {")
    for reset in ("n_recv       = 0;", "n_expect     = val_meta[0];",
                  "ew_done      = false;", "wait_pending = false;"):
        assert reset in spmv, reset
    zero = re.search(r"if \(n_expect == 0\) \{(.*?)\} else \{(.*?)\}", spmv, re.S)
    assert zero
    assert "@activate(ew_start_task_id);" in zero.group(1)
    assert "draining  = true;" in zero.group(1)
    assert "draining = false;" in zero.group(2)
    # the reset happens before the task is unblocked
    assert spmv.index("n_expect     = val_meta[0];") < spmv.index("@unblock(stream_task_id);")


def test_valtask_exports_what_the_worker_reads():
    exports = set(re.findall(r'@export_symbol\(\s*\w+\s*,\s*"(\w+)"',
                             (CSL / "pe_valtask.csl").read_text()))
    read = symbols_read()
    assert read - {"vals", "row_ids"} <= exports   # "vals": --encoding resident only
