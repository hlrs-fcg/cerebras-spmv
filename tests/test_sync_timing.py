"""--sync-timing / --loop-count: synced on-wafer timing over repeated passes.

The csl-examples bandwidth-test pattern: f_sync once (every PE samples a
reference clock broadcast from P0.0), then per pass f_tic -> spmv / stream /
wait_ew -> f_toc, and the host reads sync_pack.  Nothing here can run cslc or
the wafer, so the pieces are checked separately:

- the host arithmetic, against a model of per-PE clocks with arbitrary offsets
  and the CSL packing transcribed bit for bit;
- multi-pass correctness of BOTH kernels, through stateful models whose
  y_result / vals / queues persist between passes and start as garbage;
- flag threading (config, CLI, sweep, compiler, launcher, worker, CSV);
- the static CSL contract: what is gated on SYNC_TIMING, and that the sync
  module's queues, colors and tasks do not collide with the kernels'.

See docs/synced-timing.md.
"""

import re
import shlex
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from bench.benchmark import BenchmarkResult, load_benchmarks, save_benchmark
from cluster.coo_tiling import (
    build_stream_buffer,
    build_val_stream_buffer,
    extract_tile_structure,
    plan_val_transfers,
    spmv_reference,
)
from cluster.sweep_config import SweepConfig
from config import SYNC_TIMING_EXTRA_WORDS, SpMVConfig
from run import build_parser, io_overrides
from sdk.compiler import compile_options
from sdk.launcher import build_worker_command
from tests.test_val_encoding import PeValDevice, _deliver, _ew_reduce

from tests.worker_src import CLUSTER, worker_source  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSL = PROJECT_ROOT / "src" / "csl"
WORKER = PROJECT_ROOT / "cluster" / "worker_runtime.py"

HW = dict(rows=1000, cols=1000, density=0.01, num_pes_target=16)


def _worker():
    sys.path.insert(0, str(PROJECT_ROOT / "cluster"))
    sys.path.insert(0, str(PROJECT_ROOT / "bench"))
    import worker_runtime
    return worker_runtime


def _code(path: Path) -> str:
    """CSL source with // comments stripped."""
    return "\n".join(ln.split("//")[0] for ln in path.read_text().splitlines())


# --------------------------------------------------------------------------
# host arithmetic: sync_pack -> one on-wafer span per pass
# --------------------------------------------------------------------------

def _u16(v):
    v = np.asarray(v, dtype=np.int64)
    return v & 0xFFFF, (v >> 16) & 0xFFFF, (v >> 32) & 0xFFFF


def _pack_like_f_toc(tic, toc, ref):
    """pe.csl / pe_val.csl f_toc, transcribed: three 48-bit counts -> 5 x f32."""
    t0, t1, t2 = _u16(tic)
    e0, e1, e2 = _u16(toc)
    r0, r1, r2 = _u16(ref)
    words = np.stack([
        (t1 << 16) | t0,
        (e0 << 16) | t2,
        (e2 << 16) | e1,
        (r1 << 16) | r0,
        r2,
    ], axis=-1).astype(np.uint32)
    return words.view(np.float32).reshape(-1)       # (H, W, 5) row-major


def _clock_model(H, W, seed):
    """Per-PE tsc = global time + an arbitrary per-PE offset.

    The sync broadcast reaches PE (px, py) px + py cycles after P0.0, so every
    PE stamps ref at global G0 + px + py.  tic/toc happen at scattered global
    times.  Returns the packed buffer and the true global span.
    """
    rng = np.random.default_rng(seed)
    offset = rng.integers(0, 2**46, (H, W))
    py, px = np.indices((H, W))
    g_ref = 1_000 + px + py
    g_tic = 50_000 + rng.integers(0, 500, (H, W))
    g_toc = 900_000 + rng.integers(0, 70_000, (H, W))
    pack = _pack_like_f_toc(g_tic + offset, g_toc + offset, g_ref + offset)
    return pack, int(g_toc.max() - g_tic.min()), offset


@pytest.mark.parametrize("H, W", [(2, 2), (4, 4), (3, 7), (16, 16)])
def test_synced_span_recovers_the_global_span(H, W):
    w = _worker()
    pack, truth, _ = _clock_model(H, W, seed=H * 100 + W)
    tic, toc, ref = w.unpack_sync_pack(pack, H, W)
    assert w.synced_pass_cycles(tic, toc, ref) == truth


def test_unsynced_stamps_would_be_meaningless():
    """Guard the model: without the reference the offsets dominate."""
    w = _worker()
    pack, truth, offset = _clock_model(4, 4, seed=7)
    tic, toc, _ = w.unpack_sync_pack(pack, 4, 4)
    assert abs(int(toc.max() - tic.min()) - truth) > 1_000_000


def test_propagation_delay_correction_is_px_plus_py():
    """Zero offsets, every PE stamps at the same global instant: the span is 0
    only if ref is corrected by exactly px + py (bandwidth-test/run.py)."""
    w = _worker()
    H, W = 5, 6
    py, px = np.indices((H, W))
    ref = 10 + px + py
    tic = np.full((H, W), 100)
    toc = np.full((H, W), 100)
    assert w.synced_pass_cycles(tic, toc, ref) == 0
    assert w.synced_pass_cycles(tic, toc, np.full((H, W), 10)) != 0


def test_unpack_is_bit_exact_at_48_bits():
    w = _worker()
    big = np.array([[2**48 - 1, 2**47 + 12345], [0xABCDEF012345, 1]])
    pack = _pack_like_f_toc(big, big[::-1], big.T)
    tic, toc, ref = w.unpack_sync_pack(pack, 2, 2)
    assert np.array_equal(tic, big)
    assert np.array_equal(toc, big[::-1])
    assert np.array_equal(ref, big.T)


@pytest.mark.parametrize("which", ["tic", "toc", "ref"])
def test_a_pe_that_never_stamped_is_an_error_not_a_number(which):
    w = _worker()
    pack, _, _ = _clock_model(3, 3, seed=1)
    stamps = dict(zip(("tic", "toc", "ref"), w.unpack_sync_pack(pack, 3, 3)))
    stamps[which][1, 2] = 0
    with pytest.raises(ValueError, match=f"{which}==0"):
        w.synced_pass_cycles(stamps["tic"], stamps["toc"], stamps["ref"])


def test_pass_stats():
    w = _worker()
    assert w.pass_stats([]) == {"min": 0.0, "avg": 0.0, "max": 0.0, "std": 0.0}
    assert w.pass_stats([7]) == {"min": 7.0, "avg": 7.0, "max": 7.0, "std": 0.0}
    s = w.pass_stats([1, 2, 3, 6])
    assert (s["min"], s["avg"], s["max"]) == (1.0, 3.0, 6.0)
    assert s["std"] == pytest.approx(np.std([1, 2, 3, 6]))


def test_clock_constant_matches_the_examples():
    assert _worker().WSE_CLOCK_HZ == 0.85e9


# --------------------------------------------------------------------------
# multi-pass correctness of both kernels (loop_count > 1)
# --------------------------------------------------------------------------

class PeHeaderDevice:
    """Stateful model of pe.csl across passes.

    Persists what the hardware persists: y_result (starts as garbage here),
    the stream state machine and each PE's input queue.  spmv() does what
    pe.csl's spmv() does -- clear all KY rows, reset the state machine, stop
    draining -- unless clear_y=False, which models dropping that loop.  The
    data task consumes every queued word: header/value words until the
    sentinel, then swallows the padding.
    """

    GARBAGE = np.float32(12345.0)

    def __init__(self, ci, rp, x_grid, W, H, KY, clear_y=True):
        self.ci, self.rp, self.x_grid = ci, rp, x_grid
        self.W, self.H, self.KY = W, H, KY
        self.clear_y = clear_y
        self.y = np.full((H, W, KY), self.GARBAGE, np.float32)
        self.draining = np.ones((H, W), bool)
        self.queue = [[[] for _ in range(W)] for _ in range(H)]

    def stream(self, buf, n_words):
        """n_words(px) words of buf[px, py, :] to every PE of column px."""
        for py in range(self.H):
            for px in range(self.W):
                self.queue[py][px].extend(int(w) for w in buf[px, py, :n_words(px)])

    def spmv_pass(self):
        for py in range(self.H):
            for px in range(self.W):
                pe = py * self.W + px
                if self.clear_y:
                    self.y[py, px, :] = 0.0
                state, cur, rem, k = 0, 0, 0, 0
                self.draining[py, px] = False
                for word in self.queue[py][px]:
                    if self.draining[py, px]:
                        continue
                    if state == 0:
                        if word == 0:
                            self.draining[py, px] = True
                            continue
                        cur, rem = word & 0xFFFF, (word >> 16) & 0xFFFF
                        k, state = int(self.rp[pe, cur]), 1
                    else:
                        v = np.uint32(word).view(np.float32)
                        self.y[py, px, cur] = np.float32(
                            self.y[py, px, cur]
                            + np.float32(v * self.x_grid[pe, self.ci[pe, k]]))
                        k, rem = k + 1, rem - 1
                        if rem == 0:
                            state = 0
                if not self.draining[py, px]:
                    raise RuntimeError(f"PE ({px},{py}) never saw its sentinel: hang")
                self.queue[py][px] = []
        return _ew_reduce(self.y)


def _x_grid(x, W, H, KX, cols):
    g = np.zeros((W * H, KX), dtype=np.float32)
    for px in range(W):
        cs, ce = px * KX, min((px + 1) * KX, cols)
        for py in range(H):
            g[py * W + px, :max(ce - cs, 0)] = x[cs:ce]
    return g


# Shapes with empty rows (so a skipped clear would show), an exactly-full
# tile, and a non-square grid with ragged edges.
GEOMETRIES = {
    "200x200_d0.05_4x4": (200, 200, 0.05, 4, 4),
    "8x8_dense_2x2":     (8, 8, 1.0, 2, 2),
    "301x173_d0.03_5x3": (301, 173, 0.03, 5, 3),
}
LOOP_COUNT = 5


def _problem(rows, cols, density, seed):
    rng = np.random.default_rng(seed)
    if density >= 1.0:
        r, c = np.meshgrid(np.arange(rows, dtype=np.int32),
                           np.arange(cols, dtype=np.int32), indexing="ij")
        r, c = r.ravel(), c.ravel()
    else:
        nnz = int(rows * cols * density)
        flat = rng.choice(rows * cols, nnz, replace=False)
        r, c = (flat // cols).astype(np.int32), (flat % cols).astype(np.int32)
    return r, c, rng


def _passes(rng, nnz, cols):
    """LOOP_COUNT (values, x): repeats, then a new x, then new values."""
    v = rng.random(nnz).astype(np.float32)
    x = rng.integers(1, 100, cols).astype(np.float32)
    v2 = rng.random(nnz).astype(np.float32)
    x2 = rng.integers(1, 100, cols).astype(np.float32)
    return [(v, x), (v, x), (v, x2), (v2, x2), (v, x)]


@pytest.mark.parametrize("mode", ["column", "full"])
@pytest.mark.parametrize("name", list(GEOMETRIES))
def test_header_kernel_multi_pass(name, mode):
    rows, cols, density, W, H = GEOMETRIES[name]
    r, c, rng = _problem(rows, cols, density, seed=3)
    KX, KY = -(-cols // W), -(-rows // H)
    ci, rp, counts, order, rb = extract_tile_structure(r, c, W, H, KX, KY)
    dev = None
    ys = []
    for v, x in _passes(rng, len(r), cols):
        buf, col_words = build_stream_buffer(v, order, counts, rb, W, H, num_threads=1)
        n_words = (lambda px: col_words[px]) if mode == "column" else (lambda px: buf.shape[2])
        if dev is None:
            dev = PeHeaderDevice(ci, rp, None, W, H, KY)
        dev.x_grid = _x_grid(x, W, H, KX, cols)      # a CG loop re-sends x
        dev.stream(buf, n_words)
        y = _worker().gather_y(dev.spmv_pass(), H, KY, rows)
        ref = spmv_reference(r, c, v, x, rows)
        assert np.linalg.norm(y - ref) / np.linalg.norm(ref) < 1e-6
        ys.append(y)
    # a pass after different passes is bit-identical to the first: no carry-over
    assert np.array_equal(ys[0].view(np.uint32), ys[-1].view(np.uint32))
    assert np.array_equal(ys[0].view(np.uint32), ys[1].view(np.uint32))


def test_header_model_catches_a_missing_y_reset():
    """Mutation check: drop spmv()'s clearing loop and a later pass goes wrong."""
    rows, cols, density, W, H = GEOMETRIES["200x200_d0.05_4x4"]
    r, c, rng = _problem(rows, cols, density, seed=3)
    KX, KY = -(-cols // W), -(-rows // H)
    ci, rp, counts, order, rb = extract_tile_structure(r, c, W, H, KX, KY)
    v, x = _passes(rng, len(r), cols)[0]
    buf, col_words = build_stream_buffer(v, order, counts, rb, W, H, num_threads=1)
    dev = PeHeaderDevice(ci, rp, _x_grid(x, W, H, KX, cols), W, H, KY, clear_y=False)
    dev.stream(buf, lambda px: col_words[px])
    y = _worker().gather_y(dev.spmv_pass(), H, KY, rows)
    assert not np.allclose(y, spmv_reference(r, c, v, x, rows))


@pytest.mark.parametrize("mode", ["column", "full"])
@pytest.mark.parametrize("name", list(GEOMETRIES))
def test_val_kernel_multi_pass(name, mode):
    """vals[] and the count-terminated receive are reused every pass."""
    rows, cols, density, W, H = GEOMETRIES[name]
    r, c, rng = _problem(rows, cols, density, seed=3)
    KX, KY = -(-cols // W), -(-rows // H)
    ci, rp, counts, order, _ = extract_tile_structure(r, c, W, H, KX, KY)
    plan = plan_val_transfers(counts, W, H, mode)
    dev = PeValDevice(ci, rp, np.zeros((W * H, KX), np.float32), plan.pe_words,
                      W, H, KY, max(counts))
    ys = []
    for v, x in _passes(rng, len(r), cols):
        dev.x_grid = _x_grid(x, W, H, KX, cols)
        buf = build_val_stream_buffer(v, order, counts, W, H)
        dev.stream(_deliver(plan, buf, W, H, "row"))
        y = _worker().gather_y(dev.spmv_pass(), H, KY, rows)
        ref = spmv_reference(r, c, v, x, rows)
        assert np.linalg.norm(y - ref) / np.linalg.norm(ref) < 1e-6
        ys.append(y)
    assert np.array_equal(ys[0].view(np.uint32), ys[-1].view(np.uint32))


def test_header_and_val_agree_bitwise_on_every_pass():
    rows, cols, density, W, H = GEOMETRIES["301x173_d0.03_5x3"]
    r, c, rng = _problem(rows, cols, density, seed=11)
    KX, KY = -(-cols // W), -(-rows // H)
    ci, rp, counts, order, rb = extract_tile_structure(r, c, W, H, KX, KY)
    plan = plan_val_transfers(counts, W, H, "full")
    hdr = PeHeaderDevice(ci, rp, None, W, H, KY)
    val = PeValDevice(ci, rp, np.zeros((W * H, KX), np.float32), plan.pe_words,
                      W, H, KY, max(counts))
    for v, x in _passes(rng, len(r), cols):
        hdr.x_grid = val.x_grid = _x_grid(x, W, H, KX, cols)
        buf, _ = build_stream_buffer(v, order, counts, rb, W, H, num_threads=1)
        hdr.stream(buf, lambda px: buf.shape[2])
        val.stream(_deliver(plan, build_val_stream_buffer(v, order, counts, W, H),
                            W, H, "row"))
        assert np.array_equal(hdr.spmv_pass().view(np.uint32),
                              val.spmv_pass().view(np.uint32))


# --------------------------------------------------------------------------
# flag threading
# --------------------------------------------------------------------------

def test_defaults_are_off_and_one_pass():
    cfg = SpMVConfig(**HW)
    assert (cfg.sync_timing, cfg.loop_count) == (False, 1)
    r = BenchmarkResult()
    assert (r.sync_timing, r.loop_count) == (False, 1)
    assert (r.wafer_cycles_avg, r.wafer_time_avg) == (0, 0.0)


def test_sync_timing_is_a_compile_param_and_loop_count_is_not():
    base = SpMVConfig(**HW)
    assert "SYNC_TIMING:0" in compile_options(base)
    synced = SpMVConfig(**HW, sync_timing=True)
    assert (compile_options(synced).replace("SYNC_TIMING:1", "SYNC_TIMING:0")
            == compile_options(base))
    assert compile_options(SpMVConfig(**HW, loop_count=5)) == compile_options(base)


def test_worker_command():
    base = build_worker_command(SpMVConfig(**HW))
    assert "--loop-count 1" in base and "--sync-timing" not in base
    cmd = build_worker_command(SpMVConfig(**HW, sync_timing=True, loop_count=5))
    assert "--loop-count 5" in cmd and "--sync-timing" in cmd


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3"])
def test_loop_count_must_be_a_positive_int(bad):
    with pytest.raises(ValueError, match="loop_count"):
        SpMVConfig(**HW, loop_count=bad)


@pytest.mark.parametrize("num_pes, height", [(4, 1), (4, 4)])
def test_sync_refuses_a_one_wide_or_one_tall_grid(num_pes, height):
    kw = dict(HW, num_pes_target=num_pes, grid_height_target=height)
    SpMVConfig(**kw)                                   # fine without sync
    with pytest.raises(ValueError, match="2x2"):
        SpMVConfig(**kw, sync_timing=True)


def test_sram_guard_counts_the_sync_state():
    assert SYNC_TIMING_EXTRA_WORDS == 11
    kw = dict(rows=200_000, cols=200_000, density=0.001, num_pes_target=1024)

    def reported(**extra):
        with pytest.raises(ValueError, match="usable SRAM per PE") as exc:
            SpMVConfig(**kw, **extra)
        return float(re.search(r"usage ([\d.]+) KB", str(exc.value)).group(1))

    assert reported(sync_timing=True) - reported() == pytest.approx(
        SYNC_TIMING_EXTRA_WORDS * 4 / 1024, abs=0.06)


def test_cli_flags():
    args = build_parser().parse_args(shlex.split(
        "run --rows 1000 --cols 1000 --density 0.01 --sync-timing --loop-count 5"))
    assert io_overrides(args)["sync_timing"] is True
    assert io_overrides(args)["loop_count"] == 5
    bare = io_overrides(build_parser().parse_args(
        shlex.split("run --rows 1000 --cols 1000 --density 0.01")))
    assert "sync_timing" not in bare and "loop_count" not in bare


def _toml(tmp_path, body):
    path = tmp_path / "t.toml"
    path.write_text(textwrap.dedent("""\
        [meta]
        name = "t"

        [sweep]
        type = "weak_scaling"
        nnz_per_pe = 200
        density = 0.001
        """) + body + "\n[sweep.num_pes]\npow2_up_to = 256\n")
    return path


def test_sweep_toml_and_cli(tmp_path):
    sweep = SweepConfig.from_toml(_toml(tmp_path, "sync_timing = true\nloop_count = 5\n"))
    assert {(c.sync_timing, c.loop_count) for c in sweep.expand()} == {(True, 5)}
    sweep = SweepConfig.from_toml(_toml(tmp_path, ""),
                                  cli_override_io={"sync_timing": True, "loop_count": 3})
    assert {(c.sync_timing, c.loop_count) for c in sweep.expand()} == {(True, 3)}
    with pytest.raises(ValueError, match="sync_timing"):
        SweepConfig.from_toml(_toml(tmp_path, "sync_timing = 1\n"))


def test_csv_round_trips_and_old_rows_backfill(tmp_path):
    path = tmp_path / "r.csv"
    row = BenchmarkResult(sync_timing=True, loop_count=5, wafer_cycles_avg=123,
                          wafer_time_avg=123 / 0.85e9, wafer_cycles_std=4)
    save_benchmark(BenchmarkResult(), csv_path=str(path))
    save_benchmark(row, csv_path=str(path))
    old, new = load_benchmarks(str(path))
    assert (old.sync_timing, old.loop_count, old.wafer_cycles_avg) == (False, 1, 0)
    assert (new.sync_timing, new.loop_count, new.wafer_cycles_avg) == (True, 5, 123)
    assert new.wafer_cycles_std == 4


def test_worker_records_every_new_column():
    src = worker_source()
    for col in ("loop_count", "sync_timing", "wafer_cycles_min", "wafer_cycles_avg",
                "wafer_cycles_max", "wafer_cycles_std", "wafer_time_min",
                "wafer_time_avg", "wafer_time_max", "wafer_time_std"):
        assert f"{col}=" in src, col


def test_worker_artifact_check():
    w = _worker()
    assert w.artifact_sync_timing({"params": {}}) is False
    assert w.artifact_sync_timing({"params": {"SYNC_TIMING": "0"}}) is False
    assert w.artifact_sync_timing({"params": {"SYNC_TIMING": "1"}}) is True
    src = worker_source()
    assert (src.index("args.sync_timing and not artifact_sync_timing(compile_data)")
            < src.index("SdkRuntimeWrapper(SdkRuntime"))


# --------------------------------------------------------------------------
# the worker's command sequence
# --------------------------------------------------------------------------

def _loop_src():
    """The device phase's pass loop (worker_device.py), from f_sync to the cycle readback."""
    src = (CLUSTER / "worker_device.py").read_text()
    return src[src.index('runner.launch("f_sync"'):src.index("def read_cycle_stats")]


def _stream_launch_src():
    """One streaming pass's launches (StreamMode.launch_pass in worker_stream.py)."""
    src = (CLUSTER / "worker_stream.py").read_text()
    return src[src.index("def launch_pass"):src.index("def memcpy_h2d_time")]


def test_sync_once_then_tic_spmv_stream_wait_toc_per_pass():
    loop = _loop_src()
    seq = ['runner.launch("f_sync", nonblock=True)',
           "for p_i in range(args.loop_count):",
           "mode.launch_pass(r, d, t_comp)",
           "sym_sync_pack, 0, 0, W, H, 5"]
    pos = [loop.index(s) for s in seq]
    assert pos == sorted(pos), "command order changed"
    assert loop.count('runner.launch("f_sync"') == 1      # once, before the loop
    stream = _stream_launch_src()
    seq = ['runner.launch("f_tic", nonblock=True)',
           'runner.launch("spmv", nonblock=True)',
           "self.send_stream(r, d)",
           'runner.launch("wait_ew", nonblock=True)',
           'runner.launch("f_toc", nonblock=False)']
    pos = [stream.index(s) for s in seq]
    assert pos == sorted(pos), "streaming pass order changed"


def test_unsynced_path_still_blocks_on_wait_ew():
    """With the flag off, the pass is the pre-synced-timing sequence."""
    assert 'runner.launch("wait_ew", nonblock=False)' in _stream_launch_src()


def test_every_pass_is_validated():
    src = worker_source()
    assert "y_passes.append(y_flat)" in src
    assert "for p_i, y_computed in enumerate(y_computed_passes):" in src
    assert "passed = passed and ok" in src


# --------------------------------------------------------------------------
# static CSL contract
# --------------------------------------------------------------------------

KERNELS = ["pe.csl", "pe_valtask.csl", "pe_resident.csl"]

# What layout.csl hands every kernel for the sync module.  Flat params: SDK 2.10.0
# removed comptime_struct and @concat_structs (cslc on the cluster rejected
# sync/layout.csl's `fn get_params(...) comptime_struct`).
SYNC_FLAT_PARAMS = {
    "c_recv_px": "color", "c_send_px": "color", "c_recv_py": "color",
    "c_send_py": "color", "c_bcast": "color",
    "STARTUP": "local_task_id", "SYNC_Y": "local_task_id",
    "SYNC_BCAST": "local_task_id", "EXIT": "local_task_id",
    "first_px": "bool", "last_px": "bool", "first_py": "bool", "last_py": "bool",
}


def test_every_selectable_kernel_takes_the_sync_params():
    """layout.csl passes .SYNC_TIMING / .syncParams to whichever tile file it
    picks; a kernel that does not declare them fails to compile.  (Caught when
    Synced timing was rebased onto pe_valtask.csl.)"""
    layout = (CSL / "layout.csl").read_text()
    tiles = set(re.findall(r'@set_tile_code\(px, py, "(\w+\.csl)"', layout))
    assert tiles == set(KERNELS)
    for k in tiles:
        code = _code(CSL / k)
        assert re.search(r"param SYNC_TIMING: i16;", code), k
        for name, typ in SYNC_FLAT_PARAMS.items():
            assert re.search(rf"param {name}: {typ};", code), (k, name)


def test_sync_block_is_identical_in_every_kernel():
    def block(k):
        t = (CSL / k).read_text()
        i = t.index("// ── Synced on-wafer timing (docs/synced-timing.md)")
        j = t.index("\n}\n", t.index("fn f_toc() void {"))
        return t[i:j]
    assert len({block(k) for k in KERNELS}) == 1


def test_layout_param_and_gated_exports():
    src = (CSL / "layout.csl").read_text()
    assert re.search(r"param SYNC_TIMING\s*:", src)
    gated = re.search(r"if \(SYNC_TIMING == 1\) \{\s*@export_name\(\"sync_pack\"", src)
    assert gated
    block = src[gated.start():src.index("}", gated.end())]
    for name in ("f_sync", "f_tic", "f_toc"):
        assert f'@export_name("{name}"' in block
    assert "@comptime_assert(width > 1 and height > 1)" in src
    assert ".SYNC_TIMING = SYNC_TIMING" in src
    for name in SYNC_FLAT_PARAMS:
        assert re.search(rf"\.{name}\s*=", src), name
    assert "sync/layout.csl" not in _code(CSL / "layout.csl"), "layout must not import the sync module"


@pytest.mark.parametrize("kernel", KERNELS)
def test_kernel_exports_are_gated(kernel):
    code = _code(CSL / kernel)
    gated = [m for m in re.finditer(r"if \(SYNC_TIMING == 1\) \{(.*?)\}", code, re.S)
             if "@export_symbol" in m.group(1)]
    assert len(gated) == 1
    for sym in ('@export_symbol(ptr_sync_pack, "sync_pack")', "@export_symbol(f_sync)",
                "@export_symbol(f_tic)", "@export_symbol(f_toc)"):
        assert sym in gated[0].group(1)
        assert code.count(sym) == 1                        # nowhere ungated


@pytest.mark.parametrize("kernel", KERNELS)
def test_kernel_never_touches_the_tsc_enable_under_sync(kernel):
    """The reference clock is only valid if the tsc free-runs for the whole
    program; disabling (or re-enabling) it per pass would break it."""
    code = _code(CSL / kernel)
    for call in ("tsc.disable_tsc();", "tsc.enable_tsc();"):
        for m in re.finditer(re.escape(call), code):
            before = code[:m.start()].rstrip()
            assert before.endswith("if (SYNC_TIMING == 0) {"), f"{kernel}: ungated {call}"


@pytest.mark.parametrize("kernel", KERNELS)
def test_f_toc_packs_tic_toc_and_ref(kernel):
    code = _code(CSL / kernel)
    toc = code[code.index("fn f_toc()"):]
    toc = toc[:toc.index("\n}")]
    assert "tsc.get_timestamp(&tsc_toc);" in toc
    assert toc.count("sync_pack[") == 5
    assert "sync_mod.tscRefBuffer[2]" in toc
    assert toc.rstrip().endswith("sys_mod.unblock_cmd_stream();")


def _queue_ids(code, kind):
    return {int(n) for n in re.findall(rf"@get_{kind}_queue\((\d+)\)", code)}


@pytest.mark.parametrize("kernel", KERNELS)
def test_sync_queues_clear_of_the_kernels_queues(kernel):
    code = _code(CSL / kernel)
    m = re.search(r"\.input_queues\s*=\s*\[3\]u16\{([\d,\s]+)\}", code)
    sync_iq = {int(v) for v in m.group(1).split(",")}
    m = re.search(r"\.output_queues\s*=\s*\[3\]u16\{([\d,\s]+)\}", code)
    sync_oq = {int(v) for v in m.group(1).split(",")}
    assert not sync_iq & _queue_ids(code, "input")
    assert not sync_oq & _queue_ids(code, "output")
    assert not ({0, 1} & (sync_iq | sync_oq))              # memcpy's own queues


def test_sync_colors_and_tasks_are_clear_of_ours():
    from sdk.compiler import MEMCPYH2D_DATA_1_ID
    layout = _code(CSL / "layout.csl")
    sync_colors = [int(v) for v in re.findall(r"const SYNC_C\d: color = @get_color\((\d+)\)", layout)]
    assert len(sync_colors) == 5 and sync_colors == sorted(set(sync_colors))
    ew = {int(v) for v in re.findall(r"const EW_\w+_COLOR: color = @get_color\((\d+)\)", layout)}
    assert not set(sync_colors) & (ew | {MEMCPYH2D_DATA_1_ID})
    sync_tasks = {int(v) for v in re.findall(r"@get_local_task_id\((\d+)\)", layout)}
    assert sync_tasks == {15, 16, 17, 18}
    for k in KERNELS:
        ours = {int(v) for v in re.findall(r"@get_local_task_id\((\d+)\)", _code(CSL / k))}
        assert ours and not ours & sync_tasks


def test_vendored_sync_module_is_inert_unless_enabled():
    code = _code(CSL / "sync" / "pe.csl")
    assert re.search(r"param enabled: bool;", code)
    blocks = re.findall(r"comptime \{\s*(\S.*?)\n", code)
    assert blocks and all(b.startswith("if (enabled)") for b in blocks), blocks
    assert not re.search(r"^var fab_", code, re.M)         # DSDs are const
    assert "@activate(STARTUP)" in code


def test_vendored_protocol_matches_upstream():
    """Only the documented edits: gate, const DSDs, the note."""
    upstream = (Path.home() / "research/repos/csl-examples/benchmarks/"
                "bandwidth-test/src/sync/pe.csl")
    if not upstream.is_file():
        pytest.skip("csl-examples checkout not present")

    def norm(text):
        out = []
        for ln in text.splitlines():
            ln = ln.split("//")[0].strip()
            if not ln or ln in ("if (enabled) {", "param enabled: bool;"):
                continue
            out.append(ln.replace("const fab_", "var fab_"))
        return out

    ours = norm((CSL / "sync" / "pe.csl").read_text())
    theirs = norm(upstream.read_text())
    # each gate adds one closing brace
    assert [ln for ln in ours if ln != "}"] == [ln for ln in theirs if ln != "}"]
    assert ours.count("}") - theirs.count("}") == 4


# --------------------------------------------------------------------------
# cross-check against host wall time
# --------------------------------------------------------------------------
# Recorded at 65,536 PEs (2026-10-06/07): (label, host compute+barrier s,
# y D2H s, per-PE tsc span max cycles).  The span exceeds compute+barrier
# whenever the stream is long -- the barrier returned before the device was
# done -- but always fits inside compute+barrier+D2H, which is the bracket
# wafer_host_ratio uses.
RECORDED = [
    ("val full ch1",      0.02199, 0.1930, 88_712_735),
    ("header column ch1", 0.03940, 0.1556, 73_388_098),
    ("header column ch2", 0.04488, 0.0928, 61_516_025),
    ("header full ch8",   0.01983, 0.0499, 16_099_846),
    ("header column ch8", 0.06222, 0.0011, 46_626_924),
]


@pytest.mark.parametrize("label, host, d2h, cycles", RECORDED)
def test_recorded_spans_fit_the_host_time_through_the_d2h(label, host, d2h, cycles):
    assert _worker().wafer_host_ratio(cycles, host + d2h) <= 1.0, label


def test_compute_barrier_alone_is_not_an_upper_bound():
    """Why the cross-check must not use compute_time_hw_*."""
    w = _worker()
    over = [lbl for lbl, host, _, cyc in RECORDED if w.wafer_host_ratio(cyc, host) > 1.0]
    assert "val full ch1" in over and "header column ch1" in over


def test_wafer_host_ratio():
    w = _worker()
    assert w.wafer_host_ratio(850_000, 0.002) == pytest.approx(0.5)
    assert w.wafer_host_ratio(1_700_000, 0.001) == pytest.approx(2.0)
    assert w.wafer_host_ratio(1, 0.0) == float("inf")


def test_host_bracket_contains_every_stamp():
    """t_comp starts before f_tic is issued and pass_barrier_s ends when the
    completion barrier D2H returns; that D2H is queued behind f_toc and cannot
    return before the device has finished the pass."""
    loop = _loop_src()
    stream = _stream_launch_src()
    assert stream.index('runner.launch("f_tic", nonblock=True)') < stream.index(
        'runner.launch("f_toc", nonblock=False)')
    seq = ["t_comp = time.perf_counter()",
           "t_h2d = mode.launch_pass(r, d, t_comp)",
           "barrier_t = timed_d2h(runner, sym_y_result",
           "pass_barrier_s.append(t_done - t_comp)",
           "runner.memcpy_d2h(y_flat, sym_y_result",
           "ratio = wafer_host_ratio(cycles, pass_barrier_s[-1])"]
    pos = [loop.index(s) for s in seq]
    assert pos == sorted(pos)
    assert "if ratio > 1.0:" in loop


def test_cross_check_columns_are_recorded(tmp_path):
    src = worker_source()
    for col in ("host_pass_time_avg", "host_pass_time_std", "wafer_host_ratio_max"):
        assert f"{col}=" in src, col
    path = tmp_path / "r.csv"
    save_benchmark(BenchmarkResult(), csv_path=str(path))
    save_benchmark(BenchmarkResult(wafer_host_ratio_max=0.83, host_pass_time_avg=0.2),
                   csv_path=str(path))
    old, new = load_benchmarks(str(path))
    assert (old.wafer_host_ratio_max, new.wafer_host_ratio_max) == (0.0, 0.83)


def test_synced_path_avoids_per_pass_tsc_enable_disable():
    """The per-PE span is bracketed by enable_tsc() in spmv() and disable_tsc()
    in chain_done; the synced stamps never are (the sync module enables once
    and, under SYNC_TIMING, the kernels never touch it again)."""
    for k in KERNELS:
        code = _code(CSL / k)
        for fn in ("fn f_tic()", "fn f_toc()", "fn f_sync()"):
            body = code[code.index(fn):]
            body = body[:body.index("\n}")]
            assert "enable_tsc" not in body and "disable_tsc" not in body


# --------------------------------------------------------------------------
# SDK 2.10.0 compatibility and the flag-off build
# --------------------------------------------------------------------------

def test_no_csl_uses_the_removed_comptime_struct_constructs():
    """comptime_struct and @concat_structs were removed in SDK 2.10.0
    (https://sdk.cerebras.net/csl/comptime-struct-migration)."""
    offenders = []
    for f in sorted((CSL).rglob("*.csl")):
        text = _code(f)
        for token in ("@concat_structs", "comptime_struct"):
            if token in text:
                offenders.append((f.name, token))
    # the kernels' memcpy_params stay untyped (`param memcpy_params;`), as on main
    assert offenders == [], offenders


@pytest.mark.parametrize("kernel", KERNELS)
def test_kernels_import_the_sync_module_only_when_the_flag_is_on(kernel):
    code = _code(CSL / kernel)
    assert code.count("@import_module(sync_pe_file()") == 1
    fn = code[code.index("fn sync_pe_file()"):]
    fn = fn[:fn.index("\n}\n")]
    assert re.search(r'if \(SYNC_TIMING == 1\) \{\s*return "sync/pe.csl";', fn)
    assert re.search(r'else \{\s*return "sync/pe_off.csl";', fn)
    assert '"sync/pe.csl"' not in code.replace(fn, "")     # no unconditional import


def test_flag_off_stub_takes_exactly_the_real_modules_params():
    def params(f):
        return set(re.findall(r"^param\s+(\w+)\s*:", _code(CSL / "sync" / f), re.M))
    assert params("pe_off.csl") == params("pe.csl")
    stub = _code(CSL / "sync" / "pe_off.csl")
    assert "fn f_sync()" in stub and "var tscRefBuffer" in stub
    for forbidden in ("@bind_", "@get_input_queue", "@get_dsd", "@set_color_config",
                      "@activate", "<time>"):
        assert forbidden not in stub


def test_sync_layout_module_is_gone():
    assert not (CSL / "sync" / "layout.csl").exists()
