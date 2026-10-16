"""--encoding resident: A lives on the wafer (src/csl/pe_resident.csl, docs/resident-spmv.md).

Nothing here can run cslc or the wafer.  What is checked instead:

- the host plumbing: encoding id, config validation and SRAM budget (at the 65K size and
  at the full wafer), CLI / TOML / launcher flags, CSV schema;
- the setup layout, by a numpy model of the kernel (local CSR over the resident arrays,
  then the EW chain) fed exactly what the worker copies, multi-pass, from garbage;
- the WHOLE worker, ``worker_runtime.main()``, against a fake SdkRuntime that stores
  per-PE memory, decodes memcpy_h2d / memcpy_d2h layouts independently of the worker's
  flatten, emulates ``spmv`` with per-PE clocks, and has to be driven the way
  pe_resident.csl needs (no stream, no wait_ew, tic/toc around spmv);
- the static CSL contract.

The rest -- that the kernel compiles, does not hang and is as fast as docs/resident-spmv.md
says -- only the simulator and the wafer can settle.
"""

import csv
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from bench.benchmark import BenchmarkResult, load_benchmarks, save_benchmark
from cluster.coo_tiling import (
    RESIDENT_ENCODINGS as TILING_RESIDENT,
    build_resident_vals,
    build_val_stream_buffer,
    column_rects_serialized,
    extract_tile_structure,
    spmv_reference,
)
from cluster.sweep_config import SweepConfig
from config import (
    ENCODINGS,
    PE_SRAM_BUDGET_BYTES,
    RESIDENT_ENCODINGS,
    RESIDENT_LOOP_COUNT,
    SpMVConfig,
)
from run import build_parser, io_overrides, single_run_overrides
from sdk.compiler import compile_options
from sdk.launcher import build_worker_command
from tests.test_sync_timing import _pack_like_f_toc

from tests.worker_src import symbols_read, worker_source  # noqa: E402

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
# ids, config, SRAM
# --------------------------------------------------------------------------

def test_resident_is_compile_param_3_and_every_module_agrees():
    assert ENCODINGS == ("header", "valtask", "resident")
    assert RESIDENT_ENCODINGS == TILING_RESIDENT == _worker().RESIDENT_ENCODINGS == ("resident",)
    cfg = SpMVConfig(**HW, encoding="resident", simulator=True)
    assert cfg.encoding_id == 3
    assert "ENCODING:3" in compile_options(cfg)


def test_resident_streams_nothing_so_nothing_is_serialised():
    for mode in ("column", "full"):
        assert not column_rects_serialized("resident", mode, False)


def test_resident_keeps_vals_and_row_ids_but_not_val_meta():
    base = SpMVConfig(**HW, simulator=True, encoding="valtask")
    res = SpMVConfig(**HW, simulator=True, encoding="resident")
    assert res.total_nnz_per_pe == base.total_nnz_per_pe
    assert base.val_only_extra_words == base.total_nnz_per_pe + 1      # vals + val_meta
    assert res.val_only_extra_words == 2 * res.total_nnz_per_pe        # vals and row_ids (COO scatter)
    assert SpMVConfig(**HW, simulator=True, encoding="header").val_only_extra_words == 0


# The SRAM check: resident holds vals[] and row_ids[] (2 N words) where valtask holds vals[] + val_meta.
# Priced with the same arithmetic config.py uses for its guard.
def _sram_bytes(cfg):
    return 4 * (cfg.kx + cfg.total_nnz_per_pe + (cfg.ky + 1) + 2 * cfg.ky
                + cfg.val_only_extra_words)


@pytest.mark.parametrize("name, rows, num_pes", [
    ("65K", 114_000, 65_536),
    ("full wafer, weak-scaling point", 414_000, 857_010),
])
def test_resident_fits_in_pe_sram_at_the_sizes_we_run(name, rows, num_pes):
    cfg = SpMVConfig(rows=rows, cols=rows, density=1e-3, num_pes_target=num_pes,
                     encoding="resident", sync_timing=True)
    assert _sram_bytes(cfg) <= PE_SRAM_BUDGET_BYTES, name
    # an order of magnitude of headroom: SRAM is not what limits resident A at ~200 NNZ/PE
    assert _sram_bytes(cfg) < PE_SRAM_BUDGET_BYTES // 3, name
    val = SpMVConfig(rows=rows, cols=rows, density=1e-3, num_pes_target=num_pes,
                     encoding="valtask", sync_timing=True)
    # resident minus valtask: row_ids[N] in, val_meta[1] out
    assert _sram_bytes(cfg) - _sram_bytes(val) == 4 * (cfg.total_nnz_per_pe - 1)


def test_resident_at_the_full_wafer_max_nnz_still_fits():
    """~272 NNZ/PE is the full-wafer maximum (docs/results.md); budget it explicitly."""
    cfg = SpMVConfig(rows=414_000, cols=414_000, density=1e-3, num_pes_target=857_010,
                     encoding="resident", sync_timing=True, total_nnz_per_pe=272)
    assert cfg.val_only_extra_words == 2 * 272
    assert _sram_bytes(cfg) <= PE_SRAM_BUDGET_BYTES


def test_resident_too_big_for_sram_is_refused_before_a_compile():
    # 10M NNZ on 16 PEs: col_indices + vals alone are ~5 MB per PE.
    with pytest.raises(ValueError, match="exceeds"):
        SpMVConfig(rows=100_000, cols=100_000, density=1e-3, num_pes_target=16,
                   encoding="resident", simulator=True)


# --------------------------------------------------------------------------
# CLI, TOML, launcher
# --------------------------------------------------------------------------

def _args(*argv):
    return build_parser().parse_args(["run", "--rows", "10", "--cols", "10",
                                      "--density", "0.1", *argv])


def test_cli_selects_resident_and_defaults_to_20_passes():
    a = _args("--encoding", "resident", "--sync-timing")
    ov = single_run_overrides(a)
    assert ov["encoding"] == "resident" and ov["loop_count"] == RESIDENT_LOOP_COUNT == 20
    assert "loop_count" not in io_overrides(a)          # sweeps keep their TOML's loop_count


def test_an_explicit_loop_count_wins_and_other_encodings_stay_at_one():
    assert single_run_overrides(_args("--encoding", "resident", "--loop-count", "3"))[
        "loop_count"] == 3
    assert "loop_count" not in single_run_overrides(_args("--encoding", "valtask"))
    assert "loop_count" not in single_run_overrides(_args())


def test_sweep_toml_can_carry_resident(tmp_path):
    toml = tmp_path / "r.toml"
    toml.write_text('''
[meta]
name = "r"
[output]
work_dir = "w"
[[run]]
rows = 1000
cols = 1000
density = 0.01
num_pes_target = 16
simulator = true
encoding = "resident"
loop_count = 20
sync_timing = true
''')
    (cfg,) = SweepConfig.from_toml(toml).expand()
    assert (cfg.encoding, cfg.loop_count) == ("resident", 20)
    swept = tmp_path / "s.toml"
    swept.write_text('''
[meta]
name = "s"
[sweep]
type = "weak_scaling"
simulator = true
density = 0.01
nnz_per_pe = 200
encoding = "resident"
loop_count = 7
[sweep.num_pes]
pow2_up_to = 64
points_per_octave = 1
''')
    cfgs = list(SweepConfig.from_toml(swept).expand())
    assert cfgs and all(c.encoding == "resident" and c.loop_count == 7
                        for c in cfgs)


@pytest.mark.parametrize("name", ["resident.toml", "sim-resident.toml"])
def test_the_shipped_resident_configs_load(name):
    sweep = SweepConfig.from_toml(PROJECT_ROOT / "bench" / "configs" / name)
    cfgs = list(sweep.expand())
    assert cfgs
    for c in cfgs:
        assert c.encoding == "resident" and c.sync_timing and c.loop_count == 20
        assert c.simulator == name.startswith("sim-")


# --------------------------------------------------------------------------
# the setup arrays: build_resident_vals
# --------------------------------------------------------------------------

def _problem(rows, cols, density, W, H, seed=3):
    row, col, data = _coo(rows, cols, density, seed)
    KX, KY = -(-cols // W), -(-rows // H)
    ci, rp, counts, order, _ = extract_tile_structure(row, col, W, H, KX, KY)
    return row, col, data, KX, KY, ci, rp, counts, order


def _coo(rows, cols, density, seed):
    rng = np.random.default_rng(seed)
    nnz = max(1, int(rows * cols * density))
    return (rng.integers(0, rows, nnz, dtype=np.int32), rng.integers(0, cols, nnz, dtype=np.int32),
            rng.random(nnz).astype(np.float32))


@pytest.mark.parametrize("rows, cols, W, H", [(64, 64, 4, 4), (100, 70, 5, 3), (37, 41, 3, 7)])
def test_resident_vals_are_the_val_stream_laid_out_for_the_setup_copy(rows, cols, W, H):
    _, _, data, _, _, ci, _, counts, order = _problem(rows, cols, 0.3, W, H)
    vals = build_resident_vals(data, order, counts, W, H)
    stream = build_val_stream_buffer(data, order, counts, W, H)          # (W, H, max) u32
    assert vals.dtype == np.float32 and vals.shape == ci.shape == (W * H, max(counts))
    np.testing.assert_array_equal(vals.view(np.uint32),
                                  stream.transpose(1, 0, 2).reshape(W * H, -1))
    # slot k pairs with col_indices[k]; the tail past a PE's count is zero
    for pe, n in enumerate(counts):
        assert not vals[pe, n:].any()


def test_resident_vals_of_an_empty_matrix_are_one_zero_slot_per_pe():
    vals = build_resident_vals(np.zeros(0, np.float32), np.zeros(0, np.int64), [0] * 4, 2, 2)
    assert vals.shape == (4, 1) and not vals.any()


# --------------------------------------------------------------------------
# a numpy model of pe_resident.csl, driven with what the worker copies
# --------------------------------------------------------------------------

class ResidentDevice:
    """pe_resident.csl: setup arrays persist, y_result starts as garbage."""

    def __init__(self, W, H, KX, KY, total_nnz, seed=0):
        rng = np.random.default_rng(seed)
        self.row_ids = rng.integers(0, 7, (H, W, total_nnz)).astype(np.int32)   # garbage
        self.W, self.H, self.KY = W, H, KY
        self.x = rng.random((H, W, KX)).astype(np.float32) * 1e3        # garbage
        self.col = np.zeros((H, W, total_nnz), dtype=np.int32)
        self.vals = (rng.random((H, W, total_nnz)).astype(np.float32) - 0.5) * 1e6   # garbage
        self.rp = np.zeros((H, W, KY + 1), dtype=np.int32)
        self.y = rng.random((H, W, KY)).astype(np.float32) * 1e6        # garbage

    def local(self, py, px):
        """One PE's local SpMV, the COO scatter of pe_resident.csl: y zeroed by one DSD op, then every NNZ adds
        into its row. NNZ are in CSR order, n = row_ptrs[KY]. Garbage past n is never read."""
        KY = self.KY
        y, rp, vals, x, col = self.y[py, px], self.rp[py, px], self.vals[py, px], self.x[py, px], self.col[py, px]
        assert KY < 65535 and self.vals.shape[2] < 65536 and int(rp[KY]) < 65536     # the u16 comptime limits
        y[:] = 0                                                  # @fmovs(mem_y, 0.0)
        for k in range(int(rp[KY])):
            r = int(self.row_ids[py, px, k])
            y[r] = np.float32(y[r] + np.float32(vals[k] * x[col[k]]))

    def spmv(self):
        W, H, KY = self.W, self.H, self.KY
        for py in range(H):
            for px in range(W):
                self.local(py, px)
        # EW chain: x = W-1 sends, each PE adds its own to what arrives, x = 0 stores.
        for py in range(H):
            acc = self.y[py, W - 1].copy()
            for px in range(W - 2, -1, -1):
                acc = (self.y[py, px] + acc).astype(np.float32)
            self.y[py, 0] = acc


@pytest.mark.parametrize("rows, cols, W, H", [(64, 64, 4, 4), (100, 70, 5, 3), (37, 41, 3, 7)])
def test_setup_then_spmv_reproduces_the_reference_over_several_passes(rows, cols, W, H):
    w = _worker()
    row, col, data, KX, KY, ci, rp, counts, order = _problem(rows, cols, 0.3, W, H)
    dev = ResidentDevice(W, H, KX, KY, max(counts) + 3)
    vals = build_resident_vals(data, order, counts, W, H)
    m = max(counts)
    dev.col[:, :, :m] = ci.reshape(H, W, m)
    dev.vals[:, :, :m] = vals.reshape(H, W, m)
    dev.rp[:] = rp.reshape(H, W, KY + 1)
    dev.row_ids[:, :, :m] = build_row_ids(rp, counts).reshape(H, W, m)
    rng = np.random.default_rng(1)
    x = rng.integers(0, 100, cols).astype(np.float32)
    ref = spmv_reference(row, col, data, x, rows)
    for p in range(4):
        scale = float(p + 1)    # a different x every pass, exactly representable in f32
        xs = (x * np.float32(scale)).astype(np.float32)
        for px in range(W):
            seg = xs[px * KX:(px + 1) * KX]
            dev.x[:, px, :len(seg)] = seg
        dev.spmv()
        got = w.gather_y(dev.y[:, 0, :].reshape(-1), H, KY, rows)
        np.testing.assert_allclose(got, ref * np.float32(scale), rtol=2e-5, atol=1e-3)


# --------------------------------------------------------------------------
# worker helpers
# --------------------------------------------------------------------------

def test_a_pass_is_launches_only_and_f_toc_blocks_when_synced():
    w = _worker()
    assert w.resident_pass_commands(True) == [("f_tic", True), ("spmv", True), ("f_toc", False)]
    assert w.resident_pass_commands(False) == [("spmv", False)]
    for sync in (True, False):
        assert "wait_ew" not in [n for n, _ in w.resident_pass_commands(sync)]


# --------------------------------------------------------------------------
# CSV schema
# --------------------------------------------------------------------------

NEW_COLUMNS = ("wafer_gflops_avg", "null_spmv")


def test_new_columns_default_to_zero_and_the_reused_ones_are_still_there():
    r = BenchmarkResult()
    for c in NEW_COLUMNS:
        assert c in r.to_dict()
        assert r.to_dict()[c] in (0, 0.0, False)
    for c in ("wafer_time_avg", "wafer_cycles_max", "host_pass_time_avg",
              "stream_time_corrected", "wafer_host_ratio_max", "loop_count", "sync_timing"):
        assert c in r.to_dict()


def test_a_resident_row_round_trips_through_csv(tmp_path):
    path = tmp_path / "r.csv"
    save_benchmark(BenchmarkResult(encoding="resident", wafer_gflops_avg=1.5, loop_count=20), csv_path=str(path))
    (back,) = load_benchmarks(str(path))
    assert back.encoding == "resident" and back.wafer_gflops_avg == 1.5


def test_an_older_csv_migrates_and_backfills_the_new_columns(tmp_path):
    from dataclasses import fields
    path = tmp_path / "old.csv"
    old = [f.name for f in fields(BenchmarkResult) if f.name not in NEW_COLUMNS]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=old)
        wr.writeheader()
        wr.writerow({n: "" for n in old})
    save_benchmark(BenchmarkResult(encoding="resident"), csv_path=str(path))
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 2 and set(NEW_COLUMNS) <= set(rows[0])
    assert rows[0]["wafer_gflops_avg"] in ("0", "0.0", "") and rows[1]["encoding"] == "resident"


# --------------------------------------------------------------------------
# static CSL contract
# --------------------------------------------------------------------------

def test_layout_picks_pe_resident_for_encoding_3_and_gates_the_exports():
    layout = _code(CSL / "layout.csl")
    assert re.search(r'ENCODING == 3 or ENCODING == 4\)\s*\{\s*@set_tile_code\(px, py, "pe_resident\.csl", res_params', layout)
    assert re.search(r'if \(ENCODING == 3 or ENCODING == 4\) \{\s*@export_name\("vals",\s*\[\*\]f32, true\)', layout)
    assert re.search(r'if \(ENCODING < 3\) \{\s*@export_name\("wait_ew"', layout)
    assert re.search(r'ENCODING == 2\) \{\s*@export_name\("val_meta"', layout)


def test_pe_resident_exports_every_symbol_the_worker_reads_for_resident():
    exports = set(re.findall(r'@export_symbol\(\s*\w+\s*,\s*"(\w+)"', (CSL / "pe_resident.csl").read_text()))
    exports |= set(re.findall(r'@export_symbol\((\w+)\)', _code(CSL / "pe_resident.csl")))
    read = symbols_read()
    assert read - {"val_meta"} <= exports
    assert {"vals", "x_vec", "col_indices", "row_ptrs", "y_result", "tsc_pack", "spmv"} <= exports
    launched = set(re.findall(r'launch\("(\w+)"', worker_source()))
    assert launched - {"wait_ew"} <= exports | {"f_sync", "f_tic", "f_toc"}
    assert "wait_ew" not in exports and "val_meta" not in exports


def test_pe_resident_holds_the_command_stream_until_its_pass_is_done():
    code = _code(CSL / "pe_resident.csl")

    def body(name, kw):
        i = code.index(f"{kw} {name}(")
        return code[i:code.index("\n}\n", i)]

    # the only release in spmv() is the --null-spmv early exit, before anything is started
    spmv = body("spmv", "fn")
    assert spmv.count("unblock_cmd_stream") == 1
    assert spmv.index("unblock_cmd_stream") < spmv.index("@activate")
    assert re.match(r"fn spmv\(\) void \{\s*if \(NULL_SPMV\) \{\s*sys_mod\.unblock_cmd_stream\(\);\s*return;", spmv)
    assert "@activate(ew_start_task_id)" in body("spmv", "fn")
    assert "unblock_cmd_stream" in body("chain_done", "task")
    # no stream machinery at all
    for gone in ("@bind_data_task", "@block(", "@unblock(", "val_meta", "draining",
                 "MEMCPYH2D", "wait_ew"):
        assert gone not in code, gone
    # SYNC_TIMING == 1: the tsc free-runs, so every enable/disable sits inside an
    # `if (SYNC_TIMING == 0) { ... }` block.
    bare = re.sub(r"if \(SYNC_TIMING == 0\) \{[^{}]*\}", "", code)
    assert "enable_tsc" in code and "disable_tsc" in code
    assert "enable_tsc" not in bare and "disable_tsc" not in bare


def test_ew_start_is_valtask_s_with_the_coo_scatter_and_a_dsd_copy():
    def grab(path, start):
        t = _code(path)
        i = t.index(start)
        return t[i:t.index("\n}\n", i)]
    res = grab(CSL / "pe_resident.csl", "task ew_start()")
    val = grab(CSL / "pe_valtask.csl", "task ew_start()")
    loop = ("        var k: i32 = 0;\n        while (k < KY) : (k += 1) {\n"
            "            ew_buf[k] = y_result[k];\n        }\n")
    assert loop in val
    def norm(t):
        return "\n".join(ln.rstrip() for ln in t.splitlines())
    assert norm(res) == norm(val.replace(loop, "        @fmovs(mem_ew, mem_y);\n"))


# --------------------------------------------------------------------------
# the whole worker against a fake SdkRuntime
# --------------------------------------------------------------------------

class _Enum:
    MEMCPY_32BIT = "32"
    ROW_MAJOR = "row"
    COL_MAJOR = "col"


class FakeSdkRuntime:
    """A resident-A device behind the SdkRuntime API the worker uses.

    Per-PE memory per symbol starts as garbage.  memcpy layouts are decoded HERE, from
    the layout the SDK documents (and csl-examples' bandwidth-test uses), not with the
    worker's flatten helpers: ROW_MAJOR is host [h][w][l], COL_MAJOR is host [l][w][h].
    ``spmv`` runs ResidentDevice's model; clocks are per-PE with arbitrary offsets.
    """

    log = []                  # every API call, in order
    null = False              # --null-spmv: spmv() computes nothing
    last = None

    def __init__(self, name, cmaddr=None, **kw):
        out = __import__("json").load(open(f"{name}/out.json"))["params"]
        self.W, self.H = int(out["width"]), int(out["height"])
        self.KX, self.KY, self.T = int(out["KX"]), int(out["KY"]), int(out["TOTAL_NNZ"])
        self.dev = ResidentDevice(self.W, self.H, self.KX, self.KY, self.T, seed=11)
        rng = np.random.default_rng(5)
        self.offset = rng.integers(0, 2**40, (self.H, self.W))
        self.clock = 1_000_000
        self.ref = None
        self.synced_run = False
        self.tic = self.toc = None
        self.mem = {"x_vec": self.dev.x, "col_indices": self.dev.col, "vals": self.dev.vals,
                    "row_ptrs": self.dev.rp, "y_result": self.dev.y,
                    "row_ids": self.dev.row_ids}
        self.setup_done = False
        self.pass_no = 0
        FakeSdkRuntime.log = []
        FakeSdkRuntime.last = self

    def get_id(self, name):
        if name in ("val_meta", "wait_ew"):
            raise KeyError(f"{name} is not exported by pe_resident.csl")
        return name

    def load(self): pass
    def run(self): pass
    def stop(self): pass

    def _rect(self, flat, w, h, l, order):
        flat = np.asarray(flat)
        if order == _Enum.ROW_MAJOR:
            return flat.reshape(h, w, l)                      # [h][w][l]
        return flat.reshape(l, w, h).transpose(2, 1, 0)       # -> [h][w][l]

    def memcpy_h2d(self, sym, flat, x0, y0, w, h, l, streaming, data_type, order, nonblock):
        FakeSdkRuntime.log.append(("h2d", sym, x0, y0, w, h, l, streaming, nonblock))
        assert streaming is False, "resident must not stream A"
        assert data_type == _Enum.MEMCPY_32BIT
        mem = self.mem[sym]
        assert l <= mem.shape[2], f"{sym}: {l} words into a {mem.shape[2]}-word symbol"
        blk = self._rect(flat, w, h, l, order)
        assert blk.shape == (h, w, l)
        mem[y0:y0 + h, x0:x0 + w, :l] = blk.view(mem.dtype) if blk.dtype != mem.dtype else blk
        if sym == "row_ptrs" and x0 + w == self.W:
            self.setup_done = True

    def memcpy_d2h(self, buf, sym, x0, y0, w, h, l, streaming, data_type, order, nonblock):
        FakeSdkRuntime.log.append(("d2h", sym, x0, y0, w, h, l))
        assert streaming is False
        if sym == "sync_pack":
            pack = self._sync_pack().reshape(self.H, self.W, 5)
            blk = pack[y0:y0 + h, x0:x0 + w, :l]
        elif sym == "tsc_pack":
            pack = self._sync_pack().reshape(self.H, self.W, 5)[..., :3]
            blk = pack[y0:y0 + h, x0:x0 + w, :l]
        else:
            blk = self.mem[sym][y0:y0 + h, x0:x0 + w, :l]
        if order == _Enum.ROW_MAJOR:
            out = np.ascontiguousarray(blk).reshape(-1)
        else:
            out = np.ascontiguousarray(blk.transpose(2, 1, 0)).reshape(-1)
        buf[:out.size] = out.view(buf.dtype) if out.dtype != buf.dtype else out

    def _sync_pack(self):
        return _pack_like_f_toc(self.tic + self.offset, self.toc + self.offset,
                                self.ref + self.offset)

    def launch(self, name, nonblock):
        FakeSdkRuntime.log.append(("launch", name, nonblock))
        py, px = np.indices((self.H, self.W))
        if name == "f_sync":
            self.synced_run = True
            self.ref = 1_000 + px + py + self.clock          # broadcast: px + py hops
        elif name == "f_tic":
            self.tic = self.clock + np.random.default_rng(self.pass_no).integers(0, 400, (self.H, self.W))
        elif name == "spmv":
            if not FakeSdkRuntime.null:
                self.dev.spmv()
            if self.ref is None:          # unsynced run: give tsc_pack something to read
                self.ref = np.full((self.H, self.W), 1_000, dtype=np.int64)
            # a PE finishes after its compute plus an EW chain that reaches it later the
            # further west it is: 14 cycles per hop
            start = self.clock + 600 + np.random.default_rng(self.pass_no + 7).integers(0, 50, (self.H, self.W))
            self.end = start + (40 if FakeSdkRuntime.null else 5_000 + 14 * (self.W - px))
            if self.tic is None or not self.synced_run:
                self.tic, self.toc = start, self.end + 20
                self.clock = int(self.end.max()) + 10_000
                self.pass_no += 1
        elif name == "f_toc":
            self.toc = self.end + 20
            self.clock = int(self.end.max()) + 10_000
            self.pass_no += 1
        elif name == "wait_ew":
            raise AssertionError("pe_resident.csl has no wait_ew")
        else:
            raise AssertionError(f"unexpected launch({name!r})")


@pytest.fixture
def fake_sdk(monkeypatch):
    mods = {}
    for n in ("cerebras", "cerebras.sdk", "cerebras.sdk.runtime"):
        mods[n] = types.ModuleType(n)
    py = types.ModuleType("cerebras.sdk.runtime.sdkruntimepybind")
    py.SdkRuntime, py.MemcpyDataType, py.MemcpyOrder = FakeSdkRuntime, _Enum, _Enum
    mods["cerebras.sdk.runtime.sdkruntimepybind"] = py
    for n, m in mods.items():
        monkeypatch.setitem(sys.modules, n, m)
    return FakeSdkRuntime


def _run_worker(tmp_path, monkeypatch, *extra, sync=True, rows=200, cols=190, density=0.05,
                encoding_id=3, mtx=None, total_nnz=None):
    """Run worker_runtime.main() on a 4x4 grid; returns (exit code, the CSV's one row).

    ``mtx``: a MatrixMarket file for --mtx-file; ``total_nnz``: the compiled TOTAL_NNZ (default: the
    uniform-random estimate, which a skewed matrix can exceed)."""
    w = _worker()
    cfg = SpMVConfig(rows=rows, cols=cols, density=density, num_pes_target=16, simulator=True,
                     encoding="resident", sync_timing=sync, total_nnz_per_pe=total_nnz)
    art = tmp_path / "out"
    art.mkdir(exist_ok=True)
    import json
    (art / "out.json").write_text(json.dumps({"params": {
        "width": cfg.width, "height": cfg.height, "KX": cfg.kx, "KY": cfg.ky,
        "TOTAL_NNZ": cfg.total_nnz_per_pe, "MEMCPYH2D_DATA_1_ID": 0, "ENCODING": encoding_id,
        "SYNC_TIMING": int(sync)}}))
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "res.csv"
    if csv_path.exists():
        csv_path.unlink()
    argv = ["worker_runtime.py", "--name", str(art), "--rows", str(rows), "--cols", str(cols),
            "--density", str(density), "--num-pes", "16", "--grid-width", str(cfg.width),
            "--grid-height", str(cfg.height), "--total-nnz-per-pe", str(cfg.total_nnz_per_pe),
            "--encoding", "resident", "--save-benchmark", str(csv_path), *extra]
    if sync:
        argv.append("--sync-timing")
    if mtx is not None:
        argv += ["--mtx-file", str(mtx)]
    monkeypatch.setattr(sys, "argv", argv)
    code = w.main()
    rows_out = list(csv.DictReader(open(csv_path))) if csv_path.exists() else []
    return code, (rows_out[0] if rows_out else None), cfg


@pytest.mark.parametrize("setup_order", ["row", "col"])
@pytest.mark.parametrize("strips", ["1", "3"])
def test_worker_runs_resident_end_to_end_synced(fake_sdk, tmp_path, monkeypatch, setup_order, strips):
    code, row, cfg = _run_worker(tmp_path, monkeypatch, "--loop-count", "5",
                                 "--setup-order", setup_order, "--setup-strips", strips)
    assert code == 0 and row["validation_passed"] == "True"
    assert row["encoding"] == "resident" and row["loop_count"] == "5"
    assert float(row["relative_error"]) < 1e-6
    # nothing streamed, per pass or at all
    assert row["h2d_bytes_total"] == "0" and float(row["memcpy_h2d_time"]) == 0.0
    # synced window: positive, bounded by the model (5,000 + 14 W + skew ~ 5.2-6k cycles)
    assert 5_000 < float(row["wafer_cycles_min"]) <= float(row["wafer_cycles_max"]) < 8_000
    assert float(row["wafer_time_avg"]) == pytest.approx(float(row["wafer_cycles_avg"]) / 0.85e9, rel=1e-3)
    assert 0 < float(row["wafer_host_ratio_max"]) and float(row["wafer_gflops_avg"]) > 0
    assert float(row["host_pass_time_avg"]) > 0 and float(row["stream_time_corrected"]) >= 0
    # the setup copy carries the values and the row ids: x + col_indices + vals + row_ids + row_ptrs
    W, H, KX, KY = cfg.width, cfg.height, cfg.kx, cfg.ky
    m = int(row["max_nnz_per_pe"])
    assert int(row["setup_h2d_bytes"]) == 4 * W * H * (KX + 3 * m + KY + 1)
    assert int(row["max_memory_per_pe"]) == 4 * (KX + m + (KY + 1) + 2 * KY + 2 * m)


def test_worker_drives_the_resident_kernel_the_way_it_needs(fake_sdk, tmp_path, monkeypatch):
    code, _, _ = _run_worker(tmp_path, monkeypatch, "--loop-count", "3")
    assert code == 0
    log = FakeSdkRuntime.log
    h2d = [e for e in log if e[0] == "h2d"]
    assert {e[1] for e in h2d} == {"x_vec", "col_indices", "vals", "row_ids", "row_ptrs"}   # setup only
    assert all(e[7] is False for e in h2d)                                       # copy mode
    assert [e[1] for e in log if e[0] == "launch"] == \
        ["f_sync"] + ["f_tic", "spmv", "f_toc"] * 3
    launches = {e[1]: e[2] for e in log if e[0] == "launch"}
    assert launches == {"f_sync": True, "f_tic": True, "spmv": True, "f_toc": False}
    # setup: row_ptrs is the LAST array (the setup barrier reads it)
    syms = [e[1] for e in h2d]
    assert syms[-1] == "row_ptrs" and syms.index("vals") < syms.index("row_ptrs")


def test_worker_unsynced_resident_is_one_blocking_launch_per_pass(fake_sdk, tmp_path, monkeypatch):
    code, row, _ = _run_worker(tmp_path, monkeypatch, "--loop-count", "2", sync=False)
    assert code == 0 and row["validation_passed"] == "True"
    assert [e[1:] for e in FakeSdkRuntime.log if e[0] == "launch"] == [("spmv", False)] * 2
    assert row["sync_timing"] == "False" and float(row["wafer_time_avg"]) == 0.0
    assert float(row["host_pass_time_avg"]) > 0       # the barrier-closed host time still lands


# --------------------------------------------------------------------------
# --null-spmv: the launch-latency floor (ENCODING 4, no new option-string entry)
# --------------------------------------------------------------------------

def test_null_spmv_is_encoding_4_and_changes_no_other_option_string():
    from config import NULL_SPMV_ENCODING_ID
    from cluster.coo_tiling import NULL_SPMV_ENCODING_ID as T
    assert NULL_SPMV_ENCODING_ID == T == _worker().NULL_SPMV_ENCODING_ID == 4
    res = SpMVConfig(**HW, simulator=True, encoding="resident", sync_timing=True)
    null = SpMVConfig(**HW, simulator=True, encoding="resident", sync_timing=True, null_spmv=True)
    assert null.encoding == "resident" and null.encoding_id == 4 and res.encoding_id == 3
    assert compile_options(null) == compile_options(res).replace("ENCODING:3", "ENCODING:4")


def test_null_spmv_validation_and_flags():
    with pytest.raises(ValueError, match="resident"):
        SpMVConfig(**HW, simulator=True, null_spmv=True)
    ov = single_run_overrides(_args("--encoding", "resident", "--null-spmv", "--sync-timing"))
    assert ov["null_spmv"] is True and ov["loop_count"] == 20
    assert "--null-spmv" in build_worker_command(SpMVConfig(**HW, simulator=True, **ov))
    assert "--null-spmv" not in build_worker_command(
        SpMVConfig(**HW, simulator=True, encoding="resident"))


def test_worker_reads_the_null_artifact_as_resident():
    w = _worker()
    assert w.artifact_encoding({"params": {"ENCODING": 4}}) == "resident"
    assert w.artifact_null_spmv({"params": {"ENCODING": 4}}) is True
    assert w.artifact_null_spmv({"params": {"ENCODING": 3}}) is False
    assert w.artifact_null_spmv({"params": {}}) is False


def test_worker_null_spmv_skips_validation_and_times_the_launch(fake_sdk, tmp_path, monkeypatch):
    FakeSdkRuntime.null = True
    try:
        code, row, _ = _run_worker(tmp_path, monkeypatch, "--loop-count", "4", "--null-spmv",
                                   encoding_id=4)
    finally:
        FakeSdkRuntime.null = False
    assert code == 0 and row["exit_reason"] == "SUCCESS_NULL_SPMV" and row["null_spmv"] == "True"
    assert float(row["wafer_cycles_avg"]) > 0 and float(row["wafer_gflops_avg"]) == 0.0


def test_worker_refuses_a_null_flag_that_disagrees_with_the_artifact(fake_sdk, tmp_path, monkeypatch):
    code, _, _ = _run_worker(tmp_path, monkeypatch, "--null-spmv", encoding_id=3)
    assert code == 1


# --------------------------------------------------------------------------
# progress.log: a hang names its phase
# --------------------------------------------------------------------------

def test_the_worker_logs_every_phase_to_a_file(fake_sdk, tmp_path, monkeypatch):
    code, _, _ = _run_worker(tmp_path, monkeypatch, "--loop-count", "2")
    assert code == 0
    lines = (tmp_path / "progress.log").read_text().splitlines()
    text = "\n".join(lines)
    for phase in ("start resident", "setup copy start", "setup copy and its barrier done",
                  "f_sync", "pass 1/2 start",
                  "pass 1 launches start", "pass 1 launches returned, barrier D2H start",
                  "pass 1/2 end", "pass 2/2 end"):
        assert phase in text, phase
    # in order, timestamps non-decreasing
    stamps = [float(ln.split("s ")[0].lstrip("+")) for ln in lines]
    assert stamps == sorted(stamps)
    assert text.index("setup copy start") < text.index("pass 1/2 start") < text.index("pass 2/2 end")


def test_progress_is_a_noop_without_a_file_and_never_raises(tmp_path):
    w = _worker()
    w.make_progress(None)("nothing")
    w.make_progress(str(tmp_path / "no_such_dir" / "p.log"))("unwritable")   # swallowed


def test_launcher_fetches_the_phase_log_even_when_the_run_raises(tmp_path, monkeypatch):
    import sdk.launcher as L

    class FakeLauncher:
        def __init__(self):
            self.downloads = []

        def download_artifact(self, name, dest):
            self.downloads.append(name)
            if name == "progress.log":
                Path(dest).write_text("+ 1.0s setup copy start\n")
            else:
                raise FileNotFoundError(name)

    fl = FakeLauncher()
    L._download_progress(fl, tmp_path / "w" / "progress.log")
    assert (tmp_path / "w" / "progress.log").read_text().startswith("+ 1.0s")

    class Broken:
        def download_artifact(self, name, dest):
            raise RuntimeError("job cancelled")
    L._download_progress(Broken(), tmp_path / "x" / "progress.log")        # does not raise
    src = Path(L.__file__).read_text()
    assert src.index("_download_progress(launcher") < src.index("_download_benchmark(launcher, download_path")
    assert "progress-file progress.log" in build_worker_command(SpMVConfig(**HW, simulator=True))


# --------------------------------------------------------------------------
# the one resident kernel: COO scatter, bitwise the CSR reference
# --------------------------------------------------------------------------

from cluster.coo_tiling import build_row_ids


def _local_y(rows, cols, W, H, seed, density=0.3):
    """(device y bits after the COO scatter, CSR-reference y bits), per PE, before the EW chain."""
    row, col, data, KX, KY, ci, rp, counts, order = _problem(rows, cols, density, W, H, seed)
    dev = ResidentDevice(W, H, KX, KY, max(counts) + 3, seed=seed)
    m = max(counts)
    dev.col[:, :, :m] = ci.reshape(H, W, m)
    vals = build_resident_vals(data, order, counts, W, H)
    dev.vals[:, :, :m] = vals.reshape(H, W, m)
    dev.row_ids[:, :, :m] = build_row_ids(rp, counts).reshape(H, W, m)
    dev.rp[:] = rp.reshape(H, W, KY + 1)
    x = np.random.default_rng(seed + 1).integers(0, 100, cols).astype(np.float32)
    for px in range(W):
        seg = x[px * KX:(px + 1) * KX]
        dev.x[:, px, :len(seg)] = seg
    for py in range(H):
        for px in range(W):
            dev.local(py, px)
    # independent reference: a CSR row loop over the same tile, float32 accumulation from 0.0 in row order
    ref = np.zeros((H, W, KY), dtype=np.float32)
    rp3, ci3, va3 = rp.reshape(H, W, KY + 1), ci.reshape(H, W, m), vals.reshape(H, W, m)
    for py in range(H):
        for px in range(W):
            xs = dev.x[py, px]
            for r in range(KY):
                acc = np.float32(0)
                for k in range(rp3[py, px, r], rp3[py, px, r + 1]):
                    acc = np.float32(acc + np.float32(va3[py, px, k] * xs[ci3[py, px, k]]))
                ref[py, px, r] = acc
    return dev.y.copy().view(np.uint32), ref.view(np.uint32)


@pytest.mark.parametrize("rows, cols, W, H, density", [
    (64, 64, 4, 4, 0.3), (100, 70, 5, 3, 0.05), (37, 41, 3, 7, 0.02),   # hypersparse: empty rows
    (200, 30, 2, 2, 0.5)])                                              # dense, duplicate-heavy
def test_the_coo_scatter_is_bitwise_the_csr_reference(rows, cols, W, H, density):
    got, ref = _local_y(rows, cols, W, H, seed=5, density=density)
    assert np.array_equal(got, ref)


def test_the_coo_scatter_zeroes_empty_rows_and_ignores_garbage_row_ids():
    # a matrix with most rows empty: the garbage in y_result and past-the-count row_ids must not leak
    got, ref = _local_y(90, 50, 3, 3, seed=9, density=0.01)
    assert np.array_equal(got, ref)
    assert (ref == 0).mean() > 0.5           # the case this guards: mostly empty rows


@pytest.mark.parametrize("rows, cols, W, H", [(64, 64, 4, 4), (100, 70, 5, 3), (37, 41, 3, 7)])
def test_row_ids_are_the_row_of_every_nnz_in_csr_order(rows, cols, W, H):
    row, col, data, KX, KY, ci, rp, counts, order = _problem(rows, cols, 0.2, W, H, seed=2)
    ids = build_row_ids(rp, counts)
    assert ids.dtype == np.int32 and ids.shape == ci.shape
    # independent construction from the COO: sorted (pe, row, col) order == `order`
    pe = (row // KY) * W + col // KX
    sorted_pe, sorted_row = pe[order], (row % KY)[order]
    pos = 0
    for p, n in enumerate(counts):
        assert np.array_equal(ids[p, :n], sorted_row[pos:pos + n])
        assert not ids[p, n:].any()                       # padding is 0 and never read
        # and it is consistent with row_ptrs
        for r in range(KY):
            assert (ids[p, rp[p, r]:rp[p, r + 1]] == r).all()
        pos += n
    assert (sorted_pe[:pos] == np.repeat(np.arange(W * H), counts)).all()


def test_row_ids_of_an_empty_matrix():
    ids = build_row_ids(np.zeros((4, 6), dtype=np.int32), [0, 0, 0, 0])
    assert ids.shape == (4, 1) and not ids.any()


def test_row_ids_chunking_does_not_change_the_result():
    row, col, data, KX, KY, ci, rp, counts, order = _problem(120, 60, 0.1, 6, 5, seed=4)
    big = np.tile(rp, (300, 1))                         # 9,000 PEs: three 4,096-PE chunks
    ids = build_row_ids(big, list(counts) * 300)
    assert np.array_equal(ids[:30], build_row_ids(rp, counts))
    assert np.array_equal(ids[8999], ids[29])


def test_there_is_no_local_kernel_flag_or_param_any_more():
    base = SpMVConfig(**HW, simulator=True, encoding="resident", sync_timing=True)
    assert "LOCAL_KERNEL" not in compile_options(base)
    assert "local-kernel" not in build_worker_command(base)
    assert not hasattr(base, "local_kernel")
    with pytest.raises(TypeError):
        SpMVConfig(**HW, simulator=True, encoding="resident", local_kernel=2)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--rows", "8", "--cols", "8", "--density", "0.5", "--local-kernel", "2"])
    ov = single_run_overrides(_args("--encoding", "resident", "--null-spmv"))
    assert ov["null_spmv"] is True and SpMVConfig(**HW, simulator=True, **ov).encoding_id == 4


@pytest.mark.parametrize("name, rows, num_pes", [("65K", 114_000, 65_536),
                                                 ("full wafer", 414_000, 857_010)])
def test_resident_fits_in_pe_sram_with_row_ids(name, rows, num_pes):
    cfg = SpMVConfig(rows=rows, cols=rows, density=1e-3, num_pes_target=num_pes,
                     encoding="resident", sync_timing=True)
    base = SpMVConfig(rows=rows, cols=rows, density=1e-3, num_pes_target=num_pes, encoding="header",
                      sync_timing=True)
    words = cfg.kx + cfg.total_nnz_per_pe + cfg.ky + 1 + 2 * cfg.ky + cfg.val_only_extra_words + 11
    assert 4 * words <= PE_SRAM_BUDGET_BYTES, name
    assert 4 * words < PE_SRAM_BUDGET_BYTES // 3, name
    assert cfg.val_only_extra_words == 2 * cfg.total_nnz_per_pe        # vals[] and row_ids[]
    assert cfg.ky < 65535 and cfg.kx < 65536 and cfg.total_nnz_per_pe < 65536    # the u16 comptime limits
    assert base.val_only_extra_words == 0


@pytest.mark.parametrize("setup_order", ["row", "col"])
def test_worker_end_to_end_sends_row_ids_before_row_ptrs(fake_sdk, tmp_path, monkeypatch, setup_order):
    code, row, cfg = _run_worker(tmp_path, monkeypatch, "--loop-count", "3", "--setup-strips", "3",
                                 "--setup-order", setup_order)
    assert code == 0 and row["validation_passed"] == "True" and float(row["relative_error"]) < 1e-6
    syms = [e[1] for e in FakeSdkRuntime.log if e[0] == "h2d"]
    assert syms.index("row_ids") < syms.index("row_ptrs")          # row_ptrs stays last: the barrier reads it
    W, H, KX, KY = cfg.width, cfg.height, cfg.kx, cfg.ky
    m = int(row["max_nnz_per_pe"])
    assert int(row["setup_h2d_bytes"]) == 4 * W * H * (KX + 3 * m + KY + 1)
    assert int(row["max_memory_per_pe"]) == 4 * (KX + m + (KY + 1) + 2 * KY + 2 * m)
    assert "local_kernel" not in row


def test_worker_refuses_an_artifact_compiled_for_a_csr_kernel(fake_sdk, tmp_path, monkeypatch):
    """Old artifacts carry LOCAL_KERNEL; 0 and 1 are CSR loops without row_ids, 2 is the COO scatter."""
    import json
    w = _worker()
    assert w.artifact_local_kernel({"params": {"LOCAL_KERNEL": 2}}) == 2
    assert w.artifact_local_kernel({"params": {}}) is None
    code, _, _ = _run_worker(tmp_path, monkeypatch)
    assert code == 0
    art = tmp_path / "out" / "out.json"
    params = json.loads(art.read_text())
    for old, expect in ((0, 1), (1, 1), (2, 0)):
        params["params"]["LOCAL_KERNEL"] = old
        art.write_text(json.dumps(params))
        monkeypatch.setattr(sys, "argv", ["worker_runtime.py", "--name", str(tmp_path / "out"), "--rows", "200",
                                          "--cols", "190", "--density", "0.05", "--num-pes", "16",
                                          "--grid-width", str(params["params"]["width"]),
                                          "--grid-height", str(params["params"]["height"]),
                                          "--total-nnz-per-pe", str(params["params"]["TOTAL_NNZ"]),
                                          "--encoding", "resident", "--sync-timing",
                                          "--save-benchmark", str(tmp_path / "x.csv")])
        assert w.main() == expect, old


def test_csl_has_one_resident_kernel_and_always_exports_row_ids():
    layout = _code(CSL / "layout.csl")
    assert "LOCAL_KERNEL" not in layout
    assert re.search(r'if \(ENCODING == 3 or ENCODING == 4\) \{\s*@export_name\("vals",\s*\[\*\]f32, true\);\s*'
                     r'@export_name\("row_ids",\s*\[\*\]i32, true\);', layout)
    code = _code(CSL / "pe_resident.csl")
    assert "LOCAL_KERNEL" not in code and "local_spmv_u16" not in code and "copy_to_ew_buf" not in code
    assert '@export_symbol(ptr_row_ids,     "row_ids");' in code
    assert "var row_ids     = @zeros([TOTAL_NNZ]i32);" in code
    # the 16-bit limits of the u16 loop state are a compile-time assert, not a silent wrap
    assert "@comptime_assert(TOTAL_NNZ < 65536 and KY < 65535 and KX < 65536)" in code
    body = code[code.index("fn local_spmv()"):code.index("task ew_start()")]
    assert "@fmovs(mem_y, 0.0)" in body and "row_ptrs[KY]" in body and "row_ids[k]" in body
    assert "@fmovs(mem_ew, mem_y)" in code


# --------------------------------------------------------------------------
# host_prep (the merged fast tiler) for --encoding resident
# --------------------------------------------------------------------------

def _hp():
    sys.path.insert(0, str(PROJECT_ROOT / "cluster"))
    import host_prep, tile_cache
    return host_prep, tile_cache


def _prep_resident(encoding="resident", fp=None, rows=60, cols=70, nnz=900, W=5, H=4, seed=2):
    host_prep, _ = _hp()
    r, c, d = _coo(rows, cols, nnz / (rows * cols), seed)
    x = np.arange(cols, dtype=np.float32)
    KX, KY = -(-cols // W), -(-rows // H)
    return host_prep.prepare_host_buffers(r, c, d, x, cols, W, H, KX, KY, encoding, "column",
                                          cache_fingerprint=fp), (r, c, d, KX, KY, W, H)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_host_prep_resident_vals_are_build_resident_vals_whichever_tiler_ran(backend, monkeypatch):
    if backend == "numba":
        pytest.importorskip("numba")
    monkeypatch.setenv("SPMV_TILE_BACKEND", backend)
    hb, (r, c, d, KX, KY, W, H) = _prep_resident()
    assert hb.backend == backend and hb.val_only is False
    assert hb.vals_grid.dtype == np.float32 and hb.vals_grid.shape == hb.ci.shape
    assert hb.val_buf is None and hb.val_plan is None and hb.stream_buf is None
    ci, rp, counts, order, _ = extract_tile_structure(r, c, W, H, KX, KY)
    np.testing.assert_array_equal(hb.vals_grid, build_resident_vals(d, order, counts, W, H))
    np.testing.assert_array_equal(hb.ci, ci)
    np.testing.assert_array_equal(hb.rp, rp)


def test_host_prep_resident_streams_nothing_and_skips_row_bounds():
    hb, _ = _prep_resident()
    assert (hb.total_stream_words, hb.h2d_words_col, hb.h2d_words_full, hb.common_words) == (0, 0, 0, 0)


def test_the_tile_cache_is_shared_by_resident_and_valtask_and_gives_the_same_values(tmp_path, monkeypatch):
    host_prep, tile_cache = _hp()
    monkeypatch.setenv("SPMV_TILE_CACHE_DIR", str(tmp_path / "tiles"))
    monkeypatch.delenv("SPMV_TILE_CACHE", raising=False)
    fp = tile_cache.random_fingerprint(60, 70, 0.2, 2)
    cold, _ = _prep_resident(fp=fp)
    assert not cold.cache_hit and cold.save_thread is not None
    cold.save_thread.join()
    warm, _ = _prep_resident(fp=fp)
    assert warm.cache_hit and warm.backend == "cache"
    np.testing.assert_array_equal(warm.vals_grid, cold.vals_grid)
    # the valtask encoding hits the entry resident wrote
    val, _ = _prep_resident(encoding="valtask", fp=fp)
    assert val.cache_hit
    np.testing.assert_array_equal(
        val.val_buf.transpose(1, 0, 2).reshape(val.ci.shape[0], -1).view(np.float32), warm.vals_grid)


def test_a_resident_worker_run_goes_through_host_prep_not_a_stream_build(fake_sdk, tmp_path, monkeypatch):
    code, row, _ = _run_worker(tmp_path, monkeypatch, "--loop-count", "2")
    assert code == 0 and row["tile_backend"] in ("numpy", "numba", "aot", "cache")
    assert row["h2d_bytes_total"] == "0"
