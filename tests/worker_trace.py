"""Record what the worker does to the device, so a refactor of it can be checked bit for bit.

Runs ``worker_runtime.main()`` against a recording fake SdkRuntime and prints one JSON document:
every runner call in order (symbol, rectangle, nonblock, a hash of the host buffer), the exit code,
the progress.log phases and the CSV row's deterministic columns.  The goldens in tests/golden/worker
were generated from the 4fb6594 worker; tests/test_worker_golden.py replays every case here against
the current one.

    python tests/worker_trace.py run CASE            # trace of the worker under CLUSTER_DIR (default: this repo)
    python tests/worker_trace.py regen OUT_DIR [CASE..]  # write goldens (twice, keep columns that repeat)

Each case is its own process: the worker keeps module-level state (vendor status, tile cache).
"""

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CLUSTER_DIR = Path(os.environ.get("CLUSTER_DIR", ROOT / "cluster"))

ROWS, COLS, DENSITY = 200, 190, 0.05

# name -> (encoding, extra worker args, compile params). Every flag a refactor could break.
CASES = {
    "header_column": ("header", ["--loop-count", "1"], {}),
    "header_full_colmajor_serialized": (
        "header", ["--h2d-mode", "full", "--memcpy-order", "col", "--serialize-column-rects",
                   "--setup-order", "row", "--setup-strips", "3"], {}),
    "valtask_column_sync": ("valtask", ["--loop-count", "2", "--sync-timing",
                                        "--setup-strips", "3"], {"SYNC_TIMING": 1}),
    "valtask_full": ("valtask", ["--h2d-mode", "full", "--memcpy-order", "col"], {}),
    "resident_sync": ("resident", ["--loop-count", "3", "--sync-timing"], {"SYNC_TIMING": 1}),
    "resident_null": ("resident", ["--loop-count", "2", "--sync-timing", "--null-spmv"],
                      {"SYNC_TIMING": 1}),
}

# Columns that can legitimately differ between two runs of the same worker.
VOLATILE = re.compile(r"(^timestamp$|time|cycles|gflops|ratio|_s$|numba|jit|host_cpu|python|"
                      r"platform|hostname|threads|vendor|cache)", re.I)


class _Enum:
    ROW_MAJOR, COL_MAJOR, MEMCPY_32BIT, MEMCPY_16BIT = "ROW", "COL", "32", "16"


class RecordingRuntime:
    log: list = []

    def __init__(self, name, cmaddr=None, **kw):
        RecordingRuntime.log.append(["init", sorted(kw), cmaddr])

    def get_id(self, name):
        RecordingRuntime.log.append(["get_id", name])
        return name

    def load(self): RecordingRuntime.log.append(["load"])
    def run(self): RecordingRuntime.log.append(["run"])
    def stop(self): RecordingRuntime.log.append(["stop"])

    def launch(self, name, nonblock):
        RecordingRuntime.log.append(["launch", name, bool(nonblock)])

    def memcpy_h2d(self, sym, flat, x0, y0, w, h, l, streaming, data_type, order, nonblock):
        arr = np.ascontiguousarray(flat)
        RecordingRuntime.log.append(
            ["h2d", sym if isinstance(sym, str) else int(sym), str(arr.dtype), int(arr.size),
             hashlib.sha256(arr.tobytes()).hexdigest()[:16], x0, y0, w, h, l, bool(streaming),
             str(data_type), str(order), bool(nonblock)])

    def memcpy_d2h(self, buf, sym, x0, y0, w, h, l, streaming, data_type, order, nonblock):
        RecordingRuntime.log.append(["d2h", sym, x0, y0, w, h, l, bool(streaming), str(data_type),
                                     str(order), bool(nonblock)])
        # deterministic non-zero bits, so the unpack / synced-window / validation paths all run
        words = (np.arange(buf.size, dtype=np.uint64) * 2654435761 + 12345) & 0x7FFFFFFF
        buf[:] = (words | 0x00010001).astype(np.uint32).view(np.float32) if sym != "y_result" \
            else (np.arange(buf.size) % 97).astype(np.float32)


def _install_fake_sdk():
    for n in ("cerebras", "cerebras.sdk", "cerebras.sdk.runtime"):
        sys.modules[n] = types.ModuleType(n)
    mod = types.ModuleType("cerebras.sdk.runtime.sdkruntimepybind")
    mod.SdkRuntime, mod.MemcpyDataType, mod.MemcpyOrder = RecordingRuntime, _Enum, _Enum
    sys.modules[mod.__name__] = mod


def run_case(case: str) -> dict:
    encoding, extra, params = CASES[case]
    sys.path[:0] = [str(ROOT), str(ROOT / "bench"), str(CLUSTER_DIR)]
    from config import SpMVConfig

    cfg = SpMVConfig(rows=ROWS, cols=COLS, density=DENSITY, num_pes_target=16, simulator=True,
                     encoding=encoding)
    work = Path(tempfile.mkdtemp(prefix="worker_trace_"))
    art = work / "out"
    art.mkdir()
    enc_id = 4 if "--null-spmv" in extra else cfg.encoding_id
    (art / "out.json").write_text(json.dumps({"params": {
        "width": cfg.width, "height": cfg.height, "KX": cfg.kx, "KY": cfg.ky,
        "TOTAL_NNZ": cfg.total_nnz_per_pe, "MEMCPYH2D_DATA_1_ID": 0, "ENCODING": enc_id,
        "SYNC_TIMING": 0, **params}}))
    os.chdir(work)
    argv = ["worker_runtime.py", "--name", str(art), "--rows", str(ROWS), "--cols", str(COLS),
            "--density", str(DENSITY), "--num-pes", "16", "--grid-width", str(cfg.width),
            "--grid-height", str(cfg.height), "--total-nnz-per-pe", str(cfg.total_nnz_per_pe),
            "--encoding", encoding, "--save-benchmark", "res.csv", *extra]
    sys.argv = argv
    _install_fake_sdk()
    import worker_runtime

    code = worker_runtime.main()
    progress = [re.sub(r"^\+\s*[\d.]+s ", "", ln.rstrip("\n"))
                for ln in open("progress.log")] if os.path.exists("progress.log") else []
    row = next(csv.DictReader(open("res.csv"))) if os.path.exists("res.csv") else {}
    return {"exit": code, "calls": RecordingRuntime.log, "progress": progress, "row": row}


def _stable_row(a: dict, b: dict) -> dict:
    return {k: v for k, v in a.items() if b.get(k) == v and not VOLATILE.search(k)}


def trace(case: str, cluster_dir: Path = CLUSTER_DIR) -> dict:
    out = subprocess.run([sys.executable, str(Path(__file__).resolve()), "run", case],
                         capture_output=True, text=True, check=True,
                         env={**os.environ, "CLUSTER_DIR": str(cluster_dir), "PYTHONHASHSEED": "0"})
    return json.loads(out.stdout.strip().splitlines()[-1])


def regen(out_dir: Path, only=None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for case in (only or CASES):
        a, b = trace(case), trace(case)
        assert a["calls"] == b["calls"] and a["progress"] == b["progress"], f"{case}: not deterministic"
        golden = {"exit": a["exit"], "calls": a["calls"], "progress": a["progress"],
                  "row": _stable_row(a["row"], b["row"])}
        (out_dir / f"{case}.json").write_text(json.dumps(golden, indent=0) + "\n")
        print(case, a["exit"], len(a["calls"]), "calls", len(golden["row"]), "stable columns")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        import logging
        logging.disable(logging.CRITICAL)
        print(json.dumps(run_case(sys.argv[2])))
    elif sys.argv[1] == "regen":
        regen(Path(sys.argv[2]), sys.argv[3:] or None)
