#!/usr/bin/env python3
"""Host-only timing of the SpMV preparation pipeline (no wafer, no SDK).

Runs the same code the worker runs (cluster/host_prep.prepare_host_buffers, then
the setup-array flatten) on a synthetic matrix from the same generator and seed
as generate_random_coo, and prints the time of every phase.

    uv run --no-project --python 3.11 --with numpy --with loguru python bench/host_tiling_bench.py 65k
    ... python bench/host_tiling_bench.py full --checksum

Presets: 65k (114000^2, 256x256 PEs), full (414000^2, 742x1155 PEs); d=1e-3.
"""

import argparse
import hashlib
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "cluster"), str(ROOT / "bench")]   # the worker imports these flat

import host_prep  # noqa: E402
import tile_cache  # noqa: E402
from worker_runtime import setup_strip_bounds, setup_strip_calls  # noqa: E402

PRESETS = {
    "65k":  dict(rows=114_000, cols=114_000, W=256, H=256),
    "full": dict(rows=414_000, cols=414_000, W=742, H=1155),
}


def synthetic_coo(rows, cols, density, seed):
    """Exactly generate_random_coo's generator, minus the on-disk cache."""
    rng = np.random.default_rng(seed)
    nnz = max(1, int(rows * cols * density))
    r = rng.integers(0, rows, size=nnz, dtype=np.int32)
    c = rng.integers(0, cols, size=nnz, dtype=np.int32)
    d = rng.random(size=nnz).astype(np.float32)
    return r, c, d


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).data)
    return h.hexdigest()[:16]


def run(rows, cols, W, H, density, seed, encoding, h2d_mode, setup_order, setup_strips, checksum, use_cache):
    KX, KY = math.ceil(cols / W), math.ceil(rows / H)
    t = time.perf_counter()
    r, c, d = synthetic_coo(rows, cols, density, seed)
    gen = time.perf_counter() - t
    x = np.random.default_rng(seed).integers(0, 100, size=cols).astype(np.float32)
    print(f"{rows}x{cols} d={density} NNZ={len(d):,}  grid {W}x{H}={W*H:,} PEs  KX={KX} KY={KY}  "
          f"cpus={os.cpu_count()}  (generate {gen:.1f}s, not counted)", flush=True)

    t0 = time.perf_counter()
    fp = tile_cache.random_fingerprint(rows, cols, density, seed) if use_cache else None
    hb = host_prep.prepare_host_buffers(r, c, d, x, cols, W, H, KX, KY, encoding, h2d_mode,
                                        cache_fingerprint=fp)
    total = time.perf_counter() - t0
    times = dict(hb.times)
    del r, c

    val_only = hb.val_only
    max_nnz = max(hb.nnz_counts)
    arrays = [hb.x_grid, hb.ci]
    if val_only:
        arrays.append(hb.val_plan.pe_words)
    arrays.append(hb.rp)
    k = len(setup_strip_bounds(W, setup_strips))
    t = time.perf_counter()
    calls = [setup_strip_calls(a, W, H, setup_order, k, last_array=(i == len(arrays) - 1))
             for i, a in enumerate(arrays)]
    times["setup_prep"] = time.perf_counter() - t
    del calls

    print(f"\nbackend={hb.backend}  cache_hit={hb.cache_hit}")
    print(f"{'phase':<14}{'seconds':>10}")
    for name, v in times.items():
        print(f"{name:<14}{v:>10.3f}")
    print(f"{'host_prep':<14}{total:>10.3f}   (tile+x_grid+stream_build)")
    # ru_maxrss is bytes on macOS, KiB on Linux
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**30 if sys.platform == "darwin" else 2**20)
    print(f"max_nnz/PE={max_nnz}  peak RSS={rss:.2f} GB")
    if hb.save_thread is not None:                       # the worker overlaps this with program load
        t = time.perf_counter()
        hb.save_thread.join()
        print(f"cache write (background) finished {time.perf_counter() - t:.1f}s after the pipeline")
    if checksum:
        parts = [hb.ci, hb.rp, hb.x_grid, np.asarray(hb.nnz_counts, dtype=np.int64)]
        parts.append(hb.val_buf if val_only else hb.stream_buf)
        print(f"checksum {digest(*parts)}")
    return times


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("preset", nargs="?", choices=sorted(PRESETS), default="65k")
    ap.add_argument("--rows", type=int)
    ap.add_argument("--cols", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--density", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoding", choices=["header", "valtask"], default="valtask")
    ap.add_argument("--h2d-mode", choices=["column", "full"], default="column")
    ap.add_argument("--setup-order", choices=["row", "col"], default="col")
    ap.add_argument("--setup-strips", type=int, default=32)
    ap.add_argument("--cache", action="store_true",
                    help="use the on-disk tile cache (SPMV_TILE_CACHE_DIR); run twice for miss, hit")
    ap.add_argument("--checksum", action="store_true", help="print a hash of every output buffer")
    a = ap.parse_args()
    if a.cache and not os.environ.get("SPMV_TILE_CACHE_DIR"):
        ap.error("--cache needs SPMV_TILE_CACHE_DIR (the cache is off unless it is set)")
    p = dict(PRESETS[a.preset])
    p.update({k: v for k, v in dict(rows=a.rows, cols=a.cols, W=a.width, H=a.height).items() if v})
    run(p["rows"], p["cols"], p["W"], p["H"], a.density, a.seed, a.encoding, a.h2d_mode,
        a.setup_order, a.setup_strips, a.checksum, a.cache)


if __name__ == "__main__":
    main()
