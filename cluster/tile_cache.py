"""On-disk cache of the tiled matrix (col_indices, row_ptrs, val buffer) for val/valtask runs.

Tiling is a pure function of the matrix and the grid, and a sweep repeats the same matrix
(n=3, one matrix over several channel counts), so the result is stored once as plain .npy files
and memory-mapped on the next run.  Header runs are not cached: their stream needs the sort
order and row bounds, which the fused tiler never builds.

Layout, one directory per entry::

    <cache dir>/<key>/{ci,rp,val_buf,counts}.npy  meta.json

The key hashes everything the arrays depend on: matrix fingerprint, W, H, KX, KY, the stream
layout and CODE_VERSION.  Entries are written to a temporary directory and renamed, so a killed
run never leaves a half-written entry that a later run would trust.

Environment:
    SPMV_TILE_CACHE_DIR     cache directory; the cache is OFF unless this is set, and should be a
                            path that outlives the job (shared filesystem), not the staged dir
    SPMV_TILE_CACHE=0       disable even when the directory is set
    SPMV_TILE_CACHE_MAX_GB  evict least-recently-used entries beyond this size (default 60)
"""

import hashlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CODE_VERSION = 1
"""Bump whenever the bytes produced by the tiler or the val buffer change."""

LAYOUT = "val-csr-v1"
"""ci (num_pes, max_nnz) i32, rp (num_pes, KY+1) i32, val_buf (W, H, max_nnz) u32; shared by
the val and valtask encodings, which stream the same buffer."""

_ARRAYS = ("ci", "rp", "val_buf", "counts")


def enabled() -> bool:
    """On only when SPMV_TILE_CACHE_DIR is set: the worker's working directory is a staged copy
    that is deleted after the job, so a default location would store ~GBs nobody ever reads."""
    return bool(os.environ.get("SPMV_TILE_CACHE_DIR")) and \
        os.environ.get("SPMV_TILE_CACHE", "1").lower() not in ("0", "false", "no", "off")


def disabled_reason() -> str:
    if not os.environ.get("SPMV_TILE_CACHE_DIR"):
        return "SPMV_TILE_CACHE_DIR is not set (the worker's staged directory is thrown away)"
    return "SPMV_TILE_CACHE=0"


def cache_dir() -> Path:
    return Path(os.environ["SPMV_TILE_CACHE_DIR"])


def random_fingerprint(rows: int, cols: int, density: float, seed: int) -> str:
    """The generator is deterministic in these four numbers (generate_random_coo)."""
    return f"random:{rows}x{cols}:d{density!r}:s{seed}"


def file_fingerprint(path: str) -> str:
    """Content hash of a matrix file (blake2b, streamed)."""
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return f"file:{h.hexdigest()}"


def make_key(fingerprint: str, W: int, H: int, KX: int, KY: int) -> str:
    spec = json.dumps({"matrix": fingerprint, "W": W, "H": H, "KX": KX, "KY": KY,
                       "layout": LAYOUT, "code": CODE_VERSION}, sort_keys=True)
    return hashlib.sha256(spec.encode()).hexdigest()[:24]


def load(key: str, root: Path | None = None):
    """``(ci, rp, val_buf, counts, nonempty)`` memory-mapped, or None on a miss.

    Mapped copy-on-write (``mmap_mode="c"``): reads page in from disk lazily, the arrays are
    writable like the ones tiling returns (the SDK copy path never meets a read-only buffer), and
    nothing is ever written back to the entry.
    """
    d = (root or cache_dir()) / key
    try:
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("key") != key or meta.get("code") != CODE_VERSION:
            return None
        arrs = [np.load(d / f"{n}.npy", mmap_mode="c") for n in _ARRAYS]
        for a, n in zip(arrs, _ARRAYS):
            if list(a.shape) != meta["shapes"][n]:
                logger.warning(f"tile cache {key}: {n} has shape {a.shape}, expected "
                               f"{meta['shapes'][n]}; ignoring the entry")
                return None
        os.utime(d / "meta.json")                 # recency for eviction
        return (*arrs, int(meta["nonempty"]))
    except (OSError, ValueError, KeyError):
        return None


def _write(key: str, root: Path, ci, rp, val_buf, counts, nonempty: int, info: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".tmp-{key}-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        arrs = dict(zip(_ARRAYS, (ci, rp, val_buf, np.asarray(counts, dtype=np.int64))))
        for n, a in arrs.items():
            np.save(tmp / f"{n}.npy", a, allow_pickle=False)
        meta = {"key": key, "code": CODE_VERSION, "layout": LAYOUT, "nonempty": int(nonempty),
                "shapes": {n: list(a.shape) for n, a in arrs.items()}, "info": info,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (tmp / "meta.json").write_text(json.dumps(meta, indent=1))      # last: marks it complete
        final = root / key
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _entry_size(d: Path) -> int:
    return sum(f.stat().st_size for f in d.iterdir() if f.is_file())


def evict(root: Path, keep: str, max_bytes: int) -> None:
    """Drop least-recently-used entries (never ``keep``) until the cache fits ``max_bytes``."""
    entries = []
    for d in root.iterdir():
        meta = d / "meta.json"
        if d.is_dir() and meta.exists():
            entries.append((meta.stat().st_mtime, d.name, _entry_size(d), d))
    total = sum(e[2] for e in entries)
    for _, name, size, d in sorted(entries):
        if total <= max_bytes:
            break
        if name != keep:
            shutil.rmtree(d, ignore_errors=True)
            total -= size
            logger.info(f"tile cache: evicted {name} ({size / 1e9:.2f} GB)")


def save_async(key: str, ci, rp, val_buf, counts, nonempty: int, info: dict,
               root: Path | None = None) -> threading.Thread:
    """Write the entry on a background thread (numpy releases the GIL while writing), so a cold
    run pays for it during program load, not in front of it.  ``join()`` the result before the
    process exits.  Never raises: a failed write only costs the next run its hit."""
    root = root or cache_dir()

    def work():
        t = time.perf_counter()
        try:
            _write(key, root, ci, rp, val_buf, counts, nonempty, info)
            logger.info(f"tile cache: stored {key} in {time.perf_counter() - t:.1f}s ({root})")
            evict(root, key, int(float(os.environ.get("SPMV_TILE_CACHE_MAX_GB", "60")) * 1e9))
        except Exception as e:
            logger.warning(f"tile cache: could not store {key} under {root}: {e}")

    th = threading.Thread(target=work, name="tile-cache-save")
    th.start()
    return th
