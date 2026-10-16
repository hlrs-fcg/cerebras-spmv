"""The fast host paths must reproduce the 34d1fdd implementation bit for bit.

tests/legacy_tiling.py is a verbatim copy of the old numpy code (3-key lexsort, Python
row-bounds loop, per-PE stream encoder); every case below runs both and compares bytes.
Cases cover W != H, tiles that do not divide the matrix, many empty PEs, and duplicate
coordinates (whose values must keep their input order).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# the worker imports these as top-level modules (SdkLauncher stages them flat); so do we
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cluster"))

import coo_tiling as new  # noqa: E402
import legacy_tiling as old  # noqa: E402
from host_prep import build_x_grid  # noqa: E402

# (rows, cols, nnz, W, H)
SHAPES = [
    (40, 40, 200, 4, 4),
    (37, 53, 150, 5, 3),       # ragged tiles, W != H
    (53, 37, 150, 3, 7),
    (64, 64, 40, 8, 8),        # mostly empty PEs
    (30, 30, 600, 3, 2),       # heavy duplicates (600 NNZ on 900 cells)
    (9, 100, 80, 7, 2),        # KY=5, KX=15
    (100, 9, 80, 2, 9),
    (16, 16, 1, 4, 4),         # a single NNZ
    (12, 12, 0, 3, 3),         # no NNZ at all
]


def _coo(rows, cols, nnz, seed):
    rng = np.random.default_rng(seed)
    r = rng.integers(0, rows, size=nnz, dtype=np.int32)
    c = rng.integers(0, cols, size=nnz, dtype=np.int32)
    return r, c, rng.random(size=nnz).astype(np.float32)


def _legacy_lists(rb):
    return [[(int(a), int(b), int(c)) for a, b, c in pe] for pe in rb]


def _as_lists(rb: "new.RowBounds"):
    out = []
    for pe in range(len(rb.ptr) - 1):
        lo, hi = rb.ptr[pe], rb.ptr[pe + 1]
        out.append([(int(rb.row[j]), int(rb.start[j]), int(rb.count[j])) for j in range(lo, hi)])
    return out


@pytest.mark.parametrize("rows,cols,nnz,W,H", SHAPES)
@pytest.mark.parametrize("seed", [0, 1])
def test_extract_matches_legacy(rows, cols, nnz, W, H, seed):
    r, c, _ = _coo(rows, cols, nnz, seed)
    KX, KY = -(-cols // W), -(-rows // H)
    o = old.extract_tile_structure(r, c, W, H, KX, KY)
    n = new.extract_tile_structure(r, c, W, H, KX, KY)
    np.testing.assert_array_equal(n[0], o[0])
    np.testing.assert_array_equal(n[1], o[1])
    assert n[2] == o[2]
    np.testing.assert_array_equal(n[3], o[3])
    assert _as_lists(n[4]) == _legacy_lists(o[4])


@pytest.mark.parametrize("rows,cols,nnz,W,H", SHAPES)
def test_header_stream_matches_legacy(rows, cols, nnz, W, H):
    r, c, d = _coo(rows, cols, nnz, 3)
    KX, KY = -(-cols // W), -(-rows // H)
    _, _, cnt_o, order_o, rb_o = old.extract_tile_structure(r, c, W, H, KX, KY)
    _, _, cnt_n, order_n, rb_n = new.extract_tile_structure(r, c, W, H, KX, KY)
    buf_o, cw_o = old.build_stream_buffer(d, order_o, cnt_o, rb_o, W, H, num_threads=3)
    buf_n, cw_n = new.build_stream_buffer(d, order_n, cnt_n, rb_n, W, H)
    assert cw_n == cw_o
    np.testing.assert_array_equal(buf_n, buf_o)


@pytest.mark.parametrize("rows,cols,nnz,W,H", SHAPES)
def test_val_buffer_without_row_bounds(rows, cols, nnz, W, H):
    r, c, d = _coo(rows, cols, nnz, 4)
    KX, KY = -(-cols // W), -(-rows // H)
    o = old.extract_tile_structure(r, c, W, H, KX, KY)
    n = new.extract_tile_structure(r, c, W, H, KX, KY, want_row_bounds=False)
    assert n[4] is None
    np.testing.assert_array_equal(new.build_val_stream_buffer(d, n[3], n[2], W, H),
                                  old.build_val_stream_buffer(d, o[3], o[2], W, H))
    # the log-only header word count, now read off row_ptrs instead of the bounds lists
    assert new.count_nonempty_rows(n[1]) == sum(len(b) for b in o[4])


@pytest.mark.parametrize("cols,W,H", [(100, 4, 3), (97, 7, 2), (10, 16, 2), (558, 5, 4)])
def test_x_grid_matches_the_loop(cols, W, H):
    KX = -(-cols // W)
    x = np.random.default_rng(0).integers(0, 100, size=cols).astype(np.float32)
    want = np.zeros((W * H, KX), dtype=np.float32)
    for px in range(W):
        cs, ce = px * KX, min((px + 1) * KX, cols)
        if cs < cols:
            for py in range(H):
                want[py * W + px, :ce - cs] = x[cs:ce]
    np.testing.assert_array_equal(build_x_grid(x, cols, W, H, KX), want)


def test_oversized_row_still_raises():
    # one row, 40000 NNZ in a single tile (> MAX_ROW_NNZ)
    n = 40000
    r = np.zeros(n, dtype=np.int32)
    c = np.zeros(n, dtype=np.int32)
    d = np.ones(n, dtype=np.float32)
    _, _, cnt, order, rb = new.extract_tile_structure(r, c, 1, 1, 4, 4)
    with pytest.raises(ValueError, match="at most 32767"):
        new.build_stream_buffer(d, order, cnt, rb, 1, 1)


# --------------------------------------------------------------------------
# fused tiler backends: numba (nogil kernels on a thread pool) and AOT (serial extension module)
# --------------------------------------------------------------------------

import host_prep  # noqa: E402
import tile_kernels  # noqa: E402

numba_installed = tile_kernels._numba_installed()
needs_numba = pytest.mark.skipif(not numba_installed, reason="numba not installed")


@pytest.fixture(scope="session")
def aot_module(tmp_path_factory):
    """Build spmv_tile_aot once with numba.pycc and make tile_kernels load it."""
    if not numba_installed:
        pytest.skip("numba not installed")
    pytest.importorskip("setuptools")
    from sdk import build_aot
    out = tmp_path_factory.mktemp("aot")
    try:
        build_aot.build(out)
    except Exception as e:                      # no C compiler / platform without pycc
        pytest.skip(f"cannot build the AOT module here: {type(e).__name__}: {e}")
    sys.path.insert(0, str(out))
    tile_kernels._aot, tile_kernels._aot_tried = None, False
    yield out
    sys.path.remove(str(out))
    sys.modules.pop(tile_kernels.AOT_MODULE, None)
    tile_kernels._aot, tile_kernels._aot_tried = None, False


@pytest.fixture(params=["numba", "aot"])
def backend(request):
    if request.param == "numba":
        if not numba_installed:
            pytest.skip("numba not installed")
    else:
        request.getfixturevalue("aot_module")
    return request.param


BIG_SHAPES = [s for s in SHAPES if s[2] > 0] + [
    (500, 400, 20000, 6, 5),       # ~670 NNZ/PE: long segments, long rows, many duplicates
    (50, 4000, 20000, 20, 2),      # wide tiles
    (4000, 50, 20000, 2, 20),      # tall tiles
]


@pytest.mark.parametrize("rows,cols,nnz,W,H", BIG_SHAPES)
@pytest.mark.parametrize("seed", [0, 1])
def test_fused_tiler_matches_legacy(backend, rows, cols, nnz, W, H, seed):
    r, c, d = _coo(rows, cols, nnz, seed)
    KX, KY = -(-cols // W), -(-rows // H)
    o = old.extract_tile_structure(r, c, W, H, KX, KY)
    ci, rp, counts, vb, nonempty = tile_kernels.tile_fused(r, c, d, W, H, KX, KY, backend=backend)
    np.testing.assert_array_equal(ci, o[0])
    np.testing.assert_array_equal(rp, o[1])
    assert counts.tolist() == o[2]
    np.testing.assert_array_equal(vb, old.build_val_stream_buffer(d, o[3], o[2], W, H))
    assert nonempty == sum(len(b) for b in o[4])
    assert ci.dtype == np.int32 and rp.dtype == np.int32 and vb.dtype == np.uint32


@needs_numba
def test_numba_backend_is_independent_of_thread_count():
    r, c, d = _coo(300, 300, 30000, 5)
    outs = []
    for n in (1, 3, 8):
        tile_kernels.set_threads(n)
        outs.append(tile_kernels.tile_fused(r, c, d, 7, 6, 43, 50, backend="numba"))
    tile_kernels.set_threads()
    for other in outs[1:]:
        for a, b in zip(outs[0][:4], other[:4]):
            np.testing.assert_array_equal(a, b)
        assert other[4] == outs[0][4]


def test_duplicates_keep_input_order(backend):
    # three NNZ on one cell: the values must come out in input order, like the stable lexsort
    r = np.array([2, 2, 2, 0], dtype=np.int32)
    c = np.array([1, 1, 1, 3], dtype=np.int32)
    d = np.array([7.0, 5.0, 6.0, 1.0], dtype=np.float32)
    _, _, _, vb, _ = tile_kernels.tile_fused(r, c, d, 1, 1, 4, 4, backend=backend)
    assert vb[0, 0, :4].view(np.float32).tolist() == [1.0, 7.0, 5.0, 6.0]


def test_out_of_range_index_is_rejected_not_read(backend):
    r = np.array([0, 9], dtype=np.int32)
    c = np.array([0, 0], dtype=np.int32)
    with pytest.raises(ValueError, match="outside"):
        tile_kernels.tile_fused(r, c, np.ones(2, np.float32), 2, 2, 2, 2, backend=backend)  # 4 x 4
    with pytest.raises(ValueError, match="outside"):
        tile_kernels.tile_fused(r - 1, c, np.ones(2, np.float32), 2, 2, 2, 2, backend=backend)


def test_aot_module_does_not_pull_in_numba_at_import(aot_module):
    """The point of AOT: a worker with only numpy can run it."""
    import subprocess
    code = (f"import sys; sys.path.insert(0, {str(aot_module)!r}); import {tile_kernels.AOT_MODULE}; "
            f"assert 'numba' not in sys.modules, 'numba was imported'; print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


@pytest.mark.parametrize("encoding", ["valtask"])
@pytest.mark.parametrize("rows,cols,nnz,W,H", [(37, 53, 150, 5, 3), (64, 64, 40, 8, 8), (500, 400, 20000, 6, 5)])
def test_prepare_host_buffers_backends_agree(monkeypatch, backend, encoding, rows, cols, nnz, W, H):
    """Whatever backend runs, the worker gets the same bytes."""
    r, c, d = _coo(rows, cols, nnz, 9)
    x = np.arange(cols, dtype=np.float32)
    KX, KY = -(-cols // W), -(-rows // H)
    monkeypatch.setenv("SPMV_TILE_BACKEND", backend)
    fast = host_prep.prepare_host_buffers(r, c, d, x, cols, W, H, KX, KY, encoding, "column")
    assert fast.backend == backend
    monkeypatch.setenv("SPMV_TILE_BACKEND", "numpy")
    slow = host_prep.prepare_host_buffers(r, c, d, x, cols, W, H, KX, KY, encoding, "column")
    assert slow.backend == "numpy"
    for name in ("ci", "rp", "x_grid", "val_buf"):
        np.testing.assert_array_equal(getattr(fast, name), getattr(slow, name), err_msg=name)
    assert fast.nnz_counts == slow.nnz_counts
    assert fast.header_words == slow.header_words
    assert (fast.h2d_words_col, fast.h2d_words_full, fast.common_words) == \
           (slow.h2d_words_col, slow.h2d_words_full, slow.common_words)
    np.testing.assert_array_equal(fast.val_plan.pe_words, slow.val_plan.pe_words)


def test_tiler_that_cannot_pack_its_key_is_not_used():
    # (loc_row, loc_col) is packed in a u32; a tile of 2^16 x 2^16 or more must fall back
    assert not tile_kernels.supports(65536, 65536, 10)
    assert not tile_kernels.supports(4, 4, 0)


# --------------------------------------------------------------------------
# backend selection by size
# --------------------------------------------------------------------------

@pytest.fixture
def no_aot(monkeypatch):
    monkeypatch.setattr(tile_kernels, "_aot", None)
    monkeypatch.setattr(tile_kernels, "_aot_tried", True)
    monkeypatch.delenv("SPMV_TILE_BACKEND", raising=False)
    monkeypatch.delenv("SPMV_TILE_NUMBA_MIN_NNZ", raising=False)


@needs_numba
def test_auto_uses_numpy_below_the_threshold_and_numba_above(no_aot):
    t = tile_kernels.NUMBA_MIN_NNZ_DEFAULT
    assert tile_kernels.select_backend(t - 1, 400, 400)[0] == "numpy"
    assert tile_kernels.select_backend(t, 400, 400)[0] == "numba"
    assert tile_kernels.select_backend(0, 400, 400)[0] == "numpy"          # nothing to tile


@needs_numba
def test_threshold_and_backend_can_be_overridden(no_aot, monkeypatch):
    monkeypatch.setenv("SPMV_TILE_NUMBA_MIN_NNZ", "1000")
    assert tile_kernels.select_backend(1000, 40, 40)[0] == "numba"
    assert tile_kernels.select_backend(999, 40, 40)[0] == "numpy"
    monkeypatch.setenv("SPMV_TILE_BACKEND", "numba")                       # forced: no threshold
    assert tile_kernels.select_backend(5, 40, 40)[0] == "numba"
    monkeypatch.setenv("SPMV_TILE_BACKEND", "numpy")
    assert tile_kernels.select_backend(10**9, 40, 40) == ("numpy", "SPMV_TILE_BACKEND=numpy")


def test_unknown_backend_name_is_an_error(no_aot, monkeypatch):
    monkeypatch.setenv("SPMV_TILE_BACKEND", "cuda")
    with pytest.raises(ValueError, match="auto|aot|numba|numpy"):
        tile_kernels.select_backend(10, 4, 4)


def test_forced_aot_without_the_module_falls_back_to_numpy_and_says_why(no_aot, monkeypatch):
    monkeypatch.setenv("SPMV_TILE_BACKEND", "aot")
    name, why = tile_kernels.select_backend(10**9, 400, 400)
    assert name == "numpy" and "AOT requested but unavailable" in why


def test_auto_without_any_compiled_backend_is_numpy(no_aot, monkeypatch):
    monkeypatch.setattr(tile_kernels, "_numba_installed", lambda: False)
    assert tile_kernels.select_backend(10**9, 400, 400)[0] == "numpy"


def test_aot_is_preferred_at_any_size(aot_module):
    assert tile_kernels.select_backend(10, 4, 4)[0] == "aot"
    assert tile_kernels.select_backend(10**9, 400, 400)[0] == "aot"


def test_a_backend_that_fails_its_warmup_degrades_to_numpy(monkeypatch):
    def boom(backend):
        raise OSError("libc.so.6: version `GLIBC_2.38' not found")
    monkeypatch.setenv("SPMV_TILE_BACKEND", "auto")
    monkeypatch.setattr(tile_kernels, "select_backend", lambda *a: ("aot", "test"))
    monkeypatch.setattr(tile_kernels, "warmup", boom)
    monkeypatch.setattr(tile_kernels, "IMPORT_ERROR", "")
    r, c, d = _coo(40, 40, 200, 1)
    hb = host_prep.prepare_host_buffers(r, c, d, np.zeros(40, np.float32), 40, 4, 4, 10, 10,
                                        "valtask", "column")
    assert hb.backend == "numpy" and "GLIBC_2.38" in tile_kernels.IMPORT_ERROR


# --------------------------------------------------------------------------
# on-disk tile cache
# --------------------------------------------------------------------------

import tile_cache  # noqa: E402


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SPMV_TILE_CACHE_DIR", str(tmp_path / "tiles"))
    monkeypatch.delenv("SPMV_TILE_CACHE", raising=False)
    return tmp_path / "tiles"


def _prep(fp, rows=60, cols=70, nnz=900, W=5, H=4, seed=2, encoding="valtask"):
    r, c, d = _coo(rows, cols, nnz, seed)
    x = np.arange(cols, dtype=np.float32)
    KX, KY = -(-cols // W), -(-rows // H)
    return host_prep.prepare_host_buffers(r, c, d, x, cols, W, H, KX, KY, encoding, "column",
                                          cache_fingerprint=fp)


def test_cache_miss_then_hit_returns_identical_buffers(cache_root):
    fp = tile_cache.random_fingerprint(60, 70, 0.2, 2)
    cold = _prep(fp)
    assert not cold.cache_hit and cold.save_thread is not None
    cold.save_thread.join()
    warm = _prep(fp)
    assert warm.cache_hit and warm.backend == "cache" and warm.save_thread is None
    for name in ("ci", "rp", "x_grid", "val_buf"):
        np.testing.assert_array_equal(getattr(warm, name), getattr(cold, name), err_msg=name)
    assert warm.nnz_counts == cold.nnz_counts and warm.header_words == cold.header_words
    assert isinstance(warm.ci, np.memmap) and warm.ci.flags.writeable   # mapped copy-on-write
    np.testing.assert_array_equal(warm.val_plan.pe_words, cold.val_plan.pe_words)


def test_cache_hits_for_the_same_grid_but_not_a_different_one(cache_root):
    fp = tile_cache.random_fingerprint(60, 70, 0.2, 2)
    _prep(fp, encoding="valtask").save_thread.join()
    assert _prep(fp, encoding="valtask").cache_hit
    other = _prep(fp, W=7, H=3)
    assert not other.cache_hit
    other.save_thread.join()
    assert _prep(fp).cache_hit and _prep(fp, W=7, H=3).cache_hit


def test_cache_key_depends_on_every_input():
    base = dict(fingerprint="f", W=4, H=4, KX=8, KY=8)
    keys = {tile_cache.make_key(**base)}
    for k, v in (("fingerprint", "g"), ("W", 5), ("H", 5), ("KX", 9), ("KY", 9)):
        keys.add(tile_cache.make_key(**{**base, k: v}))
    assert len(keys) == 6
    assert tile_cache.random_fingerprint(10, 10, 0.1, 1) != tile_cache.random_fingerprint(10, 10, 0.1, 2)
    assert tile_cache.random_fingerprint(10, 10, 0.1, 1) != tile_cache.random_fingerprint(10, 10, 0.01, 1)


def test_header_runs_and_disabled_cache_never_touch_disk(cache_root, monkeypatch):
    fp = tile_cache.random_fingerprint(60, 70, 0.2, 2)
    assert _prep(fp, encoding="header").save_thread is None
    monkeypatch.setenv("SPMV_TILE_CACHE", "0")
    assert _prep(fp).save_thread is None
    assert not cache_root.exists()


def test_incomplete_or_stale_entries_are_misses(cache_root):
    fp = tile_cache.random_fingerprint(60, 70, 0.2, 2)
    _prep(fp).save_thread.join()
    entry = next(p for p in cache_root.iterdir() if p.is_dir())
    (entry / "meta.json").unlink()                      # a write that never finished
    assert not _prep(fp).cache_hit
    _prep(fp).save_thread.join()
    (entry / "rp.npy").write_bytes(b"garbage")          # corrupt array
    assert not _prep(fp).cache_hit


def test_matrix_with_a_different_nnz_is_not_served_from_a_stale_entry(cache_root):
    fp = tile_cache.random_fingerprint(60, 70, 0.2, 2)
    _prep(fp, nnz=900).save_thread.join()
    again = _prep(fp, nnz=901)                          # same fingerprint, different content
    assert not again.cache_hit


def test_eviction_keeps_the_newest_and_the_entry_just_written(cache_root):
    for i in range(3):
        _prep(tile_cache.random_fingerprint(60, 70, 0.2, i), seed=i).save_thread.join()
        entry_dirs = sorted(p for p in cache_root.iterdir() if p.is_dir())
    one = sum(f.stat().st_size for f in entry_dirs[0].iterdir())
    newest = max(entry_dirs, key=lambda p: (p / "meta.json").stat().st_mtime)
    tile_cache.evict(cache_root, keep=newest.name, max_bytes=int(one * 1.5))
    left = [p for p in cache_root.iterdir() if p.is_dir()]
    assert [p.name for p in left] == [newest.name]


def test_file_fingerprint_follows_content(tmp_path):
    a, b = tmp_path / "a.mtx", tmp_path / "b.mtx"
    a.write_bytes(b"x" * 100)
    b.write_bytes(b"x" * 100)
    assert tile_cache.file_fingerprint(str(a)) == tile_cache.file_fingerprint(str(b))
    b.write_bytes(b"x" * 99 + b"y")
    assert tile_cache.file_fingerprint(str(a)) != tile_cache.file_fingerprint(str(b))


# --------------------------------------------------------------------------
# radix replacement for the global lexsort (numpy fallback)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n,pe_max,row_max,col_max", [
    (5000, 40, 30, 30),            # all keys fit one 16-bit digit
    (5000, 70000, 100, 100),       # pe_id needs two digits
    (5000, 10, 100, 200000),       # loc_col needs two digits
    (5000, 300000, 70000, 100000), # everything does
    (1, 3, 3, 3),
    (0, 3, 3, 3),
])
def test_stable_tile_order_equals_lexsort(n, pe_max, row_max, col_max):
    rng = np.random.default_rng(n + pe_max)
    pe = rng.integers(0, pe_max, size=n).astype(np.int32)
    lr = rng.integers(0, row_max, size=n).astype(np.int32)
    lc = rng.integers(0, col_max, size=n).astype(np.int32)
    want = np.lexsort((lc, lr, pe))
    np.testing.assert_array_equal(new.stable_tile_order(pe, lr, lc), want)


def test_stable_tile_order_keeps_duplicates_in_input_order():
    z = np.zeros(6, dtype=np.int32)
    np.testing.assert_array_equal(new.stable_tile_order(z, z, z), np.arange(6))
