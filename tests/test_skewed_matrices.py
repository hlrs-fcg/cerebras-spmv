"""The resident path on matrices that are not uniform random: what is checked, and where it stops.

Every resident number so far came from uniform-random matrices, whose NNZ per PE is balanced. Real
matrices are not: a banded matrix puts three NNZ per row on the diagonal-block PEs and nothing elsewhere,
a power-law matrix has hubs, a block-diagonal matrix leaves most PEs empty. The resident model depends on
NNZ_max (SRAM: col_indices and vals are [TOTAL_NNZ] on every PE; time: the slowest PE sets the window) and on
KX/KY. These tests pin what the code does with such input, step by step:

1. the MatrixMarket reader (cluster/mm_io.py): symmetric lower triangle, pattern, skew, rectangular,
   duplicates, explicit zeros, and what it refuses;
2. sizing without tiling (mm_io.tile_nnz_counts) equals the real tiler's counts;
3. the uniform estimate of TOTAL_NNZ undersizes skewed matrices, and the host refuses with a clear error
   (after the compile, before the device); an oversized TOTAL_NNZ is refused by SpMVConfig before the compile;
4. mm_io.fit_report agrees with SpMVConfig's SRAM guard;
5. the whole worker on a skewed .mtx against the fake device: validated results, and the refusals.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PROJECT_ROOT / "cluster"), str(PROJECT_ROOT / "bench")]

import mm_io  # noqa: E402
from config import SpMVConfig, total_nnz_estimate  # noqa: E402
from cluster.coo_tiling import spmv_reference  # noqa: E402
from cluster.host_prep import prepare_host_buffers  # noqa: E402

# the fake device and the worker harness of the resident tests
from tests.test_resident_encoding import _run_worker, fake_sdk  # noqa: E402,F401


# --------------------------------------------------------------------------
# synthetic matrices
# --------------------------------------------------------------------------

def banded(n, bw=1, seed=0):
    rng = np.random.default_rng(seed)
    r, c = zip(*[(i, j) for i in range(n) for j in range(max(0, i - bw), min(n, i + bw + 1))])
    return n, n, np.array(r), np.array(c), rng.integers(1, 9, len(r)).astype(np.float32)


def power_law(n, nnz, a=1.2, seed=0):
    """Rows and columns drawn with probability ~ 1/(i+1)^a: hubs at low indices (rows AND columns)."""
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, n + 1) ** a
    p /= p.sum()
    r = rng.choice(n, nnz, p=p)
    c = rng.choice(n, nnz, p=p)
    return n, n, r, c, rng.integers(1, 9, nnz).astype(np.float32)


def block_diagonal(n, blocks, seed=0):
    rng = np.random.default_rng(seed)
    size = n // blocks
    rr, cc = [], []
    for b in range(blocks):
        lo = b * size
        for i in range(size):
            for j in range(size):
                rr.append(lo + i)
                cc.append(lo + j)
    return n, n, np.array(rr), np.array(cc), rng.integers(1, 9, len(rr)).astype(np.float32)


def dense_row_and_column(n, seed=0):
    """A sparse diagonal plus one completely dense row and one completely dense column."""
    rng = np.random.default_rng(seed)
    r = np.concatenate([np.arange(n), np.full(n, n // 2), np.arange(n)])
    c = np.concatenate([np.arange(n), np.arange(n), np.full(n, n // 3)])
    return n, n, r, c, rng.integers(1, 9, len(r)).astype(np.float32)


MATRICES = {
    "banded": lambda: banded(240, 2),
    "power_law": lambda: power_law(240, 3000),
    "block_diagonal": lambda: block_diagonal(200, 4),
    "dense_row_and_column": lambda: dense_row_and_column(240),
}


def write_mtx(path, rows, cols, r, c, v, symmetry="general", field="real", comment=""):
    lines = [f"%%MatrixMarket matrix coordinate {field} {symmetry}", comment or "% test",
             f"{rows} {cols} {len(r)}"]
    for i, j, x in zip(r, c, v):
        lines.append(f"{i + 1} {j + 1}" if field == "pattern" else f"{i + 1} {j + 1} {x:.9g}")
    path.write_text("\n".join(lines) + "\n")
    return path


def dense_of(rows, cols, r, c, v):
    a = np.zeros((rows, cols))
    np.add.at(a, (r, c), v)
    return a


# --------------------------------------------------------------------------
# 1. the reader
# --------------------------------------------------------------------------

def test_symmetric_lower_triangle_is_expanded_with_the_diagonal_once(tmp_path):
    n, _, r, c, v = banded(8, 1)
    keep = r >= c                                         # stored as a lower triangle only
    path = write_mtx(tmp_path / "s.mtx", n, n, r[keep], c[keep], v[keep], "symmetric")
    mm = mm_io.read_matrix_market(path)
    assert mm.symmetry == "symmetric" and mm.stored_entries == int(keep.sum())
    full = dense_of(n, n, mm.row, mm.col, mm.data)
    ref = dense_of(n, n, r[keep], c[keep], v[keep])
    np.testing.assert_array_equal(full, ref + ref.T - np.diag(np.diag(ref)))
    assert mm.nnz == 2 * int((r[keep] != c[keep]).sum()) + int((r[keep] == c[keep]).sum())


def test_skew_symmetric_negates_the_mirror(tmp_path):
    path = write_mtx(tmp_path / "k.mtx", 3, 3, [1, 2], [0, 1], [1.5, -2.0], "skew-symmetric")
    a = dense_of(3, 3, *(lambda m: (m.row, m.col, m.data))(mm_io.read_matrix_market(path)))
    np.testing.assert_array_equal(a, -a.T)
    assert a[1, 0] == 1.5 and a[0, 1] == -1.5


def test_pattern_entries_are_ones_and_rectangular_shapes_are_kept(tmp_path):
    path = write_mtx(tmp_path / "p.mtx", 3, 5, [0, 1, 2], [0, 2, 4], [0, 0, 0], field="pattern")
    mm = mm_io.read_matrix_market(path)
    assert (mm.rows, mm.cols) == (3, 5) and mm.data.tolist() == [1.0, 1.0, 1.0]


def test_duplicates_and_explicit_zeros_are_kept_as_separate_entries(tmp_path):
    path = write_mtx(tmp_path / "d.mtx", 3, 3, [0, 0, 1, 2], [0, 0, 1, 2], [1.0, 2.0, 0.0, 7.0])
    mm = mm_io.read_matrix_market(path)
    assert mm.nnz == 4                                    # not merged, not dropped: they occupy PE slots
    assert dense_of(3, 3, mm.row, mm.col, mm.data)[0, 0] == 3.0   # SpMV sums them


@pytest.mark.parametrize("name", ["banded", "power_law"])
def test_the_reader_agrees_with_scipy(tmp_path, name):
    scipy_io = pytest.importorskip("scipy.io")
    rows, cols, r, c, v = MATRICES[name]()
    path = write_mtx(tmp_path / "m.mtx", rows, cols, r, c, v)
    mm = mm_io.read_matrix_market(path)
    ref = scipy_io.mmread(path).toarray()
    np.testing.assert_allclose(dense_of(rows, cols, mm.row, mm.col, mm.data), ref, rtol=1e-6)


@pytest.mark.parametrize("text, match", [
    ("%%MatrixMarket matrix array real general\n2 2\n1\n2\n3\n4\n", "dense 'array'"),
    ("%%MatrixMarket matrix coordinate complex general\n1 1 1\n1 1 1.0 2.0\n", "complex"),
    ("%%MatrixMarket matrix coordinate real general\n2 2 1\n3 1 1.0\n", "outside"),
    ("%%MatrixMarket matrix coordinate real general\n2 2 3\n1 1 1.0\n", "header says 3 entries"),
    ("%%MatrixMarket matrix coordinate real symmetric\n2 3 1\n1 1 1.0\n", "not square"),
    ("not a matrix\n", "not a MatrixMarket"),
])
def test_what_the_reader_refuses_and_says_why(tmp_path, text, match):
    path = tmp_path / "bad.mtx"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        mm_io.read_matrix_market(path)


# --------------------------------------------------------------------------
# 2. sizing without tiling == the tiler
# --------------------------------------------------------------------------

def _tile(rows, cols, r, c, v, W, H, encoding="resident", total=None):
    KX, KY = math.ceil(cols / W), math.ceil(rows / H)
    x = np.arange(cols, dtype=np.float32)
    return prepare_host_buffers(r.astype(np.int32), c.astype(np.int32), v, x, cols, W, H, KX, KY,
                                encoding, "column", total_nnz_per_pe=total, num_threads=1), KX, KY


@pytest.mark.parametrize("name", sorted(MATRICES))
@pytest.mark.parametrize("W, H", [(4, 4), (5, 3), (1, 6)])
def test_tile_nnz_counts_equal_the_tilers(name, W, H):
    rows, cols, r, c, v = MATRICES[name]()
    hb, _, _ = _tile(rows, cols, r, c, v, W, H)
    counts = mm_io.tile_nnz_counts(r, c, rows, cols, W, H)
    assert counts.reshape(-1).tolist() == list(hb.nnz_counts)
    assert counts.sum() == len(v)


def test_a_cyclic_mapping_would_balance_a_banded_matrix_that_the_block_tiling_piles_up():
    """Sizing only (the tiler is block): row i on PE row i % H, column j on PE column j % W. The kernels
    would not change; the host tiling, x_grid and gather_y would."""
    rows, cols, r, c, v = banded(960, 2)
    for W, H in [(5, 3), (7, 4)]:                      # coprime sides: the diagonal is spread over all PEs
        block = mm_io.tile_nnz_counts(r, c, rows, cols, W, H, "block")
        cyclic = mm_io.tile_nnz_counts(r, c, rows, cols, W, H, "cyclic")
        assert block.sum() == cyclic.sum() == len(v)
        assert block.max() / block.mean() > 2.0 and cyclic.max() / cyclic.mean() < 1.3
    # a pitfall of the plain cyclic mapping: on a SQUARE grid (i, i) always lands on PE (i % n, i % n), so the
    # main diagonal's NNZ stay on four PEs; the 742 x 1155 wafer has gcd 7, which only mostly avoids it
    sq = mm_io.tile_nnz_counts(r, c, rows, cols, 4, 4, "cyclic")
    assert sq.max() / sq.mean() > 1.4                   # vs ~1.0 on the coprime grids above
    with pytest.raises(ValueError, match="mapping"):
        mm_io.tile_nnz_counts(r, c, rows, cols, 4, 4, "diagonal")


def test_the_skew_is_real_not_a_property_of_the_generators():
    """max / mean on a 4x4 grid: uniform random is ~1.3, these are 2.5 to 4."""
    ratios = {}
    for name, make in MATRICES.items():
        rows, cols, r, c, v = make()
        counts = mm_io.tile_nnz_counts(r, c, rows, cols, 4, 4)
        ratios[name] = counts.max() / counts.mean()
    assert all(x > 2.0 for x in ratios.values()), ratios
    assert block_diagonal_empty_pes() >= 12


def block_diagonal_empty_pes():
    rows, cols, r, c, v = block_diagonal(240, 4)
    return int((mm_io.tile_nnz_counts(r, c, rows, cols, 4, 4) == 0).sum())


# --------------------------------------------------------------------------
# 3. TOTAL_NNZ: the estimate undersizes, and who says so
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(MATRICES))
def test_the_uniform_estimate_undersizes_skewed_matrices(name):
    rows, cols, r, c, v = MATRICES[name]()
    cfg = SpMVConfig(rows=rows, cols=cols, density=len(v) / (rows * cols), num_pes_target=16,
                     encoding="resident", simulator=True)
    counts = mm_io.tile_nnz_counts(r, c, rows, cols, cfg.width, cfg.height)
    assert cfg.total_nnz_per_pe == total_nnz_estimate(int(len(v) / cfg.num_pes), cfg.num_pes)
    assert counts.max() > cfg.total_nnz_per_pe, (counts.max(), cfg.total_nnz_per_pe)


def test_the_host_refuses_a_skewed_matrix_before_the_device_with_the_needed_size():
    rows, cols, r, c, v = banded(240, 2)
    W = H = 4
    mx = int(mm_io.tile_nnz_counts(r, c, rows, cols, W, H).max())
    with pytest.raises(ValueError, match=rf"busiest PE holds {mx} NNZ .* --total-nnz-per-pe >= {mx}"):
        _tile(rows, cols, r, c, v, W, H, total=mx - 1)
    hb, _, _ = _tile(rows, cols, r, c, v, W, H, total=mx)          # exactly enough
    assert max(hb.nnz_counts) == mx


def test_an_oversized_total_nnz_is_refused_by_the_config_before_any_compile():
    with pytest.raises(ValueError, match="usable SRAM per PE"):
        SpMVConfig(rows=240, cols=240, density=0.05, num_pes_target=16, encoding="resident",
                   simulator=True, total_nnz_per_pe=3500)
    SpMVConfig(rows=240, cols=240, density=0.05, num_pes_target=16, encoding="resident",
               simulator=True, total_nnz_per_pe=3000)             # 3 x 3000 words + the rest still fits


# --------------------------------------------------------------------------
# 4. fit_report == SpMVConfig's guard
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sync", [True, False])
@pytest.mark.parametrize("n, pes, total", [
    (240, 16, 2000), (240, 16, 3000), (240, 16, 3200), (20_000, 16, 1), (20_000, 16, 2_000),
    (6_000, 64, 3_000), (6_000, 64, 4_300), (50_000, 256, 100),
])
def test_fit_report_agrees_with_the_config_guard(n, pes, total, sync):
    kw = dict(rows=n, cols=n, density=1e-4, num_pes_target=pes, encoding="resident", simulator=True,
              sync_timing=sync, total_nnz_per_pe=total)
    try:
        SpMVConfig(**kw)
        guard_fits = True
    except ValueError as e:
        assert "usable SRAM" in str(e), e
        guard_fits = False
    grid = SpMVConfig(rows=64, cols=64, density=0.1, num_pes_target=pes, simulator=True)   # W and H only

    class cfg:                                                        # noqa: N801
        width, height = grid.width, grid.height
        kx = ky = -(-n // grid.width)
        ky = -(-n // grid.height)
    counts = np.zeros((cfg.height, cfg.width), dtype=int)
    counts[0, 0] = total
    rep = mm_io.fit_report(counts, n, n, sync_timing=sync)
    assert rep.fits == guard_fits, (rep.verdict(), guard_fits)
    assert (rep.kx, rep.ky) == (cfg.kx, cfg.ky)
    if guard_fits:
        assert mm_io.resident_words(rep.kx, rep.ky, total, sync) <= mm_io.PE_SRAM_WORDS
    # max_fit_nnz is the exact boundary
    assert mm_io.resident_words(rep.kx, rep.ky, max(rep.max_fit_nnz, 0), sync) <= mm_io.PE_SRAM_WORDS \
        or rep.max_fit_nnz < 0
    assert mm_io.resident_words(rep.kx, rep.ky, rep.max_fit_nnz + 1, sync) > mm_io.PE_SRAM_WORDS


def test_resident_holds_row_ids_where_header_holds_nothing_extra():
    counts = np.full((4, 4), 1000)
    res = mm_io.fit_report(counts, 1000, 1000, encoding="resident")
    hdr = mm_io.fit_report(counts, 1000, 1000, encoding="header")
    val = mm_io.fit_report(counts, 1000, 1000, encoding="valtask")
    assert res.words - hdr.words == 2 * 1000                  # vals[] and row_ids[]
    assert val.words - hdr.words == 1000 + 1                  # vals[] and val_meta[1]


def test_a_tall_tile_is_refused_even_when_it_would_fit_in_sram():
    """KY > 65535 is a 16-bit local-row limit of the stream headers. worker_host applies it to every
    encoding, resident included (the resident CSR loop is i32, local kernel 1 is the one that needs u16)."""
    counts = np.zeros((2, 1), dtype=int)
    rep = mm_io.fit_report(counts, 140_000, 10, sync_timing=False)
    assert not rep.fits and "16-bit" in rep.reason


def test_the_worker_rejects_ky_over_65535_for_resident_too():
    from types import SimpleNamespace
    import worker_host
    args = SimpleNamespace(rows=140_000, cols=10, grid_width=1, grid_height=2, total_nnz_per_pe=10,
                           num_stripes=1, num_pes=2)
    run = SimpleNamespace(args=args)
    with pytest.raises(ValueError, match="exceeds 65535"):
        worker_host.set_geometry(run)


# --------------------------------------------------------------------------
# 5. the worker on a skewed .mtx, resident, against the fake device
# --------------------------------------------------------------------------

def _worker_total(rows, cols, r, c, W=4, H=4):
    return int(mm_io.tile_nnz_counts(r, c, rows, cols, W, H).max())


@pytest.mark.parametrize("name", sorted(MATRICES))
def test_resident_worker_validates_skewed_matrices(fake_sdk, tmp_path, monkeypatch, name):
    rows, cols, r, c, v = MATRICES[name]()
    path = write_mtx(tmp_path / f"{name}.mtx", rows, cols, r, c, v)
    code, row, cfg = _run_worker(tmp_path, monkeypatch, "--loop-count", "3", rows=rows, cols=cols,
                                 mtx=path, total_nnz=_worker_total(rows, cols, r, c))
    assert code == 0 and row["validation_passed"] == "True", row
    assert float(row["relative_error"]) < 1e-6
    assert int(row["total_nnz"]) == len(v) and int(row["max_nnz_per_pe"]) == _worker_total(rows, cols, r, c)
    assert float(row["load_imbalance_pct"]) > 100          # the skew is visible in the row


def test_resident_worker_on_a_symmetric_lower_triangle_equals_the_full_matrix(fake_sdk, tmp_path, monkeypatch):
    n, _, r, c, v = banded(120, 2)
    keep = r >= c
    lower = write_mtx(tmp_path / "lower.mtx", n, n, r[keep], c[keep], v[keep], "symmetric")
    mm = mm_io.read_matrix_market(lower)
    # symmetrise so the file really is symmetric (banded() values are not)
    sym = {}
    for i, j, x in zip(r[keep], c[keep], v[keep]):
        sym[(i, j)] = x
    rr, cc = zip(*sym)
    lower = write_mtx(tmp_path / "lower.mtx", n, n, np.array(rr), np.array(cc), np.array(list(sym.values())),
                      "symmetric")
    mm = mm_io.read_matrix_market(lower)
    total = _worker_total(n, n, mm.row, mm.col)
    code, row, _ = _run_worker(tmp_path, monkeypatch, rows=n, cols=n, mtx=lower, total_nnz=total)
    assert code == 0 and row["validation_passed"] == "True"
    assert int(row["total_nnz"]) == mm.nnz > len(sym)       # the worker ran the EXPANDED matrix


def test_resident_worker_on_a_rectangular_matrix(fake_sdk, tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    rows, cols, nnz = 150, 410, 900
    r, c = rng.integers(0, rows, nnz), rng.integers(0, cols, nnz)
    v = rng.integers(1, 9, nnz).astype(np.float32)
    path = write_mtx(tmp_path / "rect.mtx", rows, cols, r, c, v)
    code, row, _ = _run_worker(tmp_path, monkeypatch, rows=rows, cols=cols, mtx=path,
                               total_nnz=_worker_total(rows, cols, r, c))
    assert code == 0 and row["validation_passed"] == "True"


def test_resident_worker_with_duplicates_and_explicit_zeros(fake_sdk, tmp_path, monkeypatch):
    r = np.array([0, 0, 0, 5, 5, 9, 39])
    c = np.array([1, 1, 2, 5, 5, 9, 39])
    v = np.array([1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 6.0], dtype=np.float32)
    path = write_mtx(tmp_path / "dz.mtx", 40, 40, r, c, v)
    code, row, _ = _run_worker(tmp_path, monkeypatch, rows=40, cols=40, mtx=path,
                               total_nnz=_worker_total(40, 40, r, c))
    assert code == 0 and row["validation_passed"] == "True" and int(row["total_nnz"]) == 7


def test_worker_refuses_a_matrix_whose_shape_differs_from_the_run(fake_sdk, tmp_path, monkeypatch):
    rows, cols, r, c, v = banded(60, 1)
    path = write_mtx(tmp_path / "b.mtx", rows, cols, r, c, v)
    from tests.test_resident_encoding import FakeSdkRuntime
    FakeSdkRuntime.last = None
    code, row, _ = _run_worker(tmp_path, monkeypatch, rows=200, cols=190, mtx=path)
    assert code == 1 and row is None
    assert FakeSdkRuntime.last is None, "the device was created for a matrix of the wrong shape"


def test_worker_refuses_a_dense_array_matrix_with_a_message_not_a_traceback(fake_sdk, tmp_path, monkeypatch):
    path = tmp_path / "dense.mtx"
    path.write_text("%%MatrixMarket matrix array real general\n2 2\n1\n2\n3\n4\n")
    from tests.test_resident_encoding import FakeSdkRuntime
    FakeSdkRuntime.last = None
    code, row, _ = _run_worker(tmp_path, monkeypatch, rows=2, cols=2, mtx=path)
    assert code == 1 and row is None and FakeSdkRuntime.last is None


def test_worker_aborts_on_an_undersized_total_nnz_before_touching_the_device(fake_sdk, tmp_path, monkeypatch):
    """The only protection once the artifact is compiled with too small a TOTAL_NNZ is this host check. It is
    raised as a ValueError out of main() (a traceback with the message, exit via exception), after the compile
    and before SdkRuntime is created."""
    from tests.test_resident_encoding import FakeSdkRuntime
    rows, cols, r, c, v = power_law(240, 3000)
    path = write_mtx(tmp_path / "p.mtx", rows, cols, r, c, v)
    FakeSdkRuntime.last = None
    with pytest.raises(ValueError, match="busiest PE holds"):
        _run_worker(tmp_path, monkeypatch, rows=rows, cols=cols, mtx=path)       # the uniform estimate
    assert FakeSdkRuntime.last is None, "the device was created before the size check"
