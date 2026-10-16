"""run.py run --mtx-file: sized from the file on the login side, refused before any compile when it cannot fit."""

import logging
import shlex
import sys
from pathlib import Path

import numpy as np
import pytest

import run
from cluster import mm_io
from tests.test_skewed_matrices import banded, power_law, write_mtx


@pytest.fixture
def spy(monkeypatch):
    """run_one and the compiler replaced: records what a run would have been given."""
    seen = {"cfg": None, "compiled": False}
    monkeypatch.setattr(run, "run_one", lambda cfg, results: seen.update(cfg=cfg, results=results) or True)
    monkeypatch.setattr(run, "compile_csl", lambda cfg: seen.update(compiled=True))
    return seen


def _main(monkeypatch, tmp_path, line):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run.py", *shlex.split(line)])
    return run.main()


def test_rows_cols_and_total_nnz_come_from_the_file(monkeypatch, tmp_path, spy):
    rows, cols, r, c, v = power_law(240, 3000)
    path = write_mtx(tmp_path / "pl.mtx", rows, cols, r, c, v)
    assert _main(monkeypatch, tmp_path, f"run --mtx-file {path} --num-pes 16 --encoding resident --simulator") == 0
    cfg = spy["cfg"]
    mx = int(mm_io.tile_nnz_counts(r, c, rows, cols, cfg.width, cfg.height).max())
    assert (cfg.rows, cfg.cols, cfg.total_nnz_per_pe) == (rows, cols, mx)     # exact, not the uniform estimate
    assert cfg.mtx_file == path and cfg.density == pytest.approx(len(v) / (rows * cols))
    assert spy["results"].name == "sim_single_pl.csv" and not spy["compiled"]


def test_a_symmetric_lower_triangle_is_sized_expanded(monkeypatch, tmp_path, spy):
    n, _, r, c, v = banded(240, 2)
    keep = r >= c
    path = write_mtx(tmp_path / "s.mtx", n, n, r[keep], c[keep], v[keep], "symmetric")
    assert _main(monkeypatch, tmp_path, f"run --mtx-file {path} --num-pes 16 --encoding resident --simulator") == 0
    cfg = spy["cfg"]
    mm = mm_io.read_matrix_market(path)
    assert cfg.total_nnz_per_pe == int(mm_io.tile_nnz_counts(mm.row, mm.col, n, n, cfg.width, cfg.height).max())
    assert cfg.density == pytest.approx(mm.nnz / n ** 2) and mm.nnz > keep.sum()


def test_a_matrix_that_does_not_fit_is_refused_before_the_compile(monkeypatch, tmp_path, spy, caplog):
    rows, cols, r, c, v = banded(20_000, 1)
    path = write_mtx(tmp_path / "big.mtx", rows, cols, r, c, v)
    with caplog.at_level(logging.INFO):
        code = _main(monkeypatch, tmp_path, f"run --mtx-file {path} --num-pes 16 --encoding resident --simulator")
    assert code == 1 and spy["cfg"] is None and not spy["compiled"]
    text = caplog.text
    assert "does not fit on 4x4 = 16 PEs" in text and "KB per PE > 38 KB" in text
    assert "smallest grid that fits is" in text and "Nothing was compiled" in text


def test_the_hint_names_a_grid_that_really_fits(monkeypatch, tmp_path, spy, caplog):
    rows, cols, r, c, v = banded(20_000, 1)
    path = write_mtx(tmp_path / "big.mtx", rows, cols, r, c, v)
    with caplog.at_level(logging.INFO):
        _main(monkeypatch, tmp_path, f"run --mtx-file {path} --num-pes 16 --encoding resident --simulator")
    import re
    pes = int(re.search(r"--num-pes (\d+)\)", caplog.text).group(1))
    assert _main(monkeypatch, tmp_path, f"run --mtx-file {path} --num-pes {pes} --encoding resident") == 0
    assert spy["cfg"].num_pes == pes


def test_the_budget_follows_the_encoding(monkeypatch, tmp_path, spy):
    """header holds no vals[], so a matrix too big for resident (and valtask) can fit header."""
    from tests.test_skewed_matrices import block_diagonal
    rows, cols, r, c, v = block_diagonal(284, 4)                # 5,041 NNZ on each diagonal PE of a 4x4 grid
    path = write_mtx(tmp_path / "bd.mtx", rows, cols, r, c, v)
    counts = mm_io.tile_nnz_counts(r, c, rows, cols, 4, 4)
    assert mm_io.fit_report(counts, rows, cols, False, "header").fits
    assert not mm_io.fit_report(counts, rows, cols, False, "resident").fits
    assert _main(monkeypatch, tmp_path, f"run --mtx-file {path} --num-pes 16 --encoding header --simulator") == 0
    assert spy["cfg"].total_nnz_per_pe == 5041
    spy["cfg"] = None
    assert _main(monkeypatch, tmp_path, f"run --mtx-file {path} --num-pes 16 --encoding resident --simulator") == 1
    assert spy["cfg"] is None


@pytest.mark.parametrize("encoding", ["header", "valtask", "resident"])
def test_fit_report_matches_the_config_guard_for_every_encoding(encoding):
    from config import SpMVConfig
    for total in (500, 2000, 3500, 4300):
        counts = np.zeros((4, 4), dtype=int)
        counts[0, 0] = total
        rep = mm_io.fit_report(counts, 8000, 8000, True, encoding)
        try:
            SpMVConfig(rows=8000, cols=8000, density=1e-4, num_pes_target=16, simulator=True, encoding=encoding,
                       sync_timing=True, total_nnz_per_pe=total)
            ok = True
        except ValueError as e:
            assert "usable SRAM" in str(e)
            ok = False
        assert rep.fits == ok, (encoding, total, rep.verdict())


def test_contradictions_and_missing_input_are_refused(monkeypatch, tmp_path, spy, caplog):
    rows, cols, r, c, v = banded(60, 1)
    path = write_mtx(tmp_path / "b.mtx", rows, cols, r, c, v)
    assert _main(monkeypatch, tmp_path, f"run --mtx-file {path} --rows 61 --simulator") == 1
    assert "contradicts" in caplog.text
    assert _main(monkeypatch, tmp_path, "run --simulator") == 1
    assert "--mtx-file" in caplog.text
    dense = tmp_path / "d.mtx"
    dense.write_text("%%MatrixMarket matrix array real general\n2 2\n1\n2\n3\n4\n")
    assert _main(monkeypatch, tmp_path, f"run --mtx-file {dense} --simulator") == 1
    assert "dense 'array'" in caplog.text
    assert spy["cfg"] is None and not spy["compiled"]


def test_the_plain_rows_cols_density_path_is_unchanged(monkeypatch, tmp_path, spy):
    assert _main(monkeypatch, tmp_path, "run --rows 1000 --cols 1000 --density 0.01 --num-pes 16 --simulator") == 0
    cfg = spy["cfg"]
    assert (cfg.rows, cfg.mtx_file) == (1000, None)
    assert spy["results"].name == "sim_single_1000x1000.csv"
