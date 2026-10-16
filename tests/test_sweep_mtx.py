"""Sweep [[run]] entries that name a MatrixMarket file: sized from it, refused before any run when it cannot fit."""

import re
import shlex
import sys
import textwrap
from pathlib import Path

import pytest

import run
from cluster import mm_io, mtx_sizing
from cluster.sweep_config import SweepConfig
from tests.test_skewed_matrices import banded, power_law, write_mtx


def _toml(tmp_path, *runs, extra=""):
    body = ['[meta]\nname = "m"\n[output]\nwork_dir = "w"\n']
    for r in runs:
        body.append("[[run]]\n" + "\n".join(f"{k} = {v}" for k, v in r.items()) + "\n")
    path = tmp_path / "m.toml"
    path.write_text("".join(body) + extra)
    return path


def _expand(path):
    return list(SweepConfig.from_toml(path).expand())


def test_a_run_with_a_file_is_sized_from_it(tmp_path):
    rows, cols, r, c, v = power_law(240, 3000)
    mtx = write_mtx(tmp_path / "pl.mtx", rows, cols, r, c, v)
    (cfg,) = _expand(_toml(tmp_path, dict(mtx_file=f'"{mtx}"', num_pes_target=16, simulator="true",
                                          encoding='"resident"')))
    mx = int(mm_io.tile_nnz_counts(r, c, rows, cols, cfg.width, cfg.height).max())
    assert (cfg.rows, cfg.cols, cfg.total_nnz_per_pe) == (rows, cols, mx)         # exact, not the uniform estimate
    assert cfg.mtx_file == mtx and cfg.density == pytest.approx(len(v) / (rows * cols))


def test_the_sweep_sizes_exactly_as_run_py_run_does(tmp_path, monkeypatch):
    rows, cols, r, c, v = banded(240, 2)
    mtx = write_mtx(tmp_path / "b.mtx", rows, cols, r, c, v)
    (cfg,) = _expand(_toml(tmp_path, dict(mtx_file=f'"{mtx}"', num_pes_target=16, simulator="true",
                                          encoding='"resident"', sync_timing="true")))
    seen = {}
    monkeypatch.setattr(run, "run_one", lambda c, results: seen.update(cfg=c) or True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run.py", *shlex.split(
        f"run --mtx-file {mtx} --num-pes 16 --encoding resident --sync-timing --simulator")])
    assert run.main() == 0
    assert (cfg.rows, cfg.cols, cfg.total_nnz_per_pe) == (seen["cfg"].rows, seen["cfg"].cols, seen["cfg"].total_nnz_per_pe)
    assert cfg.density == pytest.approx(seen["cfg"].density)


def test_a_run_that_does_not_fit_is_refused_with_the_fit_and_a_grid_that_does(tmp_path):
    rows, cols, r, c, v = banded(20_000, 1)
    mtx = write_mtx(tmp_path / "big.mtx", rows, cols, r, c, v)
    with pytest.raises(ValueError) as e:
        _expand(_toml(tmp_path, dict(mtx_file=f'"{mtx}"', num_pes_target=16, simulator="true", encoding='"resident"',
                                     comment='"too big"')))
    msg = str(e.value)
    assert "[[run]] #1 (too big)" in msg and "does not fit on 4x4 = 16 PEs" in msg and "KB per PE > 38 KB" in msg
    assert "smallest grid that fits" in msg and "Nothing was compiled" in msg
    pes = int(re.search(r"num_pes_target = (\d+)\)", msg).group(1))
    (cfg,) = _expand(_toml(tmp_path, dict(mtx_file=f'"{mtx}"', num_pes_target=pes, simulator="true",
                                          encoding='"resident"')))          # the suggested grid works
    assert cfg.num_pes == pes


def test_cmd_sweep_refuses_before_any_run_when_a_later_run_does_not_fit(tmp_path, monkeypatch, caplog):
    rows, cols, r, c, v = banded(240, 1)
    ok = write_mtx(tmp_path / "ok.mtx", rows, cols, r, c, v)
    rb, cb, rr, cc, vv = banded(20_000, 1)
    big = write_mtx(tmp_path / "big.mtx", rb, cb, rr, cc, vv)
    path = _toml(tmp_path,
                 dict(mtx_file=f'"{ok}"', num_pes_target=16, simulator="true", encoding='"resident"'),
                 dict(mtx_file=f'"{big}"', num_pes_target=16, simulator="true", encoding='"resident"'))
    calls = []
    monkeypatch.setattr(run, "run_one", lambda cfg, results: calls.append(cfg) or True)
    monkeypatch.setattr(run, "compile_csl", lambda cfg: calls.append("compile"))
    assert run.cmd_sweep([path], simulator=True) == 1
    assert calls == []                                              # the first run never started either
    assert "[[run]] #2" in caplog.text and "does not fit" in caplog.text


def test_a_sweep_with_fitting_files_runs_them_all(tmp_path, monkeypatch):
    rows, cols, r, c, v = banded(240, 1)
    mtx = write_mtx(tmp_path / "ok.mtx", rows, cols, r, c, v)
    path = _toml(tmp_path,
                 dict(mtx_file=f'"{mtx}"', num_pes_target=16, simulator="true", encoding='"resident"'),
                 dict(mtx_file=f'"{mtx}"', num_pes_target=64, simulator="true", encoding='"resident"'))
    calls = []
    monkeypatch.setattr(run, "run_one", lambda cfg, results: calls.append(cfg.num_pes) or True)
    assert run.cmd_sweep([path], simulator=True) == 0
    assert calls == [16, 64]


def test_the_budget_follows_the_encoding_in_a_sweep_too(tmp_path):
    from tests.test_skewed_matrices import block_diagonal
    rows, cols, r, c, v = block_diagonal(284, 4)                  # 5,041 NNZ on each diagonal PE of a 4x4 grid
    mtx = write_mtx(tmp_path / "bd.mtx", rows, cols, r, c, v)
    (hdr,) = _expand(_toml(tmp_path, dict(mtx_file=f'"{mtx}"', num_pes_target=16, simulator="true", encoding='"header"')))
    assert hdr.total_nnz_per_pe == 5041
    with pytest.raises(ValueError, match="does not fit"):
        _expand(_toml(tmp_path, dict(mtx_file=f'"{mtx}"', num_pes_target=16, simulator="true", encoding='"resident"')))


def test_given_rows_must_match_and_a_given_total_must_be_enough(tmp_path):
    rows, cols, r, c, v = banded(240, 1)
    mtx = write_mtx(tmp_path / "b.mtx", rows, cols, r, c, v)
    base = dict(mtx_file=f'"{mtx}"', num_pes_target=16, simulator="true", encoding='"resident"')
    with pytest.raises(ValueError, match="contradicts"):
        _expand(_toml(tmp_path, {**base, "rows": 241}))
    mx = mtx_sizing.size_run({"mtx_file": mtx, "num_pes_target": 16, "simulator": True,
                              "encoding": "resident"})["total_nnz_per_pe"]
    with pytest.raises(ValueError, match="below the busiest PE"):
        _expand(_toml(tmp_path, {**base, "total_nnz_per_pe": mx - 1}))
    (cfg,) = _expand(_toml(tmp_path, {**base, "rows": rows, "cols": cols, "total_nnz_per_pe": mx + 5}))
    assert cfg.total_nnz_per_pe == mx + 5                           # a larger one is kept (SpMVConfig's guard still checks)


def test_a_dense_array_file_or_a_missing_file_is_refused_with_a_message(tmp_path):
    dense = tmp_path / "d.mtx"
    dense.write_text("%%MatrixMarket matrix array real general\n2 2\n1\n2\n3\n4\n")
    with pytest.raises(ValueError, match="dense 'array'"):
        _expand(_toml(tmp_path, dict(mtx_file=f'"{dense}"', num_pes_target=16, simulator="true")))
    with pytest.raises(ValueError, match="\\[\\[run\\]\\] #1"):
        _expand(_toml(tmp_path, dict(mtx_file=f'"{tmp_path / "nope.mtx"}"', num_pes_target=16, simulator="true")))


def test_runs_without_a_file_are_untouched(tmp_path):
    (cfg,) = _expand(_toml(tmp_path, dict(rows=1000, cols=1000, density=0.01, num_pes_target=16, simulator="true")))
    assert (cfg.rows, cfg.mtx_file) == (1000, None)
