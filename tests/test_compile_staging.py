"""Compiles run on a staged copy of the CSL sources, never on the live repo tree.

SdkCompiler packages the directory it is given; handed the repo root it read .git, and a merge
on the synced tree during a compile failed it with FileNotFoundError on .git/objects/...
"""

import re
import sys
import types
from pathlib import Path

import pytest

from config import SpMVConfig
import sdk.compiler as sc

ROOT = Path(__file__).resolve().parent.parent


def _fake_repo(tmp_path):
    root = tmp_path / "spmv-rewrite"
    for rel, text in {
        "src/csl/layout.csl": "L", "src/csl/pe.csl": "P", "src/csl/sync/pe.csl": "S",
        "src/csl/NOTES.md": "n", "src/csl/gen.py": "x", "src/csl/sync/.hidden": "h",
        "bench/other/stray.csl": "no", "run.py": "r", "data/big.bin": "b",
        ".git/objects/a7/c0834af": "obj", ".git/HEAD": "ref", ".git/hooks/sample.csl": "g",
    }.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    return root


def _files(app):
    return sorted(str(p.relative_to(app)) for p in app.rglob("*") if p.is_file())


def test_staged_set_is_exactly_the_csl_sources(tmp_path):
    root = _fake_repo(tmp_path)
    app = sc.stage_csl(("src/csl",), tmp_path / "stage", root)
    assert app == tmp_path / "stage" / "spmv-rewrite", "same directory name as the repo"
    assert _files(app) == ["src/csl/layout.csl", "src/csl/pe.csl", "src/csl/sync/pe.csl"]
    assert (app / "src/csl/sync/pe.csl").read_text() == "S"
    assert not any(".git" in p.parts for p in app.rglob("*"))


def test_git_is_never_staged_even_when_asked_for_the_whole_tree(tmp_path):
    root = _fake_repo(tmp_path)
    app = sc.stage_csl((".",), tmp_path / "stage", root)       # a careless csl_dirs
    assert not any(".git" in p.parts for p in app.rglob("*"))
    assert all(f.endswith(".csl") for f in _files(app))


def test_real_repo_stage_has_no_git_and_only_csl(tmp_path):
    dirs = sc.CSL_DIRS
    app = sc.stage_csl(dirs, tmp_path, ROOT)
    files = _files(app)
    assert files and all(f.endswith(".csl") for f in files)
    assert not any(".git" in Path(f).parts for f in files)
    assert files == sorted(str(p.relative_to(ROOT)) for d in dirs for p in (ROOT / d).rglob("*.csl"))


def test_staged_copy_is_independent_of_the_live_tree(tmp_path):
    root = _fake_repo(tmp_path)
    app = sc.stage_csl(("src/csl",), tmp_path / "stage", root)
    (root / "src/csl/layout.csl").write_text("changed")
    (root / "src/csl/pe.csl").unlink()
    assert (app / "src/csl/layout.csl").read_text() == "L" and (app / "src/csl/pe.csl").exists()


def _imports_resolve(files_root, csl_dirs, mains):
    """Every literal @import_module("x.csl") path resolves relative to the importing file."""
    for d in csl_dirs:
        for f in (files_root / d).rglob("*.csl"):
            code = re.sub(r"//.*", "", f.read_text())
            for rel in re.findall(r'@import_module\(\s*"([^"<>]+\.csl)"', code):
                assert (f.parent / rel).resolve().is_file(), (str(f.relative_to(files_root)), rel)
    for m in mains:
        assert (files_root / m).is_file(), m


def test_every_csl_import_is_inside_the_staged_set(tmp_path):
    spmv = sc.stage_csl(sc.CSL_DIRS, tmp_path / "a", ROOT)
    _imports_resolve(spmv, sc.CSL_DIRS, [sc.CSL_MAIN])


class _FakeSdk:
    """cerebras.sdk.client.SdkCompiler stand-in that records what it was given."""
    calls = []

    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def compile(self, app_path, csl_main, options, out_path):
        app = Path(app_path)
        # while "compiling", the live tree changes underneath (a merge): the staged copy must not care
        live = ROOT / "src/csl/layout.csl"
        _FakeSdk.calls.append(dict(
            app_path=app, csl_main=csl_main, options=options, out_path=out_path, kw=self.kw,
            files=_files(app), main_exists=(app / csl_main).is_file(), live_exists=live.exists()))
        return "artifact-123"


@pytest.fixture
def fake_sdk(monkeypatch):
    _FakeSdk.calls = []
    client = types.ModuleType("cerebras.sdk.client")
    client.SdkCompiler = _FakeSdk
    for name in ("cerebras", "cerebras.sdk"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "cerebras.sdk.client", client)
    return _FakeSdk


def test_sdk_compile_runs_on_a_temporary_staged_dir_and_cleans_up(fake_sdk, tmp_path):
    out = sc.sdk_compile(sc.CSL_DIRS, sc.CSL_MAIN, "--opts", str(tmp_path), True)
    (call,) = fake_sdk.calls
    assert out == "artifact-123"
    assert call["app_path"] != ROOT and call["app_path"].name == ROOT.name
    assert call["csl_main"] == sc.CSL_MAIN and call["options"] == "--opts"
    assert call["main_exists"] and not any(".git" in Path(f).parts for f in call["files"])
    assert all(f.endswith(".csl") for f in call["files"])
    assert call["kw"] == {"resource_cpu": 24000, "resource_mem": 64 << 30,
                          "disable_version_check": True}
    assert not call["app_path"].exists(), "the temporary directory is removed afterwards"


def test_spmv_compile_passes_the_unchanged_options_and_main(fake_sdk, tmp_path):
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.01, num_pes_target=16,
                     out_path=tmp_path / "out")
    sc.compile(cfg)
    (call,) = fake_sdk.calls
    assert call["options"] == sc.compile_options(cfg)
    assert call["csl_main"] == "src/csl/layout.csl"
    assert call["files"] == sorted(str(p.relative_to(ROOT)) for p in (ROOT / "src/csl").rglob("*.csl"))
    assert cfg.artifact_id == "artifact-123"
    assert (tmp_path / "out" / "artifact.json").is_file()
