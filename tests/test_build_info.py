"""Commit provenance for benchmark rows (sdk/build_info.py): the cluster copy has no .git."""

import json
import subprocess
from pathlib import Path

import pytest

from bench.benchmark import BenchmarkResult, load_benchmarks, save_benchmark
from sdk.build_info import BuildInfo, STAMP_NAME, resolve, sweep_refusal, write_stamp

FULL = "0123456789abcdef0123456789abcdef01234567"


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "sdk").mkdir()
    (tmp_path / "run.py").write_text("print(1)\n")
    (tmp_path / "sdk" / "x.py").write_text("X = 1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def _copy_without_git(repo, dest):
    dest.mkdir()
    for rel in ("run.py", "sdk/x.py"):
        (dest / rel).parent.mkdir(exist_ok=True)
        (dest / rel).write_text((repo / rel).read_text())
    (dest / STAMP_NAME).write_text((repo / STAMP_NAME).read_text())
    return dest


def test_git_checkout_is_resolved_clean_and_dirty(repo):
    info = resolve(repo, env={})
    assert (info.source, info.dirty, info.known) == ("git", False, True)
    assert len(info.commit_full) == 40 and info.commit == info.commit_full[:7]
    (repo / "run.py").write_text("print(2)\n")
    assert resolve(repo, env={}).dirty


def test_stamp_survives_the_copy_that_has_no_git(repo, tmp_path):
    write_stamp(repo)
    head = resolve(repo, env={}).commit_full
    remote = _copy_without_git(repo, tmp_path / "cluster")
    info = resolve(remote, env={})
    assert (info.source, info.dirty, info.commit_full) == ("stamp", False, head)


def test_edit_after_the_stamp_is_reported_dirty(repo, tmp_path):
    write_stamp(repo)
    remote = _copy_without_git(repo, tmp_path / "cluster")
    (remote / "sdk" / "x.py").write_text("X = 2\n")
    info = resolve(remote, env={})
    assert (info.source, info.dirty) == ("stamp-stale", True)
    assert info.commit_full == resolve(repo, env={}).commit_full


def test_a_file_missing_from_the_sync_is_stale_too(repo, tmp_path):
    write_stamp(repo)
    remote = _copy_without_git(repo, tmp_path / "cluster")
    (remote / "sdk" / "x.py").unlink()
    assert resolve(remote, env={}).dirty


def test_stamp_of_a_dirty_tree_records_the_diff(repo):
    (repo / "run.py").write_text("print(3)\n")
    data = json.loads(write_stamp(repo).read_text())
    assert data["dirty"] is True and len(data["diff_sha256"]) == 64
    assert resolve(repo, env={}).dirty


def test_env_overrides_everything(repo):
    write_stamp(repo)
    info = resolve(repo, env={"SPMV_GIT_COMMIT": FULL, "SPMV_GIT_DIRTY": "1"})
    assert (info.source, info.commit, info.commit_full, info.dirty) == ("env", FULL[:7], FULL, True)


def test_no_git_no_stamp_is_unknown(tmp_path):
    info = resolve(tmp_path, env={})
    assert not info.known and info.commit == "unknown"


def test_garbage_stamp_is_ignored(tmp_path):
    (tmp_path / STAMP_NAME).write_text("{not json")
    assert resolve(tmp_path, env={}).source == "unknown"
    (tmp_path / STAMP_NAME).write_text(json.dumps({"version": 1, "commit_full": "abc",
                                                   "dirty": False, "code_sha256": "x"}))
    assert resolve(tmp_path, env={}).source == "unknown"


def test_write_stamp_needs_a_checkout(tmp_path):
    with pytest.raises(RuntimeError):
        write_stamp(tmp_path)


def test_sweep_refusal_rules():
    assert sweep_refusal(BuildInfo()) is not None
    assert sweep_refusal(BuildInfo(), allow_dirty=True) is not None      # unknown is never allowed
    dirty = BuildInfo(FULL[:7], FULL, True, "x-dirty", "git")
    assert sweep_refusal(dirty) is not None
    assert sweep_refusal(dirty, allow_dirty=True) is None
    assert sweep_refusal(BuildInfo(FULL[:7], FULL, False, "x", "stamp")) is None


def test_hardware_sweep_refuses_before_loading_configs(monkeypatch, tmp_path):
    import run

    monkeypatch.setattr(run, "resolve_build", lambda: BuildInfo())
    missing = tmp_path / "does-not-exist.toml"
    called = []

    def _from_toml(cls, *a, **k):
        called.append(1)
        raise ValueError("stop")

    monkeypatch.setattr(run.SweepConfig, "from_toml", classmethod(_from_toml))
    assert run.cmd_sweep([missing]) == 1
    assert not called                       # refused before any config was read
    run.cmd_sweep([missing], simulator=True)
    assert called                           # the simulator is exempt


def test_csv_rows_carry_the_provenance_columns(tmp_path):
    from sdk.launcher import _download_benchmark

    worker_csv = tmp_path / "worker.csv"
    save_benchmark(BenchmarkResult(num_pes=64), csv_path=str(worker_csv))

    class _Launcher:
        def download_artifact(self, _name, dest):
            Path(dest).write_text(worker_csv.read_text())

    build = BuildInfo(FULL[:7], FULL, True, "v2.0-3-gabcdef0-dirty", "stamp-stale")
    out = tmp_path / "results.csv"
    _download_benchmark(_Launcher(), out, git_commit=build.commit, build=build)
    row = load_benchmarks(str(out))[0]
    assert (row.git_commit, row.git_commit_full, row.git_dirty, row.repo_describe) == (
        FULL[:7], FULL, True, "v2.0-3-gabcdef0-dirty")


def test_old_csvs_without_the_columns_still_load(tmp_path):
    p = tmp_path / "old.csv"
    p.write_text("timestamp,git_commit,num_pes\n2026-01-01T00:00:00,abc1234,4\n")
    row = load_benchmarks(str(p))[0]
    assert (row.git_commit, row.git_commit_full, row.git_dirty) == ("abc1234", "", False)


# --- git archive extracts (bench runs releases from `git archive`, no .git, no stamp) ---

def _archive_extract(repo, ref, dest):
    import tarfile
    import io
    tar = subprocess.run(["git", "-C", str(repo), "archive", ref], check=True,
                         capture_output=True).stdout
    dest.mkdir()
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(dest)
    return dest


TEMPLATE = """{
  "version": 1,
  "commit_full": "$Format:%H$",
  "describe": "$Format:%(describe:tags)$",
  "dirty": false
}
"""


def _repo_with_template(repo):
    """The committed template and its export-subst attribute (written here: in an archive extract the
    repo's own copy is already filled in)."""
    (repo / ".gitattributes").write_text("BUILD_INFO.archive.json export-subst\n")
    (repo / "BUILD_INFO.archive.json").write_text(TEMPLATE)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "template")


def test_archive_of_a_tag_reads_clean_and_describes_as_the_tag(repo, tmp_path):
    _repo_with_template(repo)
    _git(repo, "tag", "v2.0")
    head = resolve(repo, env={}).commit_full
    extract = _archive_extract(repo, "v2.0", tmp_path / "x")
    assert not (extract / ".git").exists()
    info = resolve(extract, env={})
    assert (info.source, info.dirty, info.commit_full, info.describe) == ("archive", False, head, "v2.0")
    assert info.known and sweep_refusal(info) is None


def test_archive_of_an_untagged_commit_still_names_the_commit(repo, tmp_path):
    _repo_with_template(repo)
    extract = _archive_extract(repo, "HEAD", tmp_path / "x")
    info = resolve(extract, env={})
    assert (info.source, info.dirty) == ("archive", False)
    assert info.commit_full == resolve(repo, env={}).commit_full and info.describe


def test_unexpanded_template_in_a_checkout_is_ignored(repo):
    _repo_with_template(repo)
    assert resolve(repo, env={}).source == "git"          # the placeholders are not a stamp
    assert (repo / "BUILD_INFO.archive.json").read_text().count("$Format") == 2


def test_the_local_stamp_wins_over_the_archive_template(repo, tmp_path):
    _repo_with_template(repo)
    write_stamp(repo)
    extract = _archive_extract(repo, "HEAD", tmp_path / "x")
    (extract / STAMP_NAME).write_text((repo / STAMP_NAME).read_text())
    assert resolve(extract, env={}).source in ("stamp", "stamp-stale")
