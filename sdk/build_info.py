"""Which commit a benchmark row came from.

The cluster copy of the repo has no ``.git`` (it arrives by a file sync or ``git archive``), so asking
git on the user node answers ``unknown``.  A run is only reproducible if its row names the commit
and says whether the tree differed from it, so the commit is resolved in this order:

1. ``SPMV_GIT_COMMIT`` (+ optional ``SPMV_GIT_DIRTY``, ``SPMV_GIT_DESCRIBE``): an explicit override.
2. ``BUILD_INFO.json`` at the project root, written on the machine that has the ``.git`` by
   ``python -m sdk.build_info write`` right before syncing.  It records a hash of the code files, and
   it is checked against the tree it was read in: a tree edited after the stamp is reported dirty.
3. ``BUILD_INFO.archive.json``: a tracked template that ``git archive`` fills in (``.gitattributes``
   export-subst), so an extract of a commit or tag names itself and cannot be forgotten.  An archive
   is a commit, so it reads as clean; in a checkout the placeholders stay unexpanded and it is ignored.
4. ``git`` in the project root.
5. ``unknown``.

``python -m sdk.build_info show`` prints what a launch from this tree would record.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAMP_NAME = "BUILD_INFO.json"
ARCHIVE_STAMP_NAME = "BUILD_INFO.archive.json"
STAMP_VERSION = 1
UNKNOWN = "unknown"

# What a run executes.  Tests, docs and results do not change a measurement, so they do not make a
# stamp stale (and a sync may exclude them).
CODE_GLOBS = (
    "run.py", "config.py", "pyproject.toml", "uv.lock",
    "sdk/**/*.py", "utils/**/*.py", "cluster/*.py", "bench/*.py", "src/**/*.csl",
)

_FULL = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class BuildInfo:
    commit: str = UNKNOWN          # short hash, the CSV's git_commit
    commit_full: str = ""
    dirty: bool = False
    describe: str = ""
    source: str = UNKNOWN          # env | stamp | stamp-stale | git | unknown

    @property
    def known(self) -> bool:
        return self.commit != UNKNOWN

    def csv_columns(self) -> dict:
        """Columns of BenchmarkResult this fills in on the login side."""
        return {
            "git_commit": self.commit,
            "git_commit_full": self.commit_full,
            "git_dirty": str(self.dirty),
            "repo_describe": self.describe,
        }

    def label(self) -> str:
        return f"{self.commit}{'+dirty' if self.dirty else ''} ({self.source})"


def code_files(root: Path = PROJECT_ROOT) -> list[Path]:
    seen: set[Path] = set()
    for pattern in CODE_GLOBS:
        for p in root.glob(pattern):
            if p.is_file() and "__pycache__" not in p.parts:
                seen.add(p)
    return sorted(seen)


def code_sha256(root: Path = PROJECT_ROOT) -> str:
    """One hash over (relative path, content hash) of every code file under ``root``."""
    h = hashlib.sha256()
    for p in code_files(root):
        h.update(p.relative_to(root).as_posix().encode() + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                         check=True, timeout=10)
    return out.stdout.strip()


def from_git(root: Path = PROJECT_ROOT) -> Optional[BuildInfo]:
    """The checkout's HEAD, dirty if anything but the stamp itself differs from it."""
    try:
        full = _git(root, "rev-parse", "HEAD")
        status = [line for line in _git(root, "status", "--porcelain").splitlines()
                  if not line.endswith(STAMP_NAME)]
        try:
            describe = _git(root, "describe", "--tags", "--always")
        except Exception:
            describe = full[:7]
    except Exception:
        return None
    return BuildInfo(full[:7], full, bool(status), describe + ("-dirty" if status else ""), "git")


def _from_env(env: Mapping[str, str]) -> Optional[BuildInfo]:
    value = env.get("SPMV_GIT_COMMIT", "").strip().lower()
    if not value:
        return None
    dirty = env.get("SPMV_GIT_DIRTY", "0").strip().lower() in ("1", "true", "yes")
    describe = env.get("SPMV_GIT_DESCRIBE", "").strip() or value[:7]
    return BuildInfo(value[:7], value if _FULL.match(value) else "", dirty, describe, "env")


def _from_stamp(root: Path) -> Optional[BuildInfo]:
    path = root / STAMP_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
        full = str(data["commit_full"])
        if data.get("version") != STAMP_VERSION or not _FULL.match(full):
            return None
        stamped_dirty = bool(data["dirty"])
        describe = str(data.get("describe", full[:7]))
        recorded = str(data["code_sha256"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if code_sha256(root) != recorded:
        # Edited (or half-synced) after the stamp: the commit no longer describes this tree.
        return BuildInfo(full[:7], full, True, describe + "-dirty", "stamp-stale")
    return BuildInfo(full[:7], full, stamped_dirty, describe, "stamp")


def _from_archive(root: Path) -> Optional[BuildInfo]:
    path = root / ARCHIVE_STAMP_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
        full = str(data["commit_full"]).strip().lower()
        describe = str(data.get("describe", "")).strip()
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not _FULL.match(full):
        return None                       # still "$Format:%H$": not made by git archive
    if not describe or describe.startswith("$"):
        describe = full[:7]
    return BuildInfo(full[:7], full, False, describe, "archive")


def resolve(root: Path = PROJECT_ROOT, env: Optional[Mapping[str, str]] = None) -> BuildInfo:
    env = os.environ if env is None else env
    return (_from_env(env) or _from_stamp(root) or _from_archive(root) or from_git(root)
            or BuildInfo())


def write_stamp(root: Path = PROJECT_ROOT) -> Path:
    """Write ``BUILD_INFO.json`` from the git checkout at ``root`` (needs ``.git``)."""
    info = from_git(root)
    if info is None:
        raise RuntimeError(f"{root} is not a git checkout; cannot stamp it")
    data = {
        "version": STAMP_VERSION,
        "commit_full": info.commit_full,
        "dirty": info.dirty,
        "describe": info.describe,
        "code_sha256": code_sha256(root),
    }
    if info.dirty:
        data["diff_sha256"] = hashlib.sha256(_git(root, "diff", "HEAD").encode()).hexdigest()
    path = root / STAMP_NAME
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def sweep_refusal(info: BuildInfo, allow_dirty: bool = False) -> Optional[str]:
    """Why a hardware sweep must not start from this build, or None.

    Sweep rows are the data of record, so every one has to name a commit and a clean tree."""
    if not info.known:
        return ("cannot tell which commit this tree is: no BUILD_INFO.json, no .git, no "
                "SPMV_GIT_COMMIT.  Run `python -m sdk.build_info write` before syncing")
    if info.dirty and not allow_dirty:
        return (f"the tree differs from commit {info.commit} ({info.source}); commit first, or "
                "pass --allow-dirty to record the run as dirty")
    return None


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["write"]:
        path = write_stamp()
        print(f"wrote {path}")
        args = ["show"]
    if args == ["show"]:
        info = resolve()
        print(info.label(), info.describe)
        return 0 if info.known else 1
    print("usage: python -m sdk.build_info write|show", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
