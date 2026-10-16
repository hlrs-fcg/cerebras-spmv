"""Every run.py command line printed in the docs must parse against the real parser.

A live cluster run failed on `run.py: error: unrecognized arguments: --grid-height 1 -v`
because memory/AGENTS.md documented flags the parser did not have.  These tests
extract the commands from the docs and feed them through run.build_parser(), so
that drift fails here instead of on the cluster.
"""

import re
import shlex
from pathlib import Path

import pytest

from run import build_parser

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Docs whose fenced bash blocks are checked. Each must contribute >= 1 command.
DOCUMENTED_FILES = ("README.md", "memory/AGENTS.md", "docs/val-only-encoding.md",
                    "docs/host-timing-barrier.md", "docs/synced-timing.md",
                    "docs/resident-spmv.md")

# Fence languages treated as shell. An untagged fence counts; ```python does not.
_SHELL_TAGS = {"", "bash", "sh", "shell", "console"}

# run.py at a command position: bare, or behind an interpreter prefix. Anchored
# so prose and directory-tree lines like "└── run.py  # Entry point" are ignored.
_COMMAND = re.compile(r"^(?:(?:uv\s+run\s+)?(?:python3?|py)\s+|\./)?run\.py\b(.*)$")


def extract_commands(text: str) -> list[str]:
    """Return the argument strings that follow `run.py` inside shell fences."""
    commands = []
    fence_tag = None  # None means "not inside a fence"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            fence_tag = stripped[3:].strip().lower() if fence_tag is None else None
            continue
        if fence_tag is None or fence_tag not in _SHELL_TAGS:
            continue
        if not stripped or stripped.startswith("#"):
            continue
        match = _COMMAND.match(stripped)
        if match:
            commands.append(match.group(1).strip())
    return commands


def _documented() -> list[tuple[str, str]]:
    pairs = []
    for rel in DOCUMENTED_FILES:
        path = PROJECT_ROOT / rel
        if path.is_file():
            pairs.extend((rel, cmd) for cmd in extract_commands(path.read_text()))
    return pairs


DOCUMENTED_COMMANDS = _documented()


# --------------------------------------------------------------------------
# extractor sanity — otherwise the parametrized test below is vacuous
# --------------------------------------------------------------------------

def test_every_documented_file_contributes_commands():
    for rel in DOCUMENTED_FILES:
        path = PROJECT_ROOT / rel
        if not path.is_file():
            pytest.skip(f"{rel} not present")
        assert extract_commands(path.read_text()), f"no run.py commands found in {rel}"


def test_extractor_finds_commands_and_skips_prose():
    text = "\n".join([
        "prose mentioning run.py --not-a-command",
        "```bash",
        "# a comment",
        "uv run python3 run.py run --rows 8 --cols 8 --density 0.5",
        "",
        "uv run python run.py sweep a.toml",
        "```",
        "more prose",
    ])
    assert extract_commands(text) == [
        "run --rows 8 --cols 8 --density 0.5",
        "sweep a.toml",
    ]


def test_extractor_ignores_non_bash_fences():
    text = "```python\nrun.py run --rows 1\n```\n"
    assert extract_commands(text) == []


def test_extractor_ignores_directory_tree_lines():
    """README's structure block mentions run.py but is not an invocation."""
    text = "\n".join([
        "```",
        ".",
        "├── config.py               # SpMVConfig dataclass",
        "└── run.py                  # Entry point: single run or sweep",
        "```",
    ])
    assert extract_commands(text) == []


# --------------------------------------------------------------------------
# the actual contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "source, command",
    DOCUMENTED_COMMANDS,
    ids=[f"{src}::{cmd[:48]}" for src, cmd in DOCUMENTED_COMMANDS],
)
def test_documented_command_parses(source, command):
    parser = build_parser()
    try:
        parser.parse_args(shlex.split(command))
    except SystemExit as exc:
        pytest.fail(
            f"{source} documents a command run.py cannot parse "
            f"(argparse exit {exc.code}):\n    run.py {command}"
        )


# --------------------------------------------------------------------------
# the two gaps the cluster run exposed
# --------------------------------------------------------------------------

def test_grid_height_is_accepted_and_reaches_grid_height_target():
    args = build_parser().parse_args(
        shlex.split("run --rows 64 --cols 64 --density 0.5 --num-pes 1 --grid-height 1")
    )
    assert args.grid_height == 1


def test_grid_height_defaults_to_none():
    """None lets grid_shape pick the near-square shape."""
    args = build_parser().parse_args(
        shlex.split("run --rows 64 --cols 64 --density 0.5 --num-pes 64")
    )
    assert args.grid_height is None


def test_explicit_grid_height_still_hits_config_validation():
    from run import cmd_run

    args = build_parser().parse_args(shlex.split(
        "run --rows 1000 --cols 1000 --density 0.01 --num-pes 4 --grid-height 99999"
    ))
    assert cmd_run(args) == 1  # loud failure, not a silent clamp


@pytest.mark.parametrize("command", [
    "-v run --rows 64 --cols 64 --density 0.5",
    "run --rows 64 --cols 64 --density 0.5 -v",
    "--verbose run --rows 64 --cols 64 --density 0.5",
    "run --rows 64 --cols 64 --density 0.5 --verbose",
])
def test_verbose_parses_before_and_after_the_subcommand(command):
    args = build_parser().parse_args(shlex.split(command))
    assert args.verbose is True


@pytest.mark.parametrize("command", [
    "-v sweep bench/configs/sim-weak-scaling.toml",
    "sweep bench/configs/sim-weak-scaling.toml -v",
])
def test_verbose_works_on_the_sweep_subcommand_too(command):
    args = build_parser().parse_args(shlex.split(command))
    assert args.verbose is True


def test_verbose_defaults_to_false():
    args = build_parser().parse_args(
        shlex.split("run --rows 64 --cols 64 --density 0.5")
    )
    assert args.verbose is False


def test_subparser_does_not_clobber_a_leading_verbose():
    """argparse would reset verbose=False here without default=SUPPRESS."""
    args = build_parser().parse_args(
        shlex.split("-v run --rows 64 --cols 64 --density 0.5")
    )
    assert args.verbose is True
