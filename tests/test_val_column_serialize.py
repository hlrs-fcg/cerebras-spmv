"""--encoding valtask with --h2d-mode column serialises its column rects.

On hardware, val's W one-column streaming memcpys issued back to back with
nonblock=True hang once a PE gets more words than the fabric buffers (4x4,
1000^2 d=0.01, ~650 words/PE: 112 s and 103 s; <= 76 words/PE passed).  The
same artifact passes with every rect but the last nonblock=False (5.32e-08,
22 s); the simulator passes either way.  valtask hung the same way (4x4,
d=0.01, cancelled at 2m54s).  So val / valtask +
column now serialise automatically (coo_tiling.column_rects_serialized),
nothing refuses the combination, and the header kernel's column path stays
overlapping unless --serialize-column-rects opts it in.
See docs/val-only-encoding.md, Risk 7.
"""

import logging
import shlex
import sys
import textwrap
from pathlib import Path

import pytest

from cluster.coo_tiling import column_rects_serialized
from cluster.sweep_config import BOOL_KEYS, SweepConfig
from config import DEFAULT_H2D_MODE, SpMVConfig
from run import build_parser, io_overrides, main
from sdk.compiler import compile_options
from sdk.launcher import build_worker_command

from tests.worker_src import worker_source  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKER = PROJECT_ROOT / "cluster" / "worker_runtime.py"
DOC = PROJECT_ROOT / "docs" / "val-only-encoding.md"

# Hardware: nothing about this combination is simulator-only any more.
SMALL = dict(rows=1000, cols=1000, density=0.01, num_pes_target=16)


def _worker():
    sys.path.insert(0, str(PROJECT_ROOT / "cluster"))
    sys.path.insert(0, str(PROJECT_ROOT / "bench"))
    import worker_runtime
    return worker_runtime


def _run_args(extra: str):
    return build_parser().parse_args(shlex.split(
        f"run --rows 1000 --cols 1000 --density 0.01 --num-pes 16 {extra}"))


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------

@pytest.mark.parametrize("encoding, h2d_mode, requested, expected", [
    ("valtask", "column", False, True),  # automatic: overlapping valtask rects hang (hardware 10-07)
    ("valtask", "column", True, True),
    ("valtask", "full", False, False),
    ("header", "column", False, False),  # header column path unchanged
    ("header", "column", True, True),    # ... unless opted in
    ("header", "full", True, True),      # requested, harmless (one rect)
])
def test_column_rects_serialized(encoding, h2d_mode, requested, expected):
    assert column_rects_serialized(encoding, h2d_mode, requested) is expected


def test_the_rule_ignores_word_counts():
    """The buffering threshold (76 < t <~ 650 words/PE) is unknown, so the
    decision must not depend on how many words a PE gets."""
    import inspect
    assert list(inspect.signature(column_rects_serialized).parameters) == [
        "encoding", "h2d_mode", "requested"]


def test_the_default_mode_is_column():
    """A bare --encoding valtask runs column mode, so it must be the serialised one."""
    assert DEFAULT_H2D_MODE == "column"
    assert column_rects_serialized("valtask", DEFAULT_H2D_MODE, False)


@pytest.mark.parametrize("n, serialize, expected", [
    (4, False, [True, True, True, True]),     # overlapping: what hung for val
    (4, True, [False, False, False, True]),   # one at a time; wait_ew behind the last
    (1, True, [True]),                        # full mode: a no-op
    (0, True, []),                            # all-zero matrix: nothing sent
])
def test_rect_nonblock_flags(n, serialize, expected):
    assert _worker().rect_nonblock_flags(n, serialize) == expected


# --------------------------------------------------------------------------
# the worker applies it
# --------------------------------------------------------------------------

def _src():
    return worker_source()


def test_worker_decides_once_before_the_device():
    src = _src()
    decide = src.index("serialize_rects = column_rects_serialized(args.encoding, args.h2d_mode,")
    assert src.index("args = parse_args()") < decide
    assert decide < src.index("from cerebras.sdk.runtime")
    assert src.count("column_rects_serialized(") == 1
    # logged once, there, not per rect
    assert src.count("column rects SERIALISED") == 1


def test_val_send_loop_uses_the_decision():
    src = _src()
    assert "nonblock = rect_nonblock_flags(len(r.val_plan.rects), r.serialize_rects)" in src
    loop = src[src.index("for (x0, width, n_words), nb in zip(r.val_plan.rects, nonblock):"):]
    assert "nonblock=nb)" in loop[:600]


def test_header_column_loop_uses_the_decision():
    """Same helper, so --serialize-column-rects reaches the header path too;
    column_rects_serialized keeps it overlapping by default."""
    src = _src()
    assert "nonblock = rect_nonblock_flags(W, r.serialize_rects)" in src
    loop = src[src.index("for px, nb in zip(range(W), nonblock):"):]
    assert "nonblock=nb)" in loop[:700]


def test_csv_records_what_ran():
    assert ('serialize_column_rects=serialize_rects and args.h2d_mode == "column",'
            in _src())


def test_the_refusal_is_gone_everywhere():
    for path in (WORKER, PROJECT_ROOT / "config.py",
                 PROJECT_ROOT / "cluster" / "sweep_config.py",
                 PROJECT_ROOT / "sdk" / "launcher.py"):
        text = path.read_text()
        assert "check_encoding_h2d_mode" not in text, path


# --------------------------------------------------------------------------
# config, CLI, sweep: val + column is an ordinary configuration
# --------------------------------------------------------------------------

def test_config_accepts_valtask_column_on_hardware_without_warning(recwarn):
    cfg = SpMVConfig(**SMALL, encoding="valtask")
    assert (cfg.encoding, cfg.h2d_mode, cfg.simulator) == ("valtask", "column", False)
    assert cfg.serialize_column_rects is False   # the worker decides; no rewrite
    assert not [w for w in recwarn if "column" in str(w.message)]


def test_cli_run_valtask_column_builds_a_config():
    args = _run_args("--encoding valtask")
    cfg = SpMVConfig(**SMALL, **io_overrides(args))
    assert (cfg.encoding, cfg.h2d_mode) == ("valtask", "column")


def test_sweep_accepts_valtask_at_the_default_mode(tmp_path):
    path = tmp_path / "t.toml"
    path.write_text(textwrap.dedent("""\
        [meta]
        name = "t"

        [sweep]
        type = "weak_scaling"
        nnz_per_pe = 200
        density = 0.001
        encoding = "valtask"

        [sweep.num_pes]
        pow2_up_to = 256
        """))
    runs = list(SweepConfig.from_toml(path).expand())
    assert {(c.encoding, c.h2d_mode) for c in runs} == {("valtask", "column")}


def test_explicit_opt_in_threads_through(tmp_path):
    args = _run_args("--serialize-column-rects")
    assert io_overrides(args)["serialize_column_rects"] is True
    assert "serialize_column_rects" not in io_overrides(_run_args(""))
    assert "serialize_column_rects" in BOOL_KEYS
    on = SpMVConfig(**SMALL, serialize_column_rects=True)
    assert "--serialize-column-rects" in build_worker_command(on)
    assert "--serialize-column-rects" not in build_worker_command(SpMVConfig(**SMALL))
    # runtime-only: the same artifact either way
    assert compile_options(on) == compile_options(SpMVConfig(**SMALL))


def test_valtask_column_needs_no_flag_on_the_worker_command():
    """The worker serialises valtask + column by itself."""
    cmd = build_worker_command(SpMVConfig(**SMALL, encoding="valtask"))
    assert "--serialize-column-rects" not in cmd


# --------------------------------------------------------------------------
# the CSV column
# --------------------------------------------------------------------------

def test_serialize_csv_column_round_trips(tmp_path):
    from bench.benchmark import BenchmarkResult, load_benchmarks, save_benchmark
    path = tmp_path / "r.csv"
    for flag in (False, True):
        save_benchmark(BenchmarkResult(serialize_column_rects=flag), csv_path=str(path))
    assert [r.serialize_column_rects for r in load_benchmarks(str(path))] == [False, True]
    assert BenchmarkResult().serialize_column_rects is False


# --------------------------------------------------------------------------
# docs
# --------------------------------------------------------------------------

def test_doc_records_cause_evidence_and_recommendation():
    text = DOC.read_text()
    for fact in ("112 s", "103 s", "same artifact hash", "simulator passes either",
                 "4.14e-08", "between 76 and ~650", "unconditionally",
                 "160.8K", "39.3K", "Full mode stays the recommended path",
                 "column_rects_serialized", "--serialize-column-rects"):
        assert fact in text, fact


def test_doc_documents_a_valtask_column_command():
    from tests.test_documented_commands import extract_commands
    cmds = [build_parser().parse_args(shlex.split(c))
            for c in extract_commands(DOC.read_text())]
    assert any(a.cmd == "run" and a.encoding == "valtask" and a.h2d_mode in (None, "column")
               for a in cmds)
