"""Tests for the appliance staging contract between sdk/launcher.py and the worker.

cluster/worker_runtime.py runs on the appliance worker node, staged flat into a
working directory by SdkLauncher.  Nothing here can be checked by running the
worker for real on this machine, so these tests pin the parts that are
statically checkable: the command line matches the worker's argparse, the
staged file set closes the worker's import graph, and the compile options carry
every param src/csl/layout.csl declares.
"""

import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from config import SpMVConfig
from sdk.compiler import compile_options
from sdk.launcher import STAGED_FILES, WORKER_PYTHON, build_worker_command

from tests.worker_src import worker_source  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKER = PROJECT_ROOT / "cluster" / "worker_runtime.py"
LAYOUT_CSL = PROJECT_ROOT / "src" / "csl" / "layout.csl"


@pytest.fixture
def cfg():
    return SpMVConfig(rows=10_000, cols=10_000, density=0.001,
                      num_pes_target=256, simulator=True)


def _worker_argparse_flags() -> dict[str, bool]:
    """Map flag -> required, parsed out of the worker's argparse calls.

    Brace-matched rather than line-anchored: the previous regex required the
    closing paren on the same line, so it silently skipped every multi-line
    add_argument call -- which is how --h2d-mode stayed invisible to the
    contract tests below while being fully implemented in the worker.
    """
    source = worker_source()
    flags = {}
    for match in re.finditer(r"add_argument\(", source):
        depth, i = 1, match.end()
        while i < len(source) and depth:
            depth += {"(": 1, ")": -1}.get(source[i], 0)
            i += 1
        body = source[match.end():i - 1]
        name = re.search(r'"(--[a-z0-9-]+)"', body)
        if name:
            flags[name.group(1)] = "required=True" in body
    return flags


def _command_flags(cmd: str) -> set[str]:
    return {tok for tok in cmd.split() if tok.startswith("--")}


# --------------------------------------------------------------------------
# worker command vs worker argparse
# --------------------------------------------------------------------------

def test_worker_argparse_was_parsed():
    """Guard the regex above: if it silently matches nothing, later tests are vacuous."""
    flags = _worker_argparse_flags()
    assert len(flags) >= 10, f"only parsed {flags} from {WORKER}"
    assert flags["--rows"] is True
    # --h2d-mode is declared across several lines; a line-anchored parser misses
    # it and every contract test here goes quietly vacuous for that flag.
    assert flags["--h2d-mode"] is False


def test_every_emitted_flag_exists_in_the_worker(cfg):
    known = set(_worker_argparse_flags())
    for flag in _command_flags(build_worker_command(cfg)):
        assert flag in known, f"{flag} is not accepted by cluster/worker_runtime.py"


def test_every_required_worker_flag_is_emitted(cfg):
    required = {f for f, req in _worker_argparse_flags().items() if req}
    emitted = _command_flags(build_worker_command(cfg))
    assert required <= emitted, f"missing required flag(s): {sorted(required - emitted)}"


# Worker flags that are deliberately never emitted, each with its reason.  This
# allowlist is the whole point of the test below: an optional worker flag that
# is a real knob must either be emitted or be justified here.
WORKER_ONLY_FLAGS = {
    "--verbose": "worker-side log level, not a property of the run",
}


def test_every_optional_worker_knob_is_emitted_too():
    """The gap that hid --h2d-mode for a whole campaign.

    test_every_required_worker_flag_is_emitted only covers required=True, so an
    optional flag could be fully implemented in the worker, never emitted by
    build_worker_command, and silently take its argparse default -- which is
    exactly what --h2d-mode did.  Union over both matrix sources, since
    --mtx-file and --density/--seed are mutually exclusive by construction.
    """
    emitted = _command_flags(build_worker_command(
        SpMVConfig(rows=10_000, cols=10_000, density=0.001,
                   num_pes_target=256, simulator=True)))
    emitted |= _command_flags(build_worker_command(
        SpMVConfig(rows=10_000, cols=10_000, density=0.001,
                   num_pes_target=256, simulator=True,
                   mtx_file=Path("matrices/example.mtx"))))
    # --cmaddr is hardware-only (the simulator refuses %CMADDR%).
    emitted |= _command_flags(build_worker_command(
        SpMVConfig(rows=10_000, cols=10_000, density=0.001,
                   num_pes_target=256, simulator=False)))
    # --serialize-column-rects is emitted only when requested.
    emitted |= _command_flags(build_worker_command(
        SpMVConfig(rows=10_000, cols=10_000, density=0.001,
                   num_pes_target=256, simulator=True,
                   serialize_column_rects=True, sync_timing=True)))
    # --null-spmv is emitted only for a resident run that asks for it.
    emitted |= _command_flags(build_worker_command(
        SpMVConfig(rows=10_000, cols=10_000, density=0.001,
                   num_pes_target=256, simulator=True,
                   encoding="resident", null_spmv=True)))

    # --vendor is emitted only when a numba vendor zip is staged (SPMV_VENDOR_ZIP / _DIR).
    emitted |= _command_flags(build_worker_command(
        SpMVConfig(rows=10_000, cols=10_000, density=0.001,
                   num_pes_target=256, simulator=True), vendor="numba_vendor.zip"))

    unreachable = set(_worker_argparse_flags()) - emitted - set(WORKER_ONLY_FLAGS)
    assert not unreachable, (
        f"worker accepts {sorted(unreachable)} but sdk/launcher.py never emits "
        f"them, so the worker silently takes its argparse default. Emit them in "
        f"build_worker_command, or justify them in WORKER_ONLY_FLAGS."
    )


def test_worker_only_allowlist_stays_honest():
    """Every allowlisted flag must still exist, or the entry is stale."""
    flags = _worker_argparse_flags()
    for flag in WORKER_ONLY_FLAGS:
        assert flag in flags, f"{flag} is allowlisted but no longer a worker flag"


def test_max_rows_per_pe_is_gone(cfg):
    """HEAD passed --max-rows-per-pe; the field and the worker flag are both gone."""
    assert "--max-rows-per-pe" not in build_worker_command(cfg)
    assert "--max-rows-per-pe" not in _worker_argparse_flags()


def test_num_stripes_is_passed(cfg):
    cmd = build_worker_command(cfg)
    assert "--num-stripes" in _worker_argparse_flags()
    assert f"--num-stripes {cfg.num_col_stripes}" in cmd


def test_random_matrix_passes_density_and_seed(cfg):
    cmd = build_worker_command(cfg)
    assert f"--density {cfg.density}" in cmd
    assert f"--seed {cfg.seed}" in cmd
    assert "--mtx-file" not in cmd


def test_mtx_file_is_passed_as_a_basename():
    """Staging flattens, so the worker sees the file without its directory."""
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.01,
                     mtx_file=Path("matrices/nested/foo.mtx"),
                     num_pes_target=64, simulator=True)
    cmd = build_worker_command(cfg)
    assert "--mtx-file foo.mtx" in cmd
    assert "matrices/" not in cmd
    assert "--density" not in cmd


def test_cmaddr_placeholder_is_left_for_sdklauncher_on_hardware():
    """On hardware SdkLauncher substitutes the system address for %CMADDR%."""
    cfg = SpMVConfig(rows=1000, cols=1000, density=0.01, num_pes_target=16,
                     simulator=False)
    assert "--cmaddr %CMADDR%" in build_worker_command(cfg)


def test_simulator_command_never_contains_the_placeholder(cfg):
    """SdkLauncher(simulator=True).run() raises on any command containing
    %CMADDR% (sdk_appliance_client.py, SdkLauncher.run), so it must be absent."""
    assert cfg.simulator
    cmd = build_worker_command(cfg)
    assert "%CMADDR%" not in cmd and "--cmaddr" not in cmd


def test_worker_cmaddr_is_optional_and_defaults_to_the_simulator():
    src = worker_source()
    assert 'parser.add_argument("--cmaddr",   default=None)' in src
    assert 'runtime_kwargs = {"cmaddr": args.cmaddr}' in src


def test_grid_values_come_from_the_config(cfg):
    cmd = build_worker_command(cfg)
    assert f"--grid-width {cfg.width}" in cmd
    assert f"--grid-height {cfg.height}" in cmd
    assert f"--num-pes {cfg.num_pes}" in cmd
    assert f"--total-nnz-per-pe {cfg.total_nnz_per_pe}" in cmd


# --------------------------------------------------------------------------
# staged file set
# --------------------------------------------------------------------------

def test_staged_files_all_exist():
    for rel in STAGED_FILES:
        assert (PROJECT_ROOT / rel).is_file(), f"staged file missing: {rel}"


def test_coo_tiling_is_staged():
    """New module imported at cluster/worker_runtime.py:19; HEAD's list predates it."""
    assert any(Path(f).name == "coo_tiling.py" for f in STAGED_FILES)


def test_no_staged_file_shadows_a_stdlib_module_the_worker_imports():
    """Staging is flat, so a staged utils/logging.py would land as logging.py."""
    worker_imports = set()
    for node in ast.walk(ast.parse(worker_source())):
        if isinstance(node, ast.Import):
            worker_imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            worker_imports.add(node.module.split(".")[0])

    for rel in STAGED_FILES:
        stem = Path(rel).stem
        if stem == "__init__":
            continue
        assert stem not in sys.stdlib_module_names or stem not in worker_imports, (
            f"staged {rel} flattens to {stem}.py and shadows stdlib {stem}"
        )


def test_flat_staging_closes_the_worker_import_graph(tmp_path):
    """Import the worker from a flattened staging dir with the repo off sys.path.

    Catches a missing stage entry and the utils/loguru dependency that
    bench/benchmark.py used to have.  worker_runtime.py imports cerebras only
    inside main(), so importing the module needs no SDK.
    """
    staging = tmp_path / "staged"
    staging.mkdir()
    for rel in STAGED_FILES:
        shutil.copy(PROJECT_ROOT / rel, staging / Path(rel).name)

    result = subprocess.run(
        [sys.executable, "-c", "import worker_runtime; print('IMPORT_OK')"],
        cwd=staging,
        capture_output=True,
        text=True,
        timeout=120,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert "IMPORT_OK" in result.stdout, (
        f"worker import graph is not closed by STAGED_FILES\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_benchmark_module_imports_without_utils(tmp_path):
    """bench/benchmark.py must degrade to stdlib logging where utils/loguru is absent."""
    staging = tmp_path / "staged"
    staging.mkdir()
    shutil.copy(PROJECT_ROOT / "bench" / "benchmark.py", staging / "benchmark.py")

    result = subprocess.run(
        [sys.executable, "-c",
         "import benchmark; benchmark.logger.info('x'); print('IMPORT_OK')"],
        cwd=staging,
        capture_output=True,
        text=True,
        timeout=120,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert "IMPORT_OK" in result.stdout, (
        f"benchmark.py still needs utils/loguru\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_worker_python_is_a_bare_interpreter_name():
    assert WORKER_PYTHON in ("python", "python3", "cs_python")


# --------------------------------------------------------------------------
# compile options
# --------------------------------------------------------------------------

def test_compile_options_pass_every_param_layout_declares(cfg):
    declared = set(re.findall(r"^param\s+([A-Za-z_][A-Za-z0-9_]*)\s*:",
                              LAYOUT_CSL.read_text(), re.M))
    assert declared, f"parsed no params from {LAYOUT_CSL}"
    # a param with a default in layout.csl would be named only when it is not the default, so the option
    # string of every existing run stays byte-identical; there is none now (LOCAL_KERNEL is gone)
    defaulted = set(re.findall(r"^param\s+([A-Za-z_][A-Za-z0-9_]*)\s*:[^=;]*=",
                               LAYOUT_CSL.read_text(), re.M))
    assert defaulted == set()

    params_arg = compile_options(cfg).split("--params=")[1].split()[0]
    passed = {kv.split(":")[0] for kv in params_arg.split(",")}

    assert declared - defaulted <= passed, f"missing --params: {sorted(declared - defaulted - passed)}"
    assert passed <= declared, f"undeclared --params: {sorted(passed - declared)}"


def test_compile_options_carry_fabric_and_memcpy_flags(cfg):
    options = compile_options(cfg)
    assert f"--fabric-dims={cfg.fabric_dims[0]},{cfg.fabric_dims[1]}" in options
    assert f"--fabric-offsets={cfg.fabric_offsets[0]},{cfg.fabric_offsets[1]}" in options
    assert f"--arch={cfg.arch}" in options
    assert "--memcpy" in options
    assert "--channels=1" in options


def test_compile_options_use_kx_not_kx_stripe(cfg):
    """Striping is not implemented end to end; the PE sizes x_vec for the column."""
    params_arg = compile_options(cfg).split("--params=")[1].split()[0]
    assert f"KX:{cfg.kx}" in params_arg


def test_every_module_the_worker_imports_from_this_repo_is_staged():
    """SdkLauncher stages STAGED_FILES flat; an import of a sibling that is not in the list fails
    on the appliance and nowhere else (the SDK is Linux-only), so check the closure here."""
    staged = {Path(f).stem: Path(f) for f in STAGED_FILES}
    repo_modules = {p.stem for d in ("cluster", "bench") for p in (PROJECT_ROOT / d).glob("*.py")}
    for stem, rel in staged.items():
        tree = ast.parse((PROJECT_ROOT / rel).read_text())
        for node in ast.walk(tree):
            names = ([a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import)
                     else [node.module.split(".")[0]] if isinstance(node, ast.ImportFrom) and node.module
                     and node.level == 0 else [])
            for name in names:
                if name in repo_modules and name not in ("run", "sweep_config", "analyze"):
                    assert name in staged, f"{rel} imports {name}, which is not in STAGED_FILES"
