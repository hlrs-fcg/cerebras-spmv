"""Thin SdkLauncher wrapper: stage the worker onto the appliance and run it.

Execution runs on the appliance, not locally: the CS-3 user node has neither
`cslc` nor `cs_python` on PATH.  Simulator vs hardware is the
``SdkLauncher(simulator=...)`` argument.  On hardware SdkLauncher substitutes
the system address for the ``%CMADDR%`` placeholder; on the simulator it
*refuses* any command containing ``%CMADDR%``, so simulator runs omit --cmaddr
and the worker builds ``SdkRuntime(name, cmaddr=None)``.
"""

import csv
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from bench.benchmark import append_rows
from sdk.build_info import BuildInfo, resolve as resolve_build
from config import SpMVConfig
from utils.logging import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Interpreter for the worker command *on the appliance worker node*, where the
# Cerebras runtime is already on the default python.  This string is never run
# on the user node, so the missing local cs_python does not apply here.  If the
# worker ever fails importing cerebras.sdk.runtime, this is the one token to
# change to "cs_python".
WORKER_PYTHON = "python"

# Modules the worker needs.  SdkLauncher stages files flat into the worker's
# working directory, which is why worker_runtime.py runs bare and imports
# `benchmark` / `coo_tiling` as top-level modules rather than by package path.
STAGED_FILES = (
    "cluster/worker_runtime.py",
    "cluster/worker_cli.py",
    "cluster/worker_common.py",
    "cluster/worker_device.py",
    "cluster/worker_host.py",
    "cluster/worker_layout.py",
    "cluster/worker_modes.py",
    "cluster/worker_report.py",
    "cluster/worker_resident.py",
    "cluster/worker_stream.py",
    "cluster/worker_timing.py",
    "cluster/coo_tiling.py",
    "cluster/host_prep.py",
    "cluster/mm_io.py",
    "cluster/tile_kernels.py",
    "cluster/tile_cache.py",
    "cluster/worker_env.py",
    "bench/__init__.py",
    "bench/benchmark.py",
)


def resolve_vendor_zip() -> Path | None:
    """The numba vendor zip to stage, from the user-node environment (opt-in).

    ``SPMV_VENDOR_ZIP``  a zip built by ``python -m sdk.vendor_pack build``
    ``SPMV_VENDOR_DIR``  an unpacked directory of numba + llvmlite (uv pip install --target);
                         zipped into a temporary file here
    A single zip is staged rather than the directory: it is one ~50 MB file (145 MB unpacked,
    thousands of entries), and the worker has to unpack it anyway because zipimport cannot load
    the compiled extensions.
    """
    zip_env, dir_env = os.environ.get("SPMV_VENDOR_ZIP"), os.environ.get("SPMV_VENDOR_DIR")
    if zip_env:
        z = Path(zip_env)
        if not z.is_file():
            raise RuntimeError(f"SPMV_VENDOR_ZIP={zip_env} does not exist")
        return z
    if dir_env:
        from sdk.vendor_pack import pack
        return pack(dir_env, Path(tempfile.mkdtemp(prefix="spmv-vendor-")) / "numba_vendor.zip")
    return None


def build_worker_command(cfg: SpMVConfig, vendor: str | None = None) -> str:
    """Worker command line for cfg.

    Every flag here must exist in cluster/worker_runtime.py's argparse, and
    every required flag there must appear here.  Paths are basenames because
    staging flattens the directory structure.
    """
    worker_cmd = (
        f"{WORKER_PYTHON} worker_runtime.py"
        f" --name out"
        f"{'' if cfg.simulator else ' --cmaddr %CMADDR%'}"
        f" --rows {cfg.rows}"
        f" --cols {cfg.cols}"
        f" --num-pes {cfg.num_pes}"
        f" --grid-width {cfg.width}"
        f" --grid-height {cfg.height}"
        f" --arch {cfg.arch}"
        f" --total-nnz-per-pe {cfg.total_nnz_per_pe}"
        f" --num-stripes {cfg.num_col_stripes}"
        f" --h2d-mode {cfg.h2d_mode}"
        f" --memcpy-order {cfg.memcpy_order}"
        f" --encoding {cfg.encoding}"
        f" --setup-order {cfg.setup_order}"
        f" --setup-strips {cfg.resolved_setup_strips}"
        f" --loop-count {cfg.loop_count}"
        f" --progress-file progress.log"
        f" --save-benchmark benchmark_results.csv"
    )

    if vendor:
        worker_cmd += f" --vendor {vendor}"

    if cfg.serialize_column_rects:
        worker_cmd += " --serialize-column-rects"

    if cfg.sync_timing:
        worker_cmd += " --sync-timing"

    if cfg.null_spmv:
        worker_cmd += " --null-spmv"

    if cfg.simulator:
        worker_cmd += " --sim-trace"

    if cfg.mtx_file:
        worker_cmd += f" --mtx-file {Path(cfg.mtx_file).name}"
    else:
        worker_cmd += f" --density {cfg.density} --seed {cfg.seed}"

    return worker_cmd


def run(cfg: SpMVConfig, download_path: Path = Path("bench/data/benchmark_results.csv")) -> None:
    """Stage and run worker_runtime.py via SdkLauncher. Downloads benchmark CSV on completion."""
    vendor_zip = resolve_vendor_zip()
    worker_cmd = build_worker_command(cfg, vendor=vendor_zip.name if vendor_zip else None)

    build = resolve_build()
    cfg.git_commit = build.commit
    logger.info(f"Launching with artifact_id={cfg.artifact_id}, build={build.label()}")
    if not build.known:
        logger.warning("Build unknown: this run's rows will say git_commit=unknown and cannot be "
                       "traced to a commit (python -m sdk.build_info write, then sync)")
    logger.debug(f"Worker command: {worker_cmd}")

    # Fail here rather than with an opaque error from the appliance.
    missing = [f for f in STAGED_FILES if not (_PROJECT_ROOT / f).is_file()]
    if missing:
        raise RuntimeError(f"Cannot stage worker — missing file(s): {', '.join(missing)}")

    # Imported here, not at module scope: the SDK is Linux-only and run.py must
    # stay importable (--help, tests) on machines without it.
    from cerebras.sdk.client import SdkLauncher

    with SdkLauncher(
        cfg.artifact_id,
        simulator=cfg.simulator,
        disable_version_check=cfg.disable_version_check,
    ) as launcher:
        for rel_path in STAGED_FILES:
            launcher.stage(rel_path)

        if vendor_zip:
            logger.info(f"Staging numba vendor zip: {vendor_zip} ({vendor_zip.stat().st_size / 1e6:.1f} MB)")
            launcher.stage(str(vendor_zip))

        if cfg.mtx_file:
            logger.info(f"Staging matrix file: {cfg.mtx_file}")
            launcher.stage(str(cfg.mtx_file))

        target = "simulator" if cfg.simulator else "hardware"
        logger.info(f"Executing on worker node ({target})...")
        # Download in finally: a simulator deadlock raises out of run(), and
        # its sim.log / traces are exactly what is needed to diagnose it.
        try:
            response = launcher.run(worker_cmd)
            logger.info(f"Worker output:\n{response}")
        finally:
            _download_progress(launcher, Path(cfg.out_path) / "progress.log")
            _download_benchmark(launcher, download_path, cfg.git_commit,
                                io_settings=compile_io_settings(cfg), build=build)
            if cfg.simulator:
                _download_sim_artifacts(launcher, Path(cfg.out_path) / "sim")


def _download_progress(launcher, dest: Path) -> None:
    """Fetch the worker's phase log (progress.log); never raise.  It is fsynced per line, so
    if the worker's directory is still reachable after a hang or a cancel it names the
    last phase reached.  Whether SdkLauncher can still download from a cancelled job is
    unverified."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        launcher.download_artifact("progress.log", str(dest))
        logger.info(f"Worker phase log -> {dest}:\n{dest.read_text()}")
    except Exception as e:
        logger.warning(f"progress.log not fetched: {e}")


def compile_io_settings(cfg: SpMVConfig) -> dict:
    """The memcpy I/O configuration this run was compiled with.

    Stamped into the downloaded rows because the worker cannot know it: it is
    fixed at compile time by sdk/compiler.py, and cluster/worker_runtime.py only
    ever sees the finished artifact.
    """
    return {
        "channels": cfg.channels,
        "width_west_buf": cfg.width_west_buf,
        "width_east_buf": cfg.width_east_buf,
    }


# Simulator evidence.  Where the runtime writes these (the appliance working
# directory or the --name bindir) is not documented, so both are tried.
# Directories arrive as <name>.tar.gz (SdkLauncher.download_artifact).
SIM_ARTIFACTS = ("sim.log", "sim_stats.json", "simfab_traces")
SIM_ARTIFACT_DIRS = ("", "out/")


def _download_sim_artifacts(launcher, dest: Path) -> list[Path]:
    """Fetch every simulator artifact that exists; never raise."""
    dest.mkdir(parents=True, exist_ok=True)
    got = []
    for name in SIM_ARTIFACTS:
        for prefix in SIM_ARTIFACT_DIRS:
            target = dest / (prefix.replace("/", "_") + name)
            try:
                launcher.download_artifact(prefix + name, str(target))
            except Exception as e:
                logger.debug(f"sim artifact {prefix + name} not fetched: {e}")
                continue
            got.append(target)
            logger.info(f"Downloaded simulator artifact {prefix + name} -> {target}")
    if not got:
        logger.warning(f"No simulator artifacts found ({', '.join(SIM_ARTIFACTS)})")
    return got


def _download_benchmark(launcher, download_path: Path, git_commit: str = "unknown",
                        io_settings: Optional[dict] = None,
                        build: Optional[BuildInfo] = None) -> None:
    download_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp:
        tmp_path = Path(tmp.name)

    try:
        launcher.download_artifact("benchmark_results.csv", str(tmp_path))

        with open(tmp_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            logger.warning("Downloaded CSV has no data rows")
            return

        fieldnames = reader.fieldnames
        if not fieldnames:
            logger.error("Downloaded CSV has rows but no field names — skipping write")
            return

        # Stamp values the worker could not know, exactly as git_commit is.
        stamped = {"git_commit": git_commit, **(build.csv_columns() if build else {}),
                   **(io_settings or {})}
        for column, value in stamped.items():
            if column in fieldnames:
                for row in rows:
                    row[column] = value

        # append_rows never writes positionally: if download_path predates a
        # column it migrates the file, and if the headers are incompatible it
        # diverts to a sibling rather than shifting values under wrong keys.
        written = append_rows(download_path, fieldnames, rows)
        logger.info(f"Appended {len(rows)} benchmark(s) to {written}")

    except Exception:
        logger.opt(exception=True).warning("Failed to download benchmark results")
    finally:
        tmp_path.unlink(missing_ok=True)
