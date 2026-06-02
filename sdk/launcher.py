"""Thin SdkLauncher wrapper."""

import csv
import subprocess
import tempfile
from pathlib import Path

from config import SpMVConfig
from utils.logging import logger


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, check=True, timeout=1,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def run(cfg: SpMVConfig, download_path: Path = Path("bench/data/benchmark_results.csv")) -> None:
    """Stage and run worker_runtime.py via SdkLauncher. Downloads benchmark CSV on completion."""
    worker_cmd = (
        f"python worker_runtime.py"
        f" --name out"
        f" --cmaddr %CMADDR%"
        f" --rows {cfg.rows}"
        f" --cols {cfg.cols}"
        f" --num-pes {cfg.num_pes}"
        f" --grid-width {cfg.width}"
        f" --grid-height {cfg.height}"
        f" --arch {cfg.arch}"
        f" --total-nnz-per-pe {cfg.total_nnz_per_pe}"
        f" --max-rows-per-pe {cfg.max_rows_per_pe}"
        f" --save-benchmark benchmark_results.csv"
    )

    if cfg.mtx_file:
        worker_cmd += f" --mtx-file {cfg.mtx_file}"
    else:
        worker_cmd += f" --density {cfg.density} --seed {cfg.seed}"

    cfg.git_commit = _get_git_commit()
    logger.info(f"Launching with artifact_id={cfg.artifact_id}, git_commit={cfg.git_commit}")
    logger.debug(f"Worker command: {worker_cmd}")

    from cerebras.sdk.client import SdkLauncher

    with SdkLauncher(
        cfg.artifact_id,
        simulator=cfg.simulator,
        disable_version_check=cfg.disable_version_check,
    ) as launcher:
        launcher.stage("cluster/worker_runtime.py")
        launcher.stage("bench/__init__.py")
        launcher.stage("bench/benchmark.py")

        if cfg.mtx_file:
            logger.info(f"Staging matrix file: {cfg.mtx_file}")
            launcher.stage(str(cfg.mtx_file))

        logger.info("Executing on worker node...")
        response = launcher.run(worker_cmd)
        logger.info(f"Worker output:\n{response}")

        _download_benchmark(launcher, download_path, cfg.git_commit)


def _download_benchmark(launcher, download_path: Path, git_commit: str = "unknown") -> None:
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

        if "git_commit" in fieldnames:
            for row in rows:
                row["git_commit"] = git_commit

        file_exists = download_path.exists()
        with open(download_path, "a" if file_exists else "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Appended {len(rows)} benchmark(s) to {download_path}")

    except Exception as e:
        logger.opt(exception=True).warning("Failed to download benchmark results")
    finally:
        tmp_path.unlink(missing_ok=True)
