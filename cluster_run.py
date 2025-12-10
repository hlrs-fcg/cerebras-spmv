#!/usr/bin/env python3
"""
Cluster Run Script
Uses SdkLauncher to submit worker_runtime.py for execution on cluster worker node.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from cerebras.sdk.client import SdkLauncher

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Launch worker runtime via SdkLauncher."""
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description='SpMV execution on cluster worker node',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use artifact from cluster_compile.py
  python cluster_run.py

  # Use custom artifact path
  python cluster_run.py --artifact-path my_artifact.json

  # Run in simulator mode (for testing)
  python cluster_run.py --simulator

  # Download benchmark results to custom location
  python cluster_run.py --download-benchmark results/my_benchmark.csv
        """
    )
    parser.add_argument('--artifact-path', type=str, default='artifact_path.json',
                       help='Path to artifact metadata JSON from cluster_compile.py (default: artifact_path.json)')
    parser.add_argument('--simulator', action='store_true',
                       help='Run in simulator mode instead of real hardware')
    parser.add_argument('--download-benchmark', type=str, default='',
                       help='Local path to download benchmark CSV (if empty, will use tests/benchmark_results.csv)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    args = parser.parse_args()

    # resolve artifact path
    if not Path(args.artifact_path).is_absolute():
        args.artifact_path = str(script_dir / args.artifact_path)

    # configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    logger.info("="*60)
    logger.info("Cluster Launcher - Login Node")
    logger.info("="*60)

    # load artifact metadata
    logger.info(f"Loading artifact metadata from: {args.artifact_path}")
    try:
        with open(args.artifact_path, 'r', encoding='utf8') as f:
            artifact_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Artifact metadata not found: {args.artifact_path}")
        logger.error("Did you run cluster_compile.py first?")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse artifact metadata: {e}")
        return 1

    # load other required data
    artifact_id = artifact_data.get('artifact_id')
    if not artifact_id:
        logger.error("artifact_id not found in metadata")
        return 1

    matrix_config = artifact_data.get('matrix_config', {})
    params = artifact_data.get('params', {})
    num_pes = artifact_data.get('num_pes')

    # validate required fields
    required_matrix_fields = ['rows', 'cols', 'density', 'seed']
    missing_fields = [f for f in required_matrix_fields if f not in matrix_config]
    if missing_fields:
        logger.error(f"Missing matrix config fields: {missing_fields}")
        return 1

    required_param_fields = ['width', 'height']
    missing_params = [f for f in required_param_fields if f not in params]
    if missing_params:
        logger.error(f"Missing param fields: {missing_params}")
        return 1

    if num_pes is None:
        logger.error("num_pes not found in metadata")
        return 1

    logger.info(f"Artifact ID: {artifact_id}")
    logger.info(f"Matrix: {matrix_config['rows']}×{matrix_config['cols']}, density={matrix_config['density']}")
    logger.info(f"PEs: {num_pes}, Grid: {params['width']}×{params['height']}")

    arch = 'wse3'  # default, could be stored in compiled artifact

    # construct worker_runtime.py command
    # NOTE: Use %CMADDR% template variable - SdkLauncher will substitute it at runtime
    worker_cmd = (
        f"python worker_runtime.py "
        f"--name out "  # should match the out directory from compilation
        f"--cmaddr %CMADDR% "
        f"--rows {matrix_config['rows']} "
        f"--cols {matrix_config['cols']} "
        f"--density {matrix_config['density']} "
        f"--seed {matrix_config['seed']} "
        f"--num-pes {num_pes} "
        f"--grid-width {params['width']} "
        f"--grid-height {params['height']} "
        f"--arch {arch} "
        f"--save-benchmark benchmark_results.csv"
    )

    if args.verbose:
        worker_cmd += " --verbose"

    logger.info("\nLaunching worker runtime on cluster...")
    logger.info(f"Command: {worker_cmd}")

    try:
        with SdkLauncher(
            artifact_id,
            simulator=args.simulator,
            disable_version_check=True
        ) as launcher:
            # stage worker_runtime.py and dependencies
            logger.info("Staging worker_runtime.py...")
            launcher.stage("worker_runtime.py")
            launcher.stage("spmv_prototype.py")
            launcher.stage("spmv_runner")  # stage entire package
            launcher.stage("tests/__init__.py")
            launcher.stage("tests/benchmark.py")

            logger.info("Executing on worker node...")
            response = launcher.run(worker_cmd)

            logger.info("\n" + "="*60)
            logger.info("Worker Output:")
            logger.info("="*60)
            print(response)
            logger.info("="*60)

            download_path = args.download_benchmark if args.download_benchmark else 'tests/benchmark_results.csv'
            download_path_resolved = str(script_dir / download_path) if not Path(download_path).is_absolute() else download_path

            logger.info(f"\nDownloading benchmark results...")
            try:
                # ensure parent directory exists
                Path(download_path_resolved).parent.mkdir(parents=True, exist_ok=True)

                # Download to temporary file first (worker has only one entry)
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv') as tmp:
                    tmp_path = tmp.name

                launcher.download_artifact("benchmark_results.csv", tmp_path)

                # Append downloaded results to main CSV
                import csv
                with open(tmp_path, 'r') as tmp_f:
                    reader = csv.DictReader(tmp_f)
                    rows = list(reader)

                    if rows:
                        # Check if main file exists
                        main_exists = Path(download_path_resolved).exists()

                        with open(download_path_resolved, 'a' if main_exists else 'w', newline='') as main_f:
                            writer = csv.DictWriter(main_f, fieldnames=reader.fieldnames)

                            if not main_exists:
                                writer.writeheader()

                            writer.writerows(rows)

                        logger.info(f"✓ Appended {len(rows)} benchmark(s) to {download_path_resolved}")
                    else:
                        logger.warning("Downloaded CSV has no data rows")

                # Clean up temp file
                Path(tmp_path).unlink()

            except Exception as e:
                logger.warning(f"Failed to download benchmark results: {e}")
                logger.warning("This may be normal if the run failed or benchmarking was disabled")

    except Exception as e:
        logger.error(f"Launcher execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    logger.info("\n" + "="*60)
    logger.info("Execution complete!")
    logger.info("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
