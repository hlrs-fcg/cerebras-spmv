#!/usr/bin/env python3
"""Launch SpMV execution on cluster worker node."""

import argparse
import json
import sys
from pathlib import Path

from utils.logging import logger, setup_logging
from config import SpMVConfig
from sdk.launcher import run


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Run SpMV on cluster worker node")
    parser.add_argument("--out-path", type=Path, default=Path("cluster_work/out"),
                        help="Output directory used during compilation (default: cluster_work/out)")
    parser.add_argument("--simulator", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        setup_logging(level="DEBUG")

    artifact_path = args.out_path / "artifact.json"

    try:
        with open(artifact_path, "r", encoding="utf8") as f:
            cfg = SpMVConfig.from_dict(json.load(f))
    except FileNotFoundError:
        logger.error(f"Artifact not found at {artifact_path} — run cluster/compile.py first")
        return 1

    cfg.simulator = args.simulator

    try:
        run(cfg)
    except Exception as e:
        logger.exception("Execution failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
