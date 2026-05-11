#!/usr/bin/env python3
"""Compile CSL code for cluster execution."""

import argparse
import sys
from pathlib import Path

from utils.logging import logger, setup_logging
from config import SpMVConfig
from sdk.compiler import compile


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Compile SpMV CSL code for cluster execution")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--random", action="store_true", help="Generate random matrix")
    source.add_argument("--mtx-file", type=Path, help="Path to MatrixMarket (.mtx) file")

    parser.add_argument("--rows", type=int, help="Matrix rows (required with --random)")
    parser.add_argument("--cols", type=int, help="Matrix cols (required with --random)")
    parser.add_argument("--density", type=float, help="Sparsity density (required with --random)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-path", type=Path, default=Path("cluster_work/out"))
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        setup_logging(level="DEBUG")

    if args.random and not all([args.rows, args.cols, args.density]):
        logger.error("--rows, --cols, and --density are required with --random")
        return 1

    cfg = SpMVConfig(
        rows=args.rows or 0,
        cols=args.cols or 0,
        density=args.density or 0.0,
        seed=args.seed,
        mtx_file=args.mtx_file,
        out_path=args.out_path,
    )

    try:
        compile(cfg)
    except Exception as e:
        logger.exception("Compilation failed")
        return 1

    logger.info(f"Artifact saved to: {cfg.out_path / 'artifact.json'}")
    logger.info("Next: python cluster/run.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
