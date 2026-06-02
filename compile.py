"""Compile CSL code on login node for later execution via SdkLauncher."""

import sys

from utils.logging import logger, setup_logging
from config import SpMVConfig
from sdk.compiler import compile


if __name__ == "__main__":
    setup_logging()

    cfg = SpMVConfig(
        rows=10000,
        cols=10000,
        density=0.01,
    )

    logger.info(f"Config: {cfg.width}×{cfg.height} grid, {cfg.num_pes:,} PEs")
    sys.exit(compile(cfg))
