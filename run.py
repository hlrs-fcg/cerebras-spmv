"""Run compiled SpMV artifact on Cerebras hardware via SdkLauncher."""

import json
import sys
from pathlib import Path

from utils.logging import logger, setup_logging
from config import SpMVConfig
from sdk.launcher import run


if __name__ == "__main__":
    setup_logging()

    artifact_path = Path("cluster_work/out/artifact.json")

    try:
        with open(artifact_path, "r", encoding="utf8") as f:
            cfg = SpMVConfig.from_dict(json.load(f))
    except FileNotFoundError:
        logger.error(f"Artifact not found at {artifact_path} — run compile.py first")
        sys.exit(1)

    run(cfg)
