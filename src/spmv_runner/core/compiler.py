"""CSL compilation using Cerebras SDK compiler."""

import json
import logging
import time
from pathlib import Path

from cerebras.sdk.client import SdkCompiler

logger = logging.getLogger(__name__)


def compile_csl(
    width: int,
    height: int,
    total_nnz_per_pe: int,
    max_rows_per_pe: int,
    fabric_dims: tuple = (762, 1172),
    fabric_offsets: tuple = (4, 1),
    artifact_path: str = "artifact.json",
    out_path: str = "out"
):
    logger.info(f"Compiling CSL for {width}×{height} PE grid...")

    # Resolve paths - find project root (where run.py is)
    # Assuming this file is in spmv_runner/core/, project root is 2 levels up
    project_root = Path(__file__).resolve().parent.parent.parent
    src_path = str(project_root / "src")

    out_path = str(Path(out_path).resolve())
    Path(out_path).mkdir(parents=True, exist_ok=True)

    ARGS = (
        f"--fabric-dims={fabric_dims[0]},{fabric_dims[1]} "
        f"--fabric-offsets={fabric_offsets[0]},{fabric_offsets[1]} "
        f"--memcpy --channels=1 "
        f"--params=width:{width},height:{height},"
        f"TOTAL_NNZ:{total_nnz_per_pe},MAX_ROWS:{max_rows_per_pe}"
    )

    compile_start = time.time()

    with SdkCompiler(resource_cpu=24000, resource_mem=64<<30, disable_version_check=True) as compiler:
        artifact_id = compiler.compile(
            app_path=src_path,
            csl_main="layout.csl",
            options=ARGS,
            out_path=out_path,
        )

    compile_time = time.time() - compile_start

    if artifact_id is None:
        logger.error("Compilation failed! No artifact was generated!")
        raise RuntimeError("CSL compilation failed")

    # Ensure all values are Python native types for JSON serialization
    artifact_data = {
        "artifact_id": artifact_id,
        "fabric_dims": [int(fabric_dims[0]), int(fabric_dims[1])],
        "params": {
            "width": int(width),
            "height": int(height),
            "TOTAL_NNZ": int(total_nnz_per_pe),
            "MAX_ROWS": int(max_rows_per_pe),
        },
        "compile_time_seconds": float(compile_time)
    }

    artifact_file = Path(artifact_path)
    with open(artifact_file, "w", encoding="utf8") as f:
        json.dump(artifact_data, f, indent=2)

    return out_path, artifact_id, compile_time
