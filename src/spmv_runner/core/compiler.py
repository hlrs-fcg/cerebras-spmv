"""CSL compilation using Cerebras SDK compiler."""

import json
import time
from pathlib import Path

from cerebras.sdk.client import SdkCompiler

from utils.logging import logger


def compile_csl(
    width: int,
    height: int,
    total_nnz_per_pe: int,
    max_rows_per_pe: int,
    arch: str = "wse3",
    fabric_dims: tuple = (762, 1172),
    fabric_offsets: tuple = (4, 1),
    artifact_path: str = "artifact.json",
    out_path: str = "out"
):
    logger.info(f"Compiling CSL for {width}×{height} PE grid...")

    # Resolve paths - find project root (where run.py is)
    # Assuming this file is in spmv_runner/core/, project root is 2 levels up
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    src_path = str(project_root / "src")

    out_path = str(Path(out_path).resolve())
    Path(out_path).mkdir(parents=True, exist_ok=True)

    ARGS = (
        f"--fabric-dims={fabric_dims[0]},{fabric_dims[1]} "
        f"--fabric-offsets={fabric_offsets[0]},{fabric_offsets[1]} "
        f"--memcpy --channels=1 "
        f"--arch=wse3 "
        f"--params=width:{width},height:{height},"
        f"TOTAL_NNZ:{total_nnz_per_pe},MAX_ROWS:{max_rows_per_pe}"
    )

    compile_start = time.time()

    logger.info(f"DEBUG: Compiling with app_path={src_path}")
    logger.info("DEBUG: CSL main file: csl/layout.csl")
    logger.info(f"DEBUG: Compiler options: {ARGS}")
    logger.info(f"DEBUG: Output path: {out_path}")

    with SdkCompiler(resource_cpu=24000, resource_mem=64<<30, disable_version_check=True) as compiler:
        artifact_id = compiler.compile(
            app_path=src_path,
            csl_main="csl/layout.csl",
            options=ARGS,
            out_path=out_path,
        )

    compile_time = time.time() - compile_start

    logger.info(f"DEBUG: Compilation returned artifact_id={artifact_id}")
    logger.info(f"DEBUG: Artifact ID type: {type(artifact_id)}")
    logger.info(f"DEBUG: Artifact ID length: {len(artifact_id) if artifact_id else 0}")

    # Check what's in the output directory
    out_dir = Path(out_path)
    if out_dir.exists():
        files = list(out_dir.iterdir())
        logger.info(f"DEBUG: Files in output directory ({out_path}): {[f.name for f in files[:10]]}")
        logger.info(f"DEBUG: Total files in output: {len(files)}")
    else:
        logger.warning(f"DEBUG: Output directory {out_path} does not exist!")

    if artifact_id is None:
        logger.error("Compilation failed! No artifact was generated!")
        raise RuntimeError("CSL compilation failed")

    # Ensure all values are Python native types for JSON serialization
    artifact_data = {
        "artifact_id": artifact_id,
        "arch": "wse3",
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

    logger.info(f"DEBUG: Saved artifact metadata to {artifact_file}")
    logger.info(f"DEBUG: Artifact data: {json.dumps(artifact_data, indent=2)}")

    return out_path, artifact_id, compile_time
