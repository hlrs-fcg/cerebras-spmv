"""
CSL Compilation Module

Handles compilation of CSL code using the Cerebras SDK compiler.
"""

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
    arch: str = "wse3",
    fabric_dims: tuple = (762, 1172),
    fabric_offsets: tuple = (4, 1),
    artifact_path: str = "artifact.json",
    out_path: str = "out"
):
    """Compile the CSL code using SdkCompiler."""
    logger.info("Compiling CSL...")
    logger.debug(f"  PE grid: {width}×{height}")
    logger.debug(f"  TOTAL_NNZ per PE: {total_nnz_per_pe}")
    logger.debug(f"  MAX_ROWS per PE: {max_rows_per_pe}")
    logger.debug(f"  Architecture: {arch}")
    logger.debug(f"  Fabric dims: {fabric_dims}")

    # Ensure out_path exists and is absolute
    out_path = str(Path(out_path).resolve())
    Path(out_path).mkdir(parents=True, exist_ok=True)

    logger.debug(f"  Output path: {out_path}")

    # Build compilation arguments
    ARGS = (
        f"--arch={arch} "
        f"--fabric-dims={fabric_dims[0]},{fabric_dims[1]} "
        f"--fabric-offsets={fabric_offsets[0]},{fabric_offsets[1]} "
        f"--memcpy --channels=1 "
        f"--params=width:{width},height:{height},"
        f"TOTAL_NNZ:{total_nnz_per_pe},MAX_ROWS:{max_rows_per_pe}"
    )

    logger.debug(f"  Compilation args: {ARGS}")

    # Compile using SdkCompiler
    compile_start = time.time()

    with SdkCompiler(resource_cpu=24000, resource_mem=64<<30, disable_version_check=True) as compiler:
        artifact_id = compiler.compile(
            app_path="src",
            csl_main="layout.csl",
            options=ARGS,
            out_path=out_path,
        )

    compile_time = time.time() - compile_start

    if artifact_id is None:
        logger.error("Compilation failed! No artifact was generated!")
        raise RuntimeError("CSL compilation failed")

    # Save artifact metadata
    artifact_data = {
        "artifact_id": artifact_id,
        "fabric_dims": list(fabric_dims),
        "params": {
            "width": width,
            "height": height,
            "TOTAL_NNZ": total_nnz_per_pe,
            "MAX_ROWS": max_rows_per_pe,
        },
        "compile_time_seconds": compile_time
    }

    artifact_file = Path(artifact_path)
    with open(artifact_file, "w", encoding="utf8") as f:
        json.dump(artifact_data, f, indent=2)

    logger.info(f"✅ Compilation successful ({compile_time:.2f}s)")
    logger.debug(f"  Artifact ID: {artifact_id}")
    logger.debug(f"  Artifact saved to: {artifact_file.resolve()}")

    return out_path, artifact_id, compile_time
