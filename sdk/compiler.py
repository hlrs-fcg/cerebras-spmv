"""CSL compilation via Cerebras SDK."""

import json
import time
from pathlib import Path

from config import SpMVConfig
from utils.logging import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def compile(cfg: SpMVConfig) -> None:
    """Compile CSL for cfg and save artifact.json. Sets cfg.artifact_id on success."""
    logger.info(f"Compiling for {cfg.width}×{cfg.height} grid ({cfg.num_pes:,} PEs)")
    logger.info(f"  KX={cfg.kx}, KY={cfg.ky}, TOTAL_NNZ={cfg.total_nnz_per_pe}")

    cfg.out_path.mkdir(parents=True, exist_ok=True)
    out_path = str(cfg.out_path.resolve())

    args = (
        f"--fabric-dims={cfg.fabric_dims[0]},{cfg.fabric_dims[1]} "
        f"--fabric-offsets={cfg.fabric_offsets[0]},{cfg.fabric_offsets[1]} "
        f"--memcpy --channels=1 "
        f"--arch={cfg.arch} "
        f"--params=width:{cfg.width},height:{cfg.height},"
        f"KX:{cfg.kx},KY:{cfg.ky},TOTAL_NNZ:{cfg.total_nnz_per_pe}"
    )

    from cerebras.sdk.client import SdkCompiler

    t0 = time.time()
    with SdkCompiler(resource_cpu=24000, resource_mem=64 << 30, disable_version_check=True) as compiler:
        artifact_id = compiler.compile(
            app_path=str(_PROJECT_ROOT),
            csl_main="src/csl/layout.csl",
            options=args,
            out_path=out_path,
        )
    compile_time = time.time() - t0

    if artifact_id is None:
        raise RuntimeError("CSL compilation failed — no artifact produced")

    cfg.artifact_id = artifact_id
    logger.info(f"Compiled in {compile_time:.2f}s — artifact: {artifact_id}")

    with open(cfg.out_path / "artifact.json", "w", encoding="utf8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    logger.info(f"Artifact saved to {cfg.out_path / 'artifact.json'}")
