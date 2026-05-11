"""Thin CSL compiler wrapper."""

import json

from spmv_runner.core import compile_csl
from config import SpMVConfig
from utils.logging import logger


def compile(cfg: SpMVConfig) -> None:
    """Compile CSL for cfg. Sets cfg.artifact_id on success."""
    logger.info(f"Compiling for {cfg.width}×{cfg.height} grid ({cfg.num_pes} PEs)")
    logger.info(f"  TOTAL_NNZ_PER_PE: {cfg.total_nnz_per_pe}, MAX_ROWS_PER_PE: {cfg.max_rows_per_pe}")

    cfg.out_path.mkdir(parents=True, exist_ok=True)

    _, artifact_id, compile_time = compile_csl(
        width=cfg.width,
        height=cfg.height,
        total_nnz_per_pe=cfg.total_nnz_per_pe,
        max_rows_per_pe=cfg.max_rows_per_pe,
        arch=cfg.arch,
        fabric_dims=cfg.fabric_dims,
        fabric_offsets=cfg.fabric_offsets,
        artifact_path=str(cfg.out_path / "artifact.json"),
        out_path=str(cfg.out_path),
    )

    logger.info(f"Compilation done in {compile_time:.2f}s — artifact: {artifact_id}")

    cfg.artifact_id = artifact_id

    with open(cfg.out_path / "artifact.json", "w", encoding="utf8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    logger.info(f"Config saved to {cfg.out_path / 'artifact.json'}")
