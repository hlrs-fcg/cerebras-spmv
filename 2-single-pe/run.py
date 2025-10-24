#!/usr/bin/env python3
"""
SpMV CSL Implementation - Single PE Test
Phase 2: Minimal working example with SdkCompiler
"""

import argparse
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import json
import logging
import sys
import time

from cerebras.sdk.client import SdkRuntime, SdkCompiler
from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder

# Import our prototype code
from spmv_prototype import generate_random_sparse_matrix, MatchedPairsSpMV, validate_results

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    #level = logging.DEBUG if args.verbose else logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def compile_csl(
    width: int,
    height: int,
    total_nnz: int,
    max_rows: int,
    arch: str = "wse3",
    fabric_dims: tuple = (762, 1172),
    fabric_offsets: tuple = (4, 1),
    artifact_path: str = "artifact.json",
    out_path: str = "out"
):
    """Compile the CSL code using SdkCompiler."""
    logger.info("="*60)
    logger.info("Compiling CSL code")
    logger.info("="*60)
    logger.info(f"  PE grid: {width}x{height}")
    logger.info(f"  TOTAL_NNZ: {total_nnz}")
    logger.info(f"  MAX_ROWS: {max_rows}")
    logger.info(f"  Architecture: {arch}")
    logger.info(f"  Fabric dims: {fabric_dims}")

    # Ensure out_path exists and is absolute
    out_path = str(Path(out_path).resolve())
    Path(out_path).mkdir(parents=True, exist_ok=True)

    logger.info(f"  Output path: {out_path}")

    # Build compilation arguments
    ARGS = (
        f"--arch={arch} "
        f"--fabric-dims={fabric_dims[0]},{fabric_dims[1]} "
        f"--fabric-offsets={fabric_offsets[0]},{fabric_offsets[1]} "
        f"--memcpy --channels=1 "
        f"--params=width:{width},height:{height},"
        f"TOTAL_NNZ:{total_nnz},MAX_ROWS:{max_rows}"
    )

    logger.info(f"  Compilation args: {ARGS}")

    # Compile using SdkCompiler
    with SdkCompiler(resource_cpu=24000, resource_mem=64<<30, disable_version_check=True) as compiler:
        artifact_id = compiler.compile(
            app_path=".",  # Current directory
            csl_main="layout.csl",
            options=ARGS,
            out_path=out_path,
        )

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
            "TOTAL_NNZ": total_nnz,
            "MAX_ROWS": max_rows,
        }
    }

    artifact_file = Path(artifact_path)
    with open(artifact_file, "w", encoding="utf8") as f:
        json.dump(artifact_data, f, indent=2)

    logger.info("✅ Compilation successful!")
    logger.info(f"  Artifact ID: {artifact_id}")
    logger.info(f"  Artifact saved to: {artifact_file.resolve()}")
    logger.info("="*60 + "\n")

    return out_path, artifact_id  


def run_spmv_on_device(
    csr_matrix: sp.csr_matrix,
    vector_x: np.ndarray,
    artifact_path: str,
    num_pes: int = 1
):
    """Run SpMV on the Cerebras device (or simulator)."""

    # For now, we only support single PE
    assert num_pes == 1, "Multi-PE not yet implemented"

    # Distribute data (even though it's 1 PE, we use same logic)
    spmv = MatchedPairsSpMV(csr_matrix, num_pes)
    pe_data = spmv.distribute_by_nnz()
    spmv.prepare_matched_pairs(pe_data, vector_x)

    # Get data for PE 0
    pe = pe_data[0]

    logger.info("="*60)
    logger.info("PE 0 Data Summary")
    logger.info("="*60)
    logger.info(f"  Rows: {pe['num_rows']}")
    logger.info(f"  NNZ: {pe['total_nnz']}")
    logger.info(f"  Row offsets shape: {pe['row_offsets'].shape}")
    logger.info(f"  Matrix values shape: {pe['matrix_values'].shape}")
    logger.info(f"  Vector values shape: {pe['vector_values'].shape}")
    logger.info("="*60 + "\n")

    # Initialize SDK runtime
    logger.info("Initializing SdkRuntime...")
    runtime_start = time.perf_counter()
    with SdkRuntime(artifact_path, suppress_simfab_trace=True, disable_version_check=True) as runner:

        # Get symbol IDs
        sym_num_rows = runner.get_id("num_rows")
        sym_row_offsets = runner.get_id("row_offsets")
        sym_matrix_values = runner.get_id("matrix_values")
        sym_vector_values = runner.get_id("vector_values")
        sym_results = runner.get_id("results")

        # Send data to PE
        logger.info("Sending data to PE...")

        runner.memcpy_h2d(
            sym_num_rows,
            np.array([pe['num_rows']], dtype=np.int32),  
            0, 0, 1, 1,
            1,  # elt_per_pe: 1 element
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT,  # Changed to 32BIT
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False
        )

        # Send row_offsets
        runner.memcpy_h2d(
            sym_row_offsets,
            pe['row_offsets'].astype(np.int32),
            0, 0, 1, 1,
            len(pe['row_offsets']),
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT,
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False
        )

        # Send matrix_values
        runner.memcpy_h2d(
            sym_matrix_values,
            pe['matrix_values'],
            0, 0, 1, 1,
            len(pe['matrix_values']),  # elt_per_pe: number of matrix values
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT,
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False
        )

        # Send vector_values
        runner.memcpy_h2d(
            sym_vector_values,
            pe['vector_values'],
            0, 0, 1, 1,
            len(pe['vector_values']),  # elt_per_pe: number of vector values
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT,
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False
        )

        logger.info("✅ Data sent to PE")

        # Launch computation
        logger.info("Launching SpMV computation...")
        runner.launch("spmv", nonblock=False)
        logger.info("✅ Computation complete")

        # Retrieve results
        logger.info("Retrieving results...")
        results_raw = np.zeros(pe['num_rows'], dtype=np.float32)
        runner.memcpy_d2h(
            results_raw,
            sym_results,
            0, 0, 1, 1,
            pe['num_rows'],  # elt_per_pe: number of results
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT,
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False
        )

        elapsed = time.perf_counter() - runtime_start
        print(f"runtime took {elapsed:.4f}s")


        logger.info("✅ Results retrieved")

    # Reconstruct full result vector
    result_vector = np.zeros(csr_matrix.shape[0], dtype=np.float32)
    for local_idx, global_row in enumerate(pe['row_ids']):
        result_vector[global_row] = results_raw[local_idx]

    return result_vector


def main():
    program_start = time.perf_counter()
    parser = argparse.ArgumentParser(description='SpMV CSL Single PE Test')
    parser.add_argument('--rows', type=int, default=10, help='Number of rows')
    parser.add_argument('--cols', type=int, default=10, help='Number of columns')
    parser.add_argument('--density', type=float, default=0.1, help='Sparsity density')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--compile-only', action='store_true',
                       help='Only compile, do not run')
    parser.add_argument('--artifact-path', type=str, default='artifact.json',
                       help='Path to save artifact metadata')
    parser.add_argument('--out-path', type=str, default='out',
                       help='Compilation output directory')
    parser.add_argument('--arch', type=str, default='wse3',
                       choices=['wse2', 'wse3'],
                       help='Target architecture')
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("SpMV CSL Single PE Test")
    logger.info("="*60 + "\n")

    # Generate test matrix
    logger.info(f"Generating random matrix: {args.rows}x{args.cols}, density={args.density}")
    A = generate_random_sparse_matrix(args.rows, args.cols, args.density, args.seed)
    x = np.random.randint(0, 100, size=args.cols, dtype=np.int32).astype(np.float32)

    logger.info(f"Matrix: {A.shape}, NNZ: {A.nnz}")
    logger.info(f"Vector: {x.shape}\n")

    # Determine required sizes
    # For single PE, all data goes to one PE
    total_nnz = A.nnz
    max_rows = A.shape[0]

    # Add some padding for safety
    total_nnz_padded = max(int(total_nnz * 1.2) + 10, 10)  # At least 10
    max_rows_padded = max_rows + 5

    logger.info(f"CSL Parameters:")
    logger.info(f"  TOTAL_NNZ: {total_nnz_padded} (actual: {total_nnz})")
    logger.info(f"  MAX_ROWS: {max_rows_padded} (actual: {max_rows})\n")

    # Compile CSL
    out_path, artifact_id = compile_csl(
        width=1,
        height=1,
        total_nnz=total_nnz_padded,  # Use the padded version
        max_rows=max_rows_padded,    # Use the padded version
        arch=args.arch,
        artifact_path=args.artifact_path,
        out_path=args.out_path
    )

    if args.compile_only:
        logger.info("Compilation complete. Exiting (--compile-only specified).")
        return 0

    # Run on device
    logger.info("Running SpMV on device...\n")
    y_computed = run_spmv_on_device(A, x, artifact_id, num_pes=1)

    # Compute reference
    logger.info("Computing reference solution...")
    y_reference = A.dot(x).astype(np.float32)

    # Validate
    passed = validate_results(y_computed, y_reference, tolerance=1e-4)

    program_time = time.perf_counter() - program_start
    print(f"program took: {program_time:.4f}s")

    if passed:
        logger.info("\n🎉 SUCCESS! SpMV on CSL works correctly!")
        return 0
    else:
        logger.error("\n❌ FAILED! Results do not match.")
        return 1


if __name__ == "__main__":
    exit(main())
