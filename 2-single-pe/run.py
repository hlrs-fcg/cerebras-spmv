#!/usr/bin/env python3
"""
SpMV CSL Implementation - Single PE Test
Phase 2: Minimal working example
"""

import argparse
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import subprocess
import json

from cerebras.sdk.runtime.sdkruntimepym import SdkRuntime, MemcpyDataType, MemcpyOrder

# Import our prototype code
import sys
sys.path.append(str(Path(__file__).parent))
from spmv_prototype import generate_random_sparse_matrix, MatchedPairsSpMV, validate_results


def compile_csl(
    width: int,
    height: int,
    total_nnz: int,
    max_rows: int,
    fabric_dims: tuple = (8, 3),
    fabric_offsets: tuple = (4, 1),
    compile_dir: str = "out"
):
    """Compile the CSL code."""
    print(f"Compiling CSL code...")
    print(f"  PE grid: {width}x{height}")
    print(f"  TOTAL_NNZ: {total_nnz}")
    print(f"  MAX_ROWS: {max_rows}")
    
    # Build compilation command
    cmd = [
        "cslc",
        "./layout.csl",
        f"--fabric-dims={fabric_dims[0]},{fabric_dims[1]}",
        f"--fabric-offsets={fabric_offsets[0]},{fabric_offsets[1]}",
        f"--params=width:{width}",
        f"--params=height:{height}",
        f"--params=TOTAL_NNZ:{total_nnz}",
        f"--params=MAX_ROWS:{max_rows}",
        f"-o={compile_dir}",
        "--memcpy",
        "--channels=1",
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Compilation failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError("CSL compilation failed")
    
    print("✅ Compilation successful!")
    return compile_dir


def run_spmv_on_device(
    csr_matrix: sp.csr_matrix,
    vector_x: np.ndarray,
    compile_dir: str,
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
    
    print(f"\n{'='*60}")
    print(f"PE 0 Data:")
    print(f"  Rows: {pe['num_rows']}")
    print(f"  NNZ: {pe['total_nnz']}")
    print(f"  Row offsets shape: {pe['row_offsets'].shape}")
    print(f"  Matrix values shape: {pe['matrix_values'].shape}")
    print(f"  Vector values shape: {pe['vector_values'].shape}")
    print(f"{'='*60}\n")
    
    # Initialize SDK runtime
    print("Initializing SdkRuntime...")
    runner = SdkRuntime(compile_dir, suppress_simfab_trace=True)
    
    # Load the program
    print("Loading program...")
    runner.load()
    runner.run()
    
    # Send data to PE
    print("Sending data to PE...")
    
    # Send num_rows
    runner.memcpy_h2d(
        "num_rows",
        np.array([pe['num_rows']], dtype=np.int16),
        0, 0, 1, 1,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_16BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False
    )
    
    # Send row_offsets
    runner.memcpy_h2d(
        "row_offsets",
        pe['row_offsets'],
        0, 0, 1, 1,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_16BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False
    )
    
    # Send matrix_values
    runner.memcpy_h2d(
        "matrix_values",
        pe['matrix_values'],
        0, 0, 1, 1,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False
    )
    
    # Send vector_values
    runner.memcpy_h2d(
        "vector_values",
        pe['vector_values'],
        0, 0, 1, 1,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False
    )
    
    print("✅ Data sent to PE")
    
    # Launch computation
    print("Launching SpMV computation...")
    runner.launch("spmv", nonblock=False)
    print("✅ Computation complete")
    
    # Retrieve results
    print("Retrieving results...")
    results_raw = np.zeros(pe['num_rows'], dtype=np.float32)
    runner.memcpy_d2h(
        results_raw,
        "results",
        0, 0, 1, 1,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False
    )
    
    print("✅ Results retrieved")
    
    # Stop the runner
    runner.stop()
    
    # Reconstruct full result vector
    result_vector = np.zeros(csr_matrix.shape[0], dtype=np.float32)
    for local_idx, global_row in enumerate(pe['row_ids']):
        result_vector[global_row] = results_raw[local_idx]
    
    return result_vector


def main():
    parser = argparse.ArgumentParser(description='SpMV CSL Single PE Test')
    parser.add_argument('--rows', type=int, default=10, help='Number of rows')
    parser.add_argument('--cols', type=int, default=10, help='Number of columns')
    parser.add_argument('--density', type=float, default=0.1, help='Sparsity density')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--compile-only', action='store_true', 
                       help='Only compile, do not run')
    parser.add_argument('--compile-dir', type=str, default='out',
                       help='Compilation output directory')
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print("SpMV CSL Single PE Test")
    print(f"{'='*60}\n")
    
    # Generate test matrix
    print(f"Generating random matrix: {args.rows}x{args.cols}, density={args.density}")
    A = generate_random_sparse_matrix(args.rows, args.cols, args.density, args.seed)
    x = np.random.randint(0, 100, size=args.cols, dtype=np.int32).astype(np.float32)
    
    print(f"Matrix: {A.shape}, NNZ: {A.nnz}")
    print(f"Vector: {x.shape}\n")
    
    # Determine required sizes
    # For single PE, all data goes to one PE
    total_nnz = A.nnz
    max_rows = A.shape[0]
    
    # Add some padding for safety
    total_nnz_padded = int(total_nnz * 1.2) + 10
    max_rows_padded = max_rows + 5
    
    print(f"CSL Parameters:")
    print(f"  TOTAL_NNZ: {total_nnz_padded} (actual: {total_nnz})")
    print(f"  MAX_ROWS: {max_rows_padded} (actual: {max_rows})\n")
    
    # Compile CSL
    compile_dir = compile_csl(
        width=1,
        height=1,
        total_nnz=total_nnz_padded,
        max_rows=max_rows_padded,
        compile_dir=args.compile_dir
    )
    
    if args.compile_only:
        print("Compilation complete. Exiting (--compile-only specified).")
        return 0
    
    # Run on device
    print("\nRunning SpMV on device...")
    y_computed = run_spmv_on_device(A, x, compile_dir, num_pes=1)
    
    # Compute reference
    print("Computing reference solution...")
    y_reference = A.dot(x).astype(np.float32)
    
    # Validate
    passed = validate_results(y_computed, y_reference, tolerance=1e-4)
    
    if passed:
        print("\n🎉 SUCCESS! SpMV on CSL works correctly!")
        return 0
    else:
        print("\n❌ FAILED! Results do not match.")
        return 1


if __name__ == "__main__":
    exit(main())
