#!/usr/bin/env python3
"""
SpMV CSL Implementation - Multi-PE Support
Phase 2: Extends single PE to multiple PEs with timing and memory checks
"""

import argparse
import numpy as np
import logging
import time

# Import our prototype code
from spmv_prototype import generate_random_sparse_matrix, MatchedPairsSpMV, validate_results

# Import modularized components
from spmv_runner.core import (
    calculate_pe_grid,
    print_grid_info,
    validate_grid_fits_fabric,
    compile_csl,
    run_spmv_on_device
)

from spmv_runner.utils import (
    check_memory_feasibility,
    print_memory_analysis,
    calculate_min_pes_needed,
    parse_num_pes_arg,
    suggest_pe_counts,
    print_timing_summary
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='SpMV CSL Multi-PE Implementation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-calculate minimum PEs needed
  python run.py --rows 1000 --cols 1000 --density 0.01 --num-pes min

  # Auto-calculate maximum useful PEs
  python run.py --rows 1000 --cols 1000 --density 0.01 --num-pes max

  # Specific number of PEs
  python run.py --rows 1000 --cols 1000 --density 0.01 --num-pes 16

  # Show PE count suggestions
  python run.py --rows 1000 --cols 1000 --density 0.01 --suggest-pes
        """
    )
    parser.add_argument('--rows', type=int, default=10, help='Number of rows')
    parser.add_argument('--cols', type=int, default=10, help='Number of columns')
    parser.add_argument('--density', type=float, default=0.1, help='Sparsity density')
    parser.add_argument('--num-pes', type=str, default='1',
                       help='Number of PEs: integer, "min", or "max"')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--suggest-pes', action='store_true',
                       help='Show PE count suggestions and exit')
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

    total_start = time.time()

    logger.info("SpMV - Multi-PE Implementation")

    # Fabric constraints based on architecture
    if args.arch == 'wse3':
        fabric_dims = (762, 1172)
        fabric_offsets = (4, 1)
    else:  # wse2
        fabric_dims = (757, 996)
        fabric_offsets = (4, 1)

    max_width = fabric_dims[0] - fabric_offsets[0]
    max_height = fabric_dims[1] - fabric_offsets[1]

    # Generate test matrix
    logger.info(f"Matrix: {args.rows}×{args.cols}, density={args.density}")
    A = generate_random_sparse_matrix(args.rows, args.cols, args.density, args.seed)
    x = np.random.randint(0, 100, size=args.cols, dtype=np.int32).astype(np.float32)
    logger.info(f"NNZ: {A.nnz}")

    # Show PE suggestions if requested
    if args.suggest_pes:
        suggest_pe_counts(A.nnz, A.shape[0], max_width, max_height)
        return 0

    # Parse num_pes argument (can be 'min', 'max', or number)
    try:
        num_pes = parse_num_pes_arg(args.num_pes, A, max_width, max_height)
    except ValueError as e:
        logger.error(str(e))
        return 1

    logger.info(f"Number of PEs: {num_pes}")

    # Validate num_pes is reasonable
    if num_pes < 1:
        logger.error("Number of PEs must be at least 1")
        return 1

    if num_pes > A.shape[0]:
        logger.warning(f"⚠️  More PEs ({num_pes}) than rows ({A.shape[0]}). Some PEs will be idle.")

    # Calculate optimal PE grid
    try:
        width, height = calculate_pe_grid(num_pes, max_width, max_height)
    except ValueError as e:
        logger.error(str(e))
        return 1

    # Validate grid fits
    if not validate_grid_fits_fabric(width, height, fabric_dims, fabric_offsets):
        return 1

    print_grid_info(width, height, num_pes, fabric_dims, fabric_offsets)

    # Estimate required sizes per PE and check memory
    spmv_temp = MatchedPairsSpMV(A, num_pes)
    pe_data_temp = spmv_temp.distribute_by_nnz()
    spmv_temp.prepare_matched_pairs(pe_data_temp, x)

    # Memory feasibility check
    is_feasible, max_memory, offending_pe = check_memory_feasibility(pe_data_temp)

    print_memory_analysis(pe_data_temp)

    if not is_feasible:
        logger.error("="*60)
        logger.error("MEMORY LIMIT EXCEEDED!")
        logger.error("="*60)
        logger.error(f"PE {offending_pe} requires {max_memory} bytes ({max_memory/1024:.2f} KB)")
        logger.error("This exceeds the 48 KB (48,000 bytes) limit per PE.")
        logger.error("\nSuggestions:")
        min_pes_needed = calculate_min_pes_needed(A.nnz, A.shape[0])
        logger.error(f"  1. Use at least {min_pes_needed} PEs (try --num-pes {min_pes_needed})")
        logger.error("  2. Or use --num-pes min for automatic calculation")
        logger.error(f"  3. Reduce matrix size (current: {args.rows}×{args.cols})")
        logger.error(f"  4. Reduce density (current: {args.density})")
        logger.error("="*60)
        return 1

    # Find max across all PEs for compilation
    max_nnz = max(pe['total_nnz'] for pe in pe_data_temp)
    max_rows = max(pe['num_rows'] for pe in pe_data_temp)

    # Add padding
    total_nnz_padded = max(int(max_nnz * 1.2) + 10, 10)
    max_rows_padded = max_rows + 5

    logger.info("CSL Parameters (per PE):")
    logger.info(f"  PE Grid: {width}×{height}")
    logger.info(f"  TOTAL_NNZ: {total_nnz_padded} (max actual: {max_nnz})")
    logger.info(f"  MAX_ROWS: {max_rows_padded} (max actual: {max_rows})\n")

    # Compile CSL
    out_path, artifact_id, compile_time = compile_csl(
        width=width,
        height=height,
        total_nnz_per_pe=total_nnz_padded,
        max_rows_per_pe=max_rows_padded,
        arch=args.arch,
        fabric_dims=fabric_dims,
        fabric_offsets=fabric_offsets,
        artifact_path=args.artifact_path,
        out_path=args.out_path
    )

    if args.compile_only:
        logger.info("Compilation complete. Exiting (--compile-only specified).")
        total_time = time.time() - total_start
        logger.info(f"\nTotal time: {total_time:.2f} seconds")
        return 0

    # Run on device
    logger.info("Running SpMV on device...\n")
    y_computed, timing_info = run_spmv_on_device(A, x, artifact_id, num_pes, width, height)

    # Compute reference
    logger.info("Computing reference solution...")
    y_reference = A.dot(x).astype(np.float32)

    # Validate
    passed = validate_results(y_computed, y_reference, tolerance=1e-4)

    total_time = time.time() - total_start

    # Print timing summary
    print_timing_summary(compile_time, timing_info, total_time)

    if passed:
        logger.info("🎉 SUCCESS! Multi-PE SpMV works correctly!")
        return 0
    else:
        logger.error("❌ FAILED! Results do not match.")
        return 1


if __name__ == "__main__":
    exit(main())
