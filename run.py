#!/usr/bin/env python3
"""
SpMV CSL Implementation - Multi-PE Support
Phase 2: Extends single PE to multiple PEs with timing and memory checks
"""

import argparse
import numpy as np
import logging
import time
from enum import Enum

from spmv_prototype import generate_random_sparse_matrix, MatchedPairsSpMV, validate_results


class ExitReason(Enum):
    """Exit reasons for benchmark runs."""
    SUCCESS = "SUCCESS"
    MEMORY_EXCEEDED = "MEMORY_EXCEEDED"
    COMPILATION_FAILED = "COMPILATION_FAILED"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    TIMEOUT = "TIMEOUT"

from tests.benchmark import BenchmarkResult, save_benchmark

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
    parser.add_argument('--save-benchmark', type=str, default='tests/benchmark_results.csv',
                       help='Path to save benchmark results CSV (use empty string to disable)')
    parser.add_argument('--save-results', type=str, default='',
                       help='Path to save result vectors as .npz file (computed, reference, input matrix/vector)')
    args = parser.parse_args()

    # Initialize state variables
    total_start = time.time()
    exit_reason = ExitReason.SUCCESS
    error_message = ""

    # Variables that may not be initialized if early failure
    A = None
    x = None
    num_pes = 1
    width = height = 1
    fabric_dims = (762, 1172)
    compile_time = 0.0
    total_nnz_padded = 0
    max_rows_padded = 0
    max_memory = 0
    avg_memory = 0.0
    memory_utilization = 0.0
    passed = False
    timing_info = None
    pe_data_temp = []

    try:
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
        try:
            logger.info(f"Matrix: {args.rows}×{args.cols}, density={args.density}")
            A = generate_random_sparse_matrix(args.rows, args.cols, args.density, args.seed)
            x = np.random.randint(0, 100, size=args.cols, dtype=np.int32).astype(np.float32)
            logger.info(f"NNZ: {A.nnz}")
        except Exception as e:
            exit_reason = ExitReason.RUNTIME_ERROR
            error_message = f"Matrix generation failed: {str(e)}"
            logger.error(error_message)
            return 1

        # Show PE suggestions if requested
        if args.suggest_pes:
            suggest_pe_counts(A.nnz, A.shape[0], max_width, max_height)
            return 0

        # Parse num_pes argument (can be 'min', 'max', or number)
        try:
            num_pes = parse_num_pes_arg(args.num_pes, A, max_width, max_height)
        except ValueError as e:
            exit_reason = ExitReason.RUNTIME_ERROR
            error_message = f"PE calculation failed: {str(e)}"
            logger.error(error_message)
            return 1

        logger.info(f"Number of PEs: {num_pes}")

        # Validate num_pes is reasonable
        if num_pes < 1:
            exit_reason = ExitReason.RUNTIME_ERROR
            error_message = "Number of PEs must be at least 1"
            logger.error(error_message)
            return 1

        if num_pes > A.shape[0]:
            logger.warning(f"⚠️  More PEs ({num_pes}) than rows ({A.shape[0]}). Some PEs will be idle.")

        # Calculate optimal PE grid
        try:
            width, height = calculate_pe_grid(num_pes, max_width, max_height)
        except ValueError as e:
            exit_reason = ExitReason.RUNTIME_ERROR
            error_message = f"Grid calculation failed: {str(e)}"
            logger.error(error_message)
            return 1

        # Validate grid fits
        if not validate_grid_fits_fabric(width, height, fabric_dims, fabric_offsets):
            exit_reason = ExitReason.RUNTIME_ERROR
            error_message = f"Grid {width}x{height} does not fit fabric dimensions"
            logger.error(error_message)
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
            exit_reason = ExitReason.MEMORY_EXCEEDED
            error_message = f"PE {offending_pe} requires {max_memory} bytes, exceeds 48KB limit"
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
        try:
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
        except Exception as e:
            exit_reason = ExitReason.COMPILATION_FAILED
            error_message = f"Compilation failed: {str(e)}"
            logger.error(error_message)
            return 1

        if args.compile_only:
            return 0

        # Run on device
        try:
            y_computed, timing_info = run_spmv_on_device(A, x, artifact_id, num_pes, width, height)
        except Exception as e:
            exit_reason = ExitReason.RUNTIME_ERROR
            error_message = f"Device execution failed: {str(e)}"
            logger.error(error_message)
            return 1

        # Validate results
        y_reference = A.dot(x).astype(np.float32)
        passed = validate_results(y_computed, y_reference, tolerance=1e-4)

        if not passed:
            exit_reason = ExitReason.VALIDATION_FAILED
            error_message = "Results do not match reference within tolerance"
            logger.error("❌ Validation failed: results do not match reference")

        # Save result vectors if requested
        if args.save_results:
            try:
                from pathlib import Path
                import scipy.sparse as sp

                result_path = Path(args.save_results)
                result_path.parent.mkdir(parents=True, exist_ok=True)

                # Save as npz with all relevant data
                np.savez(
                    result_path,
                    y_computed=y_computed,
                    y_reference=y_reference,
                    x=x,
                    # Save sparse matrix in CSR format
                    matrix_data=A.data,
                    matrix_indices=A.indices,
                    matrix_indptr=A.indptr,
                    matrix_shape=A.shape,
                    # Metadata
                    num_pes=num_pes,
                    validation_passed=passed,
                    relative_error=np.linalg.norm(y_computed - y_reference) / np.linalg.norm(y_reference) if np.linalg.norm(y_reference) > 0 else 0
                )
                logger.info(f"Results saved to: {result_path}")
            except Exception as e:
                logger.warning(f"Failed to save results: {e}")

        total_time = time.time() - total_start

        print_timing_summary(compile_time, timing_info, total_time, A.nnz)

        # Calculate load balancing statistics
        nnz_counts = [pe['total_nnz'] for pe in pe_data_temp if pe['total_nnz'] > 0]
        if nnz_counts:
            min_nnz_pe = min(nnz_counts)
            max_nnz_pe = max(nnz_counts)
            avg_nnz_pe = np.mean(nnz_counts)
            load_imbalance = (max_nnz_pe - min_nnz_pe) / avg_nnz_pe * 100 if avg_nnz_pe > 0 else 0
        else:
            min_nnz_pe = max_nnz_pe = avg_nnz_pe = load_imbalance = 0

        # Calculate memory statistics
        avg_memory = np.mean([check_memory_feasibility([pe])[1] for pe in pe_data_temp])
        memory_utilization = (max_memory / 48000) * 100 if max_memory > 0 else 0

        # Count active vs idle PEs
        num_active = sum(1 for pe in pe_data_temp if pe['num_rows'] > 0)
        num_idle = num_pes - num_active

        # Calculate performance metrics
        flops = 2 * A.nnz
        gflops_hw = (flops / timing_info['compute_hw_avg']) / 1e9 if timing_info['compute_hw_avg'] > 0 else 0
        gflops_python = (flops / timing_info['compute_python']) / 1e9 if timing_info['compute_python'] > 0 else 0

    except Exception as e:
        # Catch any unexpected exceptions
        if exit_reason == ExitReason.SUCCESS:
            exit_reason = ExitReason.RUNTIME_ERROR
            error_message = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        passed = False

        # Initialize defaults for missing variables
        if A is None:
            A = type('obj', (object,), {'nnz': 0, 'shape': (args.rows, args.cols)})()
        if timing_info is None:
            timing_info = {
                'memcpy_h2d': 0, 'compute_python': 0, 'compute_hw_min': 0,
                'compute_hw_avg': 0, 'compute_hw_max': 0, 'compute_hw_std': 0,
                'compute_hw_cycles_min': 0, 'compute_hw_cycles_avg': 0,
                'compute_hw_cycles_max': 0, 'compute_hw_cycles_std': 0,
                'memcpy_d2h': 0, 'total_run': 0
            }
        if not pe_data_temp:
            pe_data_temp = [{'total_nnz': 0, 'num_rows': 0}]

        min_nnz_pe = max_nnz_pe = avg_nnz_pe = load_imbalance = 0
        num_active = num_idle = 0
        gflops_hw = gflops_python = 0

    finally:
        # Always save benchmark results
        if args.save_benchmark and A is not None:
            total_time = time.time() - total_start

            # Ensure we have values for all fields
            if timing_info is None:
                timing_info = {
                    'memcpy_h2d': 0, 'compute_python': 0, 'compute_hw_min': 0,
                    'compute_hw_avg': 0, 'compute_hw_max': 0, 'compute_hw_std': 0,
                    'compute_hw_cycles_min': 0, 'compute_hw_cycles_avg': 0,
                    'compute_hw_cycles_max': 0, 'compute_hw_cycles_std': 0,
                    'memcpy_d2h': 0, 'total_run': 0
                }

            if not pe_data_temp:
                min_nnz_pe = max_nnz_pe = avg_nnz_pe = load_imbalance = 0
                num_active = num_idle = 0
                gflops_hw = gflops_python = 0
            else:
                nnz_counts = [pe['total_nnz'] for pe in pe_data_temp if pe['total_nnz'] > 0]
                if nnz_counts:
                    min_nnz_pe = min(nnz_counts)
                    max_nnz_pe = max(nnz_counts)
                    avg_nnz_pe = np.mean(nnz_counts)
                    load_imbalance = (max_nnz_pe - min_nnz_pe) / avg_nnz_pe * 100 if avg_nnz_pe > 0 else 0
                else:
                    min_nnz_pe = max_nnz_pe = avg_nnz_pe = load_imbalance = 0

                avg_memory = np.mean([check_memory_feasibility([pe])[1] for pe in pe_data_temp]) if pe_data_temp else 0
                memory_utilization = (max_memory / 48000) * 100 if max_memory > 0 else 0
                num_active = sum(1 for pe in pe_data_temp if pe['num_rows'] > 0)
                num_idle = num_pes - num_active

                if timing_info and timing_info.get('compute_hw_avg', 0) > 0:
                    flops = 2 * A.nnz
                    gflops_hw = (flops / timing_info['compute_hw_avg']) / 1e9
                    gflops_python = (flops / timing_info['compute_python']) / 1e9 if timing_info['compute_python'] > 0 else 0
                else:
                    gflops_hw = gflops_python = 0

            benchmark = BenchmarkResult(
                validation_passed=passed,
                exit_reason=exit_reason.value,
                error_message=error_message,
                matrix_rows=args.rows,
                matrix_cols=args.cols,
                density=args.density,
                total_nnz=A.nnz if hasattr(A, 'nnz') else 0,
                seed=args.seed,
                num_pes=num_pes,
                grid_width=width,
                grid_height=height,
                architecture=args.arch,
                fabric_dims_w=fabric_dims[0],
                fabric_dims_h=fabric_dims[1],
                max_memory_per_pe=max_memory,
                avg_memory_per_pe=avg_memory,
                memory_limit_per_pe=48000,
                memory_utilization_pct=memory_utilization,
                compile_time=compile_time,
                memcpy_h2d_time=timing_info.get('memcpy_h2d', 0),
                compute_time_python=timing_info.get('compute_python', 0),
                compute_time_hw_min=timing_info.get('compute_hw_min', 0),
                compute_time_hw_avg=timing_info.get('compute_hw_avg', 0),
                compute_time_hw_max=timing_info.get('compute_hw_max', 0),
                compute_time_hw_std=timing_info.get('compute_hw_std', 0),
                compute_hw_cycles_min=timing_info.get('compute_hw_cycles_min', 0),
                compute_hw_cycles_avg=timing_info.get('compute_hw_cycles_avg', 0),
                compute_hw_cycles_max=timing_info.get('compute_hw_cycles_max', 0),
                compute_hw_cycles_std=timing_info.get('compute_hw_cycles_std', 0),
                memcpy_d2h_time=timing_info.get('memcpy_d2h', 0),
                total_run_time=timing_info.get('total_run', 0),
                total_time=total_time,
                gflops_hw=gflops_hw,
                gflops_python=gflops_python,
                min_nnz_per_pe=min_nnz_pe,
                max_nnz_per_pe=max_nnz_pe,
                avg_nnz_per_pe=avg_nnz_pe,
                load_imbalance_pct=load_imbalance,
                num_active_pes=num_active,
                num_idle_pes=num_idle,
                total_nnz_padded=total_nnz_padded,
                max_rows_padded=max_rows_padded
            )

            try:
                save_benchmark(benchmark, csv_path=args.save_benchmark)
            except Exception as e:
                logger.warning(f"Failed to save benchmark: {e}")

    if not passed:
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
