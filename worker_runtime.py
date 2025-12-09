#!/usr/bin/env python3
"""
Worker Runtime Script
Runs on worker node via SdkLauncher. Executes SpMV on Cerebras hardware.
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from cerebras.sdk.client import SdkRuntime
from cerebras.sdk.client.sdk_utils import calculate_cycles
from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder

from spmv_prototype import generate_random_sparse_matrix, MatchedPairsSpMV, validate_results
from tests.benchmark import BenchmarkResult, save_benchmark

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """Get last commit hash for later reference of the program state"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            timeout=1
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def main():
    """Worker node entry point - runs SpMV via SdkRuntime."""
    parser = argparse.ArgumentParser(
        description='SpMV Worker Runtime (runs on worker node via SdkLauncher)'
    )
    # required by SdkLauncher pattern
    parser.add_argument('--name', required=True, help='Compiled artifact output directory')
    parser.add_argument('--cmaddr', required=True, help='IP:port for Cerebras system')

    # matrix configuration
    parser.add_argument('--rows', type=int, required=True, help='Number of matrix rows')
    parser.add_argument('--cols', type=int, required=True, help='Number of matrix columns')
    parser.add_argument('--density', type=float, required=True, help='Sparsity density')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # PE configuration
    parser.add_argument('--num-pes', type=int, required=True, help='Number of PEs')
    parser.add_argument('--grid-width', type=int, required=True, help='PE grid width')
    parser.add_argument('--grid-height', type=int, required=True, help='PE grid height')

    # architecture
    parser.add_argument('--arch', type=str, default='wse3', choices=['wse2', 'wse3'])

    # benchmark output
    parser.add_argument('--save-benchmark', type=str, default='benchmark_results.csv',
                       help='Path to save benchmark CSV')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    logger.info("="*60)
    logger.info("Worker Runtime - Executing on Worker Node")
    logger.info("="*60)
    logger.info(f"Artifact: {args.name}")
    logger.info(f"System: {args.cmaddr}")

    total_start = time.time()

    # Generate matrix
    logger.info(f"Generating matrix: {args.rows}×{args.cols}, density={args.density}")
    try:
        A = generate_random_sparse_matrix(args.rows, args.cols, args.density, args.seed)
        x = np.random.randint(0, 100, size=args.cols, dtype=np.int32).astype(np.float32)
        logger.info(f"Matrix NNZ: {A.nnz:,}")
    except Exception as e:
        logger.error(f"Matrix generation failed: {e}")
        return 1

    # distribute data across PEs
    logger.info(f"Distributing data across {args.num_pes} PEs...")
    lb_start = time.time()
    spmv = MatchedPairsSpMV(A, args.num_pes)

    distribute_start = time.time()
    pe_data = spmv.distribute_by_nnz()
    distribute_time = time.time() - distribute_start

    prepare_start = time.time()
    spmv.prepare_matched_pairs(pe_data, x)
    prepare_matched_pairs_time = time.time() - prepare_start

    total_lb_time = time.time() - lb_start
    logger.info(f"Load balancing: {total_lb_time*1000:.2f}ms")

    # count active PEs
    num_active = sum(1 for pe in pe_data if pe['num_rows'] > 0)
    logger.info(f"Active PEs: {num_active}/{args.num_pes}")

    # Execute on device using SdkRuntime with cmaddr
    logger.info("Connecting to Cerebras system...")
    try:
        with SdkRuntime(args.name, cmaddr=args.cmaddr) as runner:
            # Get symbol IDs
            sym_num_rows = runner.get_id("num_rows")
            sym_row_offsets = runner.get_id("row_offsets")
            sym_matrix_values = runner.get_id("matrix_values")
            sym_vector_values = runner.get_id("vector_values")
            sym_results = runner.get_id("results")
            sym_time_buf = runner.get_id("time_buf_f32")

            # Load and start
            logger.info("Loading program...")
            runner.load()
            runner.run()

            # H2D memcpy
            logger.info("Transferring data to device (H2D)...")
            memcpy_h2d_start = time.time()

            for pe in pe_data:
                pe_id = pe['pe_id']
                x_coord = pe_id % args.grid_width
                y_coord = pe_id // args.grid_width

                # Send num_rows
                runner.memcpy_h2d(
                    sym_num_rows,
                    np.array([pe['num_rows']], dtype=np.int32),
                    x_coord, y_coord, 1, 1, 1,
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_32BIT,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False
                )

                # Send row_offsets
                row_offsets = pe['row_offsets'].astype(np.int32) if pe['num_rows'] > 0 else np.array([0], dtype=np.int32)
                runner.memcpy_h2d(
                    sym_row_offsets,
                    row_offsets,
                    x_coord, y_coord, 1, 1, len(row_offsets),
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_32BIT,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False
                )

                # Send matrix_values
                matrix_values = pe['matrix_values'] if pe['total_nnz'] > 0 else np.array([0.0], dtype=np.float32)
                runner.memcpy_h2d(
                    sym_matrix_values,
                    matrix_values,
                    x_coord, y_coord, 1, 1, len(matrix_values),
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_32BIT,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False
                )

                # Send vector_values
                vector_values = pe['vector_values'] if pe['total_nnz'] > 0 else np.array([0.0], dtype=np.float32)
                runner.memcpy_h2d(
                    sym_vector_values,
                    vector_values,
                    x_coord, y_coord, 1, 1, len(vector_values),
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_32BIT,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False
                )

            memcpy_h2d_time = time.time() - memcpy_h2d_start
            logger.info(f"H2D transfer: {memcpy_h2d_time:.3f}s")

            # Launch kernel
            logger.info("Launching kernel...")
            runner.launch("spmv", nonblock=False)

            # Retrieve hardware timestamps
            logger.info("Retrieving hardware timestamps...")
            all_pe_cycles = []

            for pe in pe_data:
                if pe['num_rows'] == 0:
                    continue

                pe_id = pe['pe_id']
                x_coord = pe_id % args.grid_width
                y_coord = pe_id // args.grid_width

                ts_buf_f32 = np.zeros(3, dtype=np.float32)
                runner.memcpy_d2h(
                    ts_buf_f32,
                    sym_time_buf,
                    x_coord, y_coord, 1, 1, 3,
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_32BIT,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False
                )

                elapsed_cycles = calculate_cycles(ts_buf_f32)
                all_pe_cycles.append(elapsed_cycles)

            # Calculate timing statistics
            all_pe_cycles = np.array(all_pe_cycles)
            min_cycles = np.min(all_pe_cycles)
            max_cycles = np.max(all_pe_cycles)
            avg_cycles = np.mean(all_pe_cycles)
            std_cycles = np.std(all_pe_cycles)

            CLOCK_FREQ_GHZ = 0.85
            min_time_s = (min_cycles / CLOCK_FREQ_GHZ) / 1e6
            max_time_s = (max_cycles / CLOCK_FREQ_GHZ) / 1e6
            avg_time_s = (avg_cycles / CLOCK_FREQ_GHZ) / 1e6
            std_time_s = (std_cycles / CLOCK_FREQ_GHZ) / 1e6

            logger.info(f"Compute time - Min: {min_time_s*1e6:.3f}µs, Avg: {avg_time_s*1e6:.3f}µs, Max: {max_time_s*1e6:.3f}µs")

            # Retrieve results (D2H)
            logger.info("Retrieving results from device (D2H)...")
            memcpy_d2h_start = time.time()

            all_results = []
            for pe in pe_data:
                pe_id = pe['pe_id']
                x_coord = pe_id % args.grid_width
                y_coord = pe_id // args.grid_width

                if pe['num_rows'] > 0:
                    results_raw = np.zeros(pe['num_rows'], dtype=np.float32)
                    runner.memcpy_d2h(
                        results_raw,
                        sym_results,
                        x_coord, y_coord, 1, 1, pe['num_rows'],
                        streaming=False,
                        data_type=MemcpyDataType.MEMCPY_32BIT,
                        order=MemcpyOrder.ROW_MAJOR,
                        nonblock=False
                    )
                else:
                    results_raw = np.array([], dtype=np.float32)

                all_results.append(results_raw)

            memcpy_d2h_time = time.time() - memcpy_d2h_start
            logger.info(f"D2H transfer: {memcpy_d2h_time:.3f}s")


    except Exception as e:
        logger.error(f"Runtime execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    # Gather and validate results
    y_computed = spmv.gather_results(pe_data, all_results)
    y_reference = A.dot(x).astype(np.float32)
    passed = validate_results(y_computed, y_reference, tolerance=1e-4)

    if passed:
        logger.info("✓ Validation PASSED")
    else:
        logger.error("✗ Validation FAILED")

    total_time = time.time() - total_start

    # Calculate benchmark metrics
    nnz_counts = [pe['total_nnz'] for pe in pe_data if pe['total_nnz'] > 0]
    min_nnz_pe = min(nnz_counts) if nnz_counts else 0
    max_nnz_pe = max(nnz_counts) if nnz_counts else 0
    avg_nnz_pe = np.mean(nnz_counts) if nnz_counts else 0
    load_imbalance = (max_nnz_pe - min_nnz_pe) / avg_nnz_pe * 100 if avg_nnz_pe > 0 else 0

    # Memory statistics
    PE_MEMORY_LIMIT = 48000
    max_memory = 0
    for pe in pe_data:
        mem = (pe['total_nnz'] * 8 +  # matrix + vector values
               (pe['num_rows'] + 1) * 4)  # row_offsets
        max_memory = max(max_memory, mem)

    avg_memory = np.mean([
        pe['total_nnz'] * 8 + (pe['num_rows'] + 1) * 4
        for pe in pe_data
    ])
    memory_utilization = (max_memory / PE_MEMORY_LIMIT) * 100

    # Performance metrics
    flops = 2 * A.nnz
    gflops_hw_avg = (flops / avg_time_s) / 1e9 if avg_time_s > 0 else 0
    gflops_hw_max = (flops / max_time_s) / 1e9 if max_time_s > 0 else 0

    # Save benchmark results
    if args.save_benchmark:
        benchmark = BenchmarkResult(
            git_commit=get_git_commit_hash(),
            validation_passed=passed,
            exit_reason="SUCCESS" if passed else "VALIDATION_FAILED",
            error_message="" if passed else "Results do not match reference",
            relative_error=np.linalg.norm(y_computed - y_reference) / np.linalg.norm(y_reference) if np.linalg.norm(y_reference) > 0 else 0.0,
            matrix_rows=args.rows,
            matrix_cols=args.cols,
            density=args.density,
            total_nnz=A.nnz,
            seed=args.seed,
            num_pes=args.num_pes,
            grid_width=args.grid_width,
            grid_height=args.grid_height,
            architecture=args.arch,
            max_memory_per_pe=max_memory,
            avg_memory_per_pe=avg_memory,
            memory_utilization_pct=memory_utilization,
            distribute_time=distribute_time,
            prepare_matched_pairs_time=prepare_matched_pairs_time,
            total_precond_time=total_lb_time,
            compile_time=0.0,  # compilation happened on login node
            memcpy_h2d_time=memcpy_h2d_time,
            compute_time_hw_min=min_time_s,
            compute_time_hw_avg=avg_time_s,
            compute_time_hw_max=max_time_s,
            compute_time_hw_std=std_time_s,
            compute_hw_cycles_min=int(min_cycles),
            compute_hw_cycles_avg=int(avg_cycles),
            compute_hw_cycles_max=int(max_cycles),
            compute_hw_cycles_std=int(std_cycles),
            memcpy_d2h_time=memcpy_d2h_time,
            total_run_time=memcpy_h2d_time + avg_time_s + memcpy_d2h_time,
            total_time=total_time,
            gflops_hw_avg=gflops_hw_avg,
            gflops_hw_max=gflops_hw_max,
            arithmetic_intensity=flops / (A.nnz * 16 + args.rows * 8) if A.nnz > 0 else 0,
            min_nnz_per_pe=min_nnz_pe,
            max_nnz_per_pe=max_nnz_pe,
            avg_nnz_per_pe=avg_nnz_pe,
            load_imbalance_pct=load_imbalance,
            num_idle_pes=args.num_pes - num_active,
            pe_efficiency_pct=(num_active / args.num_pes * 100) if args.num_pes > 0 else 0,
            num_active_pes=num_active
        )

        try:
            save_benchmark(benchmark, csv_path=args.save_benchmark)
            logger.info(f"✓ Benchmark saved to: {args.save_benchmark}")
        except Exception as e:
            logger.warning(f"Failed to save benchmark: {e}")

    logger.info("="*60)
    logger.info(f"Total execution time: {total_time:.2f}s")
    logger.info("="*60)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
