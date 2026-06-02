#!/usr/bin/env python3
"""
Worker Runtime Script
Runs on worker node via SdkLauncher. Executes SpMV on Cerebras hardware.
Self-contained: no imports from login-node utils/ or sdk/.
"""

import argparse
import logging
import subprocess
import sys
import time

import numpy as np
import scipy.sparse as sp

from benchmark import BenchmarkResult, save_benchmark

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Matrix generation
# ---------------------------------------------------------------------------

def generate_random_sparse_matrix(
    rows: int, cols: int, density: float, seed: int = 42, cache_dir: str = "matrices"
) -> sp.csr_matrix:
    """Generate a random sparse CSR matrix with caching."""
    from pathlib import Path

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"random_{rows}x{cols}_d{density}_s{seed}.npz"

    if cache_file.exists():
        logger.debug(f"Loading cached matrix from {cache_file}")
        matrix = sp.load_npz(cache_file)
        logger.info(f"Loaded cached matrix: {matrix.shape}, NNZ={matrix.nnz}")
        return matrix

    logger.info(f"Generating matrix (caching to {cache_file.name})...")
    np.random.seed(seed)

    if density < 0.01:
        expected_nnz = int(rows * cols * density)
        row_idx = np.random.randint(0, rows, size=expected_nnz, dtype=np.int32)
        col_idx = np.random.randint(0, cols, size=expected_nnz, dtype=np.int32)
        data = np.random.rand(expected_nnz).astype(np.float32)
        matrix = sp.coo_matrix((data, (row_idx, col_idx)), shape=(rows, cols), dtype=np.float32).tocsr()
    else:
        matrix = sp.random(rows, cols, density=density, format='csr', dtype=np.float32)
        matrix.data = np.random.rand(matrix.nnz).astype(np.float32)

    sp.save_npz(cache_file, matrix)
    logger.info(f"Generated and cached: {matrix.shape}, NNZ={matrix.nnz}")
    return matrix


# ---------------------------------------------------------------------------
# Result validation
# ---------------------------------------------------------------------------

def validate_results(y_computed: np.ndarray, y_reference: np.ndarray, tolerance: float = 1e-4) -> bool:
    """Validate via relative norm: ||y_computed - y_ref|| / ||y_ref|| < tolerance."""
    if y_computed.shape != y_reference.shape:
        logger.error(f"Shape mismatch: computed {y_computed.shape} vs reference {y_reference.shape}")
        return False

    if np.any(np.isnan(y_computed)) or np.any(np.isinf(y_computed)):
        logger.error("Invalid values (NaN or Inf) in computed results")
        return False

    diff_norm = np.linalg.norm(y_computed - y_reference)
    ref_norm = np.linalg.norm(y_reference)

    if ref_norm < 1e-15:
        passed = diff_norm < tolerance
        if passed:
            logger.info(f"Validation PASSED (both near zero, ||diff||={diff_norm:.2e})")
        else:
            logger.error(f"Validation FAILED: reference is zero, computed is not")
        return passed

    relative_error = diff_norm / ref_norm
    passed = relative_error < tolerance

    if passed:
        logger.info(f"Validation PASSED — relative error: {relative_error:.2e}")
    else:
        logger.error(f"Validation FAILED — relative error: {relative_error:.2e} (tolerance: {tolerance:.2e})")
        abs_diff = np.abs(y_computed - y_reference)
        for idx in np.argsort(abs_diff)[-5:][::-1]:
            logger.error(f"  Row {idx}: computed={y_computed[idx]:.6e}, reference={y_reference[idx]:.6e}")

    return passed


# ---------------------------------------------------------------------------
# Distribution (matched pairs)
# ---------------------------------------------------------------------------

def distribute_by_nnz(matrix: sp.csr_matrix, num_pes: int) -> list:
    """
    Distribute rows across PEs via histogram-based bucketing (O(n)).
    Groups rows by NNZ magnitude (powers of 2), distributes round-robin from
    largest bucket first to spread heavy rows evenly.
    """
    row_nnz = np.diff(matrix.indptr)
    max_nnz = int(np.max(row_nnz)) if len(row_nnz) > 0 else 0

    if max_nnz == 0:
        buckets = {0: list(range(matrix.shape[0]))}
    else:
        num_buckets = int(np.ceil(np.log2(max_nnz + 1))) + 1
        buckets: dict[int, list] = {i: [] for i in range(num_buckets)}
        for row, nnz in enumerate(row_nnz):
            bucket_id = 0 if nnz == 0 else int(np.floor(np.log2(nnz))) + 1
            buckets[bucket_id].append(row)

    pe_data = [{'pe_id': i, 'row_ids': [], 'total_nnz': 0, 'num_rows': 0} for i in range(num_pes)]

    current_pe = 0
    for bucket_id in sorted(buckets.keys(), reverse=True):
        for row in buckets[bucket_id]:
            pe_data[current_pe]['row_ids'].append(row)
            pe_data[current_pe]['total_nnz'] += int(row_nnz[row])
            pe_data[current_pe]['num_rows'] += 1
            current_pe = (current_pe + 1) % num_pes

    nnz_counts = [pe['total_nnz'] for pe in pe_data if pe['total_nnz'] > 0]
    num_empty = num_pes - len(nnz_counts)
    if nnz_counts:
        avg = np.mean(nnz_counts)
        imbalance = (max(nnz_counts) - min(nnz_counts)) / avg * 100 if avg > 0 else 0
        logger.info(f"Load balance: active={len(nnz_counts)}/{num_pes}, "
                    f"min={min(nnz_counts)}, max={max(nnz_counts)}, imbalance={imbalance:.1f}%")
        if num_empty > 0:
            logger.warning(f"{num_empty} empty PEs ({num_empty / num_pes * 100:.1f}%)")
    else:
        logger.warning(f"All {num_pes} PEs are empty")

    return pe_data


def prepare_matched_pairs(matrix: sp.csr_matrix, pe_data: list, vector_x: np.ndarray) -> None:
    """
    Pre-gather vector values into matched (matrix_val, vector_val) pairs per PE.
    Modifies pe_data in place, adding 'row_offsets', 'matrix_values', 'vector_values'.
    """
    for pe in pe_data:
        if pe['num_rows'] == 0:
            pe['row_offsets'] = np.array([0], dtype=np.int32)
            pe['matrix_values'] = np.array([], dtype=np.float32)
            pe['vector_values'] = np.array([], dtype=np.float32)
            continue

        row_offsets = [0]
        matrix_values = []
        vector_values = []

        for global_row in pe['row_ids']:
            start = matrix.indptr[global_row]
            end = matrix.indptr[global_row + 1]
            matrix_values.extend(matrix.data[start:end])
            vector_values.extend(vector_x[matrix.indices[start:end]])
            row_offsets.append(len(matrix_values))

        pe['row_offsets'] = np.array(row_offsets, dtype=np.int32)
        pe['matrix_values'] = np.array(matrix_values, dtype=np.float32)
        pe['vector_values'] = np.array(vector_values, dtype=np.float32)


def gather_results(num_rows: int, pe_data: list, pe_results: list) -> np.ndarray:
    """Assemble per-PE partial results into the full result vector b."""
    result = np.zeros(num_rows, dtype=np.float32)
    for pe, results in zip(pe_data, pe_results):
        for local_idx, global_row in enumerate(pe['row_ids']):
            result[global_row] = results[local_idx]
    return result


# ---------------------------------------------------------------------------
# SDK runtime wrapper
# ---------------------------------------------------------------------------

class SdkRuntimeWrapper:
    """Context manager for SdkRuntime (which doesn't natively support 'with')."""

    def __init__(self, runner_cls, *args, **kwargs):
        self.runner_cls = runner_cls
        self.args = args
        self.kwargs = kwargs
        self.runner = None

    def __enter__(self):
        self.runner = self.runner_cls(*self.args, **self.kwargs)
        return self.runner

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.runner is not None:
            try:
                self.runner.stop()
            except Exception as e:
                logger.warning(f"Error stopping SdkRuntime: {e}")
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, check=True, timeout=1
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _pad(arr: np.ndarray, target_len: int, dtype) -> np.ndarray:
    """Right-pad arr to target_len with zeros."""
    out = np.zeros(target_len, dtype=dtype)
    out[:len(arr)] = arr
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SpMV Worker Runtime (runs on worker node via SdkLauncher)')
    parser.add_argument('--name', required=True, help='Compiled artifact output directory')
    parser.add_argument('--cmaddr', required=True, help='IP:port for Cerebras system')
    parser.add_argument('--rows', type=int, required=True)
    parser.add_argument('--cols', type=int, required=True)
    parser.add_argument('--density', type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mtx-file', type=str)
    parser.add_argument('--num-pes', type=int, required=True)
    parser.add_argument('--grid-width', type=int, required=True)
    parser.add_argument('--grid-height', type=int, required=True)
    parser.add_argument('--arch', type=str, default='wse3', choices=['wse2', 'wse3'])
    parser.add_argument('--total-nnz-per-pe', type=int, required=True,
                        help='Compile-time TOTAL_NNZ (used for bulk memcpy padding)')
    parser.add_argument('--max-rows-per-pe', type=int, required=True,
                        help='Compile-time MAX_ROWS (used for bulk memcpy padding)')
    parser.add_argument('--save-benchmark', type=str, default='benchmark_results.csv')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
    from cerebras.sdk.sdk_utils import calculate_cycles

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("Worker Runtime")
    logger.info(f"Artifact: {args.name}  System: {args.cmaddr}")
    logger.info("=" * 60)

    total_start = time.time()
    W, H = args.grid_width, args.grid_height
    TOTAL_NNZ = args.total_nnz_per_pe
    MAX_ROWS = args.max_rows_per_pe

    # ------------------------------------------------------------------
    # Load / generate matrix
    # ------------------------------------------------------------------
    t = time.time()
    try:
        if args.mtx_file:
            try:
                import fast_matrix_market as fmm
            except ImportError:
                logger.error("fast-matrix-market not installed")
                return 1
            A = fmm.mmread(args.mtx_file)
            logger.info(f"Loaded matrix: {A.shape[0]}×{A.shape[1]}, NNZ={A.nnz:,}")
        else:
            if args.density is None:
                logger.error("--density required for random generation")
                return 1
            A = generate_random_sparse_matrix(args.rows, args.cols, args.density, args.seed)
    except Exception as e:
        logger.error(f"Matrix load/generation failed: {e}")
        return 1
    matrix_generation_time = time.time() - t

    t = time.time()
    x = np.random.randint(0, 100, size=args.cols, dtype=np.int32).astype(np.float32)
    vector_generation_time = time.time() - t

    # ------------------------------------------------------------------
    # Distribute
    # ------------------------------------------------------------------
    logger.info(f"Distributing across {args.num_pes} PEs...")
    t0 = time.time()

    t = time.time()
    pe_data = distribute_by_nnz(A, args.num_pes)
    distribute_time = time.time() - t

    t = time.time()
    prepare_matched_pairs(A, pe_data, x)
    prepare_matched_pairs_time = time.time() - t

    total_lb_time = time.time() - t0
    num_active = sum(1 for pe in pe_data if pe['num_rows'] > 0)
    logger.info(f"Active PEs: {num_active}/{args.num_pes}  (load balance: {total_lb_time*1000:.1f}ms)")

    # ------------------------------------------------------------------
    # Build bulk transfer arrays (pad each PE to compile-time constants)
    # ------------------------------------------------------------------
    # Shape: (num_pes,) or (num_pes, size) in PE order (pe_id = y*W + x)
    all_num_rows = np.zeros(args.num_pes, dtype=np.int32)
    all_row_offsets = np.zeros((args.num_pes, MAX_ROWS + 1), dtype=np.int32)
    all_matrix_values = np.zeros((args.num_pes, TOTAL_NNZ), dtype=np.float32)
    all_vector_values = np.zeros((args.num_pes, TOTAL_NNZ), dtype=np.float32)

    for pe in pe_data:
        pid = pe['pe_id']
        all_num_rows[pid] = pe['num_rows']
        ro = pe['row_offsets']
        all_row_offsets[pid, :len(ro)] = ro
        mv = pe['matrix_values']
        all_matrix_values[pid, :len(mv)] = mv
        vv = pe['vector_values']
        all_vector_values[pid, :len(vv)] = vv

    # ------------------------------------------------------------------
    # Execute on device
    # ------------------------------------------------------------------
    logger.info("Connecting to Cerebras system...")
    sdk_init_start = time.time()
    try:
        with SdkRuntimeWrapper(SdkRuntime, args.name, cmaddr=args.cmaddr) as runner:
            sdk_init_time = time.time() - sdk_init_start

            sym_num_rows = runner.get_id("num_rows")
            sym_row_offsets = runner.get_id("row_offsets")
            sym_matrix_values = runner.get_id("matrix_values")
            sym_vector_values = runner.get_id("vector_values")
            sym_results = runner.get_id("results")
            sym_time_buf = runner.get_id("time_buf_f32")

            logger.info("Loading program...")
            t = time.time()
            runner.load()
            runner.run()
            program_load_time = time.time() - t

            # Bulk H2D — one call per symbol covers all W×H PEs
            logger.info("H2D transfer (bulk)...")
            t = time.time()

            runner.memcpy_h2d(
                sym_num_rows, all_num_rows,
                0, 0, W, H, 1,
                streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.ROW_MAJOR, nonblock=True
            )
            runner.memcpy_h2d(
                sym_row_offsets, all_row_offsets.flatten(),
                0, 0, W, H, MAX_ROWS + 1,
                streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.ROW_MAJOR, nonblock=True
            )
            runner.memcpy_h2d(
                sym_matrix_values, all_matrix_values.flatten(),
                0, 0, W, H, TOTAL_NNZ,
                streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.ROW_MAJOR, nonblock=True
            )
            runner.memcpy_h2d(
                sym_vector_values, all_vector_values.flatten(),
                0, 0, W, H, TOTAL_NNZ,
                streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.ROW_MAJOR, nonblock=False  # last one: sync
            )

            memcpy_h2d_time = time.time() - t
            logger.info(f"H2D: {memcpy_h2d_time:.3f}s")

            logger.info("Launching kernel...")
            runner.launch("spmv", nonblock=False)

            # Bulk D2H — timestamps and results
            logger.info("D2H transfer (bulk)...")
            t = time.time()

            all_time_buf = np.zeros(args.num_pes * 3, dtype=np.float32)
            runner.memcpy_d2h(
                all_time_buf, sym_time_buf,
                0, 0, W, H, 3,
                streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.ROW_MAJOR, nonblock=True
            )

            all_results_flat = np.zeros(args.num_pes * MAX_ROWS, dtype=np.float32)
            runner.memcpy_d2h(
                all_results_flat, sym_results,
                0, 0, W, H, MAX_ROWS,
                streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.ROW_MAJOR, nonblock=False  # sync
            )

            memcpy_d2h_time = time.time() - t
            logger.info(f"D2H: {memcpy_d2h_time:.3f}s")

    except Exception as e:
        logger.error(f"Runtime execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    # ------------------------------------------------------------------
    # Parse timestamps
    # ------------------------------------------------------------------
    all_time_buf = all_time_buf.reshape(args.num_pes, 3)
    all_pe_cycles = []
    for pe in pe_data:
        if pe['num_rows'] == 0:
            continue
        elapsed = calculate_cycles(all_time_buf[pe['pe_id']])
        all_pe_cycles.append(elapsed)

    all_pe_cycles = np.array(all_pe_cycles)
    min_cycles = int(np.min(all_pe_cycles))
    max_cycles = int(np.max(all_pe_cycles))
    avg_cycles = float(np.mean(all_pe_cycles))
    std_cycles = float(np.std(all_pe_cycles))

    CLOCK_FREQ_GHZ = 0.85
    min_time_s = (min_cycles / CLOCK_FREQ_GHZ) / 1e9
    max_time_s = (max_cycles / CLOCK_FREQ_GHZ) / 1e9
    avg_time_s = (avg_cycles / CLOCK_FREQ_GHZ) / 1e9
    std_time_s = (std_cycles / CLOCK_FREQ_GHZ) / 1e9

    logger.info(f"Compute — min: {min_time_s*1e6:.3f}µs  avg: {avg_time_s*1e6:.3f}µs  max: {max_time_s*1e6:.3f}µs")

    # ------------------------------------------------------------------
    # Gather and validate
    # ------------------------------------------------------------------
    all_results = all_results_flat.reshape(args.num_pes, MAX_ROWS)
    pe_results = [all_results[pe['pe_id'], :pe['num_rows']] for pe in pe_data]

    t = time.time()
    y_computed = gather_results(A.shape[0], pe_data, pe_results)
    gather_results_time = time.time() - t

    t = time.time()
    y_reference = A.dot(x).astype(np.float32)
    passed = validate_results(y_computed, y_reference)
    validation_time = time.time() - t

    total_time = time.time() - total_start

    # ------------------------------------------------------------------
    # Benchmark metrics
    # ------------------------------------------------------------------
    nnz_counts = [pe['total_nnz'] for pe in pe_data if pe['total_nnz'] > 0]
    min_nnz_pe = min(nnz_counts) if nnz_counts else 0
    max_nnz_pe = max(nnz_counts) if nnz_counts else 0
    avg_nnz_pe = float(np.mean(nnz_counts)) if nnz_counts else 0.0
    load_imbalance = (max_nnz_pe - min_nnz_pe) / avg_nnz_pe * 100 if avg_nnz_pe > 0 else 0.0

    PE_MEMORY_LIMIT = 48_000
    max_memory = max(
        (pe['total_nnz'] * 8 + (pe['num_rows'] + 1) * 4) for pe in pe_data
    )
    avg_memory = float(np.mean([
        pe['total_nnz'] * 8 + (pe['num_rows'] + 1) * 4 for pe in pe_data
    ]))
    memory_utilization = max_memory / PE_MEMORY_LIMIT * 100

    flops = 2 * A.nnz
    gflops_hw_avg = (flops / avg_time_s) / 1e9 if avg_time_s > 0 else 0.0
    gflops_hw_max = (flops / max_time_s) / 1e9 if max_time_s > 0 else 0.0

    rel_error = 0.0
    ref_norm = np.linalg.norm(y_reference)
    if ref_norm > 0:
        rel_error = float(np.linalg.norm(y_computed - y_reference) / ref_norm)

    if args.save_benchmark:
        benchmark = BenchmarkResult(
            git_commit=get_git_commit_hash(),
            validation_passed=passed,
            exit_reason="SUCCESS" if passed else "VALIDATION_FAILED",
            error_message="" if passed else "Results do not match reference",
            relative_error=rel_error,
            matrix_rows=args.rows,
            matrix_cols=args.cols,
            density=args.density or 0.0,
            total_nnz=A.nnz,
            seed=args.seed,
            num_pes=args.num_pes,
            grid_width=W,
            grid_height=H,
            architecture=args.arch,
            max_memory_per_pe=max_memory,
            avg_memory_per_pe=avg_memory,
            memory_utilization_pct=memory_utilization,
            matrix_generation_time=matrix_generation_time,
            vector_generation_time=vector_generation_time,
            distribute_time=distribute_time,
            prepare_matched_pairs_time=prepare_matched_pairs_time,
            total_precond_time=total_lb_time,
            compile_time=0.0,
            sdk_init_time=sdk_init_time,
            program_load_time=program_load_time,
            memcpy_h2d_time=memcpy_h2d_time,
            compute_time_hw_min=min_time_s,
            compute_time_hw_avg=avg_time_s,
            compute_time_hw_max=max_time_s,
            compute_time_hw_std=std_time_s,
            compute_hw_cycles_min=min_cycles,
            compute_hw_cycles_avg=int(avg_cycles),
            compute_hw_cycles_max=max_cycles,
            compute_hw_cycles_std=int(std_cycles),
            memcpy_d2h_time=memcpy_d2h_time,
            gather_results_time=gather_results_time,
            validation_time=validation_time,
            total_run_time=memcpy_h2d_time + avg_time_s + memcpy_d2h_time,
            total_time=total_time,
            gflops_hw_avg=gflops_hw_avg,
            gflops_hw_max=gflops_hw_max,
            arithmetic_intensity=flops / (A.nnz * 16 + args.rows * 8) if A.nnz > 0 else 0.0,
            min_nnz_per_pe=min_nnz_pe,
            max_nnz_per_pe=max_nnz_pe,
            avg_nnz_per_pe=avg_nnz_pe,
            load_imbalance_pct=load_imbalance,
            num_idle_pes=args.num_pes - num_active,
            pe_efficiency_pct=(num_active / args.num_pes * 100) if args.num_pes > 0 else 0.0,
            num_active_pes=num_active,
        )
        try:
            save_benchmark(benchmark, csv_path=args.save_benchmark)
            logger.info(f"Benchmark saved to {args.save_benchmark}")
        except Exception as e:
            logger.warning(f"Failed to save benchmark: {e}")

    logger.info("=" * 60)
    logger.info(f"Total time: {total_time:.2f}s  — {'PASSED' if passed else 'FAILED'}")
    logger.info("=" * 60)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
