"""Device runtime and SpMV execution on Cerebras WSE."""

import logging
import time

import numpy as np
import scipy.sparse as sp

from cerebras.sdk.client import SdkRuntime
from cerebras.sdk.client.sdk_utils import calculate_cycles
from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder

from spmv_prototype import MatchedPairsSpMV

logger = logging.getLogger(__name__)


def run_spmv_on_device(
    csr_matrix: sp.csr_matrix,
    vector_x: np.ndarray,
    artifact_path: str,
    num_pes: int,
    width: int,
    height: int
):

    run_start = time.time()

    spmv = MatchedPairsSpMV(csr_matrix, num_pes)
    pe_data = spmv.distribute_by_nnz()
    spmv.prepare_matched_pairs(pe_data, vector_x)

    non_empty = [pe for pe in pe_data if pe['num_rows'] > 0]
    logger.info(f"Active PEs: {len(non_empty)}/{num_pes}")
    with SdkRuntime(artifact_path, suppress_simfab_trace=True, disable_version_check=True) as runner:
        sym_num_rows = runner.get_id("num_rows")
        sym_row_offsets = runner.get_id("row_offsets")
        sym_matrix_values = runner.get_id("matrix_values")
        sym_vector_values = runner.get_id("vector_values")
        sym_results = runner.get_id("results")
        sym_time_buf = runner.get_id("time_buf_u16")

        memcpy_h2d_start = time.time()

        for pe in pe_data:
            pe_id = pe['pe_id']
            x = pe_id % width
            y = pe_id // width

            runner.memcpy_h2d(
                sym_num_rows,
                np.array([pe['num_rows']], dtype=np.int32),
                x, y, 1, 1,
                1,
                streaming=False,
                data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=False
            )

            if pe['num_rows'] > 0:
                row_offsets_to_send = pe['row_offsets'].astype(np.int32)
            else:
                row_offsets_to_send = np.array([0], dtype=np.int32)

            runner.memcpy_h2d(
                sym_row_offsets,
                row_offsets_to_send,
                x, y, 1, 1,
                len(row_offsets_to_send),
                streaming=False,
                data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=False
            )

            if pe['total_nnz'] > 0:
                matrix_values_to_send = pe['matrix_values']
            else:
                matrix_values_to_send = np.array([0.0], dtype=np.float32)

            runner.memcpy_h2d(
                sym_matrix_values,
                matrix_values_to_send,
                x, y, 1, 1,
                len(matrix_values_to_send),
                streaming=False,
                data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=False
            )

            if pe['total_nnz'] > 0:
                vector_values_to_send = pe['vector_values']
            else:
                vector_values_to_send = np.array([0.0], dtype=np.float32)

            runner.memcpy_h2d(
                sym_vector_values,
                vector_values_to_send,
                x, y, 1, 1,
                len(vector_values_to_send),
                streaming=False,
                data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=False
            )

        memcpy_h2d_time = time.time() - memcpy_h2d_start
        logger.info(f"Data transfer H2D: {memcpy_h2d_time:.3f}s")

        compute_start = time.time()
        runner.launch("spmv", nonblock=False)
        compute_time_python = time.time() - compute_start
        logger.info(f"Computation (Python): {compute_time_python:.3f}s")

        logger.info("Retrieving hardware timestamps from PEs...")
        all_pe_cycles = []
        all_pe_ids = []

        for pe in pe_data:
            if pe['num_rows'] == 0:
                continue

            pe_id = pe['pe_id']
            x = pe_id % width
            y = pe_id // width

            # Retrieve as 32-bit (3 words: each u32 contains 2×u16)
            ts_raw_u32 = np.zeros(3, dtype=np.uint32)
            runner.memcpy_d2h(
                ts_raw_u32,
                sym_time_buf,
                x, y, 1, 1,
                3,
                streaming=False,
                data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=False
            )

            # Unpack: each u32 contains two u16 values (little-endian)
            ts_raw = np.zeros(6, dtype=np.uint16)
            for i in range(3):
                ts_raw[2*i] = ts_raw_u32[i] & 0xFFFF          # Lower 16 bits
                ts_raw[2*i + 1] = (ts_raw_u32[i] >> 16) & 0xFFFF  # Upper 16 bits

            start_buf = ts_raw[0:3]
            end_buf = ts_raw[3:6]

            start_cycles = calculate_cycles(start_buf)
            end_cycles = calculate_cycles(end_buf)
            elapsed_cycles = end_cycles - start_cycles

            all_pe_cycles.append(elapsed_cycles)
            all_pe_ids.append(pe_id)

        all_pe_cycles = np.array(all_pe_cycles)
        min_cycles = np.min(all_pe_cycles)
        max_cycles = np.max(all_pe_cycles)
        avg_cycles = np.mean(all_pe_cycles)
        std_cycles = np.std(all_pe_cycles)

        CLOCK_FREQ_GHZ = 0.85

        min_time_us = (min_cycles / CLOCK_FREQ_GHZ) * 1e-3
        max_time_us = (max_cycles / CLOCK_FREQ_GHZ) * 1e-3
        avg_time_us = (avg_cycles / CLOCK_FREQ_GHZ) * 1e-3
        std_time_us = (std_cycles / CLOCK_FREQ_GHZ) * 1e-3

        min_time_s = min_time_us / 1e6
        max_time_s = max_time_us / 1e6
        avg_time_s = avg_time_us / 1e6
        std_time_s = std_time_us / 1e6

        logger.info("="*60)
        logger.info("Hardware Timing (Timestamp Counter)")
        logger.info("="*60)
        logger.info(f"Active PEs measured: {len(all_pe_cycles)}")
        logger.info(f"Min PE:  {min_time_us:>10.3f} µs ({min_cycles:>10,} cycles)")
        logger.info(f"Avg PE:  {avg_time_us:>10.3f} µs ({avg_cycles:>10,.0f} cycles)")
        logger.info(f"Max PE:  {max_time_us:>10.3f} µs ({max_cycles:>10,} cycles)")
        logger.info(f"Std Dev: {std_time_us:>10.3f} µs ({std_cycles:>10,.0f} cycles)")
        logger.info(f"-"*60)
        logger.info(f"Python time (with overhead): {compute_time_python:.6f} s")
        logger.info(f"Overhead factor: {compute_time_python / avg_time_s:.1f}×")
        logger.info("="*60 + "\n")

        memcpy_d2h_start = time.time()

        all_results = []
        for pe in pe_data:
            pe_id = pe['pe_id']
            x = pe_id % width
            y = pe_id // width

            if pe['num_rows'] > 0:
                results_raw = np.zeros(pe['num_rows'], dtype=np.float32)
                runner.memcpy_d2h(
                    results_raw,
                    sym_results,
                    x, y, 1, 1,
                    pe['num_rows'],
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_32BIT,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False
                )
            else:
                results_raw = np.array([], dtype=np.float32)

            all_results.append(results_raw)

        memcpy_d2h_time = time.time() - memcpy_d2h_start
        logger.info(f"Data transfer D2H: {memcpy_d2h_time:.3f}s")

    result_vector = spmv.gather_results(pe_data, all_results)

    run_time = time.time() - run_start

    timing_info = {
        'total_run': memcpy_h2d_time + avg_time_s + memcpy_d2h_time,
        'memcpy_h2d': memcpy_h2d_time,
        'compute_python': compute_time_python,
        'compute_hw_min': min_time_s,
        'compute_hw_avg': avg_time_s,
        'compute_hw_max': max_time_s,
        'compute_hw_std': std_time_s,
        'compute_hw_cycles_min': int(min_cycles),
        'compute_hw_cycles_avg': int(avg_cycles),
        'compute_hw_cycles_max': int(max_cycles),
        'compute_hw_cycles_std': int(std_cycles),
        'memcpy_d2h': memcpy_d2h_time
    }

    return result_vector, timing_info
