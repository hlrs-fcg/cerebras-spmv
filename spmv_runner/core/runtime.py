"""Device runtime and SpMV execution on Cerebras WSE."""

import logging
import time

import numpy as np
import scipy.sparse as sp

from cerebras.sdk.client import SdkRuntime
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
        compute_time = time.time() - compute_start
        logger.info(f"Computation: {compute_time:.3f}s")

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
        'total_run': run_time,
        'memcpy_h2d': memcpy_h2d_time,
        'compute': compute_time,
        'memcpy_d2h': memcpy_d2h_time
    }

    return result_vector, timing_info
