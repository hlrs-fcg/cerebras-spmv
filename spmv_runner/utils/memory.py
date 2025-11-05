"""Memory calculation and feasibility checking for PE allocation."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def calculate_memory_per_pe(num_rows: int, nnz: int) -> int:
    """Calculate memory usage in bytes for SpMV data on a single PE."""
    row_offsets = 4 * (num_rows + 1)
    num_rows_var = 4
    matrix_values = 4 * nnz
    vector_values = 4 * nnz
    results = 4 * num_rows

    total = row_offsets + num_rows_var + matrix_values + vector_values + results
    return total


def check_memory_feasibility(pe_data: list, max_memory: int = 48_000) -> tuple:
    """Check if data distribution fits within PE memory constraints.

    Returns: (is_feasible, max_usage, offending_pe_id)
    """
    max_usage = 0
    offending_pe = None

    for pe in pe_data:
        memory = calculate_memory_per_pe(pe['num_rows'], pe['total_nnz'])
        if memory > max_usage:
            max_usage = memory
            if memory > max_memory:
                offending_pe = pe['pe_id']

    is_feasible = max_usage <= max_memory
    return is_feasible, max_usage, offending_pe


def print_memory_analysis(pe_data: list, max_memory: int = 48_000):
    max_memory_used = max(calculate_memory_per_pe(pe['num_rows'], pe['total_nnz'])
                          for pe in pe_data)
    avg_memory_used = np.mean([calculate_memory_per_pe(pe['num_rows'], pe['total_nnz'])
                                for pe in pe_data])

    logger.info(f"Memory: max {max_memory_used/1024:.2f} KB/PE, avg {avg_memory_used/1024:.2f} KB/PE (limit: {max_memory/1024:.0f} KB)")
