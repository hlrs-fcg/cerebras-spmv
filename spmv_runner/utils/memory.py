"""
Memory Calculation and Analysis

Provides utilities for calculating memory usage per PE and checking feasibility.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def calculate_memory_per_pe(num_rows: int, nnz: int) -> int:
    """
    Calculate memory usage in bytes for SpMV data on a single PE.

    Args:
        num_rows: Number of rows on this PE
        nnz: Number of nonzeros on this PE

    Returns:
        Memory usage in bytes
    """
    row_offsets = 4 * (num_rows + 1)  # i32 array
    num_rows_var = 4                   # single i32
    matrix_values = 4 * nnz            # f32 array
    vector_values = 4 * nnz            # f32 array
    results = 4 * num_rows             # f32 array

    total = row_offsets + num_rows_var + matrix_values + vector_values + results
    return total


def check_memory_feasibility(pe_data: list, max_memory: int = 48_000) -> tuple:
    """
    Check if data distribution fits within PE memory constraints.

    Args:
        pe_data: List of PE data dictionaries
        max_memory: Maximum memory per PE in bytes (default: 48 KB)

    Returns:
        (is_feasible, max_usage, offending_pe_id)
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
    """Print detailed memory analysis for all PEs."""
    # Only show non-empty PEs and first/last few
    non_empty = [pe for pe in pe_data if pe['num_rows'] > 0]

    max_memory_used = max(calculate_memory_per_pe(pe['num_rows'], pe['total_nnz'])
                          for pe in pe_data)
    avg_memory_used = np.mean([calculate_memory_per_pe(pe['num_rows'], pe['total_nnz'])
                                for pe in pe_data])

    logger.info(f"Memory: max {max_memory_used/1024:.2f} KB/PE, avg {avg_memory_used/1024:.2f} KB/PE (limit: {max_memory/1024:.0f} KB)")

    # Show detailed breakdown in debug mode
    if len(non_empty) <= 10:
        pes_to_show = non_empty
    else:
        pes_to_show = non_empty[:3] + non_empty[-3:]

    for i, pe in enumerate(pes_to_show):
        if i == 3 and len(non_empty) > 6:
            logger.debug("... (%d PEs omitted) ..." % (len(non_empty) - 6))

        memory = calculate_memory_per_pe(pe['num_rows'], pe['total_nnz'])
        percentage = (memory / max_memory) * 100

        logger.debug(f"PE {pe['pe_id']}: {pe['num_rows']} rows, {pe['total_nnz']} NNZ → {memory/1024:.2f} KB ({percentage:.1f}%)")

        if memory > max_memory:
            logger.error(f"  PE {pe['pe_id']} EXCEEDS 48 KB LIMIT!")
