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
    logger.info("="*60)
    logger.info("Memory Analysis")
    logger.info("="*60)

    # Only show non-empty PEs and first/last few
    non_empty = [pe for pe in pe_data if pe['num_rows'] > 0]

    if len(non_empty) <= 10:
        # Show all if not too many
        pes_to_show = non_empty
    else:
        # Show first 3 and last 3
        pes_to_show = non_empty[:3] + non_empty[-3:]

    for i, pe in enumerate(pes_to_show):
        if i == 3 and len(non_empty) > 6:
            logger.info("... (%d PEs omitted) ..." % (len(non_empty) - 6))

        memory = calculate_memory_per_pe(pe['num_rows'], pe['total_nnz'])
        percentage = (memory / max_memory) * 100

        row_offsets_mem = 4 * (pe['num_rows'] + 1)
        matrix_values_mem = 4 * pe['total_nnz']
        vector_values_mem = 4 * pe['total_nnz']
        results_mem = 4 * pe['num_rows']

        logger.info(f"PE {pe['pe_id']}:")
        logger.info(f"  Rows: {pe['num_rows']}, NNZ: {pe['total_nnz']}")
        logger.info("  Memory breakdown:")
        logger.info(f"    row_offsets:    {row_offsets_mem:>6} bytes")
        logger.info(f"    matrix_values:  {matrix_values_mem:>6} bytes")
        logger.info(f"    vector_values:  {vector_values_mem:>6} bytes")
        logger.info(f"    results:        {results_mem:>6} bytes")
        logger.info(f"    num_rows:       {4:>6} bytes")
        logger.info(f"  Total: {memory:>6} bytes ({memory/1024:.2f} KB, {percentage:.1f}% of limit)")

        if memory > max_memory:
            logger.error("  ❌ EXCEEDS 48 KB LIMIT!")
        else:
            logger.info("  ✅ Within limit")

    max_memory_used = max(calculate_memory_per_pe(pe['num_rows'], pe['total_nnz'])
                          for pe in pe_data)
    avg_memory_used = np.mean([calculate_memory_per_pe(pe['num_rows'], pe['total_nnz'])
                                for pe in pe_data])

    logger.info("\nAggregate Statistics:")
    logger.info(f"  Max memory/PE:  {max_memory_used:>6} bytes ({max_memory_used/1024:.2f} KB)")
    logger.info(f"  Avg memory/PE:  {avg_memory_used:>6.0f} bytes ({avg_memory_used/1024:.2f} KB)")
    logger.info(f"  Memory limit:   {max_memory:>6} bytes ({max_memory/1024:.2f} KB)")
    logger.info("="*60 + "\n")
