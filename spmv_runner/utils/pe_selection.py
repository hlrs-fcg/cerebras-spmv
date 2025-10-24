"""
PE Count Selection Utilities

Provides intelligent PE count selection and auto-calculation functionality.
"""

import logging
import numpy as np
import scipy.sparse as sp

from .memory import calculate_memory_per_pe
from ..core.grid import calculate_pe_grid

logger = logging.getLogger(__name__)


def calculate_min_pes_needed(matrix_nnz: int, matrix_rows: int, max_memory: int = 48_000) -> int:
    """
    Calculate minimum number of PEs needed to fit matrix in memory.

    Args:
        matrix_nnz: Total nonzeros in matrix
        matrix_rows: Total rows in matrix
        max_memory: Maximum memory per PE in bytes

    Returns:
        Minimum number of PEs needed
    """
    # Memory per PE = 8*rows_per_pe + 8*nnz_per_pe + 8
    # Assume perfect equal distribution with 20% overhead for imbalance

    # Try different PE counts
    for num_pes in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        rows_per_pe = matrix_rows / num_pes
        nnz_per_pe = matrix_nnz / num_pes

        # Estimate memory (pessimistic: add 20% for imbalance)
        estimated_memory = (8 * rows_per_pe + 8 * nnz_per_pe + 8) * 1.2

        if estimated_memory <= max_memory:
            return num_pes

    # If we get here, need more PEs
    # Calculate exact minimum
    min_pes = int(np.ceil((8 * matrix_rows + 8 * matrix_nnz) * 1.2 / max_memory))
    return min_pes


def calculate_max_pes_useful(matrix_rows: int, max_width: int = 758, max_height: int = 1171) -> int:
    """
    Calculate maximum useful number of PEs.

    More PEs than rows doesn't help (some PEs would be empty).
    Also limited by fabric dimensions.

    Args:
        matrix_rows: Total rows in matrix
        max_width: Maximum fabric width
        max_height: Maximum fabric height

    Returns:
        Maximum useful number of PEs
    """
    max_fabric_pes = max_width * max_height

    # Don't use more PEs than rows (some would be idle)
    max_useful = min(matrix_rows, max_fabric_pes)

    # Round down to nearest "nice" number for better grid
    nice_numbers = [1, 2, 4, 8, 16, 32, 64, 100, 128, 256, 512, 1024, 2048, 4096, 8192]

    for n in reversed(nice_numbers):
        if n <= max_useful:
            return n

    return max_useful


def parse_num_pes_arg(num_pes_str: str, matrix: sp.csr_matrix,
                     max_width: int = 758, max_height: int = 1171) -> int:
    """
    Parse --num-pes argument which can be a number, 'min', or 'max'.

    Args:
        num_pes_str: String argument from CLI
        matrix: The sparse matrix
        max_width: Maximum fabric width
        max_height: Maximum fabric height

    Returns:
        Number of PEs to use
    """
    if num_pes_str.lower() == 'min':
        num_pes = calculate_min_pes_needed(matrix.nnz, matrix.shape[0])
        logger.info(f"🔍 Auto-calculated minimum PEs needed: {num_pes}")
        return num_pes

    elif num_pes_str.lower() == 'max':
        num_pes = calculate_max_pes_useful(matrix.shape[0], max_width, max_height)
        logger.info(f"🔍 Auto-calculated maximum useful PEs: {num_pes}")
        return num_pes

    else:
        # Parse as integer
        try:
            return int(num_pes_str)
        except ValueError:
            raise ValueError(f"Invalid --num-pes value: {num_pes_str}. Use a number, 'min', or 'max'")


def suggest_pe_counts(matrix_nnz: int, matrix_rows: int,
                     max_width: int = 758, max_height: int = 1171):
    """Print suggestions for different PE counts."""
    min_pes = calculate_min_pes_needed(matrix_nnz, matrix_rows)
    max_pes = calculate_max_pes_useful(matrix_rows, max_width, max_height)

    logger.info("="*60)
    logger.info("PE Count Suggestions")
    logger.info("="*60)
    logger.info(f"Minimum needed (memory):  {min_pes:>6} PEs")
    logger.info(f"Maximum useful (rows):    {max_pes:>6} PEs")
    logger.info("")
    logger.info("Recommended options:")

    # Generate some good intermediate options
    options = []
    for n in [1, 2, 4, 8, 16, 32, 64, 100, 128, 256, 512, 1024, 2048, 4096]:
        if min_pes <= n <= max_pes:
            # Estimate performance characteristics
            rows_per_pe = matrix_rows / n
            nnz_per_pe = matrix_nnz / n
            memory_per_pe = calculate_memory_per_pe(int(rows_per_pe), int(nnz_per_pe))
            memory_pct = (memory_per_pe / 48_000) * 100

            # Calculate grid
            try:
                w, h = calculate_pe_grid(n, max_width, max_height)
                grid_str = f"{w}×{h}"
            except Exception:
                grid_str = "N/A"

            options.append((n, rows_per_pe, nnz_per_pe, memory_per_pe, memory_pct, grid_str))

    if options:
        logger.info(f"{'PEs':>6} | {'Grid':>8} | {'Rows/PE':>8} | {'NNZ/PE':>8} | {'Memory':>10}")
        logger.info("-" * 60)
        for n, rows, nnz, mem, pct, grid in options:
            logger.info(f"{n:>6} | {grid:>8} | {rows:>8.1f} | {nnz:>8.1f} | {mem:>6.0f} ({pct:>4.1f}%)")
    else:
        logger.warning("No valid PE counts found in range!")

    logger.info("="*60 + "\n")
