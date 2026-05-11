"""Intelligent PE count selection and auto-calculation."""

import numpy as np
import scipy.sparse as sp

from .memory import calculate_memory_per_pe
from ..core.grid import calculate_pe_grid

from utils.logging import logger


def calculate_min_pes_needed(matrix_nnz: int, matrix_rows: int, max_memory: int = 36_000) -> int:
    """Calculate minimum number of PEs needed to fit matrix in memory.

    Default max_memory is 36KB (75% of 48KB limit) to provide safety margin
    for compiler overhead and load balancing variations.
    """
    for num_pes in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        rows_per_pe = matrix_rows / num_pes
        nnz_per_pe = matrix_nnz / num_pes
        estimated_memory = (8 * rows_per_pe + 8 * nnz_per_pe + 8) * 1.2  # 20% overhead for imbalance

        if estimated_memory <= max_memory:
            return num_pes

    min_pes = int(np.ceil((8 * matrix_rows + 8 * matrix_nnz) * 1.2 / max_memory))
    return min_pes


def calculate_max_pes_useful(matrix_rows: int, max_width: int = 758, max_height: int = 1171) -> int:
    """Calculate maximum useful PEs (limited by rows and fabric dimensions)."""
    max_fabric_pes = max_width * max_height
    max_useful = min(matrix_rows, max_fabric_pes)

    nice_numbers = [1, 2, 4, 8, 16, 32, 64, 100, 128, 256, 512, 1024, 2048, 4096, 8192]

    for n in reversed(nice_numbers):
        if n <= max_useful:
            return n

    return max_useful


def parse_num_pes_arg(num_pes_str: str, matrix: sp.csr_matrix,
                     max_width: int = 758, max_height: int = 1171) -> int:
    """Parse --num-pes argument: number, 'min', or 'max'."""
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
