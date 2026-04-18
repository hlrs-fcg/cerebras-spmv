"""
Core execution modules for SpMV
"""

from .grid import calculate_pe_grid, print_grid_info, validate_grid_fits_fabric
from .compiler import compile_csl

__all__ = [
    'calculate_pe_grid',
    'print_grid_info',
    'validate_grid_fits_fabric',
    'compile_csl',
]
