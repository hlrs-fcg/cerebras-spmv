"""
Utility modules for SpMV
"""

from .memory import calculate_memory_per_pe, check_memory_feasibility, print_memory_analysis
from .pe_selection import (
    calculate_min_pes_needed,
    calculate_max_pes_useful,
    parse_num_pes_arg,
    suggest_pe_counts
)
from .timing import print_timing_summary

__all__ = [
    'calculate_memory_per_pe',
    'check_memory_feasibility',
    'print_memory_analysis',
    'calculate_min_pes_needed',
    'calculate_max_pes_useful',
    'parse_num_pes_arg',
    'suggest_pe_counts',
    'print_timing_summary',
]
