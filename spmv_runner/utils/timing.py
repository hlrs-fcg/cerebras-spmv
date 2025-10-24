"""
Timing Utilities

Provides timing and performance reporting functionality.
"""

import logging

logger = logging.getLogger(__name__)


def print_timing_summary(compile_time: float, timing_info: dict, total_time: float):
    """Print timing summary at the end."""
    logger.info("\nTiming Summary:")
    logger.info(f"  Compilation:    {compile_time:>8.3f}s")
    logger.info(f"  H2D transfer:   {timing_info['memcpy_h2d']:>8.3f}s")
    logger.info(f"  Computation:    {timing_info['compute']:>8.3f}s")
    logger.info(f"  D2H transfer:   {timing_info['memcpy_d2h']:>8.3f}s")
    logger.info(f"  Total runtime:  {timing_info['total_run']:>8.3f}s")
    logger.info(f"  End-to-end:     {total_time:>8.3f}s")
