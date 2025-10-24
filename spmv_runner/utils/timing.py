"""
Timing Utilities

Provides timing and performance reporting functionality.
"""

import logging

logger = logging.getLogger(__name__)


def print_timing_summary(compile_time: float, timing_info: dict, total_time: float):
    """Print timing summary at the end."""
    logger.info("\n" + "="*60)
    logger.info("Timing Summary")
    logger.info("="*60)
    logger.info(f"Compilation:        {compile_time:>8.3f} s")
    logger.info("Runtime breakdown:")
    logger.info(f"  H2D memcpy:       {timing_info['memcpy_h2d']:>8.3f} s")
    logger.info(f"  Computation:      {timing_info['compute']:>8.3f} s")
    logger.info(f"  D2H memcpy:       {timing_info['memcpy_d2h']:>8.3f} s")
    logger.info(f"  Total runtime:    {timing_info['total_run']:>8.3f} s")
    logger.info("-" * 60)
    logger.info(f"Total (end-to-end): {total_time:>8.3f} s")
    logger.info("="*60 + "\n")
