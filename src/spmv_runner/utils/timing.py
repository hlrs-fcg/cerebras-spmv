"""Timing and performance reporting."""

import logging

logger = logging.getLogger(__name__)


def print_timing_summary(compile_time: float, timing_info: dict, total_time: float, matrix_nnz: int = 0):
    logger.info("\n" + "="*60)
    logger.info("Timing Summary")
    logger.info("="*60)
    logger.info(f"Compilation:        {compile_time:>10.3f} s")
    logger.info("")
    logger.info("Runtime breakdown:")
    logger.info(f"  H2D memcpy:       {timing_info['memcpy_h2d']:>10.3f} s")
    logger.info("  Computation (HW TSC):")
    logger.info(f"    Min PE:         {timing_info['compute_hw_min']:>10.6f} s ({timing_info['compute_hw_cycles_min']:>10,} cycles)")
    logger.info(f"    Avg PE:         {timing_info['compute_hw_avg']:>10.6f} s ({timing_info['compute_hw_cycles_avg']:>10,} cycles)")
    logger.info(f"    Max PE:         {timing_info['compute_hw_max']:>10.6f} s ({timing_info['compute_hw_cycles_max']:>10,} cycles)")
    logger.info(f"    Std Dev:        {timing_info['compute_hw_std']:>10.6f} s ({timing_info['compute_hw_cycles_std']:>10,} cycles)")
    logger.info(f"  D2H memcpy:       {timing_info['memcpy_d2h']:>10.3f} s")
    logger.info("-" * 60)
    logger.info(f"Total runtime:      {timing_info['total_run']:>10.3f} s")
    logger.info(f"End-to-end:         {total_time:>10.3f} s")
    logger.info("="*60)

    if matrix_nnz > 0:
        flops = 2 * matrix_nnz
        gflops_hw = (flops / timing_info['compute_hw_avg']) / 1e9
        logger.info(f"\nPerformance: {gflops_hw:>10.2f} GFLOPS")

    logger.info("="*60 + "\n")
