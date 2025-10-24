#!/usr/bin/env python3
"""
SpMV CSL Implementation - Multi-PE Support
Phase 2: Extends single PE to multiple PEs with timing and memory checks
"""

import argparse
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import json
import logging
import sys
import time

from cerebras.sdk.client import SdkCompiler, SdkRuntime
from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder

# Import our prototype code
from spmv_prototype import generate_random_sparse_matrix, MatchedPairsSpMV, validate_results

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# PE Grid Calculation
# ==============================================================================

def calculate_pe_grid(num_pes: int, 
                     max_width: int = 758, 
                     max_height: int = 1171) -> tuple:
    """
    Calculate optimal 2D PE grid that fits within fabric constraints.
    Prefers more square-like grids for better locality.
    
    Args:
        num_pes: Total number of PEs
        max_width: Maximum fabric width (default: 758 for WSE-3)
        max_height: Maximum fabric height (default: 1171 for WSE-3)
    
    Returns:
        (width, height) tuple
    
    Raises:
        ValueError: If num_pes doesn't fit in fabric
    """
    if num_pes == 1:
        return (1, 1)
    
    if num_pes > max_width * max_height:
        raise ValueError(
            f"Cannot fit {num_pes} PEs in fabric "
            f"(max: {max_width}×{max_height} = {max_width * max_height})"
        )
    
    # Find all valid factorizations
    factorizations = []
    for w in range(1, min(num_pes, max_width) + 1):
        if num_pes % w == 0:
            h = num_pes // w
            if h <= max_height:
                # Score: prefer square-ish grids
                # Ratio closer to 1.0 is better
                aspect_ratio = max(w, h) / min(w, h)
                factorizations.append((w, h, aspect_ratio))
    
    if not factorizations:
        raise ValueError(
            f"Cannot fit {num_pes} PEs in fabric constraints "
            f"(max_width={max_width}, max_height={max_height})"
        )
    
    # Choose factorization with best (lowest) aspect ratio
    factorizations.sort(key=lambda x: x[2])
    width, height, _ = factorizations[0]
    
    return (width, height)


def print_grid_info(width: int, height: int, num_pes: int, 
                   fabric_dims: tuple = (762, 1172),
                   fabric_offsets: tuple = (4, 1)):
    """Print visual representation of PE grid with fabric info."""
    logger.info("="*60)
    logger.info(f"PE Grid Layout: {width}×{height} ({num_pes} PEs)")
    logger.info("="*60)
    logger.info(f"Fabric dimensions: {fabric_dims[0]}×{fabric_dims[1]}")
    logger.info(f"Fabric offsets: {fabric_offsets}")
    logger.info(f"Usable area: {fabric_dims[0]-fabric_offsets[0]}×{fabric_dims[1]-fabric_offsets[1]}")
    logger.info(f"Grid utilization: {(width*height)/(fabric_dims[0]*fabric_dims[1])*100:.2f}%")
    logger.info("")
    
    # Visual grid (limit display for large grids)
    if width <= 20 and height <= 20:
        for y in range(height):
            row = ""
            for x in range(width):
                pe_id = y * width + x
                row += f"[PE{pe_id:3d}] "
            logger.info(row)
    else:
        logger.info(f"Grid too large to display ({width}×{height})")
        logger.info(f"Corners:")
        logger.info(f"  Top-left:     PE 0")
        logger.info(f"  Top-right:    PE {width-1}")
        logger.info(f"  Bottom-left:  PE {(height-1)*width}")
        logger.info(f"  Bottom-right: PE {height*width-1}")
    
    logger.info("="*60 + "\n")


def validate_grid_fits_fabric(width: int, height: int, 
                              fabric_dims: tuple = (762, 1172),
                              fabric_offsets: tuple = (4, 1)) -> bool:
    """Check if PE grid fits within fabric constraints."""
    max_width = fabric_dims[0] - fabric_offsets[0]
    max_height = fabric_dims[1] - fabric_offsets[1]
    
    if width > max_width or height > max_height:
        logger.error("="*60)
        logger.error("FABRIC CONSTRAINT VIOLATION!")
        logger.error("="*60)
        logger.error(f"Requested grid: {width}×{height}")
        logger.error(f"Maximum allowed: {max_width}×{max_height}")
        logger.error(f"Fabric dims: {fabric_dims}, offsets: {fabric_offsets}")
        logger.error("="*60)
        return False
    
    return True


# ==============================================================================
# Memory Calculation
# ==============================================================================

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
        logger.info(f"  Memory breakdown:")
        logger.info(f"    row_offsets:    {row_offsets_mem:>6} bytes")
        logger.info(f"    matrix_values:  {matrix_values_mem:>6} bytes")
        logger.info(f"    vector_values:  {vector_values_mem:>6} bytes")
        logger.info(f"    results:        {results_mem:>6} bytes")
        logger.info(f"    num_rows:       {4:>6} bytes")
        logger.info(f"  Total: {memory:>6} bytes ({memory/1024:.2f} KB, {percentage:.1f}% of limit)")
        
        if memory > max_memory:
            logger.error(f"  ❌ EXCEEDS 48 KB LIMIT!")
        else:
            logger.info(f"  ✅ Within limit")
    
    max_memory_used = max(calculate_memory_per_pe(pe['num_rows'], pe['total_nnz']) 
                          for pe in pe_data)
    avg_memory_used = np.mean([calculate_memory_per_pe(pe['num_rows'], pe['total_nnz']) 
                                for pe in pe_data])
    
    logger.info(f"\nAggregate Statistics:")
    logger.info(f"  Max memory/PE:  {max_memory_used:>6} bytes ({max_memory_used/1024:.2f} KB)")
    logger.info(f"  Avg memory/PE:  {avg_memory_used:>6.0f} bytes ({avg_memory_used/1024:.2f} KB)")
    logger.info(f"  Memory limit:   {max_memory:>6} bytes ({max_memory/1024:.2f} KB)")
    logger.info("="*60 + "\n")


# ==============================================================================
# Auto PE Count Selection
# ==============================================================================

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
            except:
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


# ==============================================================================
# Compilation
# ==============================================================================

def compile_csl(
    width: int,
    height: int,
    total_nnz_per_pe: int,
    max_rows_per_pe: int,
    arch: str = "wse3",
    fabric_dims: tuple = (762, 1172),
    fabric_offsets: tuple = (4, 1),
    artifact_path: str = "artifact.json",
    out_path: str = "out"
):
    """Compile the CSL code using SdkCompiler."""
    logger.info("="*60)
    logger.info("Compiling CSL code")
    logger.info("="*60)
    logger.info(f"  PE grid: {width}×{height}")
    logger.info(f"  TOTAL_NNZ per PE: {total_nnz_per_pe}")
    logger.info(f"  MAX_ROWS per PE: {max_rows_per_pe}")
    logger.info(f"  Architecture: {arch}")
    logger.info(f"  Fabric dims: {fabric_dims}")
    
    # Ensure out_path exists and is absolute
    out_path = str(Path(out_path).resolve())
    Path(out_path).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"  Output path: {out_path}")
    
    # Build compilation arguments
    ARGS = (
        f"--arch={arch} "
        f"--fabric-dims={fabric_dims[0]},{fabric_dims[1]} "
        f"--fabric-offsets={fabric_offsets[0]},{fabric_offsets[1]} "
        f"--memcpy --channels=1 "
        f"--params=width:{width},height:{height},"
        f"TOTAL_NNZ:{total_nnz_per_pe},MAX_ROWS:{max_rows_per_pe}"
    )
    
    logger.info(f"  Compilation args: {ARGS}")
    
    # Compile using SdkCompiler
    compile_start = time.time()
    
    with SdkCompiler(resource_cpu=24000, resource_mem=64<<30, disable_version_check=True) as compiler:
        artifact_id = compiler.compile(
            app_path=".",
            csl_main="layout.csl",
            options=ARGS,
            out_path=out_path,
        )
    
    compile_time = time.time() - compile_start
    
    if artifact_id is None:
        logger.error("Compilation failed! No artifact was generated!")
        raise RuntimeError("CSL compilation failed")
    
    # Save artifact metadata
    artifact_data = {
        "artifact_id": artifact_id,
        "fabric_dims": list(fabric_dims),
        "params": {
            "width": width,
            "height": height,
            "TOTAL_NNZ": total_nnz_per_pe,
            "MAX_ROWS": max_rows_per_pe,
        },
        "compile_time_seconds": compile_time
    }
    
    artifact_file = Path(artifact_path)
    with open(artifact_file, "w", encoding="utf8") as f:
        json.dump(artifact_data, f, indent=2)
    
    logger.info("✅ Compilation successful!")
    logger.info(f"  Compile time: {compile_time:.2f} seconds")
    logger.info(f"  Artifact ID: {artifact_id}")
    logger.info(f"  Artifact saved to: {artifact_file.resolve()}")
    logger.info("="*60 + "\n")
    
    return out_path, artifact_id, compile_time


# ==============================================================================
# Runtime
# ==============================================================================

def run_spmv_on_device(
    csr_matrix: sp.csr_matrix,
    vector_x: np.ndarray,
    artifact_path: str,
    num_pes: int,
    width: int,
    height: int
):
    """Run SpMV on the Cerebras device (or simulator)."""
    
    run_start = time.time()
    
    # Distribute data using our prototype
    spmv = MatchedPairsSpMV(csr_matrix, num_pes)
    pe_data = spmv.distribute_by_nnz()
    spmv.prepare_matched_pairs(pe_data, vector_x)
    
    # Print distribution summary
    logger.info("="*60)
    logger.info("Multi-PE Data Summary")
    logger.info("="*60)
    
    # Only show non-empty PEs
    non_empty = [pe for pe in pe_data if pe['num_rows'] > 0]
    if len(non_empty) <= 10:
        for pe in non_empty:
            logger.info(f"PE {pe['pe_id']}:")
            logger.info(f"  Rows: {pe['num_rows']}")
            logger.info(f"  NNZ: {pe['total_nnz']}")
    else:
        # Show summary
        logger.info(f"Active PEs: {len(non_empty)}/{num_pes}")
        logger.info(f"First PE: {non_empty[0]['num_rows']} rows, {non_empty[0]['total_nnz']} NNZ")
        logger.info(f"Last PE: {non_empty[-1]['num_rows']} rows, {non_empty[-1]['total_nnz']} NNZ")
    
    logger.info("="*60 + "\n")
    
    # Initialize SDK runtime
    logger.info("Initializing SdkRuntime...")
    with SdkRuntime(artifact_path, suppress_simfab_trace=True, disable_version_check=True) as runner:
        
        # Get symbol IDs
        sym_num_rows = runner.get_id("num_rows")
        sym_row_offsets = runner.get_id("row_offsets")
        sym_matrix_values = runner.get_id("matrix_values")
        sym_vector_values = runner.get_id("vector_values")
        sym_results = runner.get_id("results")
        
        # Send data to ALL PEs
        logger.info("Sending data to PEs...")
        memcpy_h2d_start = time.time()
        
        for pe in pe_data:
            pe_id = pe['pe_id']
            
            # Calculate PE grid position
            x = pe_id % width
            y = pe_id // width
            
            logger.debug(f"  Sending to PE {pe_id} at ({x}, {y})")
            
            # Send num_rows
            runner.memcpy_h2d(
                sym_num_rows,
                np.array([pe['num_rows']], dtype=np.int32),
                x, y, 1, 1,
                1,
                streaming=False,
                data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=False
            )
            
            # Send row_offsets (handle empty PEs)
            if pe['num_rows'] > 0:
                row_offsets_to_send = pe['row_offsets'].astype(np.int32)
            else:
                row_offsets_to_send = np.array([0], dtype=np.int32)
            
            runner.memcpy_h2d(
                sym_row_offsets,
                row_offsets_to_send,
                x, y, 1, 1,
                len(row_offsets_to_send),
                streaming=False,
                data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=False
            )
            
            # Send matrix_values (handle empty PEs)
            if pe['total_nnz'] > 0:
                matrix_values_to_send = pe['matrix_values']
            else:
                matrix_values_to_send = np.array([0.0], dtype=np.float32)
            
            runner.memcpy_h2d(
                sym_matrix_values,
                matrix_values_to_send,
                x, y, 1, 1,
                len(matrix_values_to_send),
                streaming=False,
                data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=False
            )
            
            # Send vector_values (handle empty PEs)
            if pe['total_nnz'] > 0:
                vector_values_to_send = pe['vector_values']
            else:
                vector_values_to_send = np.array([0.0], dtype=np.float32)
            
            runner.memcpy_h2d(
                sym_vector_values,
                vector_values_to_send,
                x, y, 1, 1,
                len(vector_values_to_send),
                streaming=False,
                data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=False
            )
        
        memcpy_h2d_time = time.time() - memcpy_h2d_start
        logger.info(f"✅ Data sent to all PEs ({memcpy_h2d_time:.3f}s)")
        
        # Launch computation on all PEs
        logger.info("Launching SpMV computation...")
        compute_start = time.time()
        runner.launch("spmv", nonblock=False)
        compute_time = time.time() - compute_start
        logger.info(f"✅ Computation complete ({compute_time:.3f}s)")
        
        # Retrieve results from ALL PEs
        logger.info("Retrieving results from PEs...")
        memcpy_d2h_start = time.time()
        
        all_results = []
        for pe in pe_data:
            pe_id = pe['pe_id']
            x = pe_id % width
            y = pe_id // width
            
            # Handle empty PEs
            if pe['num_rows'] > 0:
                results_raw = np.zeros(pe['num_rows'], dtype=np.float32)
                runner.memcpy_d2h(
                    results_raw,
                    sym_results,
                    x, y, 1, 1,
                    pe['num_rows'],
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_32BIT,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False
                )
            else:
                results_raw = np.array([], dtype=np.float32)
            
            all_results.append(results_raw)
        
        memcpy_d2h_time = time.time() - memcpy_d2h_start
        logger.info(f"✅ Results retrieved from all PEs ({memcpy_d2h_time:.3f}s)")
    
    # Gather results using prototype logic
    result_vector = spmv.gather_results(pe_data, all_results)
    
    run_time = time.time() - run_start
    
    timing_info = {
        'total_run': run_time,
        'memcpy_h2d': memcpy_h2d_time,
        'compute': compute_time,
        'memcpy_d2h': memcpy_d2h_time
    }
    
    return result_vector, timing_info


# ==============================================================================
# Timing Summary
# ==============================================================================

def print_timing_summary(compile_time: float, timing_info: dict, total_time: float):
    """Print timing summary at the end."""
    logger.info("\n" + "="*60)
    logger.info("Timing Summary")
    logger.info("="*60)
    logger.info(f"Compilation:        {compile_time:>8.3f} s")
    logger.info(f"Runtime breakdown:")
    logger.info(f"  H2D memcpy:       {timing_info['memcpy_h2d']:>8.3f} s")
    logger.info(f"  Computation:      {timing_info['compute']:>8.3f} s")
    logger.info(f"  D2H memcpy:       {timing_info['memcpy_d2h']:>8.3f} s")
    logger.info(f"  Total runtime:    {timing_info['total_run']:>8.3f} s")
    logger.info(f"-" * 60)
    logger.info(f"Total (end-to-end): {total_time:>8.3f} s")
    logger.info("="*60 + "\n")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SpMV CSL Multi-PE Implementation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-calculate minimum PEs needed
  python run.py --rows 1000 --cols 1000 --density 0.01 --num-pes min
  
  # Auto-calculate maximum useful PEs
  python run.py --rows 1000 --cols 1000 --density 0.01 --num-pes max
  
  # Specific number of PEs
  python run.py --rows 1000 --cols 1000 --density 0.01 --num-pes 16
  
  # Show PE count suggestions
  python run.py --rows 1000 --cols 1000 --density 0.01 --suggest-pes
        """
    )
    parser.add_argument('--rows', type=int, default=10, help='Number of rows')
    parser.add_argument('--cols', type=int, default=10, help='Number of columns')
    parser.add_argument('--density', type=float, default=0.1, help='Sparsity density')
    parser.add_argument('--num-pes', type=str, default='1', 
                       help='Number of PEs: integer, "min", or "max"')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--suggest-pes', action='store_true',
                       help='Show PE count suggestions and exit')
    parser.add_argument('--compile-only', action='store_true', 
                       help='Only compile, do not run')
    parser.add_argument('--artifact-path', type=str, default='artifact.json',
                       help='Path to save artifact metadata')
    parser.add_argument('--out-path', type=str, default='out',
                       help='Compilation output directory')
    parser.add_argument('--arch', type=str, default='wse3',
                       choices=['wse2', 'wse3'],
                       help='Target architecture')
    args = parser.parse_args()
    
    total_start = time.time()
    
    logger.info("="*60)
    logger.info("SpMV CSL Multi-PE Implementation")
    logger.info("="*60 + "\n")
    
    # Fabric constraints based on architecture
    if args.arch == 'wse3':
        fabric_dims = (762, 1172)
        fabric_offsets = (4, 1)
    else:  # wse2
        fabric_dims = (757, 996)
        fabric_offsets = (4, 1)
    
    max_width = fabric_dims[0] - fabric_offsets[0]
    max_height = fabric_dims[1] - fabric_offsets[1]
    
    # Generate test matrix
    logger.info(f"Generating random matrix: {args.rows}×{args.cols}, density={args.density}")
    A = generate_random_sparse_matrix(args.rows, args.cols, args.density, args.seed)
    x = np.random.randint(0, 100, size=args.cols, dtype=np.int32).astype(np.float32)
    
    logger.info(f"Matrix: {A.shape}, NNZ: {A.nnz}")
    logger.info(f"Vector: {x.shape}\n")
    
    # Show PE suggestions if requested
    if args.suggest_pes:
        suggest_pe_counts(A.nnz, A.shape[0], max_width, max_height)
        return 0
    
    # Parse num_pes argument (can be 'min', 'max', or number)
    try:
        num_pes = parse_num_pes_arg(args.num_pes, A, max_width, max_height)
    except ValueError as e:
        logger.error(str(e))
        return 1
    
    logger.info(f"Number of PEs: {num_pes}\n")
    
    # Validate num_pes is reasonable
    if num_pes < 1:
        logger.error("Number of PEs must be at least 1")
        return 1
    
    if num_pes > A.shape[0]:
        logger.warning(f"⚠️  More PEs ({num_pes}) than rows ({A.shape[0]}). Some PEs will be idle.")
    
    # Calculate optimal PE grid
    try:
        width, height = calculate_pe_grid(num_pes, max_width, max_height)
    except ValueError as e:
        logger.error(str(e))
        return 1
    
    # Validate grid fits
    if not validate_grid_fits_fabric(width, height, fabric_dims, fabric_offsets):
        return 1
    
    print_grid_info(width, height, num_pes, fabric_dims, fabric_offsets)
    
    # Estimate required sizes per PE and check memory
    spmv_temp = MatchedPairsSpMV(A, num_pes)
    pe_data_temp = spmv_temp.distribute_by_nnz()
    spmv_temp.prepare_matched_pairs(pe_data_temp, x)
    
    # Memory feasibility check
    is_feasible, max_memory, offending_pe = check_memory_feasibility(pe_data_temp)
    
    print_memory_analysis(pe_data_temp)
    
    if not is_feasible:
        logger.error("="*60)
        logger.error("MEMORY LIMIT EXCEEDED!")
        logger.error("="*60)
        logger.error(f"PE {offending_pe} requires {max_memory} bytes ({max_memory/1024:.2f} KB)")
        logger.error(f"This exceeds the 48 KB (48,000 bytes) limit per PE.")
        logger.error("\nSuggestions:")
        min_pes_needed = calculate_min_pes_needed(A.nnz, A.shape[0])
        logger.error(f"  1. Use at least {min_pes_needed} PEs (try --num-pes {min_pes_needed})")
        logger.error(f"  2. Or use --num-pes min for automatic calculation")
        logger.error(f"  3. Reduce matrix size (current: {args.rows}×{args.cols})")
        logger.error(f"  4. Reduce density (current: {args.density})")
        logger.error("="*60)
        return 1
    
    # Find max across all PEs for compilation
    max_nnz = max(pe['total_nnz'] for pe in pe_data_temp)
    max_rows = max(pe['num_rows'] for pe in pe_data_temp)
    
    # Add padding
    total_nnz_padded = max(int(max_nnz * 1.2) + 10, 10)
    max_rows_padded = max_rows + 5
    
    logger.info(f"CSL Parameters (per PE):")
    logger.info(f"  PE Grid: {width}×{height}")
    logger.info(f"  TOTAL_NNZ: {total_nnz_padded} (max actual: {max_nnz})")
    logger.info(f"  MAX_ROWS: {max_rows_padded} (max actual: {max_rows})\n")
    
    # Compile CSL
    out_path, artifact_id, compile_time = compile_csl(
        width=width,
        height=height,
        total_nnz_per_pe=total_nnz_padded,
        max_rows_per_pe=max_rows_padded,
        arch=args.arch,
        fabric_dims=fabric_dims,
        fabric_offsets=fabric_offsets,
        artifact_path=args.artifact_path,
        out_path=args.out_path
    )
    
    if args.compile_only:
        logger.info("Compilation complete. Exiting (--compile-only specified).")
        total_time = time.time() - total_start
        logger.info(f"\nTotal time: {total_time:.2f} seconds")
        return 0
    
    # Run on device
    logger.info("Running SpMV on device...\n")
    y_computed, timing_info = run_spmv_on_device(A, x, artifact_id, num_pes, width, height)
    
    # Compute reference
    logger.info("Computing reference solution...")
    y_reference = A.dot(x).astype(np.float32)
    
    # Validate
    passed = validate_results(y_computed, y_reference, tolerance=1e-4)
    
    total_time = time.time() - total_start
    
    # Print timing summary
    print_timing_summary(compile_time, timing_info, total_time)
    
    if passed:
        logger.info("🎉 SUCCESS! Multi-PE SpMV works correctly!")
        return 0
    else:
        logger.error("❌ FAILED! Results do not match.")
        return 1


if __name__ == "__main__":
    exit(main())
