#!/usr/bin/env python3
"""
Cluster Compilation Script
Compiles CSL code on login node for later execution via SdkLauncher.
"""

import argparse
import json
import logging
import sys
from pathlib import Path


from spmv_prototype import generate_random_sparse_matrix, MatchedPairsSpMV
from spmv_runner.core import (
    calculate_pe_grid,
    print_grid_info,
    validate_grid_fits_fabric,
    compile_csl
)
from spmv_runner.utils import (
    check_memory_feasibility,
    print_memory_analysis,
    parse_num_pes_arg
)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_matrix_market(mtx_path):
    """Load matrix from MatrixMarket file using fast-matrix-market."""
    try:
        import fast_matrix_market as fmm
    except ImportError:
        logger.error("fast-matrix-market not installed. Run: pip install fast-matrix-market")
        raise

    logger.info(f"Loading MatrixMarket file: {mtx_path}")
    A = fmm.mmread(mtx_path)
    logger.info(f"Loaded matrix: {A.shape[0]}×{A.shape[1]}, NNZ={A.nnz:,}")
    return A


def fetch_suitesparse_matrix(identifier, output_dir='bench/matrices'):
    """Fetch matrix from SuiteSparse collection and return path to .mtx file."""
    try:
        import ssgetpy
    except ImportError:
        logger.error("ssgetpy not installed. Run: pip install ssgetpy")
        raise

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching SuiteSparse matrix: {identifier}")
    matrix_info = ssgetpy.fetch(identifier, format='MM', location=str(output_path), extract=True)

    matrix_dir = output_path / matrix_info['name']
    mtx_files = list(matrix_dir.glob('*.mtx'))
    if not mtx_files:
        raise FileNotFoundError(f"No .mtx file in {matrix_dir}")

    logger.info(f"Fetched: {matrix_info['name']} ({matrix_info['rows']}×{matrix_info['cols']}, NNZ={matrix_info['nnz']:,})")
    return mtx_files[0]


def main():
    """Compile CSL code for cluster execution."""
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description='Compile CSL code for SpMV cluster execution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Random matrix generation
  python cluster/compile.py --rows 1000 --cols 1000 --density 0.01 --num-pes min

  # From MatrixMarket file
  python cluster/compile.py --mtx-file bench/matrices/HB_bcsstk14/HB_bcsstk14.mtx --num-pes min

  # Fetch from SuiteSparse by ID
  python cluster/compile.py --fetch-id 1 --num-pes min

  # Fetch by name
  python cluster/compile.py --fetch-id HB/bcsstk14 --num-pes max
        """
    )

    # Matrix source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--mtx-file', type=str, help='Path to MatrixMarket (.mtx) file')
    source_group.add_argument('--fetch-id', type=str, help='SuiteSparse matrix ID or name (e.g., 1 or HB/bcsstk14)')
    source_group.add_argument('--random', action='store_true', help='Generate random matrix (requires --rows, --cols, --density)')

    # Random matrix parameters
    parser.add_argument('--rows', type=int, help='Number of matrix rows (for random generation)')
    parser.add_argument('--cols', type=int, help='Number of matrix columns (for random generation)')
    parser.add_argument('--density', type=float, help='Sparsity density 0-1 (for random generation)')
    parser.add_argument('--num-pes', type=str, default='min',
                       help='Number of PEs: integer, "min", or "max" (default: min)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for matrix generation (default: 42)')
    parser.add_argument('--artifact-path', type=str, default='cluster_work/artifact_path.json',
                       help='Path to save artifact metadata JSON (default: cluster_work/artifact_path.json)')
    parser.add_argument('--out-path', type=str, default='cluster_work/out',
                       help='Compilation output directory (default: cluster_work/out)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose debug output')
    args = parser.parse_args()

    # Validate arguments
    if args.random:
        if not all([args.rows, args.cols, args.density]):
            logger.error("--rows, --cols, and --density required with --random")
            return 1

    # resolve paths
    if not Path(args.artifact_path).is_absolute():
        args.artifact_path = str(script_dir / args.artifact_path)
    if not Path(args.out_path).is_absolute():
        args.out_path = str(script_dir / args.out_path)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    logger.info("="*60)
    logger.info("Cluster Compilation - Login Node")
    logger.info("="*60)

    fabric_dims = (762, 1172)
    fabric_offsets = (4, 1)

    max_width = fabric_dims[0] - fabric_offsets[0]
    max_height = fabric_dims[1] - fabric_offsets[1]

    # Load or generate matrix
    try:
        if args.mtx_file:
            A = load_matrix_market(args.mtx_file)
        elif args.fetch_id:
            # Try to parse as int, otherwise treat as name
            try:
                fetch_id = int(args.fetch_id)
            except ValueError:
                fetch_id = args.fetch_id
            mtx_path = fetch_suitesparse_matrix(fetch_id)
            A = load_matrix_market(mtx_path)
        else:  # --random
            logger.info(f"Generating random matrix: {args.rows}×{args.cols}, density={args.density}")
            A = generate_random_sparse_matrix(args.rows, args.cols, args.density, args.seed)
            logger.info(f"Matrix NNZ: {A.nnz:,}")
    except Exception as e:
        logger.error(f"Matrix loading/generation failed: {e}")
        return 1

    try:
        num_pes = parse_num_pes_arg(args.num_pes, A, max_width, max_height)
    except ValueError as e:
        logger.error(f"PE calculation failed: {e}")
        return 1

    logger.info(f"Number of PEs: {num_pes}")

    if num_pes < 1:
        logger.error("Number of PEs must be at least 1")
        return 1

    if num_pes > A.shape[0]:
        logger.warning(f"⚠️  More PEs ({num_pes}) than rows ({A.shape[0]}). Some PEs will be idle.")

    # calculate optimal PE grid
    try:
        width, height = calculate_pe_grid(num_pes, max_width, max_height)
    except ValueError as e:
        logger.error(f"Grid calculation failed: {e}")
        return 1

    # validate grid fits fabric
    if not validate_grid_fits_fabric(width, height, fabric_dims, fabric_offsets):
        logger.error(f"Grid {width}x{height} does not fit fabric dimensions")
        return 1

    print_grid_info(width, height, num_pes, fabric_dims, fabric_offsets)

    # perform load balancing to get actual max values per PE
    logger.info("Performing load balancing across PEs...")
    spmv = MatchedPairsSpMV(A, num_pes)
    pe_data = spmv.distribute_by_nnz()

    # check memory (targets 36KB = 75% of 48KB limit)
    is_feasible, max_memory, offending_pe = check_memory_feasibility(pe_data)
    print_memory_analysis(pe_data)

    if not is_feasible:
        logger.error("="*60)
        logger.error("MEMORY TARGET EXCEEDED!")
        logger.error("="*60)
        logger.error(f"PE {offending_pe} requires {max_memory} bytes ({max_memory/1024:.2f} KB)")
        logger.error("This exceeds the 36 KB target (75% of 48KB hardware limit)")
        logger.error("\nSuggestions:")
        logger.error("  1. Use --num-pes max for more PEs")
        logger.error("  2. Reduce matrix size or density")
        logger.error("="*60)
        return 1

    # use actual max values from distribution with minimal padding
    max_nnz = max(pe['total_nnz'] for pe in pe_data)
    max_rows = max(pe['num_rows'] for pe in pe_data)

    total_nnz_padded = max(max_nnz + 5, 10)
    max_rows_padded = max_rows + 2

    logger.info("\nCSL Compilation Parameters:")
    logger.info(f"  PE Grid: {width}×{height}")
    logger.info(f"  TOTAL_NNZ: {total_nnz_padded} (actual: {max_nnz}, padding: +{total_nnz_padded - max_nnz})")
    logger.info(f"  MAX_ROWS: {max_rows_padded} (actual: {max_rows}, padding: +{max_rows_padded - max_rows})")

    # compile CSL
    try:
        logger.info("\nCompiling CSL code...")
        out_path, artifact_id, compile_time = compile_csl(
            width=width,
            height=height,
            total_nnz_per_pe=total_nnz_padded,
            max_rows_per_pe=max_rows_padded,
            fabric_dims=fabric_dims,
            fabric_offsets=fabric_offsets,
            artifact_path=args.artifact_path,
            out_path=args.out_path
        )
    except Exception as e:
        logger.error(f"Compilation failed: {e}")
        return 1

    logger.info(f"✓ Compilation successful! Time: {compile_time:.2f}s")
    logger.info(f"✓ Artifact ID: {artifact_id}")
    logger.info(f"✓ Artifact metadata saved to: {args.artifact_path}")

    # load and update the artifact metadata with matrix info
    with open(args.artifact_path, 'r', encoding='utf8') as f:
        artifact_data = json.load(f)

    # add matrix configuration for reference
    # Convert numpy types to Python native types for JSON serialization
    if args.random:
        artifact_data['matrix_config'] = {
            'source': 'random',
            'rows': int(args.rows),
            'cols': int(args.cols),
            'density': float(args.density),
            'seed': int(args.seed),
            'nnz': int(A.nnz)
        }
    elif args.mtx_file:
        artifact_data['matrix_config'] = {
            'source': 'mtx_file',
            'mtx_file': str(args.mtx_file),
            'rows': int(A.shape[0]),
            'cols': int(A.shape[1]),
            'nnz': int(A.nnz)
        }
    else:  # fetch_id
        artifact_data['matrix_config'] = {
            'source': 'suitesparse',
            'fetch_id': args.fetch_id,
            'mtx_file': str(mtx_path),
            'rows': int(A.shape[0]),
            'cols': int(A.shape[1]),
            'nnz': int(A.nnz)
        }
    artifact_data['num_pes'] = int(num_pes)

    # save enhanced metadata
    with open(args.artifact_path, 'w', encoding='utf8') as f:
        json.dump(artifact_data, f, indent=2)

    logger.info("\n" + "="*60)
    logger.info("Next steps:")
    logger.info("  1. Run: python cluster/run.py")
    logger.info(f"     (will use artifact from {args.artifact_path})")
    logger.info("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
