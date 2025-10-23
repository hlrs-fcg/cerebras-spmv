#!/usr/bin/env python3
"""
SpMV Approach 1: Matched Pairs Prototype
Tests data distribution before CSL implementation
"""

import numpy as np
import scipy.sparse as sp
from typing import List, Dict
import argparse
import logging

class MatchedPairsSpMV:
    def __init__(self, csr_matrix: sp.csr_matrix, num_pes: int):
        self.csr_matrix = csr_matrix
        self.num_pes = num_pes
        self.num_rows = csr_matrix.shape[0]
        self.num_cols = csr_matrix.shape[1]
        self.total_nnz = csr_matrix.nnz
        
    def distribute_by_nnz(self) -> List[Dict]:
        """
        Distribute rows to PEs balanced by NNZ count.
        Returns list of PE data structures.
        """
        target_nnz_per_pe = self.total_nnz // self.num_pes
        logging.info(f"Target NNZ per PE: {target_nnz_per_pe}")
        
        pe_data = []
        for pe_id in range(self.num_pes):
            pe_data.append({
                'pe_id': pe_id,
                'row_ids': [],
                'row_offsets': [0],
                'matrix_values': [],
                'vector_values': [],
                'total_nnz': 0,
                'num_rows': 0
            })
        
        current_pe = 0
        current_nnz = 0
        
        for row in range(self.num_rows):
            row_start = self.csr_matrix.indptr[row]
            row_end = self.csr_matrix.indptr[row + 1]
            row_nnz = row_end - row_start
            
            # Switch to next PE if current is overloaded (but not last PE)
            if current_nnz + row_nnz > target_nnz_per_pe and current_pe < self.num_pes - 1:
                logging.debug(f"PE {current_pe} filled with {current_nnz} NNZ, switching to PE {current_pe + 1}")
                current_pe += 1
                current_nnz = 0
            
            # Assign row to current PE
            if row_nnz > 0:
                pe_data[current_pe]['row_ids'].append(row)
                pe_data[current_pe]['total_nnz'] += row_nnz
                pe_data[current_pe]['num_rows'] += 1
                
                # Store offset for this row
                pe_data[current_pe]['row_offsets'].append(
                    pe_data[current_pe]['row_offsets'][-1] + row_nnz
                )
                
                current_nnz += row_nnz
        
        return pe_data
    
    def prepare_matched_pairs(self, pe_data: List[Dict], vector_x: np.ndarray):
        """
        For each PE, gather vector values to create matched pairs.
        Modifies pe_data in place.
        """
        for pe in pe_data:
            matrix_vals = []
            vector_vals = []
            
            for row in pe['row_ids']:
                row_start = self.csr_matrix.indptr[row]
                row_end = self.csr_matrix.indptr[row + 1]
                
                # Get matrix values
                mat_vals = self.csr_matrix.data[row_start:row_end]
                
                # Gather corresponding vector values
                col_indices = self.csr_matrix.indices[row_start:row_end]
                vec_vals = vector_x[col_indices]
                
                matrix_vals.extend(mat_vals)
                vector_vals.extend(vec_vals)
                
                # Debug output for first few rows
                if len(pe['row_ids']) <= 5 or row < 5:
                    pairs_str = ", ".join(
                        f"(col={col_indices[i]}, mat={mat_vals[i]:.3f}, vec={vec_vals[i]})"
                        for i in range(len(mat_vals))
                    )
                    logging.debug(f"PE {pe['pe_id']}, Row {row}: {pairs_str}")
            
            pe['matrix_values'] = np.array(matrix_vals, dtype=np.float32)
            pe['vector_values'] = np.array(vector_vals, dtype=np.float32)
            pe['row_offsets'] = np.array(pe['row_offsets'], dtype=np.int16)
            pe['row_ids'] = np.array(pe['row_ids'], dtype=np.int16)
    
    def compute_on_pe(self, pe: Dict) -> np.ndarray:
        """
        Simulate computation on a single PE.
        Returns result vector for rows assigned to this PE.
        """
        results = np.zeros(pe['num_rows'], dtype=np.float32)
        
        for r in range(pe['num_rows']):
            start = pe['row_offsets'][r]
            end = pe['row_offsets'][r + 1]
            
            # Dot product: sum(matrix_val * vector_val)
            dot_product = np.sum(
                pe['matrix_values'][start:end] * pe['vector_values'][start:end]
            )
            results[r] = dot_product
            
            logging.debug(f"PE {pe['pe_id']}, local_row={r}, global_row={pe['row_ids'][r]}: "
                         f"dot_product={dot_product:.6f}")
        
        return results
    
    def gather_results(self, pe_data: List[Dict], pe_results: List[np.ndarray]) -> np.ndarray:
        """
        Gather results from all PEs into final result vector.
        """
        result_vector = np.zeros(self.num_rows, dtype=np.float32)
        
        for pe, results in zip(pe_data, pe_results):
            for local_idx, global_row in enumerate(pe['row_ids']):
                result_vector[global_row] = results[local_idx]
        
        return result_vector
    
    def print_distribution_stats(self, pe_data: List[Dict]):
        """Print statistics about the distribution."""
        logging.info("\n" + "="*60)
        logging.info("Distribution Statistics")
        logging.info("="*60)
        
        for pe in pe_data:
            logging.info(f"PE {pe['pe_id']}:")
            logging.info(f"  Rows: {pe['num_rows']}")
            logging.info(f"  NNZ:  {pe['total_nnz']}")
            logging.info(f"  Data size: {pe['total_nnz'] * 8} bytes (matrix + vector)")
            if pe['num_rows'] > 0:
                logging.info(f"  Avg NNZ/row: {pe['total_nnz'] / pe['num_rows']:.2f}")
        
        total_nnz = sum(pe['total_nnz'] for pe in pe_data)
        max_nnz = max(pe['total_nnz'] for pe in pe_data) if pe_data else 0
        min_nnz = min(pe['total_nnz'] for pe in pe_data) if pe_data else 0
        
        logging.info(f"\nTotal NNZ: {total_nnz}")
        logging.info(f"Max NNZ on single PE: {max_nnz}")
        logging.info(f"Min NNZ on single PE: {min_nnz}")
        if total_nnz > 0:
            logging.info(f"Load imbalance: {max_nnz / (total_nnz / self.num_pes):.2f}x")
        logging.info("="*60 + "\n")


def generate_random_sparse_matrix(rows: int, cols: int, density: float, seed: int = 42):
    """Generate a random sparse matrix in CSR format."""
    np.random.seed(seed)
    matrix = sp.random(rows, cols, density=density, format='csr', dtype=np.float32)
    # Make sure values are reasonable for testing
    matrix.data = np.random.rand(matrix.nnz).astype(np.float32)
    return matrix


def validate_results(y_computed: np.ndarray, y_reference: np.ndarray, tolerance: float = 1e-4) -> bool:
    """
    Validate results with appropriate floating-point tolerances.
    
    Returns True if validation passes.
    """
    diff = np.abs(y_computed - y_reference)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Compute relative errors for non-zero entries
    nonzero_mask = np.abs(y_reference) > 1e-10
    if np.any(nonzero_mask):
        rel_diff = diff[nonzero_mask] / np.abs(y_reference[nonzero_mask])
        max_rel_diff = np.max(rel_diff)
        mean_rel_diff = np.mean(rel_diff)
    else:
        max_rel_diff = 0.0
        mean_rel_diff = 0.0
    
    logging.info("\n" + "="*60)
    logging.info("Validation")
    logging.info("="*60)
    logging.info(f"Absolute errors:")
    logging.info(f"  Max:  {max_diff:.2e}")
    logging.info(f"  Mean: {mean_diff:.2e}")
    
    if np.any(nonzero_mask):
        logging.info(f"Relative errors:")
        logging.info(f"  Max:  {max_rel_diff:.2e}")
        logging.info(f"  Mean: {mean_rel_diff:.2e}")
    
    # Check both absolute and relative tolerances
    passed = False
    if max_diff < tolerance:
        logging.info(f"✅ PASSED: Absolute error < {tolerance}")
        passed = True
    elif np.any(nonzero_mask) and max_rel_diff < tolerance:
        logging.info(f"✅ PASSED: Relative error < {tolerance}")
        passed = True
    else:
        logging.error(f"❌ FAILED: Errors exceed tolerance {tolerance}")
        logging.error(f"  Worst absolute: {max_diff:.2e}")
        if np.any(nonzero_mask):
            logging.error(f"  Worst relative: {max_rel_diff:.2e}")
        
        # Show worst offenders
        worst_idx = np.argsort(diff)[-5:][::-1]
        logging.error(f"  Worst mismatches:")
        for idx in worst_idx:
            logging.error(f"    Row {idx}: computed={y_computed[idx]:.6f}, "
                         f"reference={y_reference[idx]:.6f}, "
                         f"diff={diff[idx]:.2e}")
    
    logging.info("="*60 + "\n")
    return passed


def main():
    parser = argparse.ArgumentParser(description='SpMV Matched Pairs Prototype')
    parser.add_argument('--rows', type=int, default=10, help='Number of rows')
    parser.add_argument('--cols', type=int, default=10, help='Number of columns')
    parser.add_argument('--density', type=float, default=0.1, help='Sparsity density')
    parser.add_argument('--num-pes', type=int, default=1, help='Number of PEs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--tolerance', type=float, default=1e-4, 
                       help='Validation tolerance for f32 errors')
    parser.add_argument('--log', type=str, default='info', 
                       choices=['debug', 'info', 'warning'],
                       help='Logging level')
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log.upper())
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s:%(message)s'
    )
    
    # Generate test data
    logging.info(f"Generating random matrix: {args.rows}x{args.cols}, density={args.density}")
    A = generate_random_sparse_matrix(args.rows, args.cols, args.density, args.seed)
    x = np.random.randint(0, 100, size=args.cols, dtype=np.int32)
    
    logging.debug(f"Matrix NNZ: {A.nnz}")
    logging.debug(f"Vector x: {x}")
    
    # Initialize SpMV
    spmv = MatchedPairsSpMV(A, args.num_pes)
    
    # Distribute rows
    logging.info(f"\nDistributing {args.rows} rows over {args.num_pes} PEs...")
    pe_data = spmv.distribute_by_nnz()
    
    # Prepare matched pairs
    logging.info("Preparing matched pairs...")
    spmv.prepare_matched_pairs(pe_data, x)
    
    # Print distribution stats
    spmv.print_distribution_stats(pe_data)
    
    # Compute on each PE
    logging.info("Computing SpMV on each PE...")
    pe_results = []
    for pe in pe_data:
        results = spmv.compute_on_pe(pe)
        pe_results.append(results)
    
    # Gather results
    y_computed = spmv.gather_results(pe_data, pe_results)
    
    # Reference computation
    y_reference = A.dot(x).astype(np.float32)
    
    # Validate with appropriate tolerance
    passed = validate_results(y_computed, y_reference, tolerance=args.tolerance)
    
    if not passed:
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
