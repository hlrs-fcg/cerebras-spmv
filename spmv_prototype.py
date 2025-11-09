#!/usr/bin/env python3
"""
SpMV Matched Pairs Prototype - Phase 1
Pure Python implementation to test data distribution and computation logic
before implementing in CSL.
"""

import argparse
import numpy as np
import scipy.sparse as sp
import logging


class MatchedPairsSpMV:
    """
    Prototype implementation of SpMV using matched pairs approach.
    Pre-gathers vector values on host to create matched (matrix_val, vector_val) pairs.
    """
    
    def __init__(self, csr_matrix: sp.csr_matrix, num_pes: int):
        """
        Initialize SpMV with a CSR matrix.
        
        Args:
            csr_matrix: Input sparse matrix in CSR format
            num_pes: Number of PEs to distribute across
        """
        self.csr_matrix = csr_matrix
        self.num_pes = num_pes
        
    def distribute_by_nnz(self) -> list:
        """
        Distribute rows sequentially across PEs to balance NNZ.
        Simple greedy approach: fill each PE until target is reached.
        """
        target_nnz_per_pe = self.csr_matrix.nnz / self.num_pes
        
        logging.info(f"Distributing {self.csr_matrix.shape[0]} rows over {self.num_pes} PEs...")
        logging.info(f"Target NNZ per PE: {target_nnz_per_pe:.0f}")
        
        pe_data = []
        current_pe_id = 0
        current_nnz = 0
        current_rows = []
        
        for row in range(self.csr_matrix.shape[0]):
            row_nnz = self.csr_matrix.indptr[row+1] - self.csr_matrix.indptr[row]
            
            current_rows.append(row)
            current_nnz += row_nnz
            
            # Check if we should finalize this PE
            # Don't finalize if we're on the last PE (let it accumulate all remaining)
            should_finalize = False
            
            if current_pe_id < self.num_pes - 1:
                # Not the last PE: finalize when we hit target
                if current_nnz >= target_nnz_per_pe:
                    should_finalize = True
            else:
                # Last PE: only finalize at the very end (handled after loop)
                pass
            
            if should_finalize:
                # Finalize current PE
                pe_dict = {
                    'pe_id': current_pe_id,
                    'row_ids': current_rows.copy(),
                    'total_nnz': current_nnz,
                    'num_rows': len(current_rows)
                }
                pe_data.append(pe_dict)
                
                logging.debug(f"PE {current_pe_id}: {len(current_rows)} rows, {current_nnz} NNZ")
                
                # Move to next PE
                current_pe_id += 1
                current_rows = []
                current_nnz = 0
        
        # Finalize the last PE with any remaining rows
        if current_rows:
            pe_dict = {
                'pe_id': current_pe_id,
                'row_ids': current_rows.copy(),
                'total_nnz': current_nnz,
                'num_rows': len(current_rows)
            }
            pe_data.append(pe_dict)
            
            logging.debug(f"PE {current_pe_id}: {len(current_rows)} rows, {current_nnz} NNZ (final)")
        
        # Verify we created all PEs
        while len(pe_data) < self.num_pes:
            # Create empty PE if needed
            pe_data.append({
                'pe_id': len(pe_data),
                'row_ids': [],
                'total_nnz': 0,
                'num_rows': 0
            })
            logging.warning(f"PE {len(pe_data)-1}: empty (no rows assigned)")
        
        # Log load balancing quality
        nnz_counts = [pe['total_nnz'] for pe in pe_data if pe['total_nnz'] > 0]
        if nnz_counts:
            min_nnz = min(nnz_counts)
            max_nnz = max(nnz_counts)
            avg_nnz = np.mean(nnz_counts)
            imbalance = (max_nnz - min_nnz) / avg_nnz * 100 if avg_nnz > 0 else 0
            
            logging.info(f"Load balancing:")
            logging.info(f"  Min NNZ/PE: {min_nnz}")
            logging.info(f"  Max NNZ/PE: {max_nnz}")
            logging.info(f"  Avg NNZ/PE: {avg_nnz:.1f}")
            logging.info(f"  Imbalance: {imbalance:.1f}%")
        
        return pe_data
    
    def prepare_matched_pairs(self, pe_data: list, vector_x: np.ndarray):
        """
        Prepare matched pairs for each PE by pre-gathering vector values.
        
        This is the key transformation:
        - Input: matrix values and column indices
        - Output: matrix values and corresponding vector values (matched pairs)
        
        Args:
            pe_data: List of PE data dictionaries
            vector_x: Input vector
        """
        for pe in pe_data:
            if pe['num_rows'] == 0:
                # Empty PE
                pe['row_offsets'] = np.array([0], dtype=np.int16)
                pe['matrix_values'] = np.array([], dtype=np.float32)
                pe['vector_values'] = np.array([], dtype=np.float32)
                continue
            
            # Build CSR structure for this PE's rows
            row_offsets = [0]
            matrix_values = []
            vector_values = []
            
            for local_row_idx, global_row in enumerate(pe['row_ids']):
                # Get data for this row from original matrix
                row_start = self.csr_matrix.indptr[global_row]
                row_end = self.csr_matrix.indptr[global_row + 1]
                
                # Matrix values for this row
                row_matrix_vals = self.csr_matrix.data[row_start:row_end]
                
                # Column indices for this row
                row_col_indices = self.csr_matrix.indices[row_start:row_end]
                
                # Gather vector values (this is the "matched pairs" transformation!)
                row_vector_vals = vector_x[row_col_indices]
                
                # Append to PE's data
                matrix_values.extend(row_matrix_vals)
                vector_values.extend(row_vector_vals)
                
                # Update row offset
                row_offsets.append(len(matrix_values))
            
            # Store in PE data
            pe['row_offsets'] = np.array(row_offsets, dtype=np.int16)
            pe['matrix_values'] = np.array(matrix_values, dtype=np.float32)
            pe['vector_values'] = np.array(vector_values, dtype=np.float32)
            
            logging.debug(f"PE {pe['pe_id']}: prepared {len(matrix_values)} matched pairs")
    
    def compute_on_pe(self, pe: dict) -> np.ndarray:
        """
        Simulate PE computation: SpMV using matched pairs.
        
        This mimics what the CSL kernel will do on the actual PE.
        
        Args:
            pe: PE data dictionary with matched pairs
            
        Returns:
            Result vector for this PE's rows
        """
        if pe['num_rows'] == 0:
            return np.array([], dtype=np.float32)
        
        results = np.zeros(pe['num_rows'], dtype=np.float32)
        
        for local_row in range(pe['num_rows']):
            start = pe['row_offsets'][local_row]
            end = pe['row_offsets'][local_row + 1]
            
            # Dot product using matched pairs
            dot_product = 0.0
            for i in range(start, end):
                dot_product += pe['matrix_values'][i] * pe['vector_values'][i]
            
            results[local_row] = dot_product
        
        return results
    
    def gather_results(self, pe_data: list, pe_results: list) -> np.ndarray:
        """
        Gather results from all PEs into a single result vector.
        
        Args:
            pe_data: List of PE data dictionaries
            pe_results: List of result arrays from each PE
            
        Returns:
            Complete result vector
        """
        result_vector = np.zeros(self.csr_matrix.shape[0], dtype=np.float32)
        
        for pe, results in zip(pe_data, pe_results):
            for local_idx, global_row in enumerate(pe['row_ids']):
                result_vector[global_row] = results[local_idx]
        
        return result_vector
    
    def print_distribution_stats(self, pe_data: list):
        """Print statistics about the row distribution."""
        logging.info("\n" + "="*60)
        logging.info("Distribution Statistics")
        logging.info("="*60)
        
        for pe in pe_data:
            if pe['num_rows'] > 0:
                logging.info(f"PE {pe['pe_id']:3d}: {pe['num_rows']:4d} rows, "
                           f"{pe['total_nnz']:6d} NNZ")
        
        # Overall stats
        total_rows = sum(pe['num_rows'] for pe in pe_data)
        total_nnz = sum(pe['total_nnz'] for pe in pe_data)
        active_pes = sum(1 for pe in pe_data if pe['num_rows'] > 0)
        
        logging.info("-"*60)
        logging.info(f"Total rows: {total_rows}")
        logging.info(f"Total NNZ: {total_nnz}")
        logging.info(f"Active PEs: {active_pes}/{self.num_pes}")
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
    Validate SpMV results using relative norm of the difference.

    Computes: ||y_computed - y_reference|| / ||y_reference||

    This is a standard numerical validation approach that:
    - Scales with the magnitude of the result
    - Is robust to numerical precision issues
    - Gives a single, interpretable error metric

    Args:
        y_computed: Result from device/computation
        y_reference: Ground truth from numpy (A.dot(x))
        tolerance: Maximum acceptable relative error (default 1e-4)

    Returns:
        True if validation passes, False otherwise
    """
    # Check for shape mismatch
    if y_computed.shape != y_reference.shape:
        logging.error(f"❌ Shape mismatch: computed {y_computed.shape} vs reference {y_reference.shape}")
        return False

    # Check for NaN or Inf
    if np.any(np.isnan(y_computed)) or np.any(np.isinf(y_computed)):
        logging.error(f"❌ Invalid values in computed results (NaN or Inf)")
        return False

    # Calculate norms
    diff_norm = np.linalg.norm(y_computed - y_reference)
    ref_norm = np.linalg.norm(y_reference)

    # Avoid division by zero
    if ref_norm < 1e-15:
        # Reference is essentially zero, check if computed is also near zero
        if diff_norm < tolerance:
            logging.info(f"✓ Validation PASSED (both results near zero)")
            logging.info(f"  ||diff|| = {diff_norm:.2e}")
            return True
        else:
            logging.error(f"❌ Validation FAILED: reference is zero but computed is not")
            logging.error(f"  ||computed|| = {np.linalg.norm(y_computed):.2e}")
            return False

    # Relative error using norm
    relative_error = diff_norm / ref_norm

    passed = relative_error < tolerance

    if passed:
        logging.info(f"✓ Validation PASSED")
        logging.info(f"  Relative error: {relative_error:.2e} (tolerance: {tolerance:.2e})")
        logging.info(f"  ||y_computed - y_reference|| / ||y_reference|| = {diff_norm:.2e} / {ref_norm:.2e}")
    else:
        logging.error(f"❌ Validation FAILED")
        logging.error(f"  Relative error: {relative_error:.2e} (tolerance: {tolerance:.2e})")
        logging.error(f"  ||y_computed - y_reference|| = {diff_norm:.2e}")
        logging.error(f"  ||y_reference|| = {ref_norm:.2e}")

        # Show some worst mismatches for debugging
        abs_diff = np.abs(y_computed - y_reference)
        worst_idx = np.argsort(abs_diff)[-5:][::-1]
        logging.error(f"  Worst 5 mismatches:")
        for idx in worst_idx:
            logging.error(f"    Row {idx}: computed={y_computed[idx]:.6e}, "
                         f"reference={y_reference[idx]:.6e}, "
                         f"diff={abs_diff[idx]:.2e}")

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
        format='%(levelname)s: %(message)s'
    )
    
    # Generate test data
    logging.info(f"Generating random matrix: {args.rows}x{args.cols}, density={args.density}")
    A = generate_random_sparse_matrix(args.rows, args.cols, args.density, args.seed)
    x = np.random.randint(0, 100, size=args.cols, dtype=np.int32).astype(np.float32)
    
    logging.info(f"Matrix: {A.shape}, NNZ: {A.nnz}")
    logging.debug(f"Vector x: {x}")

    spmv = MatchedPairsSpMV(A, args.num_pes)
    pe_data = spmv.distribute_by_nnz()
    spmv.prepare_matched_pairs(pe_data, x)
    spmv.print_distribution_stats(pe_data)

    pe_results = []
    for pe in pe_data:
        results = spmv.compute_on_pe(pe)
        pe_results.append(results)

    y_computed = spmv.gather_results(pe_data, pe_results)
    y_reference = A.dot(x).astype(np.float32)
    passed = validate_results(y_computed, y_reference, tolerance=args.tolerance)
    
    if not passed:
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
