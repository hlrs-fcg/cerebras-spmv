"""Sparse matrix generation utilities (login-node use only)."""

import logging
from pathlib import Path

import numpy as np
import scipy.sparse as sp

#TODO: probably place where one should add multithreading

def generate_random_sparse_matrix(
    rows: int, cols: int, density: float, seed: int = 42, cache_dir: str = "matrices"
) -> sp.csr_matrix:
    """Generate a random sparse CSR matrix with caching."""
    repo_root = Path(__file__).resolve().parent.parent
    cache_path = repo_root / cache_dir
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"random_{rows}x{cols}_d{density}_s{seed}.npz"

    if cache_file.exists():
        logging.debug(f"Loading cached matrix from {cache_file}")
        matrix = sp.load_npz(cache_file)
        logging.info(f"Loaded cached matrix: {matrix.shape}, NNZ={matrix.nnz}")
        return matrix

    logging.info(f"Generating matrix (caching to {cache_file.name})...")
    np.random.seed(seed)

    if density < 0.01:
        expected_nnz = int(rows * cols * density)
        row_idx = np.random.randint(0, rows, size=expected_nnz, dtype=np.int32)
        col_idx = np.random.randint(0, cols, size=expected_nnz, dtype=np.int32)
        data = np.random.rand(expected_nnz).astype(np.float32)
        matrix = sp.coo_matrix((data, (row_idx, col_idx)), shape=(rows, cols), dtype=np.float32).tocsr()
    else:
        matrix = sp.random(rows, cols, density=density, format="csr", dtype=np.float32)
        matrix.data = np.random.rand(matrix.nnz).astype(np.float32)

    sp.save_npz(cache_file, matrix)
    logging.info(f"Generated and cached matrix: {matrix.shape}, NNZ={matrix.nnz}")
    return matrix
