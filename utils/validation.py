"""SpMV result validation utilities (login-node use only)."""

import logging

import numpy as np


def validate_results(y_computed: np.ndarray, y_reference: np.ndarray, tolerance: float = 1e-4) -> bool:
    """Validate SpMV results via relative norm: ||y_computed - y_ref|| / ||y_ref|| < tolerance."""
    if y_computed.shape != y_reference.shape:
        logging.error(f"Shape mismatch: computed {y_computed.shape} vs reference {y_reference.shape}")
        return False

    if np.any(np.isnan(y_computed)) or np.any(np.isinf(y_computed)):
        logging.error("Invalid values in computed results (NaN or Inf)")
        return False

    diff_norm = np.linalg.norm(y_computed - y_reference)
    ref_norm = np.linalg.norm(y_reference)

    if ref_norm < 1e-15:
        passed = diff_norm < tolerance
        if passed:
            logging.info(f"Validation PASSED (both near zero, ||diff||={diff_norm:.2e})")
        else:
            logging.error(f"Validation FAILED: reference is zero, computed is not (||y||={np.linalg.norm(y_computed):.2e})")
        return passed

    relative_error = diff_norm / ref_norm
    passed = relative_error < tolerance

    if passed:
        logging.info(f"Validation PASSED — relative error: {relative_error:.2e}")
    else:
        logging.error(f"Validation FAILED — relative error: {relative_error:.2e} (tolerance: {tolerance:.2e})")
        abs_diff = np.abs(y_computed - y_reference)
        worst = np.argsort(abs_diff)[-5:][::-1]
        for idx in worst:
            logging.error(f"  Row {idx}: computed={y_computed[idx]:.6e}, reference={y_reference[idx]:.6e}")

    return passed
