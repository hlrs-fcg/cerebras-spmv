"""Worker pieces shared by every mode: validation, the progress log, the runtime wrapper, and
what an artifact's out.json says it was compiled with.  Staged flat next to worker_runtime.py."""

import logging
import os
import time

import numpy as np

from coo_tiling import ENCODING_IDS, NULL_SPMV_ENCODING_ID

logger = logging.getLogger(__name__)


def validate_results(
    y_computed: np.ndarray, y_reference: np.ndarray, tolerance: float = 1e-4
) -> bool:
    if y_computed.shape != y_reference.shape:
        logger.error(f"Shape mismatch: computed {y_computed.shape} vs reference {y_reference.shape}")
        return False

    if np.any(np.isnan(y_computed)) or np.any(np.isinf(y_computed)):
        logger.error("Invalid values (NaN or Inf) in computed results")
        return False

    diff_norm = np.linalg.norm(y_computed - y_reference)
    ref_norm  = np.linalg.norm(y_reference)

    if ref_norm < 1e-15:
        passed = diff_norm < tolerance
        if passed:
            logger.info(f"Validation PASSED (both near zero, ||diff||={diff_norm:.2e})")
        else:
            logger.error("Validation FAILED: reference is zero but computed is not")
        return passed

    relative_error = diff_norm / ref_norm
    passed = relative_error < tolerance

    if passed:
        logger.info(f"Validation PASSED — relative error: {relative_error:.2e}")
    else:
        logger.error(
            f"Validation FAILED — relative error: {relative_error:.2e} "
            f"(tolerance: {tolerance:.2e})"
        )
        abs_diff = np.abs(y_computed - y_reference)
        for idx in np.argsort(abs_diff)[-5:][::-1]:
            logger.error(
                f"  Row {idx}: computed={y_computed[idx]:.6e}, "
                f"reference={y_reference[idx]:.6e}"
            )
    return passed


def make_progress(path, t0=None):
    """A ``progress(msg)`` that appends ``+<seconds> msg`` to ``path``, flushed and fsynced
    per line, so a job that hangs or is cancelled still shows the last phase it reached
    (the worker's stdout reaches the host only when the job completes).  ``path`` None: no-op.
    Never raises: progress must not be able to fail a run."""
    t0 = time.perf_counter() if t0 is None else t0

    def progress(msg: str) -> None:
        if path is None:
            return
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"+{time.perf_counter() - t0:9.3f}s {msg}\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            pass
    return progress


class SdkRuntimeWrapper:
    def __init__(self, runner_cls, *args, **kwargs):
        self.runner_cls = runner_cls
        self.args       = args
        self.kwargs     = kwargs
        self.runner     = None

    def __enter__(self):
        self.runner = self.runner_cls(*self.args, **self.kwargs)
        return self.runner

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.runner is not None:
            try:
                self.runner.stop()
            except Exception as e:
                logger.warning(f"Error stopping SdkRuntime: {e}")
        return False


def artifact_encoding(compile_data: dict) -> str:
    """The stream encoding an artifact was compiled for, from its out.json.

    ``ENCODING`` is a cslc --params entry (sdk/compiler.py), so it lands in
    out.json's ``params`` next to MEMCPYH2D_DATA_1_ID.  Artifacts compiled
    before the param existed are header kernels.
    """
    raw = int(compile_data.get("params", {}).get("ENCODING", 0))
    if raw == NULL_SPMV_ENCODING_ID:
        return "resident"   # the resident kernel with spmv() a no-op
    for name, encoding_id in ENCODING_IDS.items():
        if encoding_id == raw:
            return name
    raise ValueError(f"the artifact was compiled with ENCODING={raw}, which this worker does not know "
                     f"(id 1, the bulk val kernel, was removed in v2.0): recompile")


def artifact_null_spmv(compile_data: dict) -> bool:
    """Whether an artifact was compiled with ENCODING:4, the no-op spmv() (absent: no)."""
    return int(compile_data.get("params", {}).get("ENCODING", 0)) == NULL_SPMV_ENCODING_ID


def artifact_local_kernel(compile_data: dict) -> int | None:
    """The LOCAL_KERNEL an artifact from before the single resident kernel carries (absent: None).

    The param is gone: the resident kernel is the COO scatter, which an old LOCAL_KERNEL:2 artifact also is."""
    raw = compile_data.get("params", {}).get("LOCAL_KERNEL")
    return None if raw is None else int(raw)


def artifact_sync_timing(compile_data: dict) -> bool:
    """Whether an artifact was compiled with SYNC_TIMING:1 (absent: no)."""
    return int(compile_data.get("params", {}).get("SYNC_TIMING", 0)) == 1


class Run:
    """Everything one worker run knows, filled stage by stage.

    worker_runtime.main() makes it from the parsed arguments; worker_host fills the geometry, the
    matrix and the tiled host buffers, worker_device the device-side measurements, worker_report
    reads it all back for validation and the CSV row.  Plain attributes on purpose: the names are
    the ones the single-function worker used, so a line moved between stages keeps its variables.
    """

    def __init__(self, args, total_start, progress, env_info):
        self.args = args
        self.total_start = total_start
        self.progress = progress
        self.env_info = env_info
        self.resident = False       # set by worker_runtime.main()
        self.serialize_rects = False

