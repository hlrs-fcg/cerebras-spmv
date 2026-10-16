"""What differs between the worker's modes, as one interface the device phase drives.

``StreamMode`` (worker_stream.py) streams A every pass; ``ResidentMode`` (worker_resident.py) copies
A to the wafer once.  A future CG mode implements the same hooks around ResidentMode's spmv.
Staged flat next to worker_runtime.py.

``d`` below is the device context worker_device builds: ``runner``, ``MemcpyDataType``,
``MemcpyOrder``, ``setup_enum`` and ``syms`` (name -> symbol id).
"""


class Mode:
    # The host arrays a mode no longer needs once setup is on the wafer are dropped by after_setup;
    # the device phase then also drops its own references to the flattened setup calls.
    drops_setup_arrays = False

    def extra_symbol_names(self, r) -> list:
        """Symbols to fetch after tsc_pack (order matters: it is the get_id order)."""
        return []

    def setup_extra_bytes(self, r) -> int:
        return 0

    def setup_extra_arrays(self, r, d) -> list:
        """(symbol, host array, words per PE) copied after col_indices and before row_ptrs."""
        return []

    def after_setup(self, r) -> None:
        """The setup copies are on the wafer: release host arrays."""

    def log_spmv(self, r) -> None:
        raise NotImplementedError

    def prepare_passes(self, r, d) -> None:
        """Anything run once before the pass loop (and before f_sync)."""

    def launch_pass(self, r, d, t_comp) -> float:
        """Issue one pass; returns the time the host-return bracket for H2D starts at."""
        raise NotImplementedError

    def memcpy_h2d_time(self, r) -> float:
        raise NotImplementedError

    def log_pass_summary(self, r) -> None:
        raise NotImplementedError
