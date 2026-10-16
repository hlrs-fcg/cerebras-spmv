"""
Benchmark Result Storage and Analysis

Provides dataclass and CSV utilities for saving SpMV benchmark results.
"""

import csv
import os
import shutil
from dataclasses import MISSING, dataclass, field, asdict, fields
from datetime import datetime
from pathlib import Path
from typing import List

try:
    # Login node: shared loguru-backed logger.
    from utils.logging import logger
except ImportError:  # pragma: no cover - taken only on the appliance worker node
    # Worker node: SdkLauncher stages only worker_runtime.py, coo_tiling.py and
    # this module, so the utils package is absent there (and loguru is not
    # available on the worker either).  Nothing below uses loguru-only APIs, so
    # stdlib logging is a drop-in.  cluster/worker_runtime.py calls
    # logging.basicConfig(), which configures this logger too.
    import logging as _logging

    logger = _logging.getLogger("spmv.benchmark")


@dataclass
class BenchmarkResult:
    """
    Complete record of a single SpMV run with all parameters and timings.

    Organized into sections:
    - Metadata: timestamp, validation
    - Matrix parameters: rows, cols, density, nnz, seed
    - PE configuration: num_pes, grid dimensions, architecture
    - Memory statistics: max/avg memory per PE
    - Timing: compilation, H2D, compute, D2H, total
    - Load balancing: min/max/avg NNZ per PE, imbalance percentage
    """

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    git_commit: str = "unknown"  # Git commit hash
    validation_passed: bool = True
    exit_reason: str = "SUCCESS"
    error_message: str = ""
    relative_error: float = 0.0  # ||y_computed - y_reference|| / ||y_reference||

    # Matrix parameters
    matrix_rows: int = 0
    matrix_cols: int = 0
    density: float = 0.0
    total_nnz: int = 0
    seed: int = 42

    # PE configuration
    num_pes: int = 1
    grid_width: int = 1
    grid_height: int = 1
    kx: int = 0  # x elements per PE column = ceil(cols / grid_width)
    ky: int = 0  # rows per PE row = ceil(rows / grid_height)
    num_col_stripes: int = 1  # column-stripe passes over A
    architecture: str = "wse3"

    # memcpy I/O configuration this row was measured under.  Defaults match the
    # values that were hardcoded before they became configurable, so rows in
    # CSVs written before this column existed read back correctly.
    channels: int = 1
    width_west_buf: int = 0
    width_east_buf: int = 0

    # Shape of the streaming A-values H2D transfer ("column" | "full").  Set by
    # the worker, not by compile_io_settings(): unlike the three fields above
    # this is a runtime choice the worker makes and therefore knows.  The
    # default matches the historical behaviour, so rows written before this
    # column existed read back correctly.
    h2d_mode: str = "column"

    # Host-side memory order of the streaming A transfer ("row" | "col").  Also
    # set by the worker.  Default matches historical behaviour.
    memcpy_order: str = "row"

    # A-values stream encoding ("header" | "valtask" | "resident"), i.e. which kernel ran.  Set by
    # the worker, which is told it and cross-checks it against the artifact's
    # out.json.  Default matches every row written before this column existed.
    encoding: str = "header"
    # Memory order of the copy-mode setup transfers.  Always "row" before the
    # --setup-order flag existed, so the default is correct for old rows.
    setup_order: str = "row"
    # Vertical strips each copy-mode setup transfer was split into (issued serially).  1 (one
    # full rectangle) for every row written before --setup-strips existed.
    setup_strips: int = 1
    # Column mode sent its rects one at a time (all but the last nonblock=False).
    # Always True for val + column (automatic; overlapping val rects hang on
    # hardware), opt-in for header.  False for every row written before the
    # column existed, which is correct: no val column run completed before it.
    serialize_column_rects: bool = False

    # Memory statistics (bytes)
    max_memory_per_pe: int = 0
    avg_memory_per_pe: float = 0.0
    memory_utilization_pct: float = 0.0

    # Host preprocessing timing (seconds)
    matrix_generation_time: float = 0.0
    vector_generation_time: float = 0.0
    tile_extract_time: float = 0.0  # per-PE tile structure extraction
    # The host phases tile_extract_time does not cover (all 0 for rows written before they were
    # timed): x_grid build, A-stream buffer build (val buffer + plans, or header stream), the
    # setup-array flatten (col-major transpose + strips; inside the SDK context, before the
    # setup bracket), and the wall time of the whole prepare_host_buffers() call.
    x_grid_time: float = 0.0
    stream_build_time: float = 0.0
    setup_prep_time: float = 0.0
    host_prep_time: float = 0.0
    # How the tile structure was obtained: "numba" (fused threaded tiler), "numpy" (lexsort path;
    # also every row written before this column existed) or "cache" (on-disk tile cache hit, in
    # which case tile_extract_time is just the time to map the entry).  tile_jit_time is the numba
    # compile / compiled-cache load, which tile_extract_time does not include.
    tile_backend: str = "numpy"
    tile_cache_hit: bool = False
    tile_jit_time: float = 0.0
    # The worker's Python, as seen by the worker (which wheels does it need?): interpreter + numpy,
    # OS, glibc, usable CPUs, and the numba version or the import error that kept it out.
    worker_python: str = ""
    worker_platform: str = ""
    worker_glibc: str = ""
    worker_cpus: int = 0
    numba_status: str = ""

    # load balancing timing (seconds)
    distribute_time: float = 0.0  # row distribution algorithm time
    prepare_matched_pairs_time: float = 0.0  # matched pairs transformation time
    total_precond_time: float = 0.0  # total load balancing time (kept as precond for CSV compatibility)

    # Device timing (seconds)
    compile_time: float = 0.0
    sdk_init_time: float = 0.0
    program_load_time: float = 0.0
    h2d_bytes_total: int = 0  # bytes streamed host->device in ONE pass (loop_count passes ran)
    memcpy_h2d_time: float = 0.0
    # One-time setup H2D (x_vec, col_indices, row_ptrs).  These go streaming=False,
    # i.e. copy mode -- DMA'd straight into PE memory with no compute-engine
    # involvement -- so unlike memcpy_h2d_time they measure the link WITHOUT any
    # per-word kernel cost.  Cerebras' own bandwidth benchmark measures only this
    # mode (csl-examples/benchmarks/bandwidth-test, streaming=False throughout),
    # so this is the column that makes our streaming rate interpretable.
    setup_h2d_bytes: int = 0
    setup_h2d_time: float = 0.0
    # Completion barriers (host-only; docs/host-timing-barrier.md).  A
    # blocking launch / memcpy_h2d can return before the device is done, so
    # memcpy_h2d_time, compute_time_hw_* and setup_h2d_time are HOST-RETURN
    # times: lower bounds.  Each bracket is
    # closed by a tiny copy-mode D2H that cannot finish before the device has
    # (barrier_*), and that D2H is priced by repeating it on the idle device
    # (*_repeat_time).  The *_corrected columns are what rates must use.
    # 0 in every row written before these columns existed.
    setup_done_time: float = 0.0            # setup start -> setup barrier done
    setup_barrier_d2h_time: float = 0.0     # 1 word of row_ptrs from every PE
    setup_barrier_d2h_repeat_time: float = 0.0
    setup_h2d_time_corrected: float = 0.0   # setup_done - setup_barrier_repeat
    device_done_time: float = 0.0           # H2D bracket start -> barrier done
    barrier_d2h_time: float = 0.0           # y_result[0] from each column-0 PE
    barrier_d2h_repeat_time: float = 0.0
    stream_time_corrected: float = 0.0      # device_done - barrier_repeat
    compute_time_hw_min: float = 0.0
    compute_time_hw_avg: float = 0.0
    compute_time_hw_max: float = 0.0
    compute_time_hw_std: float = 0.0
    compute_hw_cycles_min: int = 0
    compute_hw_cycles_avg: int = 0
    compute_hw_cycles_max: int = 0
    compute_hw_cycles_std: int = 0
    # Passes per run (--loop-count); compute_time_hw_* are statistics over them.
    loop_count: int = 1
    # Synced on-wafer timing (--sync-timing, docs/synced-timing.md):
    # per pass, max(toc) - min(tic) across PEs on the sync module's common
    # reference clock, then min/avg/max/std over passes.  wafer_time_* are the
    # same at 850 MHz, in seconds.  0 when sync timing was off or did not read
    # back every pass -- every row before these columns existed.
    sync_timing: bool = False
    wafer_cycles_min: int = 0
    wafer_cycles_avg: int = 0
    wafer_cycles_max: int = 0
    wafer_cycles_std: int = 0
    wafer_time_min: float = 0.0
    wafer_time_avg: float = 0.0
    wafer_time_max: float = 0.0
    wafer_time_std: float = 0.0
    # Host time per pass from before f_tic / spmv to the return of the completion
    # barrier D2H -- a blocking call that cannot complete before the device has
    # finished the pass.  compute_time_hw_* stop at the wait_ew / f_toc return,
    # which can come BEFORE the device is done (val full/ch1 at 65,536 PEs:
    # 22 ms host vs a 104 ms tsc span, the rest absorbed by a 193 ms D2H).
    host_pass_time_avg: float = 0.0
    host_pass_time_std: float = 0.0
    # max over passes of wafer_time / host_pass_time; > 1 is impossible and
    # flags a tsc-rate or stamp-placement error.  0 when sync timing was off.
    wafer_host_ratio_max: float = 0.0
    # Resident A (--encoding resident, docs/resident-spmv.md).  Nothing is streamed per
    # pass, so h2d_bytes_total and memcpy_h2d_time are 0, and the reused columns mean:
    #   wafer_*                  the synced window of ONE SpMV (launch -> EW chain done)
    #   host_pass_time_*         host time per SpMV through the completion barrier
    #   stream_time_corrected    the same, minus the idle barrier D2H: the barrier-closed
    #                            host time per SpMV (name kept for schema stability)
    #   setup_h2d_*              includes the values (4 B x max_nnz x PEs more than header)
    # wafer_gflops_avg: 2 * NNZ / wafer_time_avg, in GFLOP/s; 0 without sync timing.
    wafer_gflops_avg: float = 0.0
    # --null-spmv: spmv() was a no-op (launch-latency floor); y not validated.
    null_spmv: bool = False
    memcpy_d2h_time: float = 0.0

    # Host postprocessing timing (seconds)
    gather_results_time: float = 0.0
    validation_time: float = 0.0

    total_run_time: float = 0.0
    total_time: float = 0.0

    # Performance metrics
    gflops_hw_avg: float = 0.0  # GFLOPS using average time (optimistic)
    gflops_hw_max: float = 0.0  # GFLOPS using max time (actual throughput!)
    arithmetic_intensity: float = 0.0  # FLOPS per byte transferred

    # Load balancing
    min_nnz_per_pe: int = 0
    max_nnz_per_pe: int = 0
    avg_nnz_per_pe: float = 0.0
    load_imbalance_pct: float = 0.0
    num_idle_pes: int = 0  # PEs with 0 rows
    pe_efficiency_pct: float = 100.0  # (active PEs / total PEs) * 100

    # Additional metadata
    num_active_pes: int = 0

    # Build provenance, stamped by sdk/launcher.py on the login side (sdk/build_info.py).
    # Appended last so older CSVs migrate by padding, not by shifting columns.
    git_commit_full: str = ""
    git_dirty: bool = False
    repo_describe: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'BenchmarkResult':
        """Create from dictionary, handling type conversions."""
        # Filter to only known fields
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        # Convert string booleans if needed
        for name in ('validation_passed', 'git_dirty', 'serialize_column_rects', 'sync_timing', 'null_spmv'):
            val = filtered_data.get(name)
            if isinstance(val, str):
                filtered_data[name] = val.lower() in ('true', '1', 'yes')

        return cls(**filtered_data)

    def summary(self) -> str:
        """Generate a minimal human-readable summary with key metrics."""
        return (
            f"[{self.timestamp}] "
            f"{self.matrix_rows}×{self.matrix_cols} (NNZ={self.total_nnz}) | "
            f"{self.num_pes} PEs | "
            f"compute={self.compute_time_hw_avg:.6f}s | "
            f"{'PASSED' if self.validation_passed else 'FAILED'}"
        )


def field_defaults() -> dict:
    """Column -> value used to backfill a row that predates that column.

    Fields with a default_factory (timestamp) backfill as "", which
    load_benchmarks() skips so the dataclass default applies on read.
    """
    out = {}
    for f in fields(BenchmarkResult):
        out[f.name] = "" if f.default is MISSING else f.default
    return out


def read_header(csv_file: Path) -> List[str]:
    """Existing column names of csv_file, or [] if it is absent or empty."""
    if not csv_file.exists():
        return []
    with open(csv_file, 'r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f).fieldnames or [])


def _diverted_path(csv_file: Path) -> Path:
    """First free `<stem>.schemaN<suffix>` sibling."""
    n = 2
    while True:
        candidate = csv_file.with_name(f"{csv_file.stem}.schema{n}{csv_file.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def _migrate_to_wider_schema(csv_file: Path, fieldnames: List[str]) -> None:
    """Rewrite csv_file under `fieldnames`, backfilling the new columns.

    Only ever called when the existing header is a strict subset of
    `fieldnames`, so no recorded value can be dropped.  Writes a .bak first and
    swaps atomically, because these are raw experiment results.
    """
    with open(csv_file, 'r', newline='', encoding='utf-8') as f:
        existing = list(csv.DictReader(f))

    defaults = field_defaults()
    tmp = csv_file.with_suffix(csv_file.suffix + ".tmp")
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing:
            writer.writerow({name: row.get(name, defaults.get(name, "")) for name in fieldnames})

    shutil.copyfile(csv_file, csv_file.with_suffix(csv_file.suffix + ".bak"))
    os.replace(tmp, csv_file)


def append_rows(csv_file: Path, fieldnames: List[str], rows: List[dict]) -> Path:
    """Append rows to csv_file without ever writing positionally.

    csv.DictWriter will happily write wide rows against a narrow header, which
    produces a file whose values sit under the wrong keys while every write
    reports success.  That is the one outcome this function makes impossible.

    Behaviour by case, returning the path actually written:
      * no file, or an empty one -> create it with this header
      * identical header         -> plain append
      * header is a subset       -> migrate the file to the wider schema,
                                    backfilling old rows from the dataclass
                                    defaults, then append
      * anything else            -> divert to a `<stem>.schemaN.csv` sibling and
                                    log loudly; the original is left untouched
    """
    csv_file = Path(csv_file)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    existing = read_header(csv_file)

    if not existing:
        target, write_header = csv_file, True
    elif existing == list(fieldnames):
        target, write_header = csv_file, False
    elif set(existing) < set(fieldnames):
        added = sorted(set(fieldnames) - set(existing))
        logger.warning(
            f"{csv_file.name} predates {len(added)} column(s) {added}; migrating it "
            f"to the current schema (old rows backfilled with defaults, "
            f"{csv_file.name}.bak kept)."
        )
        _migrate_to_wider_schema(csv_file, list(fieldnames))
        target, write_header = csv_file, False
    else:
        target = _diverted_path(csv_file)
        logger.error(
            f"{csv_file.name} has an incompatible header "
            f"(only there: {sorted(set(existing) - set(fieldnames))}; "
            f"only here: {sorted(set(fieldnames) - set(existing))}). "
            f"Writing to {target.name} instead of appending — appending would "
            f"put values under the wrong columns."
        )
        write_header = True

    with open(target, 'a' if not write_header else 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    return target


def save_benchmark(result: BenchmarkResult,
                   csv_path: str = "tests/benchmark_results.csv",
                   append: bool = True) -> None:
    """
    Save benchmark result to CSV file.

    Args:
        result: BenchmarkResult to save
        csv_path: Path to CSV file
        append: If True, append to existing file (migrating or diverting it when
            the schema differs). If False, overwrite it.
    """
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [f.name for f in fields(BenchmarkResult)]

    if not append and csv_file.exists():
        csv_file.unlink()

    csv_file = append_rows(csv_file, fieldnames, [result.to_dict()])

    logger.info(f"Benchmark saved to {csv_file.resolve()}")
    logger.debug(f"  Timestamp: {result.timestamp}")
    logger.debug(f"  Matrix: {result.matrix_rows}×{result.matrix_cols}, NNZ={result.total_nnz}")
    logger.debug(f"  PEs: {result.num_pes}, Time: {result.total_time:.2f}s")


def load_benchmarks(csv_path: str = "tests/benchmark_results.csv") -> List[BenchmarkResult]:
    """
    Load all benchmark results from CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of BenchmarkResult objects
    """
    csv_file = Path(csv_path)

    if not csv_file.exists():
        logger.warning(f"Benchmark file not found: {csv_file}")
        return []

    results = []

    field_types = {f.name: f.type for f in fields(BenchmarkResult)}

    skipped = 0
    with open(csv_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            typed_row = {}
            for key, value in row.items():
                if value == '':
                    continue

                field_type = field_types.get(key)
                if field_type is None:
                    typed_row[key] = value
                elif field_type is int:
                    typed_row[key] = int(float(value))
                elif field_type is float:
                    typed_row[key] = float(value)
                elif field_type is bool:
                    typed_row[key] = value.lower() in ('true', '1', 'yes')
                elif field_type is str:
                    typed_row[key] = value
                else:
                    typed_row[key] = value

            try:
                result = BenchmarkResult.from_dict(typed_row)
                results.append(result)
            except Exception as e:
                logger.warning(f"Skipping malformed row {reader.line_num}: {e}")
                skipped += 1

    if skipped:
        logger.warning(f"Skipped {skipped} malformed row(s) in {csv_file.name}")
    logger.info(f"Loaded {len(results)} benchmark results from {csv_file}")
    return results


