"""
Benchmark Result Storage and Analysis

Provides dataclass and CSV utilities for saving SpMV benchmark results.
"""

import csv
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime
from pathlib import Path
from typing import List

from utils.logging import logger


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
    architecture: str = "wse3"

    # Memory statistics (bytes)
    max_memory_per_pe: int = 0
    avg_memory_per_pe: float = 0.0
    memory_utilization_pct: float = 0.0

    # Host preprocessing timing (seconds)
    matrix_generation_time: float = 0.0
    vector_generation_time: float = 0.0

    # load balancing timing (seconds)
    distribute_time: float = 0.0  # row distribution algorithm time
    prepare_matched_pairs_time: float = 0.0  # matched pairs transformation time
    total_precond_time: float = 0.0  # total load balancing time (kept as precond for CSV compatibility)

    # Device timing (seconds)
    compile_time: float = 0.0
    sdk_init_time: float = 0.0
    program_load_time: float = 0.0
    memcpy_h2d_time: float = 0.0
    compute_time_hw_min: float = 0.0
    compute_time_hw_avg: float = 0.0
    compute_time_hw_max: float = 0.0
    compute_time_hw_std: float = 0.0
    compute_hw_cycles_min: int = 0
    compute_hw_cycles_avg: int = 0
    compute_hw_cycles_max: int = 0
    compute_hw_cycles_std: int = 0
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
        if 'validation_passed' in filtered_data:
            val = filtered_data['validation_passed']
            if isinstance(val, str):
                filtered_data['validation_passed'] = val.lower() in ('true', '1', 'yes')

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


def save_benchmark(result: BenchmarkResult,
                   csv_path: str = "tests/benchmark_results.csv",
                   append: bool = True) -> None:
    """
    Save benchmark result to CSV file.

    Args:
        result: BenchmarkResult to save
        csv_path: Path to CSV file
        append: If True, append to existing file. If False, create new file.
    """
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_file.exists() and append

    fieldnames = [f.name for f in fields(BenchmarkResult)]

    # Check if we need to migrate the CSV schema
    if file_exists:
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []

            if set(existing_fieldnames) != set(fieldnames):
                logger.info(f"Migrating {csv_file.name} to new schema...")
                existing_rows = list(reader)

                backup_path = csv_file.with_suffix('.csv.bak')
                csv_file.rename(backup_path)

                with open(csv_file, 'w', newline='', encoding='utf-8') as new_f:
                    writer = csv.DictWriter(new_f, fieldnames=fieldnames)
                    writer.writeheader()

                    for row in existing_rows:
                        migrated_row = {fn: row.get(fn, '') for fn in fieldnames}
                        writer.writerow(migrated_row)

                logger.info(f"Migrated {len(existing_rows)} rows (backup: {backup_path.name})")

    mode = 'a' if file_exists else 'w'

    with open(csv_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(result.to_dict())

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
                logger.warning(f"Failed to parse row: {e}")
                continue

    logger.info(f"Loaded {len(results)} benchmark results from {csv_file}")
    return results


