"""
Benchmark Result Storage and Analysis

Provides dataclass and CSV utilities for saving SpMV benchmark results.
"""

import csv
import logging
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


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
    validation_passed: bool = True

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
    fabric_dims_w: int = 762
    fabric_dims_h: int = 1172

    # Memory statistics (bytes)
    max_memory_per_pe: int = 0
    avg_memory_per_pe: float = 0.0
    memory_limit_per_pe: int = 48000
    memory_utilization_pct: float = 0.0  # max_memory / limit * 100

    # Timing (seconds)
    compile_time: float = 0.0
    memcpy_h2d_time: float = 0.0
    compute_time_python: float = 0.0
    compute_time_hw_min: float = 0.0
    compute_time_hw_avg: float = 0.0
    compute_time_hw_max: float = 0.0
    compute_time_hw_std: float = 0.0
    compute_hw_cycles_min: int = 0
    compute_hw_cycles_avg: int = 0
    compute_hw_cycles_max: int = 0
    compute_hw_cycles_std: int = 0
    memcpy_d2h_time: float = 0.0
    total_run_time: float = 0.0
    total_time: float = 0.0

    # Performance metrics
    gflops_hw: float = 0.0
    gflops_python: float = 0.0

    # Load balancing
    min_nnz_per_pe: int = 0
    max_nnz_per_pe: int = 0
    avg_nnz_per_pe: float = 0.0
    load_imbalance_pct: float = 0.0  # (max - min) / avg * 100

    # Additional metadata
    num_active_pes: int = 0  # PEs with actual work
    num_idle_pes: int = 0    # PEs with no work

    # Compilation parameters
    total_nnz_padded: int = 0  # Padded NNZ for compilation
    max_rows_padded: int = 0   # Padded rows for compilation

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

    # Get field names from dataclass
    fieldnames = [f.name for f in fields(BenchmarkResult)]

    mode = 'a' if file_exists else 'w'

    with open(csv_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header if new file
        if not file_exists:
            writer.writeheader()

        # Write data
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

    with open(csv_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Convert string values to appropriate types
            typed_row = {}
            for key, value in row.items():
                if value == '':
                    # Handle empty values
                    typed_row[key] = None
                    continue

                # Infer type from field definition
                field_type = None
                for field_def in fields(BenchmarkResult):
                    if field_def.name == key:
                        field_type = field_def.type
                        break

                if field_type is None:
                    typed_row[key] = value
                elif field_type == int or field_type == 'int':
                    typed_row[key] = int(float(value)) if value else 0
                elif field_type == float or field_type == 'float':
                    typed_row[key] = float(value) if value else 0.0
                elif field_type == bool or field_type == 'bool':
                    typed_row[key] = value.lower() in ('true', '1', 'yes')
                elif field_type == str or field_type == 'str':
                    typed_row[key] = value
                else:
                    typed_row[key] = value

            # Filter None values
            typed_row = {k: v for k, v in typed_row.items() if v is not None}

            try:
                result = BenchmarkResult.from_dict(typed_row)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to parse row: {e}")
                continue

    logger.info(f"Loaded {len(results)} benchmark results from {csv_file}")
    return results


