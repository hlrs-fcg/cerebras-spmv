"""Tests for the BenchmarkResult schema and its CSV round-trip.

cluster/worker_runtime.py constructs BenchmarkResult with the streaming
architecture's fields; these pin the names it depends on and check that
widening the schema did not break reading CSVs written before the change.
"""

import csv
from dataclasses import fields

import pytest

from bench.benchmark import BenchmarkResult, load_benchmarks, save_benchmark

# Names the worker passes; a rename here breaks the worker at runtime.
NEW_FIELDS = {
    "kx": 0,
    "ky": 0,
    "num_col_stripes": 1,
    "tile_extract_time": 0.0,
    "h2d_bytes_total": 0,
}


def test_new_fields_exist_with_expected_defaults():
    defaults = {f.name: f.default for f in fields(BenchmarkResult)}
    for name, default in NEW_FIELDS.items():
        assert name in defaults, f"BenchmarkResult is missing {name!r}"
        assert defaults[name] == default


def test_preexisting_timing_field_names_are_unchanged():
    """The worker uses these names; they must not be renamed to *_gen_time."""
    names = {f.name for f in fields(BenchmarkResult)}
    assert "matrix_generation_time" in names
    assert "vector_generation_time" in names
    assert "matrix_gen_time" not in names
    assert "vector_gen_time" not in names


def test_constructor_accepts_the_full_worker_kwarg_set():
    result = BenchmarkResult(
        matrix_rows=1000,
        matrix_cols=1000,
        num_pes=64,
        grid_width=8,
        grid_height=8,
        kx=125,
        ky=125,
        num_col_stripes=1,
        matrix_generation_time=0.5,
        vector_generation_time=0.1,
        tile_extract_time=2.5,
        h2d_bytes_total=4096,
    )
    assert result.kx == 125
    assert result.ky == 125
    assert result.tile_extract_time == 2.5
    assert result.h2d_bytes_total == 4096


def test_csv_round_trip_preserves_new_fields(tmp_path):
    csv_path = tmp_path / "bench.csv"
    save_benchmark(
        BenchmarkResult(kx=125, ky=250, num_col_stripes=3,
                        tile_extract_time=1.25, h2d_bytes_total=987654),
        csv_path=str(csv_path),
    )

    loaded = load_benchmarks(str(csv_path))
    assert len(loaded) == 1
    assert loaded[0].kx == 125
    assert loaded[0].ky == 250
    assert loaded[0].num_col_stripes == 3
    assert loaded[0].tile_extract_time == 1.25
    assert loaded[0].h2d_bytes_total == 987654


def test_appending_reuses_the_header(tmp_path):
    csv_path = tmp_path / "bench.csv"
    save_benchmark(BenchmarkResult(kx=1), csv_path=str(csv_path))
    save_benchmark(BenchmarkResult(kx=2), csv_path=str(csv_path))

    assert [r.kx for r in load_benchmarks(str(csv_path))] == [1, 2]
    with open(csv_path, newline="", encoding="utf-8") as f:
        assert sum(1 for _ in f) == 3  # header + two rows


def test_reads_an_older_csv_missing_the_new_columns(tmp_path):
    """A CSV written before the schema widened still loads, with defaults."""
    csv_path = tmp_path / "old.csv"
    old_fieldnames = [
        f.name for f in fields(BenchmarkResult) if f.name not in NEW_FIELDS
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=old_fieldnames)
        writer.writeheader()
        writer.writerow({"num_pes": 64, "grid_width": 8, "grid_height": 8,
                         "matrix_rows": 1000, "compute_time_hw_avg": 0.001})

    loaded = load_benchmarks(str(csv_path))
    assert len(loaded) == 1
    assert loaded[0].num_pes == 64
    for name, default in NEW_FIELDS.items():
        assert getattr(loaded[0], name) == default


def test_from_dict_ignores_unknown_columns():
    result = BenchmarkResult.from_dict({"kx": 7, "not_a_field": "ignored"})
    assert result.kx == 7


def test_save_migrates_a_narrower_csv_rather_than_dropping_the_row(tmp_path):
    """A narrower destination must be widened, not appended to positionally.

    This used to warn and return, which lost the row; appending would have put
    values under the wrong columns. See tests/test_csv_schema_safety.py.
    """
    csv_path = tmp_path / "old.csv"
    old_fieldnames = [
        f.name for f in fields(BenchmarkResult) if f.name not in NEW_FIELDS
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=old_fieldnames)
        writer.writeheader()
        writer.writerow({"num_pes": 64})

    save_benchmark(BenchmarkResult(num_pes=128, kx=9), csv_path=str(csv_path))

    rows = load_benchmarks(str(csv_path))
    assert [r.num_pes for r in rows] == [64, 128]
    assert rows[0].kx == 0   # backfilled default, correct for a row predating it
    assert rows[1].kx == 9
    assert csv_path.with_suffix(".csv.bak").exists()


@pytest.mark.parametrize("name", sorted(NEW_FIELDS))
def test_new_fields_appear_in_the_csv_header(tmp_path, name):
    csv_path = tmp_path / "bench.csv"
    save_benchmark(BenchmarkResult(), csv_path=str(csv_path))
    with open(csv_path, newline="", encoding="utf-8") as f:
        assert name in (csv.DictReader(f).fieldnames or [])
