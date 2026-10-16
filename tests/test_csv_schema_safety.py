"""Tests that a schema change can never silently mis-record a benchmark row.

A real 65,536-PE channels sweep was corrupted this way: the destination CSV had
a narrow header, csv.DictWriter appended wide rows against it, and the file read
back with compute_time_hw_avg = 112086016.0 s while the real 0.0394 s sat in
compute_hw_cycles_min.  Every write reported success.
"""

import csv
from dataclasses import fields
from pathlib import Path

import pytest

from bench.benchmark import (
    BenchmarkResult,
    append_rows,
    field_defaults,
    load_benchmarks,
    read_header,
    save_benchmark,
)

# Columns added after the first results were recorded, in two waves.
LATE_COLUMNS = ("kx", "ky", "num_col_stripes", "tile_extract_time",
                "x_grid_time", "stream_build_time", "setup_prep_time", "host_prep_time",
                "tile_backend", "tile_cache_hit", "tile_jit_time",
                "worker_python", "worker_platform", "worker_glibc", "worker_cpus", "numba_status",
                "h2d_bytes_total", "channels", "width_west_buf", "width_east_buf")

WIDE = [f.name for f in fields(BenchmarkResult)]
NARROW = [n for n in WIDE if n not in LATE_COLUMNS]


def _write_narrow_csv(path: Path, **values) -> None:
    """A results file as it existed before the schema widened."""
    row = {name: "" for name in NARROW}
    row.update({k: v for k, v in values.items() if k in NARROW})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NARROW)
        writer.writeheader()
        writer.writerow(row)


def test_the_two_schemas_really_do_differ():
    """Guard: if these ever match, the tests below stop meaning anything."""
    assert len(WIDE) - len(NARROW) == len(LATE_COLUMNS)
    assert set(NARROW) < set(WIDE)


# --------------------------------------------------------------------------
# the corruption, reproduced and then prevented
# --------------------------------------------------------------------------

def test_positional_append_corrupts_a_narrow_file(tmp_path):
    """Characterise the old behaviour, so the fix below has something to beat."""
    path = tmp_path / "results.csv"
    _write_narrow_csv(path, num_pes=65536, compute_time_hw_avg=0.039308)

    wide_row = BenchmarkResult(num_pes=65536, compute_time_hw_avg=0.039398,
                               compute_hw_cycles_min=0, channels=1).to_dict()
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=WIDE).writerows([wide_row])

    read_back = list(csv.DictReader(open(path, newline="", encoding="utf-8")))[1]
    shifted = read_back["compute_time_hw_avg"]      # wherever the shift lands it: wrong or blank
    assert shifted == "" or float(shifted) != pytest.approx(0.039398)
    assert "channels" not in read_back


def test_append_rows_migrates_instead_of_shifting(tmp_path):
    """The exact scenario, through the real code path: every value under its key."""
    path = tmp_path / "results.csv"
    _write_narrow_csv(path, num_pes=65536, compute_time_hw_avg=0.039308,
                      validation_passed=True)

    wide_row = BenchmarkResult(num_pes=65536, compute_time_hw_avg=0.039398,
                               channels=1, kx=155, ky=155).to_dict()
    written = append_rows(path, WIDE, [wide_row])
    assert written == path

    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    assert len(rows) == 2
    assert float(rows[0]["compute_time_hw_avg"]) == pytest.approx(0.039308)
    assert float(rows[1]["compute_time_hw_avg"]) == pytest.approx(0.039398)
    assert int(rows[1]["num_pes"]) == 65536
    assert int(rows[1]["kx"]) == 155


def test_migration_backfills_old_rows_with_correct_defaults(tmp_path):
    """Rows predating the columns were all measured at channels=1, no buffers."""
    path = tmp_path / "results.csv"
    _write_narrow_csv(path, num_pes=65536, compute_time_hw_avg=0.039308)

    append_rows(path, WIDE, [BenchmarkResult(channels=16).to_dict()])

    old, new = load_benchmarks(str(path))
    assert (old.channels, old.width_west_buf, old.width_east_buf) == (1, 0, 0)
    assert new.channels == 16


def test_migration_preserves_every_pre_existing_value(tmp_path):
    path = tmp_path / "results.csv"
    _write_narrow_csv(path, num_pes=65536, matrix_rows=114000, total_nnz=1234567,
                      compute_time_hw_avg=0.039308, relative_error=1.49e-07,
                      git_commit="abc1234")
    before = list(csv.DictReader(open(path, newline="", encoding="utf-8")))[0]

    append_rows(path, WIDE, [BenchmarkResult().to_dict()])
    after = list(csv.DictReader(open(path, newline="", encoding="utf-8")))[0]

    for column, value in before.items():
        assert after[column] == value, f"{column} changed during migration"


def test_migration_keeps_a_backup(tmp_path):
    path = tmp_path / "results.csv"
    _write_narrow_csv(path, num_pes=65536)
    original = path.read_text()

    append_rows(path, WIDE, [BenchmarkResult().to_dict()])

    backup = path.with_suffix(path.suffix + ".bak")
    assert backup.exists()
    assert backup.read_text() == original


def test_header_is_widened_in_place(tmp_path):
    path = tmp_path / "results.csv"
    _write_narrow_csv(path, num_pes=65536)
    assert "channels" not in read_header(path)

    append_rows(path, WIDE, [BenchmarkResult().to_dict()])
    assert read_header(path) == WIDE


# --------------------------------------------------------------------------
# the other cases append_rows must handle
# --------------------------------------------------------------------------

def test_missing_file_is_created_with_a_header(tmp_path):
    path = tmp_path / "new.csv"
    assert append_rows(path, WIDE, [BenchmarkResult(num_pes=7).to_dict()]) == path
    assert read_header(path) == WIDE
    assert load_benchmarks(str(path))[0].num_pes == 7


def test_identical_header_is_a_plain_append(tmp_path):
    path = tmp_path / "results.csv"
    append_rows(path, WIDE, [BenchmarkResult(num_pes=1).to_dict()])
    append_rows(path, WIDE, [BenchmarkResult(num_pes=2).to_dict()])

    assert [r.num_pes for r in load_benchmarks(str(path))] == [1, 2]
    assert not path.with_suffix(path.suffix + ".bak").exists()  # no needless migration


def test_incompatible_header_diverts_and_leaves_the_original_alone(tmp_path):
    """A column present in the file but absent from the schema must not be dropped."""
    path = tmp_path / "results.csv"
    foreign = ["num_pes", "some_retired_column"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=foreign)
        writer.writeheader()
        writer.writerow({"num_pes": 4, "some_retired_column": "keep me"})
    original = path.read_text()

    written = append_rows(path, WIDE, [BenchmarkResult(num_pes=8).to_dict()])

    assert written != path
    assert written.name == "results.schema2.csv"
    assert path.read_text() == original          # untouched
    assert load_benchmarks(str(written))[0].num_pes == 8


def test_repeated_diversion_does_not_overwrite(tmp_path):
    path = tmp_path / "results.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["num_pes", "gone"])
        writer.writeheader()

    first = append_rows(path, WIDE, [BenchmarkResult(num_pes=1).to_dict()])
    second = append_rows(path, WIDE, [BenchmarkResult(num_pes=2).to_dict()])
    assert first != second
    assert {first.name, second.name} == {"results.schema2.csv", "results.schema3.csv"}


def test_empty_existing_file_is_treated_as_missing(tmp_path):
    path = tmp_path / "results.csv"
    path.touch()
    append_rows(path, WIDE, [BenchmarkResult(num_pes=3).to_dict()])
    assert read_header(path) == WIDE
    assert load_benchmarks(str(path))[0].num_pes == 3


# --------------------------------------------------------------------------
# save_benchmark reaches the same path
# --------------------------------------------------------------------------

def test_save_benchmark_migrates_rather_than_dropping_the_write(tmp_path):
    """It used to warn and silently return, losing the row."""
    path = tmp_path / "results.csv"
    _write_narrow_csv(path, num_pes=65536, compute_time_hw_avg=0.039308)

    save_benchmark(BenchmarkResult(num_pes=65536, compute_time_hw_avg=0.039398,
                                   channels=2), csv_path=str(path))

    rows = load_benchmarks(str(path))
    assert len(rows) == 2, "the appended row was dropped"
    assert rows[1].compute_time_hw_avg == pytest.approx(0.039398)
    assert rows[1].channels == 2
    assert rows[0].channels == 1


def test_save_benchmark_without_append_overwrites(tmp_path):
    path = tmp_path / "results.csv"
    save_benchmark(BenchmarkResult(num_pes=1), csv_path=str(path))
    save_benchmark(BenchmarkResult(num_pes=2), csv_path=str(path), append=False)

    rows = load_benchmarks(str(path))
    assert [r.num_pes for r in rows] == [2]


# --------------------------------------------------------------------------
# the launcher path end to end
# --------------------------------------------------------------------------

def _stub_launcher(source: Path):
    class _L:
        def download_artifact(self, _name, dest):
            Path(dest).write_text(source.read_text())
    return _L()


def test_downloaded_rows_land_under_the_right_keys_on_a_narrow_file(tmp_path):
    """The full reported scenario, through sdk/launcher.py."""
    from sdk.launcher import _download_benchmark

    worker_csv = tmp_path / "worker.csv"
    save_benchmark(BenchmarkResult(num_pes=65536, compute_time_hw_avg=0.039398,
                                   relative_error=1.49e-07), csv_path=str(worker_csv))

    destination = tmp_path / "single_114000x114000.csv"
    _write_narrow_csv(destination, num_pes=65536, compute_time_hw_avg=0.039308)

    _download_benchmark(_stub_launcher(worker_csv), destination,
                        git_commit="abc1234",
                        io_settings={"channels": 16, "width_west_buf": 0,
                                     "width_east_buf": 0})

    rows = load_benchmarks(str(destination))
    assert len(rows) == 2
    assert rows[0].compute_time_hw_avg == pytest.approx(0.039308)
    assert rows[0].channels == 1                      # backfilled, correct for it
    assert rows[1].compute_time_hw_avg == pytest.approx(0.039398)
    assert rows[1].relative_error == pytest.approx(1.49e-07)
    assert rows[1].channels == 16                     # stamped, the swept variable
    assert rows[1].git_commit == "abc1234"
    assert rows[1].compute_hw_cycles_min == 0         # not the compute time


# --------------------------------------------------------------------------
# the repair helper, against the corruption it exists for
# --------------------------------------------------------------------------

def test_field_defaults_cover_every_column():
    defaults = field_defaults()
    assert set(defaults) == set(WIDE)
    assert defaults["channels"] == 1
    assert defaults["width_west_buf"] == 0
    assert defaults["timestamp"] == ""  # factory default backfills empty
