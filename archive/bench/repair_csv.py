#!/usr/bin/env python3
"""Repair a benchmark CSV corrupted by a positional append.

THE CORRUPTION
--------------
Before the fix in bench/benchmark.py:append_rows(), sdk/launcher.py appended
downloaded rows with `csv.DictWriter(f, fieldnames=<current schema>)` onto a
file whose header was written under an older, narrower schema.  DictWriter does
not check the existing header, so each appended line carries the *wide* column
count in *wide* order under a *narrow* header.

Reading that back maps wide_value[i] onto narrow_header[i], so every value
after the first inserted column sits under the wrong key.  The tell in
single_114000x114000.csv was `compute_time_hw_avg` reading 112086016.0 seconds
while the real 0.0394 s had landed in `compute_hw_cycles_min`.

Nothing is lost: the values are all present and in wide order, just re-keyed.
Repair is therefore exact — re-zip each over-long line against the wide
fieldnames.

HOW ROWS ARE CLASSIFIED
-----------------------
By field count alone, which is unambiguous as long as the two schemas differ in
width (they differ by 8 columns here):

    len(row) == len(header)      -> written under the old schema, already aligned
    len(row) == len(wide)        -> positionally appended, needs re-keying
    anything else                -> left alone and reported, never guessed at

Old rows are backfilled from the dataclass defaults, which are correct for them:
every row predating these columns was measured at channels=1, no edge buffers.

USAGE
-----
    python bench/repair_csv.py cluster/results/data/single_114000x114000.csv
    python bench/repair_csv.py <file> -o repaired.csv     # do not write in place
    python bench/repair_csv.py <file> --dry-run           # just report

In place is the default and keeps a .bak.  Always eyeball --dry-run first.
"""

import argparse
import csv
import shutil
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.benchmark import BenchmarkResult, field_defaults  # noqa: E402


def wide_fieldnames() -> list[str]:
    return [f.name for f in fields(BenchmarkResult)]


def classify(raw_rows: list[list[str]], header: list[str],
             wide: list[str]) -> tuple[list[dict], list[tuple[int, list[str]]]]:
    """Return (repaired rows keyed by `wide`, [(line_no, row)] left unhandled)."""
    if len(header) == len(wide):
        raise SystemExit(
            f"header has {len(header)} columns, the same as the current schema — "
            f"row width cannot discriminate the two. Repair this file by hand."
        )

    defaults = field_defaults()
    repaired, unhandled = [], []

    for offset, row in enumerate(raw_rows):
        line_no = offset + 2  # 1-based, and the header is line 1
        if len(row) == len(wide):
            repaired.append(dict(zip(wide, row)))
        elif len(row) == len(header):
            narrow = dict(zip(header, row))
            repaired.append({name: narrow.get(name, defaults.get(name, ""))
                             for name in wide})
        else:
            unhandled.append((line_no, row))

    return repaired, unhandled


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair a positionally-shifted benchmark CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="write here instead of repairing in place")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change and write nothing")
    args = parser.parse_args()

    if not args.csv_path.is_file():
        print(f"no such file: {args.csv_path}", file=sys.stderr)
        return 1

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        print("file is empty", file=sys.stderr)
        return 1

    header, raw_rows = rows[0], rows[1:]
    wide = wide_fieldnames()
    repaired, unhandled = classify(raw_rows, header, wide)

    shifted = sum(1 for r in raw_rows if len(r) == len(wide))
    aligned = sum(1 for r in raw_rows if len(r) == len(header))
    print(f"{args.csv_path}: header has {len(header)} columns, "
          f"current schema has {len(wide)}")
    print(f"  {aligned} row(s) already aligned, {shifted} row(s) shifted, "
          f"{len(unhandled)} unhandled")
    for line_no, row in unhandled:
        print(f"    line {line_no}: {len(row)} fields — left out, repair by hand")

    if repaired:
        sample = repaired[-1]
        print(f"  after repair, last row reads: "
              f"compute_time_hw_avg={sample.get('compute_time_hw_avg')!r} "
              f"channels={sample.get('channels')!r} "
              f"num_pes={sample.get('num_pes')!r}")

    if args.dry_run:
        print("  --dry-run: nothing written")
        return 0

    destination = args.output or args.csv_path
    if destination == args.csv_path:
        backup = args.csv_path.with_suffix(args.csv_path.suffix + ".bak")
        shutil.copyfile(args.csv_path, backup)
        print(f"  backup: {backup}")

    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=wide)
        writer.writeheader()
        writer.writerows(repaired)
    tmp.replace(destination)
    print(f"  wrote {len(repaired)} row(s) to {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
