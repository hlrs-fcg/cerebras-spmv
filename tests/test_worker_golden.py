"""The worker's device-facing behaviour, replayed against goldens recorded from the 4fb6594 worker.

tests/worker_trace.py runs cluster/worker_runtime.py main() against a recording fake SdkRuntime and
captures every runner call (symbol, rectangle, nonblock, a hash of the host buffer), the progress.log
phases and the CSV row's non-timing columns.  A refactor of the worker must reproduce all of it.
Cases for deleted features are removed together with the feature (regenerate nothing: delete the file).
"""

import json
from pathlib import Path

import pytest

from tests.worker_trace import CASES, trace

GOLDEN = Path(__file__).resolve().parent / "golden" / "worker"


@pytest.mark.parametrize("case", sorted(CASES))
def test_worker_matches_the_recorded_trace(case):
    golden_file = GOLDEN / f"{case}.json"
    if not golden_file.exists():
        pytest.skip(f"{case}: golden removed with its feature")
    golden = json.loads(golden_file.read_text())
    got = trace(case)
    assert got["exit"] == golden["exit"]
    assert got["calls"] == golden["calls"]
    assert got["progress"] == golden["progress"]
    drift = {k: (v, got["row"].get(k)) for k, v in golden["row"].items() if got["row"].get(k) != v}
    assert not drift, drift
