"""analysis/v1_vs_v2.ipynb runs top to bottom from the committed CSVs, reads the numbers the docs quote, and skips cleanly without data."""

import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pytest.importorskip("IPython")          # the notebook code displays through IPython (not in the plain test recipe)

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
NB = ANALYSIS / "v1_vs_v2.ipynb"


def _run(monkeypatch, v2_dir=None, dev_dir=None, v1_dir=None):
    import sys
    import IPython.display as D
    monkeypatch.syspath_prepend(str(ANALYSIS))
    sys.modules.pop("lib", None)
    import lib
    for name, value in (("V2_DIR", v2_dir), ("DEV_DIR", dev_dir), ("V1_DIR", v1_dir)):
        if value is not None:
            monkeypatch.setattr(lib, name, value)
    out = []
    monkeypatch.setattr(D, "display", lambda x: out.append(getattr(x, "data", x)))
    monkeypatch.setattr(plt, "show", lambda *a, **k: plt.close("all"))
    monkeypatch.chdir(ANALYSIS)
    g = {}
    for cell in json.loads(NB.read_text())["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), g)
    return g, out


def test_the_notebook_is_committed_without_outputs_and_is_short():
    cells = json.loads(NB.read_text())["cells"]
    assert 8 <= len(cells) <= 12
    assert all(not c.get("outputs") and c.get("execution_count") in (None, 0) for c in cells if c["cell_type"] == "code")


def test_it_reads_the_headline_numbers_from_the_csvs(monkeypatch):
    g, out = _run(monkeypatch)
    K = g["K"]
    import numpy as np
    cyc = lambda k, p: float(np.median(K[(k, p)][0]))
    assert (cyc(0, 16), cyc(1, 16), cyc(2, 16)) == (35449, 25757, 27857)          # docs/resident-spmv.md, the kernel table
    assert (cyc(0, 65_536), cyc(1, 65_536), cyc(2, 65_536)) == (38598, 23756, 15316)
    assert (cyc(0, 857_010), cyc(2, 857_010)) == (43293, 25193)
    assert round(g["be"]["loop"]) == pytest.approx(3668, abs=60)                    # break-even with the loop on the wafer
    assert g["c_all"] == pytest.approx(12.6, abs=0.3) and g["e_all"] == pytest.approx(40.6, abs=0.5)
    text = "\n".join(str(o) for o in out)
    assert "kernel 2 is 2.52x faster than kernel 0" in text


def test_cells_skip_cleanly_when_the_v2_data_is_missing(monkeypatch, tmp_path):
    empty = tmp_path / "none"
    empty.mkdir()
    g, out = _run(monkeypatch, v2_dir=empty, dev_dir=empty)
    text = "\n".join(str(o) for o in out)
    assert text.count("pending:") >= 3                                              # one-line markers, not errors


def test_cells_skip_cleanly_when_v1_is_missing(monkeypatch, tmp_path):
    empty = tmp_path / "none"
    empty.mkdir()
    g, out = _run(monkeypatch, v1_dir=empty)
    assert "v2.0 streaming (valtask)" in g["S"]                                     # the v2.0 series are still there
