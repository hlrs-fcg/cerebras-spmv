"""Size a sweep [[run]] that names a MatrixMarket file, and refuse it before anything is compiled.

The same check as `run.py run --mtx-file` (run.size_from_mtx), through the same arithmetic (cluster/mm_io.py):
read the file on the login side, count the NNZ of every PE on the grid the run would use, and set rows, cols, density
and TOTAL_NNZ from it. A run that does not fit the PE's SRAM raises a ValueError that names what it needs and the smallest
grid that fits; run.cmd_sweep expands every run up front, so no run of the sweep has started when that happens.
Login side only (not staged to the worker).
"""

from __future__ import annotations

import warnings
from pathlib import Path

from cluster import mm_io
from config import DEFAULT_ENCODING, SpMVConfig

_SERIES = [16 * 4 ** k for k in range(0, 9) if 16 * 4 ** k < 857_010] + [857_010]


def _grid(pes, grid_height, simulator, sync):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SpMVConfig(rows=64, cols=64, density=0.1, num_pes_target=pes, grid_height_target=grid_height,
                       simulator=simulator, sync_timing=sync)
    return c.width, c.height


def size_run(run: dict) -> dict:
    """``run`` (a [[run]] table) with rows, cols, density and total_nnz_per_pe taken from its ``mtx_file``.

    rows / cols may be given, and must then match the file; a given total_nnz_per_pe must be at least the busiest PE's
    NNZ (a larger one is kept: it only costs SRAM, and SpMVConfig's guard still checks it)."""
    path = Path(run["mtx_file"])
    mm = mm_io.read_matrix_market(path)                      # ValueError names the file's problem
    for key, have in (("rows", mm.rows), ("cols", mm.cols)):
        if run.get(key) is not None and run[key] != have:
            raise ValueError(f"{key} = {run[key]} contradicts {path.name} ({mm.rows}x{mm.cols}); drop it, the file decides")
    encoding = run.get("encoding", DEFAULT_ENCODING)
    sync = bool(run.get("sync_timing", False))
    simulator = bool(run.get("simulator", False))
    W, H = _grid(run.get("num_pes_target"), run.get("grid_height_target"), simulator, sync)
    counts = mm_io.tile_nnz_counts(mm.row, mm.col, mm.rows, mm.cols, W, H)
    rep = mm_io.fit_report(counts, mm.rows, mm.cols, sync, encoding)
    if not rep.fits:
        grids = []
        for pes in _SERIES:
            try:
                grids.append(_grid(pes, run.get("grid_height_target"), simulator, sync))
            except ValueError:
                continue
        better = mm_io.smallest_fit(mm.row, mm.col, mm.rows, mm.cols, grids, sync, encoding)
        hint = (f"; the smallest grid that fits is {better[0][0]}x{better[0][1]} = {better[0][0] * better[0][1]:,} PEs "
                f"(num_pes_target = {better[0][0] * better[0][1]})" if better else
                "; no grid up to the full wafer holds it with the block tiling")
        raise ValueError(f"{path.name} does not fit on {W}x{H} = {W * H:,} PEs with encoding {encoding}: "
                         f"{rep.reason}{hint}. Nothing was compiled.")
    given = run.get("total_nnz_per_pe")
    if given is not None and given < rep.max:
        raise ValueError(f"total_nnz_per_pe = {given} is below the busiest PE's {rep.max} NNZ for {path.name} on {W}x{H}; "
                         f"drop it (it is computed from the file) or raise it")
    return {**run, "rows": mm.rows, "cols": mm.cols, "density": mm.nnz / (mm.rows * mm.cols),
            "total_nnz_per_pe": max(given if given is not None else rep.max, 1), "mtx_file": path}
