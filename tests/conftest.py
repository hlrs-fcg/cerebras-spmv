"""Keep test runs from writing caches into the repo (matrices/tiles, numba's compiled kernels)."""

import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.gettempdir())
# stable across runs, so numba's compiled kernels are reused instead of rebuilt every session
os.environ.setdefault("NUMBA_CACHE_DIR", str(_tmp / "spmv-tests-numba"))
os.environ.pop("SPMV_TILE_CACHE_DIR", None)      # the cache is opt-in; tests that want it set it
