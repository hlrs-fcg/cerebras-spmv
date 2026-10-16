"""Host CPU SpMV baselines: the random weak-scaling sizes (same generator, seed 42, d=1e-3), or real matrices.

    python host_spmv_baseline.py                 # the three random matrices (what docs/results.md quotes)
    python host_spmv_baseline.py --mtx A.mtx ... # MatrixMarket files instead (symmetric ones expanded as on the wafer)

--mtx needs cluster/mm_io.py (numpy only): found in ../cluster, in cluster/, or next to this file. x is the wafer validation's:
default_rng(--seed, 42).integers(0, 100, cols) as float32.
"""
import argparse, time, os, platform, sys
import numpy as np, scipy.sparse as sp, numba as nb

def spmv_reference(r, c, v, x, n):  # == cluster/coo_tiling.py spmv_reference (validation ref)
    return np.bincount(r, weights=v.astype(np.float64) * x[c].astype(np.float64), minlength=n).astype(np.float32)

@nb.njit(parallel=True, fastmath=True, cache=True)
def csr_par(indptr, indices, data, x, y):
    for r in nb.prange(len(indptr) - 1):
        acc = np.float32(0.0)
        for p in range(indptr[r], indptr[r + 1]):
            acc += data[p] * x[indices[p]]
        y[r] = acc

def timeit(f, n=7):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts) // 2], ts[0], ts[-1]

def bench(label, r, c, v, x, n, cols=None):
    t = time.perf_counter(); A = sp.csr_matrix((v, (r, c)), shape=(n, cols or n), dtype=np.float32); A.sum_duplicates(); conv = time.perf_counter() - t
    y = np.empty(n, np.float32); csr_par(A.indptr, A.indices, A.data, x, y)
    ref = spmv_reference(r, c, v, x, n)
    rows = [("numpy bincount fp64 COO (our validation ref)", timeit(lambda: spmv_reference(r, c, v, x, n), 3)),
            ("scipy CSR fp32, 1 thread", timeit(lambda: A @ x))]
    for th in (1, 8, 32):
        nb.set_num_threads(th)
        rows.append((f"numba CSR fp32, {th} threads", timeit(lambda: csr_par(A.indptr, A.indices, A.data, x, y))))
    err = np.linalg.norm(y - ref) / np.linalg.norm(ref)
    print(f"\n{label} nnz={A.nnz:,}  COO->CSR one-time {conv:.2f} s  numba rel err {err:.1e}")
    for k, (med, mn, mx) in rows:
        print(f"  {k:46s} median {med*1e3:9.2f} ms  (min {mn*1e3:8.2f}, max {mx*1e3:8.2f})  {2*A.nnz/med/1e9:6.2f} GFLOP/s")


def fmt_t(t):
    return f"{t*1e6:9.2f} us" if t < 1e-3 else f"{t*1e3:9.4f} ms"


def steady(f, reps=7):
    """(median seconds per call, calls per loop): a loop of N calls divided by N, N >= 200 under 1 ms (thread launch amortised), else 10."""
    t = time.perf_counter(); f(); one = time.perf_counter() - t
    n = 200 if one < 1e-3 else 10
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(n):
            f()
        ts.append((time.perf_counter() - t) / n)
    ts.sort(); return ts[len(ts) // 2], n


def bench_mtx(label, r, c, v, x, n, cols):
    t = time.perf_counter(); A = sp.csr_matrix((v, (r, c)), shape=(n, cols), dtype=np.float32); A.sum_duplicates(); conv = time.perf_counter() - t
    y = np.empty(n, np.float32); csr_par(A.indptr, A.indices, A.data, x, y)
    ref = spmv_reference(r, c, v, x, n)
    fs = [("numpy bincount fp64 COO (our validation ref)", None, lambda: spmv_reference(r, c, v, x, n)),
          ("scipy CSR fp32, 1 thread", None, lambda: A @ x)]
    for th in (1, 8, 32):
        fs.append((f"numba CSR fp32, {th} threads", th, lambda: csr_par(A.indptr, A.indices, A.data, x, y)))
    print(f"\n{label} nnz={A.nnz:,}  COO->CSR one-time {conv*1e3:.4f} ms")
    print("  per call = each call timed on its own (includes thread launch), median of 7 (min, max); "
          "steady = loop of N calls / N, median of 7 loops")
    for k, th, f in fs:
        if th:
            nb.set_num_threads(th)
        med, mn, mx = timeit(f)
        st, reps = steady(f)
        print(f"  {k:46s} per call {fmt_t(med)} (min {fmt_t(mn)}, max {fmt_t(mx)})  steady {fmt_t(st)} (N={reps})  {2*A.nnz/st/1e9:6.2f} GFLOP/s steady")
    nb.set_num_threads(32); csr_par(A.indptr, A.indices, A.data, x, y)
    print(f"  numba rel err {np.linalg.norm(y - ref) / np.linalg.norm(ref):.1e}")


def load_mm_io():
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.path.join(here, "..", "cluster"), os.path.join(here, "cluster"), here):
        if os.path.exists(os.path.join(d, "mm_io.py")):
            sys.path.insert(0, d)
            import mm_io
            return mm_io
    raise SystemExit("--mtx needs cluster/mm_io.py (next to this file, in cluster/ or ../cluster)")


def run_mtx(files, seed):
    mm_io = load_mm_io()
    for f in files:
        mm = mm_io.read_matrix_market(f)
        x = np.random.default_rng(seed).integers(0, 100, size=mm.cols).astype(np.float32)   # worker_host.py's x
        bench_mtx(f"{os.path.basename(f)} {mm.rows}x{mm.cols} ({mm.symmetry}, {mm.stored_entries:,} stored)", mm.row, mm.col, mm.data, x, mm.rows, mm.cols)


def run_random():
    for n in (14300, 114000, 414000):
        rng = np.random.default_rng(42); nnz = int(n * n * 1e-3)
        r = rng.integers(0, n, nnz, dtype=np.int32); c = rng.integers(0, n, nnz, dtype=np.int32)
        v = rng.random(nnz).astype(np.float32); x = rng.integers(0, 100, n).astype(np.float32)
        bench(f"n={n:,}", r, c, v, x, n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mtx", nargs="+", metavar="FILE", help="MatrixMarket files instead of the random matrices")
    ap.add_argument("--seed", type=int, default=42, help="seed of x for --mtx (the wafer runs' default)")
    a = ap.parse_args()
    print(f"host {platform.node()}, {os.cpu_count()} cores, numba threads {nb.get_num_threads()}, numpy {np.__version__}")
    run_mtx(a.mtx, a.seed) if a.mtx else run_random()
