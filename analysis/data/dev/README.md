# Development-run data the v1-vs-v2 notebook needs

Copies of rows from the development runs, taken before RC1, so that `analysis/v1_vs_v2.ipynb` runs from the committed files. They are NOT
release data; the notebook labels them as development runs and prefers `cluster/results/v2.0-data/` where both exist.

| File | What | Commit that produced it |
|---|---|---|
| `2026-10-09_weak-scaling.csv` | valtask streaming, weak scaling 1K to 857K PEs (n=3) | 16c4ce3 |
| `2026-10-09_valtask-65k-sync.csv` | valtask streaming at 65K, with and without synced timing (use `sync_timing == False`) | 6bf7c59 |
| `2026-10-13_resident-k0-65k.csv`, `...-k0-full.csv` | resident kernel 0 (CSR loop) at 65K and the full wafer, with null twins | not recorded (`unknown`) |
| `2026-10-14_kernelcheck16_hw_k{0,1,2}[_null].csv` | the three local kernels at 16 PEs, hardware, with null twins | not recorded |
| `2026-10-14_kernelcheck65_hw_k{1,2}.csv` | kernels 1 and 2 at 65K | not recorded |

Raw rows, like `cluster/results/v2.0-data/`; the public export redacts them (`scripts/redact_results.py`).
