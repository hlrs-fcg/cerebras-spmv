# v2.0 release-candidate results (RC1, commit 4739dc4; smoke 580e41f)

Raw rows of the RC1 smoke and headline runs, as written by the worker (barrier-closed timings; every row records its commit, whether the
tree was clean and `git describe`). Hardware files: `smoke16_*` (16 PEs: header, valtask, resident, resident null twin), `resident_65k`,
`resident_full` and `resident_full_null`. `*_sim_*` are the same 16-PE runs in the simulator (their times are simulator wall time, not
hardware). Also here:

| files | what | commit |
|---|---|---|
| `2026-10-14_valtask_65k.csv`, `_valtask_full.csv` | valtask streaming at 65K and the full wafer | `4739dc4` |
| `2026-10-15_mtx_<matrix>_<grid>.csv` (+ `_null` twin at the full wafer) | real SuiteSparse matrices on the wafer (`--mtx-file`): bcsstk01 and nos6 at 16 PEs, bcsstk01, nos6 and bcsstk14 at 65K, parabolic_fem at the full wafer; `_sim` = simulator | `4739dc4` |
| `2026-10-15_rc2_*_sim.csv`, `2026-10-15_rc2_sweep_refusal.txt` | simulator smoke from a fresh `git archive` and the sweep's refusal of ecology2 before any compile | `580e41f` |
| `2026-10-15_host-cpu-random.txt`, `2026-10-15_host-cpu-mtx.txt` | `bench/host_spmv_baseline.py` on the user node (32 cores): random sizes; the four real matrices (per call and steady state) | `03ef71a` script |
| `2026-10-15_tilers.txt` | host tiling backends timed | |
| `matrices.md` | the matrices' sha256, shapes and NNZ | |

The simulator files (`_sim`) time the simulator, not the wafer.

These files are the raw rows. Some columns can name the appliance job or the user node (`numba_status` holds the job's work-directory
path, a `comment` can name the node a job ran on). The **public** tree and any public dataset get a redacted copy
(`scripts/redact_results.py`, run by `scripts/export_public.py`: job ids, user-node and system names, accounts and home paths become
`<job id>`, `<user node>`, `<system>`, `<user>`, `<path>`); the raw files stay here.
