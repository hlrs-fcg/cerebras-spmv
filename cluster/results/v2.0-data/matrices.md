# Real matrices used in the v2.0 benchmarks

SuiteSparse Matrix Collection files in MatrixMarket coordinate format, downloaded over plain http (no
upstream checksums), so the sha256 below pins the exact bytes the runs used. The same bytes were checked
on the benchmark node. `NNZ` is the count after expanding the stored triangle of a symmetric file (what the
kernel sees); `stored` is the count of entries in the file.

| matrix | SuiteSparse group | sha256 | n | stored | NNZ (expanded) | symmetry |
|---|---|---|---|---|---|---|
| bcsstk01 | HB | e2a34f2243bac0d2a130e1447658cdd63a0ed79c18ab4a3bd41055b232ea16a4 | 48 | 224 | 400 | symmetric |
| nos6 | HB | 199b0f92c7cda1637d3c1811de87e1f8a3f4356ef0676f62772e88998554fb83 | 675 | 1,965 | 3,255 | symmetric |
| bcsstk14 | HB | 4130d3bf6f881a4df4b22f2fd94bbf2f352e1bdb1d1ad20f4fcae64ec2ec448d | 1,806 | 32,630 | 63,454 | symmetric |
| parabolic_fem | Wissgott | 711a89faf2454b052e1e326b6bad6e5d2df4cde162d92ae6e0bc0577b5c6f488 | 525,825 | 2,100,225 | 3,674,625 | symmetric |

Run results: `2026-10-15_mtx_<matrix>_<grid>.csv` (simulator rows end in `_sim`).
