# cluster tools

## Structure
### Modules
- `compile.py` - compiler function to interface with the csl compiler.
- `worker_runtime.py` - interfaces to the WSE, not meant to be executed by the user.
- `run.py`
- `run_sweep.py`

## Usage
### `compile.py`
Has functions to normally compile a random matrix with given parameters or alternatively load Matrixmarket file or fetch
Suitesparse matrix.

```python
python compile.py --rows 1000 --cols 1000 --density 0.01 --num-pes min
python compile.py --mtx-file bench/matrices/[ABC]*.mtx --num-pes min
python cluster/compile.py --fetch-id 1 --num-pes min
python cluster/compile.py --fetch-id HB/bcsstk14 --num-pes max
```
Parameters:
- `--rows <int>`
- `--cols <int>`
- `--density <[0,1]>`
- `--num-pes <int>`
- `--seed <int>`
- `--artifact-path <path>`
- `--out-path <path>`
- `--verbose`

Mutually exclusive matrix source:
- `--random` scipy generated matrix.
- `--fetch-id <ID>` SuiteSparse ID.
- `--mtx-file <path>`, path to MatrixMarket (.mtx) file.


### `run.py`
Runs compiled artifacts.

Parameters:
- `--artifact_path`, default: `artiftact_path.json`
- `--simulator`, default: `false`
- `--download-benchmark`, local path where to store run data.
- `--verbose`

### `run_sweep.py
Runs sweeps.

Parameters:
- `--config`, path to sweep config.
- `--output`, where to save results, default: `results/data/benchmark_results.csv`
- `--work-dir`, working dir for artifacts, default: `sweep_work`
- `--simulator` 
- `--stark-index`, default: 0
- `--max-runs`, to limit runs in case of testing
- `--verbose`
