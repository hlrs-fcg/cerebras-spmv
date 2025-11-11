# SpMV on Cerebras WSE

SpMV implementation using a matched pairs approach.

## Project Structure
```
.
├── spmv_runner/              # Modular SpMV runner package
│   ├── core/                 # Core execution modules
│   │   ├── compiler.py       # CSL compilation wrapper
│   │   ├── grid.py           # PE grid calculation & validation
│   │   ├── runtime.py        # Device runtime & execution
│   │   └── __init__.py
│   ├── utils/                # Utility modules
│   │   ├── memory.py         # Memory calculation & analysis
│   │   ├── pe_selection.py   # PE count selection algorithms
│   │   ├── timing.py         # Performance timing & reporting
│   │   └── __init__.py
│   └── __init__.py
├── src/                      # CSL source code
│   ├── layout.csl            # PE grid layout definition
│   └── pe.csl                # PE kernel implementation
├── run.py                    # Main CLI entry point
└── spmv_prototype.py         # Python prototype implementation
```

## Functionality
- **Multi-PE Support**: 1 to 888K+ PEs with automatic grid calculation
- **PE Selection**: Auto-min (memory), auto-max (parallelism), or manual
- **Memory Management**: Per-PE validation with 48 KB limit enforcement
- **Performance Analysis**: Compilation, H2D/D2H transfer, and compute timing
- **Architecture Support**: WSE-2 (753K PEs) and WSE-3 (888K PEs)

## Usage

```bash
# Auto-minimum PEs (memory-constrained)
python run.py --rows 1000 --cols 1000 --density 0.01 --num-pes min

# Auto-maximum PEs (maximize parallelism)
python run.py --rows 1000 --cols 1000 --density 0.01 --num-pes max

# Specific number of PEs
python run.py --rows 1000 --cols 1000 --density 0.01 --num-pes 16

# Show PE count suggestions
python run.py --rows 5000 --cols 5000 --density 0.005 --suggest-pes

# Compile only (no execution)
python run.py --rows 1000 --cols 1000 --density 0.01 --num-pes 4 --compile-only
```

## Options
```
usage: run.py [-h] [--rows ROWS] [--cols COLS] [--density DENSITY] [--num-pes NUM_PES] [--seed SEED] [--suggest-pes] [--compile-only] [--artifact-path ARTIFACT_PATH] [--out-path OUT_PATH] [--arch {wse2,wse3}]

optional arguments:
  -h, --help            show this help message and exit
  --rows ROWS           Number of rows
  --cols COLS           Number of columns
  --density DENSITY     Sparsity density
  --num-pes NUM_PES     Number of PEs: integer, "min", or "max"
  --seed SEED           Random seed
  --suggest-pes         Show PE count suggestions and exit
  --compile-only        Only compile, do not run
  --artifact-path ARTIFACT_PATH
                        Path to save artifact metadata
  --out-path OUT_PATH   Compilation output directory
  --arch {wse2,wse3}    Target architecture
```

## Matched Pairs Approach

Traditional SpMV requires vector broadcast and index lookup during computation. The matched pairs
approach pre-gathers vector values on host using column indices, sending matrix values with their
corresponding vector values as pairs. This eliminates communication and indexing overhead on PEs.

**Memory per PE**: ~8 bytes/nonzero + ~8 bytes/row + padding

## Current Limitations 
### Static Memory Allocation
Data can be better load balanced. Currently there is safety padding in all PEs
for edge cases. Sometimes a small percentage of the PEs also doesn't run (1-5%)
due to the currently implemented load balancing. For possible improvements see
below.

### Benchmark Storage
`.csv` is used for benchmark results, which is fragile when schema changes.
There is migration logic handling this, but SQLite would provide better schema
versioning, type safety, and query capabilities.

### Load Balancing
Current distribution uses simple NNZ-based balancing. Could improve with:
- Row-based partitioning strategies
- Dynamic rebalancing based on computation patterns
- Better handling of highly skewed distributions

### Hardware Constraints
- 48 KB memory limit per PE
- Single precision (f32) only

### Matrix Input Format
Currently only supports CSR format as input. Could probably trivially implement:
- CSC format support 
- Direct COO input 
- Matrix Market file reading 
