DATE := $(shell date -I)
#PYTHON := python
PYTHON := uv run

.PHONY: help bench bench-save clean clean-cluster cluster-test cluster-run cluster-compile cluster-compile-mtx fetch-matrix cluster-sweep

help:
	@echo "Available targets:"
	@echo "  make bench SWEEP=name              Run benchmark sweep in bench/configs/"
	@echo "  make bench-save SWEEP=name         Run and save with date prefix"
	@echo "  make cluster-test                  Test cluster setup (compile tiny matrix)"
	@echo "  make cluster-compile ROWS=N COLS=N DENSITY=D   Compile for cluster (random matrix)"
	@echo "  make cluster-compile-mtx MTX=path  Compile for cluster (from MTX file)"
	@echo "  make fetch-matrix ID=N             Fetch matrix from SuiteSparse"
	@echo "  make cluster-run                   Run on cluster (after compile)"
	@echo "  make cluster-sweep CONFIG=path     Run sweep on cluster"
	@echo "  make clean                         Remove __pycache__"
	@echo "  make clean-cluster                 Remove cluster artifacts and work dirs"
	@echo ""
	@echo "Examples:"
	@echo "  make cluster-test"
	@echo "  make fetch-matrix ID=1"
	@echo "  make cluster-compile-mtx MTX=bench/matrices/HB_bcsstk14/HB_bcsstk14.mtx"
	@echo "  make cluster-compile ROWS=10000 COLS=10000 DENSITY=0.01"
	@echo "  make cluster-sweep CONFIG=bench/configs/cluster/comprehensive-01.csv"

# Run a sweep: make bench SWEEP=density-sweep
bench:
ifndef SWEEP
	$(error SWEEP is not set. Usage: make bench SWEEP=density-sweep)
endif
	$(PYTHON) bench/sweep.py --config bench/configs/$(SWEEP).csv

# Run and save with date prefix
bench-save:
ifndef SWEEP
	$(error SWEEP is not set. Usage: make bench-save SWEEP=density-sweep)
endif
	$(PYTHON) bench/sweep.py --config bench/configs/$(SWEEP).csv \
		--output results/data/$(DATE)-$(SWEEP).csv

# Fetch matrix from SuiteSparse: make fetch-matrix ID=1
fetch-matrix:
ifndef ID
	$(error ID is not set. Usage: make fetch-matrix ID=1 or ID=HB/bcsstk14)
endif
	$(PYTHON) cluster/compile.py --fetch-id $(ID) --num-pes min

# Cluster compile from MTX file: make cluster-compile-mtx MTX=path/to/file.mtx
cluster-compile-mtx:
ifndef MTX
	$(error MTX is not set. Usage: make cluster-compile-mtx MTX=bench/matrices/file.mtx)
endif
	$(PYTHON) cluster/compile.py --mtx-file $(MTX) --num-pes min

# Cluster compile (random matrix): make cluster-compile ROWS=10000 COLS=10000 DENSITY=0.01
cluster-compile:
ifndef ROWS
	$(error ROWS is not set. Usage: make cluster-compile ROWS=10000 COLS=10000 DENSITY=0.01)
endif
ifndef COLS
	$(error COLS is not set. Usage: make cluster-compile ROWS=10000 COLS=10000 DENSITY=0.01)
endif
ifndef DENSITY
	$(error DENSITY is not set. Usage: make cluster-compile ROWS=10000 COLS=10000 DENSITY=0.01)
endif
	$(PYTHON) cluster/compile.py --random --rows $(ROWS) --cols $(COLS) --density $(DENSITY)

# Cluster run (uses artifact from compile)
cluster-run:
	$(PYTHON) cluster/run.py

# Cluster sweep: make cluster-sweep CONFIG=bench/configs/cluster/density-sweep.csv
cluster-sweep:
ifndef CONFIG
	$(error CONFIG is not set. Usage: make cluster-sweep CONFIG=bench/configs/cluster/density-sweep.csv)
endif
	$(PYTHON) cluster/run_sweep.py --config $(CONFIG)

# Test cluster compilation with tiny matrix
cluster-test:
	@echo "Testing cluster compilation with tiny matrix..."
	$(PYTHON) cluster/compile.py --random --rows 100 --cols 100 --density 0.1 --num-pes 1

# Clean pycache
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Clean cluster artifacts and work directories
clean-cluster:
	rm -rf cluster_work/ sweep_work/ out/
	rm -f artifact.json artifact_path.json
