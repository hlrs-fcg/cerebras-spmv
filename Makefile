DATE := $(shell date -I)
#PYTHON := python
PYTHON := uv run

.PHONY: help bench bench-save clean cluster-run cluster-compile

help:
	@echo "Available targets:"
	@echo "  make bench SWEEP=name              Run benchmark sweep in bench/configs/"
	@echo "  make bench-save SWEEP=name         Run and save with date prefix"
	@echo "  make cluster-compile ROWS=N COLS=N DENSITY=D   Compile for cluster"
	@echo "  make cluster-run                   Run on cluster (after compile)"
	@echo "  make cluster-sweep CONFIG=path     Run sweep on cluster"
	@echo "  make clean                         Remove __pycache__"
	@echo ""
	@echo "Examples:"
	@echo "  make bench SWEEP=density-sweep"
	@echo "  make bench-save SWEEP=strong-scaling"
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

# Cluster compile: make cluster-compile ROWS=10000 COLS=10000 DENSITY=0.01
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
	$(PYTHON) cluster/compile.py --rows $(ROWS) --cols $(COLS) --density $(DENSITY)

# Cluster run (uses artifact from compile)
cluster-run:
	$(PYTHON) cluster/run.py

# Cluster sweep: make cluster-sweep CONFIG=bench/configs/cluster/density-sweep.csv
cluster-sweep:
ifndef CONFIG
	$(error CONFIG is not set. Usage: make cluster-sweep CONFIG=bench/configs/cluster/density-sweep.csv)
endif
	$(PYTHON) cluster/run_sweep.py --config $(CONFIG)

# Clean pycache
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
