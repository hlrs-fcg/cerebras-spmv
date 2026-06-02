#!/usr/bin/env bash
# Remove generated artifacts — safe to re-run at any time
set -e

rm -rf cluster_work/
rm -rf .venv/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.egg-info"  -exec rm -rf {} + 2>/dev/null || true

echo "Cleaned."
