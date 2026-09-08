#!/usr/bin/env bash
set -euo pipefail
# Keep this small CPU workload reproducible and avoid thread oversubscription.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m experiments.run_all --seeds 7 17 29 41 53 --out results
