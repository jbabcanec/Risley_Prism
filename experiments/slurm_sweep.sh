#!/bin/bash
# SLURM array template for the large-scale experimental program.
# Every task is a single-core, ~2-100 s solve; the whole program is
# ~50-100 core-days. Adjust array sizes and --count to your allocation.
#
# The identifiability atlas: 100,000 configurations, adaptive T ladder.
#   sbatch --array=0-999 experiments/slurm_sweep.sh atlas adaptive 100
# gives task k the cases [k*100, (k+1)*100).
#
# Other experiments (same sharding pattern):
#   fixedT cohorts (scaling laws):   --exp fixedT  --T 5 10 20 40 80 160 320
#   prescription quantiles:          --exp mint    --Tmax 320
#   P x T frontier:                  --exp speedsP --P 2..6 --T 10 20 40 80
#   noise phase diagram:             --exp noise   --snr 60 50 40 30 20
#   calibration at scale: add --bounds to any adaptive/fixedT run
#
# Aggregate afterwards with experiments/aggregate.py (reads all
# results/sweep_*.jsonl; shards are resume-safe and order-free).

#SBATCH --job-name=risley-sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00

set -euo pipefail
TAG=${1:-atlas}
EXP=${2:-adaptive}
COUNT=${3:-100}
SHIFT=$(( SLURM_ARRAY_TASK_ID * COUNT ))

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python experiments/sweep.py \
    --exp "$EXP" --tag "$TAG" \
    --start "$SHIFT" --count "$COUNT" \
    "${@:4}"
