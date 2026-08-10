#!/bin/bash
# Resubmit remaining TinyImageNet alexnet TypiClust runs (seeds 1 and 5).
#
# Fixes DataLoader hangs: NUM_WORKERS=0, more CPUs, L40S-only, 48h walltime.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=/vast/s219110279
DATASET=TINYIMAGENET
BACKBONE=alexnet
BUDGET=50
INITIAL_SIZE=50
WALLTIME="2-00:00:00"
CONSTRAINT="gpu-l40s"
CPUS=4
NUM_WORKERS=0
STAMP="$(date +%Y%m%d_%H%M%S)"

for seed in 1 5; do
  out_dir="$PROJECT_ROOT/TypiClust/output/$DATASET/$BACKBONE/typiclust_${seed}_${BUDGET}b"
  if [[ -f "$out_dir/benchmark_summary.json" ]]; then
    echo "Skip complete: $out_dir"
    continue
  fi
  if [[ -d "$out_dir" ]] && compgen -G "$out_dir/episode_*" > /dev/null; then
    backup_dir="${out_dir}_partial_${STAMP}"
    echo "Backing up partial run: $out_dir -> $backup_dir"
    mv "$out_dir" "$backup_dir"
  fi
  job_name="tc_tinyimagenet_alexnet_s${seed}"
  echo "Submitting $job_name (time=$WALLTIME cpus=$CPUS workers=$NUM_WORKERS)"
  sbatch \
    --job-name="$job_name" \
    --time="$WALLTIME" \
    --constraint="$CONSTRAINT" \
    --cpus-per-task="$CPUS" \
    --export=ALL,DATASET="$DATASET",BACKBONE="$BACKBONE",SEED="$seed",BUDGET="$BUDGET",INITIAL_SIZE="$INITIAL_SIZE",NUM_WORKERS="$NUM_WORKERS" \
    "$SCRIPT_DIR/run_typiclust.sbatch"
done

echo "Done. Check with: squeue --me | grep tc_tinyimagenet_alexnet"
