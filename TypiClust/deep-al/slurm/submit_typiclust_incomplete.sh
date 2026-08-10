#!/bin/bash
# Submit TypiClust runs that lack benchmark_summary.json, with per-combo walltime.
#
# train_al.py does not resume mid-run; partial output dirs are backed up before resubmit.
#
# Usage:
#   cd /vast/s219110279/TypiClust/deep-al/slurm
#   bash submit_typiclust_incomplete.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=/vast/s219110279
BUDGET=50
INITIAL_SIZE=50
STAMP="$(date +%Y%m%d_%H%M%S)"

# dataset backbone seed time constraint
JOBS=(
  "CIFAR10 alexnet 1 24:00:00 gpu-v100|gpu-l40s"
  "CIFAR10 alexnet 2 24:00:00 gpu-v100|gpu-l40s"
  "CIFAR10 alexnet 3 24:00:00 gpu-v100|gpu-l40s"
  "CIFAR10 alexnet 4 24:00:00 gpu-v100|gpu-l40s"
  "CIFAR10 alexnet 5 24:00:00 gpu-v100|gpu-l40s"
  "CIFAR100 resnet50 1 24:00:00 gpu-v100|gpu-l40s"
  "CIFAR100 resnet50 2 24:00:00 gpu-v100|gpu-l40s"
  "CIFAR100 resnet50 3 24:00:00 gpu-v100|gpu-l40s"
  "CIFAR100 resnet50 4 24:00:00 gpu-v100|gpu-l40s"
  "CIFAR100 resnet50 5 24:00:00 gpu-v100|gpu-l40s"
  "TINYIMAGENET alexnet 1 2-00:00:00 gpu-v100|gpu-l40s"
  "TINYIMAGENET alexnet 2 2-00:00:00 gpu-v100|gpu-l40s"
  "TINYIMAGENET alexnet 3 2-00:00:00 gpu-v100|gpu-l40s"
  "TINYIMAGENET alexnet 4 2-00:00:00 gpu-v100|gpu-l40s"
  "TINYIMAGENET alexnet 5 2-00:00:00 gpu-v100|gpu-l40s"
  "TINYIMAGENET resnet18 1 18:00:00 gpu-l40s"
  "TINYIMAGENET resnet18 2 18:00:00 gpu-l40s"
  "TINYIMAGENET resnet18 3 18:00:00 gpu-l40s"
  "TINYIMAGENET resnet18 4 18:00:00 gpu-l40s"
  "TINYIMAGENET resnet50 1 2-00:00:00 gpu-v100|gpu-l40s"
  "TINYIMAGENET resnet50 2 2-00:00:00 gpu-v100|gpu-l40s"
  "TINYIMAGENET resnet50 3 2-00:00:00 gpu-v100|gpu-l40s"
  "TINYIMAGENET resnet50 4 2-00:00:00 gpu-v100|gpu-l40s"
  "TINYIMAGENET resnet50 5 2-00:00:00 gpu-v100|gpu-l40s"
)

submitted=0
skipped=0

for job in "${JOBS[@]}"; do
  read -r dataset backbone seed walltime constraint <<< "$job"
  out_dir="$PROJECT_ROOT/TypiClust/output/$dataset/$backbone/typiclust_${seed}_${BUDGET}b"

  if [[ -f "$out_dir/benchmark_summary.json" ]]; then
    echo "Skip complete: $out_dir"
    skipped=$((skipped + 1))
    continue
  fi

  if [[ -d "$out_dir" ]] && compgen -G "$out_dir/episode_*" > /dev/null; then
    backup_dir="${out_dir}_partial_${STAMP}"
    echo "Backing up partial run: $out_dir -> $backup_dir"
    mv "$out_dir" "$backup_dir"
  fi

  job_name="tc_${dataset}_${backbone}_s${seed}"
  job_name="${job_name,,}"
  echo "Submitting $job_name (time=$walltime constraint=$constraint)"
  sbatch \
    --job-name="$job_name" \
    --time="$walltime" \
    --constraint="$constraint" \
    --export=ALL,DATASET="$dataset",BACKBONE="$backbone",SEED="$seed",BUDGET="$BUDGET",INITIAL_SIZE="$INITIAL_SIZE" \
    "$SCRIPT_DIR/run_typiclust.sbatch"
  submitted=$((submitted + 1))
done

echo "Done: submitted=$submitted skipped=$skipped. Check with: squeue --me | grep tc_"
