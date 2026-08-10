#!/bin/bash
# Submit TypiClust active-learning runs (all datasets x backbones x seeds).
#
# Usage:
#   cd /vast/s219110279/TypiClust/deep-al/slurm
#   bash submit_typiclust_all.sh
#
# Optional:
#   SEEDS="1 2 3" DATASETS="CIFAR10" BACKBONES="resnet18" bash submit_typiclust_all.sh

set -euo pipefail

DATASETS=(CIFAR10 CIFAR100 TINYIMAGENET)
BACKBONES=(alexnet resnet18 resnet50)
read -r -a SEED_LIST <<< "${SEEDS:-1 2 3 4 5}"

if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
  read -r -a DATASETS <<< "$DATASETS_OVERRIDE"
fi
if [[ -n "${BACKBONES_OVERRIDE:-}" ]]; then
  read -r -a BACKBONES <<< "$BACKBONES_OVERRIDE"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for dataset in "${DATASETS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
      out_dir="/vast/s219110279/TypiClust/output/$dataset/$backbone/typiclust_${seed}_50b"
      if [[ -f "$out_dir/benchmark_summary.json" ]]; then
        echo "Skip complete: $out_dir"
        continue
      fi
      job_name="tc_${dataset}_${backbone}_s${seed}"
      job_name="${job_name,,}"
      echo "Submitting $job_name"
      sbatch \
        --job-name="$job_name" \
        --export=ALL,DATASET="$dataset",BACKBONE="$backbone",SEED="$seed",BUDGET=50,INITIAL_SIZE=50 \
        "$SCRIPT_DIR/run_typiclust.sbatch"
    done
  done
done

echo "Submitted TypiClust jobs. Check with: squeue --me"
