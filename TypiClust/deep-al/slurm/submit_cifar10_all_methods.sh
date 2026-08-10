#!/bin/bash
# Submit one Slurm job per AL method (each job runs seeds 1-5), for a given backbone.
#
# Usage (resnet18 is already complete — do not resubmit):
#   BACKBONE=resnet50 bash submit_cifar10_all_methods.sh
#   BACKBONE=alexnet  bash submit_cifar10_all_methods.sh
#   bash submit_cifar10_resnet50_alexnet.sh   # both backbones

set -euo pipefail

BACKBONE="${BACKBONE:?Set BACKBONE=resnet50 or alexnet}"

if [[ "$BACKBONE" == "resnet18" ]]; then
  echo "resnet18 CIFAR10 runs are already done. Use BACKBONE=resnet50 or alexnet." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

METHODS=(
  random
  uncertainty
  entropy
  margin
  coreset
  dbal
  probcover
  knn_distance_cover
  density_cover
  distance_variance_cover
  distance_cv_cover
  idprobcover
)

for method in "${METHODS[@]}"; do
  job_label="c10_${method}_${BACKBONE}"
  job_label="${job_label//_/-}"
  echo "Submitting $job_label (METHOD=$method BACKBONE=$BACKBONE)"
  sbatch \
    --job-name="$job_label" \
    --export=ALL,BACKBONE="$BACKBONE",METHOD="$method" \
    cifar10_method_5seeds.sbatch
done

echo "Queued ${#METHODS[@]} jobs for backbone=$BACKBONE. Check: squeue -u \$USER"
