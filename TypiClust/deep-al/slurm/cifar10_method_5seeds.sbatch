#!/bin/bash
#SBATCH --job-name=c10_al
#SBATCH --output=c10_%x_%j.out
#SBATCH --error=c10_%x_%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --mail-type=END
#SBATCH --mail-user=s219110279@deakin.edu.au

# One Slurm job = one (backbone, method) pair, all 5 seeds run back-to-back.
#
# Submit examples (from this directory):
#   sbatch --job-name=c10_random_r50  --export=ALL,BACKBONE=resnet50,METHOD=random cifar10_method_5seeds.sbatch
#   sbatch --job-name=c10_random_alex --export=ALL,BACKBONE=alexnet,METHOD=random  cifar10_method_5seeds.sbatch
#   sbatch --job-name=c10_lid_r50     --export=ALL,BACKBONE=resnet50,METHOD=idprobcover cifar10_method_5seeds.sbatch
#
# Or submit the full suite:
#   BACKBONE=resnet50 bash submit_cifar10_all_methods.sh

set -euo pipefail

BACKBONE="${BACKBONE:?Set BACKBONE=resnet50 or alexnet}"

if [[ "$BACKBONE" == "resnet18" ]]; then
  echo "resnet18 is already complete. Use BACKBONE=resnet50 or alexnet." >&2
  exit 1
fi
METHOD="${METHOD:?Set METHOD=random|uncertainty|...|idprobcover}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

cd "$TOOLS"

nvidia-smi -L
nvidia-smi -q -d ECC

BUDGET="${BUDGET:-50}"
exp_method="$METHOD"
if [[ "$METHOD" == "idprobcover" ]]; then
  exp_method="LIDCOVER"
fi

for seed in 1 2 3 4 5; do
  exp_name="${exp_method}_${seed}_${BUDGET}b"
  out_dir="$PROJECT_ROOT/TypiClust/output/CIFAR10/$BACKBONE/$exp_name"
  if [[ -f "$out_dir/benchmark_summary.json" ]]; then
    echo "Skip (done): $exp_name"
    continue
  fi
  run_train_al "$BACKBONE" "$METHOD" "$seed" "$exp_name"
done

echo "Finished backbone=$BACKBONE method=$METHOD at $(date)"
