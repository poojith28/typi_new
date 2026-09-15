#!/usr/bin/env bash
#SBATCH --job-name=lidrev-kgrid
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/lidrev-kgrid-%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/lidrev-kgrid-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-v100|gpu-l40s
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --array=0-19%15
#SBATCH --mail-type=FAIL

# Vary k_ID and k_NN one at a time around the existing (50,50) control.

set -euo pipefail
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
(( TASK_ID >= 0 && TASK_ID < 20 )) || exit 2
K_IDS=(20 75 50 50)
K_NNS=(50 50 20 75)
GROUP_INDEX=$((TASK_ID / 5))
SEED=$((TASK_ID % 5 + 1))
K_ID="${K_IDS[$GROUP_INDEX]}"
K_NN="${K_NNS[$GROUP_INDEX]}"
EXP_NAME="LIDCOVER_KID${K_ID}_KNN${K_NN}_d025_coldstart_review20260818_s${SEED}_50b"

exec bash /vast/s219110279/TypiClust/deep-al/slurm/run_lidcover_review_task.sh \
  CIFAR100 resnet18 idprobcover 0.25 "$SEED" 1.0 "$K_ID" "$K_NN" "$EXP_NAME"
