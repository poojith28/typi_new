#!/usr/bin/env bash
#SBATCH --job-name=lidrev-maxherd
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/lidrev-maxherd-%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/lidrev-maxherd-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-v100|gpu-l40s
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --array=0-14%15
#SBATCH --mail-type=FAIL

# Reviewer-requested MaxHerding numbers: 3 datasets x 5 matched seeds.

set -euo pipefail
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
(( TASK_ID >= 0 && TASK_ID < 15 )) || exit 2
DATASETS=(CIFAR10 CIFAR100 TINYIMAGENET)
DATASET_INDEX=$((TASK_ID / 5))
SEED=$((TASK_ID % 5 + 1))
DATASET="${DATASETS[$DATASET_INDEX]}"
EXP_NAME="MAXHERDING_coldstart_review20260818_s${SEED}_50b"

exec bash /vast/s219110279/TypiClust/deep-al/slurm/run_lidcover_review_task.sh \
  "$DATASET" resnet18 maxherding 0.25 "$SEED" 1.0 50 50 "$EXP_NAME"
