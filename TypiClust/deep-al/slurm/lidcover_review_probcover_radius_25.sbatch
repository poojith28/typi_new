#!/usr/bin/env bash
#SBATCH --job-name=lidrev-pcradius
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/lidrev-pcradius-%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/lidrev-pcradius-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-v100|gpu-l40s
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --array=0-24%15
#SBATCH --mail-type=FAIL

# The 0.25 and 0.60 ProbCover rows already exist. These 25 missing runs make
# CIFAR-100's ProbCover sweep match LIDCover at 0.10..0.60.

set -euo pipefail
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
(( TASK_ID >= 0 && TASK_ID < 25 )) || exit 2
RADII=(0.10 0.20 0.30 0.40 0.50)
RADIUS_INDEX=$((TASK_ID / 5))
SEED=$((TASK_ID % 5 + 1))
RADIUS="${RADII[$RADIUS_INDEX]}"
RADIUS_TAG="${RADIUS/./}"
EXP_NAME="PROBCOVER_RADIUS_d${RADIUS_TAG}_coldstart_review20260818_s${SEED}_50b"

exec bash /vast/s219110279/TypiClust/deep-al/slurm/run_lidcover_review_task.sh \
  CIFAR100 resnet18 probcover "$RADIUS" "$SEED" 1.0 50 50 "$EXP_NAME"
