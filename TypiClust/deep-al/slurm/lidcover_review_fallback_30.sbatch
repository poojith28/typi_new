#!/usr/bin/env bash
#SBATCH --job-name=lidrev-fallback
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/lidrev-fallback-%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/lidrev-fallback-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-v100|gpu-l40s
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --array=0-29%15
#SBATCH --mail-type=FAIL

# 3 datasets x 2 fallback rules x 5 seeds. The control preserves historical
# min-LID gain ties and fallback; the ablation changes only zero-gain fallback.

set -euo pipefail
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
(( TASK_ID >= 0 && TASK_ID < 30 )) || exit 2

DATASETS=(CIFAR10 CIFAR100 TINYIMAGENET)
SAMPLERS=(idprobcover_tiebreak_min_id idprobcover_fallback_random)
DATASET_INDEX=$((TASK_ID / 10))
WITHIN=$((TASK_ID % 10))
METHOD_INDEX=$((WITHIN / 5))
SEED=$((WITHIN % 5 + 1))
DATASET="${DATASETS[$DATASET_INDEX]}"
SAMPLER="${SAMPLERS[$METHOD_INDEX]}"
if [[ "$SAMPLER" == idprobcover_fallback_random ]]; then TAG=RANDOM; else TAG=MINID_CONTROL; fi
EXP_NAME="LIDCOVER_FALLBACK_${TAG}_d025_coldstart_review20260818_s${SEED}_50b"

exec bash /vast/s219110279/TypiClust/deep-al/slurm/run_lidcover_review_task.sh \
  "$DATASET" resnet18 "$SAMPLER" 0.25 "$SEED" 1.0 50 50 "$EXP_NAME"
