#!/usr/bin/env bash
#SBATCH --job-name=lidrev-signals
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/lidrev-signals-%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/lidrev-signals-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-v100|gpu-l40s
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --array=0-49%15
#SBATCH --mail-type=FAIL

# Equal exponent grid {0.5, 1, 2} under the same CIFAR-100 cold-start protocol.
# Alpha=1 already exists for all five methods; this array supplies alpha=.5,2.

set -euo pipefail
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
(( TASK_ID >= 0 && TASK_ID < 50 )) || exit 2
SAMPLERS=(idprobcover density_cover knn_distance_cover distance_variance_cover distance_cv_cover)
METHOD_INDEX=$((TASK_ID / 10))
WITHIN=$((TASK_ID % 10))
ALPHA_INDEX=$((WITHIN / 5))
SEED=$((WITHIN % 5 + 1))
ALPHAS=(0.5 2.0)
SAMPLER="${SAMPLERS[$METHOD_INDEX]}"
ALPHA="${ALPHAS[$ALPHA_INDEX]}"
case "$SAMPLER" in
  idprobcover) TAG=LIDCOVER ;;
  density_cover) TAG=DENSITY ;;
  knn_distance_cover) TAG=MEANKNN ;;
  distance_variance_cover) TAG=DISTVAR ;;
  distance_cv_cover) TAG=DISTCV ;;
esac
ALPHA_TAG="${ALPHA/./}"
EXP_NAME="${TAG}_SIGNALTUNE_a${ALPHA_TAG}_d025_coldstart_review20260818_s${SEED}_50b"

exec bash /vast/s219110279/TypiClust/deep-al/slurm/run_lidcover_review_task.sh \
  CIFAR100 resnet18 "$SAMPLER" 0.25 "$SEED" "$ALPHA" 50 50 "$EXP_NAME"
