#!/bin/bash
#SBATCH --job-name=typi-al
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/al-%x-%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/al-%x-%A_%a.err
#SBATCH --partition=gpu-large
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --array=0-59%4

# Active-learning benchmark for one backbone x dataset combo.
#
# Submit examples:
#   BACKBONE=resnet50 DATASET=cifar10 sbatch --job-name=c10-r50 run_al_suite.sbatch
#   BACKBONE=alexnet  DATASET=cifar10 sbatch --job-name=c10-alex run_al_suite.sbatch
#   BACKBONE=resnet50 DATASET=cifar100 sbatch --array=0-59%2 run_al_suite.sbatch
#
# Array layout: task = seed_index * num_methods + method_index
#   12 methods x 5 seeds = 60 tasks (indices 0-59).

set -euo pipefail

BACKBONE="${BACKBONE:?Set BACKBONE (resnet50 or alexnet) before sbatch}"
DATASET="${DATASET:-cifar10}"

ROOT="/vast/s219110279/TypiClust/deep-al"
TRAIN_SCRIPT="$ROOT/tools/train_al.py"
mkdir -p /vast/s219110279/TypiClust/output/_logs

case "$DATASET" in
  cifar10)
    CFG="$ROOT/configs/cifar10/al/$(echo "$BACKBONE" | tr '[:lower:]' '[:upper:]').yaml"
    DS_TAG=CIFAR10
    INITIAL_DELTA="${INITIAL_DELTA:-0.25}"
    ;;
  cifar100)
    CFG="$ROOT/configs/cifar100/al/$(echo "$BACKBONE" | tr '[:lower:]' '[:upper:]').yaml"
    DS_TAG=CIFAR100
    INITIAL_DELTA="${INITIAL_DELTA:-0.25}"
    ;;
  tinyimagenet)
    CFG="$ROOT/configs/tinyimagenet/al/$(echo "$BACKBONE" | tr '[:lower:]' '[:upper:]').yaml"
    DS_TAG=TINYIMAGENET
    INITIAL_DELTA="${INITIAL_DELTA:-0.25}"
    ;;
  *)
    echo "Unknown DATASET=$DATASET" >&2
    exit 1
    ;;
esac

OUT_ROOT="/vast/s219110279/TypiClust/output/${DS_TAG}/${BACKBONE}"
IDPC_CACHE_ROOT="/vast/s219110279/idpc_cache/${BACKBONE}"

export TYPI_FEATURES_ROOT="/vast/s219110279/results/results"
export TYPI_FEATURE_BACKBONE="$BACKBONE"

SEEDS=(1 2 3 4 5)
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

BUDGET="${BUDGET:-50}"
INITIAL_SIZE="${INITIAL_SIZE:-50}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
NUM_METHODS="${#METHODS[@]}"
SEED_INDEX="$((TASK_ID / NUM_METHODS))"
METHOD_INDEX="$((TASK_ID % NUM_METHODS))"

seed="${SEEDS[$SEED_INDEX]}"
method="${METHODS[$METHOD_INDEX]}"
exp_method="$method"
if [[ "$method" == "idprobcover" ]]; then
  exp_method="LIDCOVER"
fi

if [[ "$DATASET" == "cifar10" ]]; then
  exp_name="${exp_method}_${seed}_${BUDGET}b"
else
  exp_name="${DATASET}_${exp_method}_s${seed}"
fi

summary_path="$OUT_ROOT/$exp_name/benchmark_summary.json"
if [[ -f "$summary_path" ]]; then
  echo "Skipping completed run: $exp_name"
  echo "Found: $summary_path"
  exit 0
fi

module purge
module load Anaconda3
source activate
conda activate typiclust

cd "$ROOT"

cmd=(
  python "$TRAIN_SCRIPT"
  --cfg "$CFG"
  --exp-name "$exp_name"
  --al "$method"
  --budget "$BUDGET"
  --initial_size "$INITIAL_SIZE"
  --seed "$seed"
)

if [[ "$method" == "probcover" || "$method" == "idprobcover" || "$method" == "knn_distance_cover" || "$method" == "density_cover" || "$method" == "distance_variance_cover" || "$method" == "distance_cv_cover" ]]; then
  cmd+=(--initial_delta "$INITIAL_DELTA")
fi

if [[ "$method" == "idprobcover" ]]; then
  cmd+=(
    --idpc_alpha 1.0
    --idpc_mode high_id_more_centers
    --idpc_k_id 50
    --idpc_k_knn 50
    --idpc_cache_root "$IDPC_CACHE_ROOT"
  )
fi

if [[ "$method" == "knn_distance_cover" || "$method" == "density_cover" || "$method" == "distance_variance_cover" || "$method" == "distance_cv_cover" ]]; then
  cmd+=(
    --arc_alpha 1.0
    --arc_k_signal 50
    --arc_k_knn 50
    --arc_cache_root "$IDPC_CACHE_ROOT/adaptive_cover"
  )
fi

echo "backbone=$BACKBONE dataset=$DATASET exp=$exp_name"
echo "features=$TYPI_FEATURES_ROOT/$BACKBONE/..."
echo "Command: ${cmd[*]}"
"${cmd[@]}"
