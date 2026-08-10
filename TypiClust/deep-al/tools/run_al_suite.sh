#!/usr/bin/env bash
# Run the full CIFAR10 active-learning benchmark for one backbone.
#
# Usage:
#   BACKBONE=resnet50 bash run_al_suite.sh
#   BACKBONE=alexnet  DATASET=cifar100 bash run_al_suite.sh
#
# Environment overrides:
#   ROOT, PYTHON_BIN, BUDGET, INITIAL_SIZE, INITIAL_DELTA, IDPC_CACHE_ROOT, OUT_ROOT

set -euo pipefail

BACKBONE="${BACKBONE:?Set BACKBONE to resnet18, resnet50, or alexnet}"
DATASET="${DATASET:-cifar10}"

ROOT="${ROOT:-/vast/s219110279/TypiClust/deep-al}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="$ROOT/tools/train_al.py"

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
    echo "Unknown DATASET=$DATASET (use cifar10, cifar100, or tinyimagenet)" >&2
    exit 1
    ;;
esac

if [[ ! -f "$CFG" ]]; then
  echo "Config not found: $CFG" >&2
  exit 1
fi

OUT_ROOT="${OUT_ROOT:-/vast/s219110279/TypiClust/output/${DS_TAG}/${BACKBONE}}"
IDPC_CACHE_ROOT="${IDPC_CACHE_ROOT:-/vast/s219110279/idpc_cache/${BACKBONE}}"
mkdir -p "$OUT_ROOT" "$IDPC_CACHE_ROOT"

export TYPI_FEATURES_ROOT="${TYPI_FEATURES_ROOT:-/vast/s219110279/results/results}"
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

cd "$ROOT"

for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
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
      continue
    fi

    cmd=(
      "$PYTHON_BIN" "$TRAIN_SCRIPT"
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

    echo "Running: ${exp_name} (backbone=${BACKBONE}, dataset=${DATASET})"
    "${cmd[@]}"
  done
done
