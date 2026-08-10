#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/s219110279/TypiClust/deep-al"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="$ROOT/tools/train_al.py"
CFG="$ROOT/configs/tinyimagenet/al/RESNET18.yaml"

SEEDS=(1 2 3 4 5)
METHODS=(random uncertainty entropy margin coreset dbal probcover knn_distance_cover density_cover distance_variance_cover distance_cv_cover idprobcover)

BUDGET="${BUDGET:-50}"
INITIAL_SIZE="${INITIAL_SIZE:-50}"
INITIAL_DELTA="${INITIAL_DELTA:-0.30}"
IDPC_CACHE_ROOT="${IDPC_CACHE_ROOT:-/scratch/s219110279/idpc_cache}"

cd "$ROOT"

for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
    exp_name="tinyimagenet_${method}_s${seed}"
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

    echo "Running: ${exp_name}"
    "${cmd[@]}"
  done
done
