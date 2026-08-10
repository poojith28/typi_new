#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/s219110279/TypiClust/deep-al"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="$ROOT/tools/train_al.py"
CFG="$ROOT/configs/cifar100/al/RESNET18.yaml"

SEEDS=(1 2 3 4 5)
BUDGET="${BUDGET:-50}"
INITIAL_SIZE="${INITIAL_SIZE:-50}"
INITIAL_DELTA="${INITIAL_DELTA:-0.25}"
ARC_CACHE_ROOT="${ARC_CACHE_ROOT:-/scratch/s219110279/idpc_cache/adaptive_cover}"

cd "$ROOT"

for seed in "${SEEDS[@]}"; do
  exp_name="cifar100_density_cover_s${seed}"
  echo "Running: ${exp_name}"
  "$PYTHON_BIN" "$TRAIN_SCRIPT" \
    --cfg "$CFG" \
    --exp-name "$exp_name" \
    --al density_cover \
    --budget "$BUDGET" \
    --initial_size "$INITIAL_SIZE" \
    --seed "$seed" \
    --initial_delta "$INITIAL_DELTA" \
    --arc_alpha 1.0 \
    --arc_k_signal 50 \
    --arc_k_knn 50 \
    --arc_cache_root "$ARC_CACHE_ROOT"
done
