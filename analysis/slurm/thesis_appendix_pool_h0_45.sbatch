#!/bin/bash
#SBATCH --job-name=thesis_pool_h0
#SBATCH --output=/vast/s219110279/outputs/thesis_appendix/pool_reference_robustness/reference/logs/slurm/h0_%A_%a.out
#SBATCH --error=/vast/s219110279/outputs/thesis_appendix/pool_reference_robustness/reference/logs/slurm/h0_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=batch-long
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-44%6

# Exact pool-only H0: 3 backbones x 3 datasets x 5 seed-specific pools.
set -euo pipefail
source /vast/s219110279/analysis/slurm/env_setup.sh

ROOT=/vast/s219110279
REF_ROOT="$ROOT/outputs/thesis_appendix/pool_reference_robustness/reference"
BACKBONES=(resnet18 resnet50 alexnet)
DATASET_KEYS=(CIFAR10 CIFAR100 TINYIMAGENET)
FEATURE_DIRS=(cifar-10 cifar-100 tiny-imagenet)

COMBO=$((SLURM_ARRAY_TASK_ID / 5))
SEED=$((SLURM_ARRAY_TASK_ID % 5 + 1))
BACKBONE="${BACKBONES[$((COMBO / 3))]}"
DATASET="${DATASET_KEYS[$((COMBO % 3))]}"
FEATURE_DIR="${FEATURE_DIRS[$((COMBO % 3))]}"
FEATURES="$ROOT/results/results/$BACKBONE/$FEATURE_DIR/pretext/features_seed1.npy"
ELIGIBLE="$REF_ROOT/eligible_indices/${DATASET}_${BACKBONE}_seed${SEED}_eligible.npy"
OUT="$REF_ROOT/h0/${DATASET}_${BACKBONE}_seed${SEED}_pool_only_h0_reference"

if [[ -s "${OUT}.csv" && -s "${OUT}_meta.json" ]]; then
    echo "SKIP complete: ${OUT}"
    exit 0
fi

mkdir -p "$REF_ROOT/h0"
cd "$ROOT"
echo "START dataset=$DATASET backbone=$BACKBONE seed=$SEED node=$(hostname) date=$(date)"
"$PY" analysis/compute_pool_h0_reference_mst.py \
    --features "$FEATURES" \
    --eligible-indices "$ELIGIBLE" \
    --out "$OUT" \
    --dataset "$DATASET" \
    --backbone "$BACKBONE" \
    --seed "$SEED"
echo "DONE date=$(date)"
