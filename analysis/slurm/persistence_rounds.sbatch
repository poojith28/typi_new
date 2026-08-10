#!/bin/bash
#SBATCH --job-name=pers-rounds
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/pers-rounds-%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/pers-rounds-%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-64

# Per-round H0+H1 persistence + TDA-vs-accuracy summary for every
# (strategy, seed) pair of the CIFAR10 resnet18 active-learning runs.
#
# Each array task processes ONE experiment directory and writes:
#   .../analysis/persistence_rounds/<strategy>_<seed>.csv          (long-form)
#   .../analysis/persistence_rounds/<strategy>_<seed>_summary.csv  (one row per round)
#
# To re-target this script for a different dataset / backbone, change
# DATASET, BACKBONE and FEATURES below; the rest auto-derives.
#
# Array layout: index = strategy_idx * len(SEEDS) + seed_idx.

source /vast/s219110279/analysis/slurm/env_setup.sh
mkdir -p /vast/s219110279/TypiClust/output/_logs

DATASET=CIFAR10
DATASET_LOWER=cifar-10
BACKBONE=resnet18

FEATURES="$EMBED_ROOT/$BACKBONE/$DATASET_LOWER/pretext/features_seed1.npy"
EXP_ROOT="$OUTPUT_ROOT/$DATASET/$BACKBONE"
OUT_DIR="$OUTPUT_ROOT/$DATASET/analysis/persistence_rounds"
mkdir -p "$OUT_DIR"

# Strategies that exist under output/CIFAR10/resnet18/<strat>_<seed>_50b.
# Edit / extend as new strategies land.
STRATEGIES=(
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
    LIDCOVER
    LIDCOVER_rerunlowid20260524
)
SEEDS=(1 2 3 4 5)
BUDGET_TAG=50b

n_strats=${#STRATEGIES[@]}
n_seeds=${#SEEDS[@]}
total=$((n_strats * n_seeds))

if (( SLURM_ARRAY_TASK_ID >= total )); then
    echo "Array id $SLURM_ARRAY_TASK_ID is outside the strategy x seed grid (size $total). Skipping."
    exit 0
fi

S_IDX=$((SLURM_ARRAY_TASK_ID / n_seeds))
SD_IDX=$((SLURM_ARRAY_TASK_ID % n_seeds))
STRAT=${STRATEGIES[$S_IDX]}
SEED=${SEEDS[$SD_IDX]}

EXP_DIR="$EXP_ROOT/${STRAT}_${SEED}_${BUDGET_TAG}"
OUT_PREFIX="$OUT_DIR/${STRAT}_${SEED}"

echo "================================================================"
echo "  experiment : $EXP_DIR"
echo "  features   : $FEATURES"
echo "  out        : $OUT_PREFIX"
echo "  cpus       : $SLURM_CPUS_PER_TASK"
echo "  start      : $(date)"
echo "================================================================"

if [[ ! -d "$EXP_DIR" ]]; then
    echo "Experiment dir missing, skipping: $EXP_DIR"
    exit 0
fi

cd "$ANALYSIS_DIR"

# H0+H1 of the labeled set per round, normalized features, capped at 2000 points/round.
$PY compute_persistence_per_round.py \
    --experiment-dir "$EXP_DIR" \
    --features "$FEATURES" \
    --out "$OUT_PREFIX" \
    --scope labeled \
    --max-dim 1 \
    --max-edge-length 1.5 \
    --normalize \
    --max-points 2000 \
    --seed 1

echo "done at $(date)"
