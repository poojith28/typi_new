#!/bin/bash
#SBATCH --job-name=thesis_id_pool
#SBATCH --output=/vast/s219110279/outputs/thesis_appendix/pool_reference_robustness/reference/logs/slurm/id_pool_%A_%a.out
#SBATCH --error=/vast/s219110279/outputs/thesis_appendix/pool_reference_robustness/reference/logs/slurm/id_pool_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=batch-long
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-44%6

# Exact seed-specific pool-only local-ID references.
set -euo pipefail
ROOT=/vast/s219110279
module purge
module load Anaconda3
source activate
PY="$ROOT/.venv_diag/bin/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

BACKBONES=(resnet18 resnet50 alexnet)
DATASET_KEYS=(CIFAR10 CIFAR100 TINYIMAGENET)
FEATURE_DIRS=(cifar-10 cifar-100 tiny-imagenet)
COMBO=$((SLURM_ARRAY_TASK_ID / 5))
SEED=$((SLURM_ARRAY_TASK_ID % 5 + 1))
BACKBONE="${BACKBONES[$((COMBO / 3))]}"
DATASET="${DATASET_KEYS[$((COMBO % 3))]}"
FEATURE_DIR="${FEATURE_DIRS[$((COMBO % 3))]}"
FEATURES="$ROOT/results/results/$BACKBONE/$FEATURE_DIR/pretext/features_seed1.npy"
REF_ROOT="$ROOT/outputs/thesis_appendix/pool_reference_robustness/reference"
ELIGIBLE="$REF_ROOT/eligible_indices/${DATASET}_${BACKBONE}_seed${SEED}_eligible.npy"
OUT="$REF_ROOT/id/pool/${DATASET}_${BACKBONE}_seed${SEED}_pool_only_local_id_k50.npy"

if [[ -s "$OUT" && -s "${OUT%.npy}.json" ]]; then
    echo "SKIP complete: $OUT"
    exit 0
fi
mkdir -p "$(dirname "$OUT")"
cd "$ROOT"
"$PY" analysis/compute_pool_id_reference.py \
    --features "$FEATURES" --eligible-indices "$ELIGIBLE" --out "$OUT" \
    --dataset "$DATASET" --backbone "$BACKBONE" --seed "$SEED" \
    --k 50 --threads "$SLURM_CPUS_PER_TASK"
