#!/bin/bash
#SBATCH --job-name=thesis_id_full
#SBATCH --output=/vast/s219110279/outputs/thesis_appendix/pool_reference_robustness/reference/logs/slurm/id_full_%A_%a.out
#SBATCH --error=/vast/s219110279/outputs/thesis_appendix/pool_reference_robustness/reference/logs/slurm/id_full_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=batch-long
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-8%6

# Exact historical full-train local-ID references: 3 backbones x 3 datasets.
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
BACKBONE="${BACKBONES[$((SLURM_ARRAY_TASK_ID / 3))]}"
DATASET="${DATASET_KEYS[$((SLURM_ARRAY_TASK_ID % 3))]}"
FEATURE_DIR="${FEATURE_DIRS[$((SLURM_ARRAY_TASK_ID % 3))]}"
FEATURES="$ROOT/results/results/$BACKBONE/$FEATURE_DIR/pretext/features_seed1.npy"
OUT="$ROOT/outputs/thesis_appendix/pool_reference_robustness/reference/id/full/${DATASET}_${BACKBONE}_full_train_local_id_k50.npy"

if [[ -s "$OUT" && -s "${OUT%.npy}.json" ]]; then
    echo "SKIP complete: $OUT"
    exit 0
fi
mkdir -p "$(dirname "$OUT")"
cd "$ROOT"
"$PY" analysis/compute_pool_id_reference.py \
    --features "$FEATURES" --out "$OUT" --dataset "$DATASET" \
    --backbone "$BACKBONE" --k 50 --threads "$SLURM_CPUS_PER_TASK"
