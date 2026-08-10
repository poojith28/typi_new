#!/bin/bash
#SBATCH --job-name=tsne-raw
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/tsne-raw-%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/tsne-raw-%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1

# t-SNE on full raw-pixel features (3072-d). Requires extract_raw.sbatch first.
# Array index: 0 -> CIFAR10, 1 -> CIFAR100.
# Submit this AFTER extract_raw.sbatch has finished.

source /vast/s219110279/analysis/slurm/env_setup.sh
mkdir -p /vast/s219110279/TypiClust/output/_logs

RAW_DATASETS=(CIFAR10 CIFAR100)
DS=${RAW_DATASETS[$SLURM_ARRAY_TASK_ID]}
LOW="${DS,,}"

RAW_DIR="$OUTPUT_ROOT/$DS/analysis/raw"
FEAT="$RAW_DIR/${LOW}_train_raw.npy"
LABELS="$RAW_DIR/${LOW}_train_raw_labels.npy"
IDS="$RAW_DIR/${LOW}_train_raw_indices.npy"

OUT_DIR="$OUTPUT_ROOT/$DS/analysis/tsne"
mkdir -p "$OUT_DIR"

echo "raw  : $FEAT"
echo "out  : $OUT_DIR/${LOW}_raw_tsne"

cd "$ANALYSIS_DIR"

$PY compute_tsne.py \
    --features "$FEAT" \
    --labels   "$LABELS" \
    --indices  "$IDS" \
    --dataset  "$DS" \
    --out      "$OUT_DIR/${LOW}_raw_tsne" \
    --backend  openTSNE --pca-dim 0 --perplexity 30 --n-iter 1000 --seed 1

echo "done at $(date)"
