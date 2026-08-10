#!/bin/bash
#SBATCH --job-name=umap-raw
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/umap-raw-%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/umap-raw-%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1

# UMAP on full raw-pixel features (3072-d). Requires extract_raw.sbatch first.

source /vast/s219110279/analysis/slurm/env_setup.sh
mkdir -p /vast/s219110279/TypiClust/output/_logs

RAW_DATASETS=(CIFAR10 CIFAR100)
DS=${RAW_DATASETS[$SLURM_ARRAY_TASK_ID]}
LOW="${DS,,}"

RAW_DIR="$OUTPUT_ROOT/$DS/analysis/raw"
FEAT="$RAW_DIR/${LOW}_train_raw.npy"
LABELS="$RAW_DIR/${LOW}_train_raw_labels.npy"
IDS="$RAW_DIR/${LOW}_train_raw_indices.npy"
OUT_DIR="$OUTPUT_ROOT/$DS/analysis/umap"
mkdir -p "$OUT_DIR"

cd "$ANALYSIS_DIR"

$PY compute_umap.py \
    --features "$FEAT" --labels "$LABELS" --indices "$IDS" \
    --dataset "$DS" \
    --out "$OUT_DIR/${LOW}_raw_umap" \
    --n-neighbors 15 --min-dist 0.1 --seed 1

echo "done at $(date)"
