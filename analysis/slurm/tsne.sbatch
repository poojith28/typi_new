#!/bin/bash
#SBATCH --job-name=tsne
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/tsne-%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/tsne-%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-8

# Compute openTSNE 2D projection on full SimCLR embeddings (512-d, no PCA).
# Array indexing (env_setup.sh): 0..2 alexnet, 3..5 resnet18, 6..8 resnet50
#   × cifar-10, cifar-100, tiny-imagenet.
#   0..2 -> alexnet  (cifar-10, cifar-100, tiny-imagenet)
#   3..5 -> resnet18 (cifar-10, cifar-100, tiny-imagenet)
#   6..8 -> resnet50 (cifar-10, cifar-100, tiny-imagenet)

source /vast/s219110279/analysis/slurm/env_setup.sh
mkdir -p /vast/s219110279/TypiClust/output/_logs

decode_combo "$SLURM_ARRAY_TASK_ID"
print_combo

FEAT="$EMBED_ROOT/$BACKBONE/$DATASET/pretext/features_seed1.npy"
LABELS="$EMBED_ROOT/$BACKBONE/$DATASET/pretext/labels_seed1.npy"
OUT_DIR="$OUTPUT_ROOT/$OUTDIR_NAME/analysis/tsne"
mkdir -p "$OUT_DIR"

LABEL_FLAG=()
if [[ -f "$LABELS" ]]; then
    LABEL_FLAG=(--labels "$LABELS")
fi

cd "$ANALYSIS_DIR"

$PY compute_tsne.py \
    --features "$FEAT" \
    "${LABEL_FLAG[@]}" \
    --dataset "$DATASET_TAG_VAL" \
    --out "$OUT_DIR/${BACKBONE}_${OUTDIR_NAME,,}_simclr_tsne" \
    --backend openTSNE \
    --pca-dim 0 \
    --perplexity 30 \
    --n-iter 1000 \
    --normalize \
    --seed 1

echo "done at $(date)"
