#!/bin/bash
#SBATCH --job-name=umap
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/umap-%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/umap-%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-8

# UMAP 2D projection for every (backbone, dataset) combination of the
# SCAN/SimCLR embeddings. Array indexing identical to tsne.sbatch.

source /vast/s219110279/analysis/slurm/env_setup.sh
mkdir -p /vast/s219110279/TypiClust/output/_logs

decode_combo "$SLURM_ARRAY_TASK_ID"
print_combo

FEAT="$EMBED_ROOT/$BACKBONE/$DATASET/pretext/features_seed1.npy"
LABELS="$EMBED_ROOT/$BACKBONE/$DATASET/pretext/labels_seed1.npy"
OUT_DIR="$OUTPUT_ROOT/$OUTDIR_NAME/analysis/umap"
mkdir -p "$OUT_DIR"

LABEL_FLAG=()
if [[ -f "$LABELS" ]]; then
    LABEL_FLAG=(--labels "$LABELS")
fi

cd "$ANALYSIS_DIR"

# Main UMAP run (standard params).
$PY compute_umap.py \
    --features "$FEAT" \
    "${LABEL_FLAG[@]}" \
    --dataset "$DATASET_TAG_VAL" \
    --out "$OUT_DIR/${BACKBONE}_${OUTDIR_NAME,,}_simclr_umap" \
    --n-neighbors 15 --min-dist 0.1 \
    --normalize --seed 1

# Optional densMAP variant (better for showing density). Comment out if not needed.
$PY compute_umap.py \
    --features "$FEAT" \
    "${LABEL_FLAG[@]}" \
    --dataset "$DATASET_TAG_VAL" \
    --out "$OUT_DIR/${BACKBONE}_${OUTDIR_NAME,,}_simclr_densmap" \
    --n-neighbors 15 --min-dist 0.1 \
    --densmap \
    --normalize --seed 1

echo "done at $(date)"
