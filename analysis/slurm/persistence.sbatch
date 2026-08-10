#!/bin/bash
#SBATCH --job-name=pers-h0
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/pers-h0-%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/pers-h0-%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-8

# H0 persistence on the FULL embedding pool (no subsampling, no H1).
# Output CSV is used by the per-batch / per-round diag mapping script.

source /vast/s219110279/analysis/slurm/env_setup.sh
mkdir -p /vast/s219110279/TypiClust/output/_logs

decode_combo "$SLURM_ARRAY_TASK_ID"
print_combo

FEAT="$EMBED_ROOT/$BACKBONE/$DATASET/pretext/features_seed1.npy"
LABELS="$EMBED_ROOT/$BACKBONE/$DATASET/pretext/labels_seed1.npy"
OUT_DIR="$OUTPUT_ROOT/$OUTDIR_NAME/analysis/persistence"
mkdir -p "$OUT_DIR"

PER_CLASS_FLAG=()
LABEL_FLAG=()
if [[ -f "$LABELS" ]]; then
    LABEL_FLAG=(--labels "$LABELS")
    PER_CLASS_FLAG=(--per-class)
fi

cd "$ANALYSIS_DIR"

$PY compute_persistence.py \
    --features "$FEAT" \
    "${LABEL_FLAG[@]}" \
    --dataset "$DATASET_TAG_VAL" \
    --out "$OUT_DIR/${BACKBONE}_${OUTDIR_NAME,,}_simclr_h0_full" \
    --max-dim 0 \
    --normalize \
    "${PER_CLASS_FLAG[@]}" \
    --seed 1

echo "done at $(date)"
