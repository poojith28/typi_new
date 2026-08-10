#!/bin/bash
#SBATCH --job-name=pers-tiny-alex
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/pers-tiny-alex-%j.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/pers-tiny-alex-%j.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00

# TinyImageNet H0 persistence — alexnet (100k pool).
# Uses existing --sparse flag (no script changes): dense VR hits GUDHI Simplex_key limit.

source /vast/s219110279/analysis/slurm/env_setup.sh
mkdir -p /vast/s219110279/TypiClust/output/_logs

BACKBONE=alexnet
DATASET=tiny-imagenet
DATASET_TAG_VAL=TINYIMAGENET
OUTDIR_NAME=TINYIMAGENET

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

echo "================================================================"
echo "  job        : ${SLURM_JOB_NAME:-pers-tiny-alex} (id=${SLURM_JOB_ID:-?})"
echo "  backbone   : $BACKBONE"
echo "  dataset    : $DATASET -> $DATASET_TAG_VAL"
echo "  python     : $PY"
echo "  node       : $(hostname)"
echo "  start      : $(date)"
echo "================================================================"

cd "$ANALYSIS_DIR"

$PY compute_persistence.py \
    --features "$FEAT" \
    "${LABEL_FLAG[@]}" \
    --dataset "$DATASET_TAG_VAL" \
    --out "$OUT_DIR/${BACKBONE}_${OUTDIR_NAME,,}_simclr_h0_full" \
    --max-dim 0 \
    --normalize \
    --sparse 1.0 \
    "${PER_CLASS_FLAG[@]}" \
    --seed 1

echo "done at $(date)"
