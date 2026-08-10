#!/bin/bash
#SBATCH --job-name=h0ref_tiny
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/h0ref_tiny_%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/h0ref_tiny_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=batch-long
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2

# Exact full-pool H0 reference PD for TinyImageNet (N≈100k) via
# Euclidean MST + GUDHI SimplexTree (VR-H0 equivalent).
# Array: 0=alexnet, 1=resnet18, 2=resnet50

set -euo pipefail
source /vast/s219110279/analysis/slurm/env_setup.sh
mkdir -p /vast/s219110279/TypiClust/output/_logs
mkdir -p /vast/s219110279/outputs/diagnostic_h0/reference

BACKBONES=(alexnet resnet18 resnet50)
BACKBONE="${BACKBONES[$SLURM_ARRAY_TASK_ID]}"
DATASET=tiny-imagenet
DATASET_TAG=TINYIMAGENET

FEAT="$EMBED_ROOT/$BACKBONE/$DATASET/pretext/features_seed1.npy"
OUT_DIR="$OUTPUT_ROOT/$DATASET_TAG/analysis/persistence"
OUT_PREFIX="$OUT_DIR/${BACKBONE}_tinyimagenet_simclr_h0_full"
mkdir -p "$OUT_DIR"

echo "================================================================"
echo "  Exact H0 reference (MST+GUDHI)"
echo "  backbone=$BACKBONE  features=$FEAT"
echo "  out=$OUT_PREFIX"
echo "  cpus=$SLURM_CPUS_PER_TASK  node=$(hostname)  start=$(date)"
echo "================================================================"

cd "$ANALYSIS_DIR"
$PY compute_h0_reference_mst.py \
    --features "$FEAT" \
    --out "$OUT_PREFIX" \
    --dataset "$DATASET_TAG" \
    --backbone "$BACKBONE" \
    --normalize

# Also copy into diagnostic_h0/reference for the chapter pipeline
REF_DIR=/vast/s219110279/outputs/diagnostic_h0/reference
cp -f "${OUT_PREFIX}.csv" "$REF_DIR/${DATASET_TAG}_${BACKBONE}_h0_reference.csv"
cp -f "${OUT_PREFIX}_meta.json" "$REF_DIR/${DATASET_TAG}_${BACKBONE}_h0_reference_meta.json"
cp -f "${OUT_PREFIX}_with_infinite.csv" "$REF_DIR/${DATASET_TAG}_${BACKBONE}_h0_reference_with_infinite.csv"

echo "done at $(date)"
