#!/bin/bash
#SBATCH --job-name=h0ref_c10
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/h0ref_cifar10_%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/h0ref_cifar10_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=batch-long
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2

# Full-pool H0 reference PD — CIFAR-10 × {alexnet, resnet18, resnet50}
# Exact Euclidean MST + GUDHI SimplexTree (VR-H0 equivalent).
# Validates death multiset against existing dense GUDHI CSV when present.
# Array: 0=alexnet, 1=resnet18, 2=resnet50

set -euo pipefail
source /vast/s219110279/analysis/slurm/env_setup.sh
mkdir -p /vast/s219110279/TypiClust/output/_logs
mkdir -p /vast/s219110279/outputs/diagnostic_h0/reference

BACKBONES=(alexnet resnet18 resnet50)
BACKBONE="${BACKBONES[$SLURM_ARRAY_TASK_ID]}"
DATASET=cifar-10
DATASET_TAG=CIFAR10
TAG_LOWER=cifar10

FEAT="$EMBED_ROOT/$BACKBONE/$DATASET/pretext/features_seed1.npy"
OUT_DIR="$OUTPUT_ROOT/$DATASET_TAG/analysis/persistence"
OUT_PREFIX="$OUT_DIR/${BACKBONE}_${TAG_LOWER}_simclr_h0_full"
EXISTING_GUDHI="${OUT_PREFIX}.csv"
mkdir -p "$OUT_DIR"

# Write MST outputs beside dense GUDHI with an explicit suffix, then promote
MST_PREFIX="${OUT_PREFIX}_mst"
VALIDATE=()
if [[ -f "$EXISTING_GUDHI" ]]; then
  VALIDATE=(--validate-against "$EXISTING_GUDHI")
fi

echo "================================================================"
echo "  Exact H0 reference (MST+GUDHI) — CIFAR-10"
echo "  backbone=$BACKBONE  features=$FEAT"
echo "  out=$MST_PREFIX"
echo "  cpus=$SLURM_CPUS_PER_TASK  node=$(hostname)  start=$(date)"
echo "================================================================"

cd "$ANALYSIS_DIR"
$PY compute_h0_reference_mst.py \
    --features "$FEAT" \
    --out "$MST_PREFIX" \
    --dataset "$DATASET_TAG" \
    --backbone "$BACKBONE" \
    --normalize \
    "${VALIDATE[@]}"

# Stage chapter reference from MST (exact H0; consistent with TinyImageNet path)
REF_DIR=/vast/s219110279/outputs/diagnostic_h0/reference
cp -f "${MST_PREFIX}.csv" "$REF_DIR/${DATASET_TAG}_${BACKBONE}_h0_reference.csv"
cp -f "${MST_PREFIX}_meta.json" "$REF_DIR/${DATASET_TAG}_${BACKBONE}_h0_reference_meta.json"
cp -f "${MST_PREFIX}_with_infinite.csv" "$REF_DIR/${DATASET_TAG}_${BACKBONE}_h0_reference_with_infinite.csv"

echo "done at $(date)"
