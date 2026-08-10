#!/bin/bash
#SBATCH --job-name=raw-feat
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/raw-feat-%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/raw-feat-%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-1

# Build raw-pixel baselines for CIFAR10 and CIFAR100 (CPU only, no PCA).
# Array index: 0 -> CIFAR10, 1 -> CIFAR100.

source /vast/s219110279/analysis/slurm/env_setup.sh
mkdir -p /vast/s219110279/TypiClust/output/_logs

RAW_DATASETS=(CIFAR10 CIFAR100)
SCAN_DATASET_DIRS=(cifar-10 cifar-100)

DS=${RAW_DATASETS[$SLURM_ARRAY_TASK_ID]}
SCAN_DS=${SCAN_DATASET_DIRS[$SLURM_ARRAY_TASK_ID]}
DS_DIR=${DS}

OUT_DIR="$OUTPUT_ROOT/$DS_DIR/analysis/raw"
mkdir -p "$OUT_DIR"

echo "================================================================"
echo "  raw extraction : $DS"
echo "  data root      : $RAW_DATA_ROOT/$SCAN_DS"
echo "  output         : $OUT_DIR"
echo "  cpus           : $SLURM_CPUS_PER_TASK"
echo "  start          : $(date)"
echo "================================================================"

cd "$ANALYSIS_DIR"

DATA_ROOT_ARG=()
SCAN_PATH="$RAW_DATA_ROOT/$SCAN_DS"
if [[ "$DS" == "CIFAR10" && -d "$SCAN_PATH/cifar-10-batches-py" ]]; then
    DATA_ROOT_ARG=(--data-root "$SCAN_PATH")
    echo "  using scan raw data: $SCAN_PATH"
elif [[ "$DS" == "CIFAR100" && -d "$SCAN_PATH/cifar-100-python" ]]; then
    DATA_ROOT_ARG=(--data-root "$SCAN_PATH")
    echo "  using scan raw data: $SCAN_PATH"
else
    echo "  scan path not ready ($SCAN_PATH) — auto-detect will fallback to TypiClust/data"
fi

$PY extract_raw_features.py \
    --dataset "$DS" --split train \
    "${DATA_ROOT_ARG[@]}" \
    --out "$OUT_DIR/${DS,,}_train_raw"

echo "done at $(date)"
