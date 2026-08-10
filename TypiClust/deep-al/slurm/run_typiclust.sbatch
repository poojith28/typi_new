#!/bin/bash
#SBATCH --job-name=typiclust_al
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=gpu-v100|gpu-l40s
#SBATCH --time=12:00:00
#SBATCH --qos=batch-short
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s219110279@deakin.edu.au

set -eo pipefail

DATASET="${DATASET:?Set DATASET=CIFAR10|CIFAR100|TINYIMAGENET}"
BACKBONE="${BACKBONE:?Set BACKBONE=alexnet|resnet18|resnet50}"
SEED="${SEED:-1}"
BUDGET="${BUDGET:-50}"
INITIAL_SIZE="${INITIAL_SIZE:-50}"
PROJECT_ROOT=/vast/s219110279

case "$DATASET" in
  CIFAR10) DATASET_CFG_DIR=cifar10 ;;
  CIFAR100) DATASET_CFG_DIR=cifar100 ;;
  TINYIMAGENET) DATASET_CFG_DIR=tinyimagenet ;;
  *) echo "Unsupported DATASET=$DATASET" >&2; exit 2 ;;
esac

case "$BACKBONE" in
  alexnet) MODEL_CFG=ALEXNET.yaml ;;
  resnet18) MODEL_CFG=RESNET18.yaml ;;
  resnet50) MODEL_CFG=RESNET50.yaml ;;
  *) echo "Unsupported BACKBONE=$BACKBONE" >&2; exit 2 ;;
esac

EXP_NAME="typiclust_${SEED}_${BUDGET}b"
OUT_DIR="$PROJECT_ROOT/TypiClust/output/$DATASET/$BACKBONE/$EXP_NAME"
if [[ -f "$OUT_DIR/benchmark_summary.json" ]]; then
  echo "Training already complete, skipping: $OUT_DIR"
  exit 0
fi

module purge
module load Anaconda3
source activate
conda activate typiclust
set -u

export TYPI_FEATURES_ROOT=/vast/s219110279/results/results
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT/TypiClust/deep-al/tools"

nvidia-smi -L
if nvidia-smi -L | grep -Eiq 'Blackwell|RTX PRO 6000'; then
  echo "ERROR: landed on Blackwell node; requeue on V100/L40S if needed." >&2
  exit 2
fi

CFG="$PROJECT_ROOT/TypiClust/deep-al/configs/$DATASET_CFG_DIR/al/$MODEL_CFG"
RUN_CFG="$CFG"
if [[ "$DATASET" == "TINYIMAGENET" ]]; then
  NUM_WORKERS="${NUM_WORKERS:-0}"
  RUN_CFG="/tmp/typiclust_${DATASET}_${BACKBONE}_s${SEED}_${SLURM_JOB_ID:-local}.yaml"
  sed "s/NUM_WORKERS: 4/NUM_WORKERS: ${NUM_WORKERS}/" "$CFG" > "$RUN_CFG"
  echo "TinyImageNet config: NUM_WORKERS=${NUM_WORKERS}"
fi

echo "Running TypiClust: DATASET=$DATASET BACKBONE=$BACKBONE SEED=$SEED EXP_NAME=$EXP_NAME"

python -u train_al.py \
  --cfg="$RUN_CFG" \
  --al=typiclust \
  --exp-name="$EXP_NAME" \
  --budget="$BUDGET" \
  --initial_size="$INITIAL_SIZE" \
  --seed "$SEED"
