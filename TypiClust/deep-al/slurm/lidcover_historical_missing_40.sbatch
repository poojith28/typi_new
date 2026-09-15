#!/usr/bin/env bash
#SBATCH --job-name=lidcover-hist
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/lidcover-hist-%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/lidcover-hist-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-v100|gpu-l40s
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --array=0-39%15
#SBATCH --mail-type=FAIL

# Complete the 40 concrete historical matched-radius jobs listed in
# evidence_pack/lidcover_missing_jobs.csv. The array cap (%15) permits at most
# 15 experiment tasks to be RUNNING simultaneously.
#
# Layout: eight configuration blocks x five seeds. Each array element runs one
# dataset/backbone/method/radius/seed experiment.

set -euo pipefail

PROJECT_ROOT=/vast/s219110279
DEEP_AL="$PROJECT_ROOT/TypiClust/deep-al"
TRAIN_SCRIPT="$DEEP_AL/tools/train_al.py"
OUTPUT_ROOT="$PROJECT_ROOT/TypiClust/output"
LOG_ROOT="$OUTPUT_ROOT/_logs"
LOCK_ROOT="$OUTPUT_ROOT/_submission_locks/lidcover-historical"

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( TASK_ID < 0 || TASK_ID > 39 )); then
  echo "Invalid array task $TASK_ID; expected 0..39" >&2
  exit 2
fi

# These eight blocks are the exact groups in lidcover_missing_jobs.csv.
DATASETS=(
  CIFAR100 CIFAR100 CIFAR100 CIFAR100
  CIFAR100 CIFAR100 CIFAR10 TINYIMAGENET
)
BACKBONES=(
  alexnet alexnet resnet18 resnet18
  resnet50 resnet50 resnet18 resnet18
)
SAMPLERS=(
  idprobcover probcover idprobcover idprobcover
  idprobcover probcover idprobcover idprobcover
)
BASE_RADII=(
  0.60 0.25 0.25 0.60
  0.60 0.25 0.60 0.60
)

GROUP_INDEX=$((TASK_ID / 5))
SEED=$((TASK_ID % 5 + 1))
DATASET="${DATASETS[$GROUP_INDEX]}"
BACKBONE="${BACKBONES[$GROUP_INDEX]}"
SAMPLER="${SAMPLERS[$GROUP_INDEX]}"
BASE_RADIUS="${BASE_RADII[$GROUP_INDEX]}"

case "$DATASET" in
  CIFAR10) DATASET_CFG=cifar10 ;;
  CIFAR100) DATASET_CFG=cifar100 ;;
  TINYIMAGENET) DATASET_CFG=tinyimagenet ;;
  *) echo "Unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

case "$BACKBONE" in
  alexnet) MODEL_CFG=ALEXNET.yaml ;;
  resnet18) MODEL_CFG=RESNET18.yaml ;;
  resnet50) MODEL_CFG=RESNET50.yaml ;;
  *) echo "Unsupported backbone: $BACKBONE" >&2; exit 2 ;;
esac

RADIUS_TAG="${BASE_RADIUS/./}"
if [[ "$SAMPLER" == idprobcover ]]; then
  METHOD_TAG=LIDCOVER
else
  METHOD_TAG=PROBCOVER
fi

# New names are intentional: historical outputs are immutable and must never
# be reused or overwritten by these canonical cold-start runs.
EXP_NAME="${METHOD_TAG}_canonical_d${RADIUS_TAG}_coldstart_20260810_${SEED}_50b"
OUT_DIR="$OUTPUT_ROOT/$DATASET/$BACKBONE/$EXP_NAME"
CFG="$DEEP_AL/configs/$DATASET_CFG/al/$MODEL_CFG"
IDPC_CACHE_ROOT="$PROJECT_ROOT/idpc_cache/$BACKBONE"

if [[ ! -f "$CFG" ]]; then
  echo "Missing configuration: $CFG" >&2
  exit 2
fi

print_run() {
  echo "task_id=$TASK_ID group=$GROUP_INDEX seed=$SEED"
  echo "dataset=$DATASET backbone=$BACKBONE sampler=$SAMPLER base_radius=$BASE_RADIUS"
  echo "exp_name=$EXP_NAME"
  echo "output=$OUT_DIR"
}

# Local preflight example:
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=0 bash lidcover_historical_missing_40.sbatch
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  print_run
  echo "protocol: budget=50 initial_size=0 max_iter=100 expected_final_budget=5050"
  exit 0
fi

mkdir -p "$LOG_ROOT" "$LOCK_ROOT"

module purge
module load Anaconda3
# Conda activation hooks may read unset variables (notably
# MKL_INTERFACE_LAYER). Temporarily disable nounset, then restore it before
# validation and training.
set +u
source activate
conda activate typiclust
set -u

export PYTHONUNBUFFERED=1
export TYPI_FEATURES_ROOT="$PROJECT_ROOT/results/results"
export TYPI_FEATURE_BACKBONE="$BACKBONE"

# An automatically released file lock prevents the same experiment from being
# executed by overlapping submissions. A completed result is validated below;
# a partial directory is never reused.
LOCK_FILE="$LOCK_ROOT/${DATASET}_${BACKBONE}_${EXP_NAME}.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another job owns the experiment lock: $LOCK_FILE"
  exit 0
fi

validate_complete() {
  python - "$OUT_DIR" "$DATASET" "$BACKBONE" "$SAMPLER" "$SEED" <<'PY'
import json
import math
import sys
from pathlib import Path

import numpy as np

out_dir = Path(sys.argv[1])
dataset, backbone, sampler = sys.argv[2:5]
seed = int(sys.argv[5])

summary_path = out_dir / "benchmark_summary.json"
root_lset = out_dir / "lSet.npy"
final_lset = out_dir / "episode_100" / "lSet.npy"
if not all(path.is_file() for path in (summary_path, root_lset, final_lset)):
    raise SystemExit(1)

try:
    summary = json.loads(summary_path.read_text())
    valid = (
        summary.get("dataset") == dataset
        and str(summary.get("model", "")).lower() == backbone
        and str(summary.get("sampling_fn", "")).lower() == sampler
        and int(summary.get("seed", -1)) == seed
        and int(summary.get("budget_per_round", -1)) == 50
        and int(summary.get("num_rounds_completed", -1)) == 101
        and len(summary.get("episode_records", [])) == 101
        # Data.makeLUVSets stores the empty root partition as an object array.
        and np.load(root_lset, allow_pickle=True).size == 0
        and np.load(final_lset, allow_pickle=True).size == 5050
        and math.isfinite(float(summary.get("final_test_accuracy")))
        and math.isfinite(float(summary.get("test_auc")))
    )
except Exception:
    valid = False

raise SystemExit(0 if valid else 1)
PY
}

if [[ -e "$OUT_DIR" ]]; then
  if validate_complete; then
    echo "Validated COMPLETE; skipping existing canonical run: $OUT_DIR"
    exit 0
  fi
  echo "ERROR: non-complete output already exists; refusing to overwrite: $OUT_DIR" >&2
  exit 3
fi

RUN_CFG="$CFG"
TEMP_CFG=""
cleanup() {
  if [[ -n "$TEMP_CFG" && -f "$TEMP_CFG" ]]; then
    rm -f -- "$TEMP_CFG"
  fi
}
trap cleanup EXIT

# Tiny-ImageNet has historically been more reliable with worker subprocesses
# disabled on these compute nodes. The source YAML remains unchanged.
if [[ "$DATASET" == TINYIMAGENET ]]; then
  TEMP_CFG="$(mktemp "/tmp/lidcover_${DATASET}_${BACKBONE}_s${SEED}_XXXXXX.yaml")"
  sed 's/NUM_WORKERS: 4/NUM_WORKERS: 0/' "$CFG" > "$TEMP_CFG"
  RUN_CFG="$TEMP_CFG"
fi

cmd=(
  python -u "$TRAIN_SCRIPT"
  --cfg "$RUN_CFG"
  --exp-name "$EXP_NAME"
  --al "$SAMPLER"
  --budget 50
  --initial_size 0
  --seed "$SEED"
  --initial_delta "$BASE_RADIUS"
)

if [[ "$SAMPLER" == idprobcover ]]; then
  cmd+=(
    --idpc_alpha 1.0
    --idpc_mode high_id_more_centers
    --idpc_k_id 50
    --idpc_k_knn 50
    --idpc_cache_root "$IDPC_CACHE_ROOT"
  )
fi

# train_al.py resolves ../../output and ../../data relative to the working
# directory. Running from tools keeps both paths under TypiClust, matching all
# canonical historical artifacts.
cd "$DEEP_AL/tools"
print_run
nvidia-smi -L
printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

if ! validate_complete; then
  echo "ERROR: training returned successfully, but canonical completion validation failed: $OUT_DIR" >&2
  exit 4
fi

echo "Validated COMPLETE: $OUT_DIR"
