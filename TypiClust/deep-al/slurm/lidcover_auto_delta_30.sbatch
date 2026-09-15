#!/usr/bin/env bash
#SBATCH --job-name=lidcover-autodelta
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/lidcover-autodelta-%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/lidcover-autodelta-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-v100|gpu-l40s
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --array=0-29%15
#SBATCH --mail-type=FAIL

# Phase-2 automatic-base-radius evidence:
#   3 datasets x 2 matched methods x 5 seeds = 30 tasks.
# At most fifteen array elements can run simultaneously.

set -euo pipefail

PROJECT_ROOT=/vast/s219110279
DEEP_AL="$PROJECT_ROOT/TypiClust/deep-al"
TRAIN_SCRIPT="$DEEP_AL/tools/train_al.py"
OUTPUT_ROOT="$PROJECT_ROOT/TypiClust/output"
LOG_ROOT="$OUTPUT_ROOT/_logs"
LOCK_ROOT="$OUTPUT_ROOT/_submission_locks/lidcover-auto-delta"
AUTO_CACHE_ROOT="$PROJECT_ROOT/idpc_cache/lidcover_auto_delta/resnet18"

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( TASK_ID < 0 || TASK_ID > 29 )); then
  echo "Invalid array task $TASK_ID; expected 0..29" >&2
  exit 2
fi

DATASETS=(CIFAR10 CIFAR100 TINYIMAGENET)
SAMPLERS=(probcover_auto_delta lidcover_auto_delta)

DATASET_INDEX=$((TASK_ID / 10))
WITHIN_DATASET=$((TASK_ID % 10))
METHOD_INDEX=$((WITHIN_DATASET / 5))
SEED=$((WITHIN_DATASET % 5 + 1))
DATASET="${DATASETS[$DATASET_INDEX]}"
SAMPLER="${SAMPLERS[$METHOD_INDEX]}"
BACKBONE=resnet18

case "$DATASET" in
  CIFAR10) DATASET_CFG=cifar10 ;;
  CIFAR100) DATASET_CFG=cifar100 ;;
  TINYIMAGENET) DATASET_CFG=tinyimagenet ;;
  *) echo "Unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

if [[ "$SAMPLER" == lidcover_auto_delta ]]; then
  METHOD_TAG=LIDCOVER_AUTODELTA
else
  METHOD_TAG=PROBCOVER_AUTODELTA
fi

RULE_TAG=mediank50q50
EXP_NAME="${METHOD_TAG}_${RULE_TAG}_coldstart_20260810_${SEED}_50b"
OUT_DIR="$OUTPUT_ROOT/$DATASET/$BACKBONE/$EXP_NAME"
CFG="$DEEP_AL/configs/$DATASET_CFG/al/RESNET18.yaml"

if [[ ! -f "$CFG" ]]; then
  echo "Missing configuration: $CFG" >&2
  exit 2
fi

print_run() {
  echo "task_id=$TASK_ID dataset_index=$DATASET_INDEX method_index=$METHOD_INDEX seed=$SEED"
  echo "dataset=$DATASET backbone=$BACKBONE sampler=$SAMPLER"
  echo "rule=median_50th_neighbour_distance k=50 quantile=0.5"
  echo "exp_name=$EXP_NAME"
  echo "output=$OUT_DIR"
}

# Local mapping preflight:
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=0 bash lidcover_auto_delta_30.sbatch
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  print_run
  echo "protocol: budget=50 initial_size=0 max_iter=100 expected_final_budget=5050"
  exit 0
fi

mkdir -p "$LOG_ROOT" "$LOCK_ROOT" "$AUTO_CACHE_ROOT"

module purge
module load Anaconda3
set +u
source activate
conda activate typiclust
set -u

export PYTHONUNBUFFERED=1
export TYPI_FEATURES_ROOT="$PROJECT_ROOT/results/results"
export TYPI_FEATURE_BACKBONE="$BACKBONE"

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
provenance_path = out_dir / "auto_delta_provenance.json"
root_lset = out_dir / "lSet.npy"
final_lset = out_dir / "episode_100" / "lSet.npy"
if not all(path.is_file() for path in (summary_path, provenance_path, root_lset, final_lset)):
    raise SystemExit(1)

try:
    summary = json.loads(summary_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    delta_auto = float(provenance.get("delta_auto"))
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
        and provenance.get("schema") == "lidcover_auto_delta_v1"
        and provenance.get("rule") == "median_50th_neighbour_distance"
        and int(provenance.get("k", -1)) == 50
        and float(provenance.get("quantile", -1.0)) == 0.5
        and provenance.get("labels_used") is False
        and provenance.get("validation_or_test_accuracy_used") is False
        and math.isfinite(delta_auto)
        and delta_auto > 0.0
        and len(str(provenance.get("feature_sha256", ""))) == 64
        and len(str(provenance.get("candidate_indices_sha256", ""))) == 64
    )
except Exception:
    valid = False

raise SystemExit(0 if valid else 1)
PY
}

if [[ -e "$OUT_DIR" ]]; then
  if validate_complete; then
    echo "Validated COMPLETE; skipping existing auto-delta run: $OUT_DIR"
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

if [[ "$DATASET" == TINYIMAGENET ]]; then
  TEMP_CFG="$(mktemp "/tmp/lidcover_autodelta_${DATASET}_s${SEED}_XXXXXX.yaml")"
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
  --initial_delta 0.0
  --auto_delta_k 50
  --auto_delta_quantile 0.5
  --auto_delta_cache_root "$AUTO_CACHE_ROOT"
)

if [[ "$SAMPLER" == lidcover_auto_delta ]]; then
  cmd+=(
    --idpc_alpha 1.0
    --idpc_mode high_id_more_centers
    --idpc_k_id 50
    --idpc_k_knn 50
    --idpc_cache_root "$AUTO_CACHE_ROOT"
  )
fi

# Keep train_al.py's relative ../../output and ../../data resolution inside
# the canonical TypiClust tree.
cd "$DEEP_AL/tools"
print_run
nvidia-smi -L
printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

if ! validate_complete; then
  echo "ERROR: training returned successfully, but auto-delta validation failed: $OUT_DIR" >&2
  exit 4
fi

echo "Validated COMPLETE: $OUT_DIR"
