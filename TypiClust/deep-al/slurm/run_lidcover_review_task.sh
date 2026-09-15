#!/usr/bin/env bash

# Shared, fail-closed runner for the ICONIP reviewer experiments. Thin sbatch
# files below map array indices to one immutable experiment and call this file.

set -euo pipefail

if (( $# != 9 )); then
  echo "usage: $0 DATASET BACKBONE SAMPLER RADIUS SEED ALPHA K_ID K_KNN EXP_NAME" >&2
  exit 2
fi

DATASET="$1"
BACKBONE="$2"
SAMPLER="$3"
BASE_RADIUS="$4"
SEED="$5"
ALPHA="$6"
K_ID="$7"
K_KNN="$8"
EXP_NAME="$9"

PROJECT_ROOT=/vast/s219110279
DEEP_AL="$PROJECT_ROOT/TypiClust/deep-al"
TRAIN_SCRIPT="$DEEP_AL/tools/train_al.py"
OUTPUT_ROOT="$PROJECT_ROOT/TypiClust/output"
OUT_DIR="$OUTPUT_ROOT/$DATASET/$BACKBONE/$EXP_NAME"
LOCK_ROOT="$OUTPUT_ROOT/_submission_locks/lidcover-review"
IDPC_CACHE_ROOT="$PROJECT_ROOT/idpc_cache/$BACKBONE"
ARC_CACHE_ROOT="$PROJECT_ROOT/adaptive_cover_cache/$BACKBONE"

case "$DATASET" in
  CIFAR10) DATASET_CFG=cifar10 ;;
  CIFAR100) DATASET_CFG=cifar100 ;;
  TINYIMAGENET) DATASET_CFG=tinyimagenet ;;
  *) echo "unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

case "$BACKBONE" in
  alexnet) MODEL_CFG=ALEXNET.yaml ;;
  resnet18) MODEL_CFG=RESNET18.yaml ;;
  resnet50) MODEL_CFG=RESNET50.yaml ;;
  *) echo "unsupported backbone: $BACKBONE" >&2; exit 2 ;;
esac

CFG="$DEEP_AL/configs/$DATASET_CFG/al/$MODEL_CFG"
if [[ ! -f "$CFG" || ! -f "$TRAIN_SCRIPT" ]]; then
  echo "missing configuration or training script: $CFG $TRAIN_SCRIPT" >&2
  exit 2
fi

print_run() {
  echo "dataset=$DATASET backbone=$BACKBONE sampler=$SAMPLER seed=$SEED"
  echo "base_radius=$BASE_RADIUS alpha=$ALPHA k_id=$K_ID k_knn=$K_KNN"
  echo "exp_name=$EXP_NAME"
  echo "output=$OUT_DIR"
  echo "protocol=matched cold-start; initial_size=0; budget=50; rounds=100; final_budget=5050"
}

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  print_run
  exit 0
fi

mkdir -p "$OUTPUT_ROOT/_logs" "$LOCK_ROOT" "$IDPC_CACHE_ROOT" "$ARC_CACHE_ROOT"

module purge
module load Anaconda3
# `source activate` inherits the caller's positional parameters. All task
# parameters have already been copied into named variables above, so clear
# `$@` before sourcing Conda; otherwise Conda treats the nine experiment
# fields as environment names and exits before Python starts.
set --
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
  echo "another job owns the experiment lock: $LOCK_FILE"
  exit 0
fi

validate_complete() {
  python - "$OUT_DIR" "$DATASET" "$BACKBONE" "$SAMPLER" "$SEED" <<'PY'
import csv
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
        and np.load(root_lset, allow_pickle=True).size == 0
        and np.load(final_lset, allow_pickle=True).size == 5050
        and math.isfinite(float(summary.get("final_test_accuracy")))
        and math.isfinite(float(summary.get("test_auc")))
    )
    if "fallback" in sampler or "tiebreak_min_id" in sampler:
        diag = out_dir / "id_probcover_rounds.csv"
        if not diag.is_file():
            valid = False
        else:
            with diag.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            required = {
                "zero_gain_fallback_count",
                "zero_gain_fallback_fraction",
                "first_zero_gain_selection_index",
            }
            valid = valid and bool(rows) and required.issubset(rows[0])
except Exception:
    valid = False

raise SystemExit(0 if valid else 1)
PY
}

if [[ -e "$OUT_DIR" ]]; then
  if validate_complete; then
    echo "validated COMPLETE; skipping: $OUT_DIR"
    exit 0
  fi
  echo "ERROR: partial or invalid output exists; refusing to overwrite: $OUT_DIR" >&2
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
  TEMP_CFG="$(mktemp "/tmp/lidcover_review_${DATASET}_${BACKBONE}_s${SEED}_XXXXXX.yaml")"
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

case "$SAMPLER" in
  idprobcover|idprobcover_tiebreak_min_id|idprobcover_fallback_random)
    cmd+=(
      --idpc_alpha "$ALPHA"
      --idpc_mode high_id_more_centers
      --idpc_k_id "$K_ID"
      --idpc_k_knn "$K_KNN"
      --idpc_cache_root "$IDPC_CACHE_ROOT"
      --idpc_log_csv id_probcover_rounds.csv
    )
    ;;
  density_cover|knn_distance_cover|distance_variance_cover|distance_cv_cover)
    cmd+=(
      --arc_alpha "$ALPHA"
      --arc_k_signal "$K_ID"
      --arc_k_knn "$K_KNN"
      --arc_cache_root "$ARC_CACHE_ROOT"
    )
    ;;
  probcover|maxherding)
    ;;
  *) echo "unsupported reviewer sampler: $SAMPLER" >&2; exit 2 ;;
esac

cd "$DEEP_AL/tools"
print_run
nvidia-smi -L
printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

if ! validate_complete; then
  echo "ERROR: training returned but completion validation failed: $OUT_DIR" >&2
  exit 4
fi

echo "validated COMPLETE: $OUT_DIR"
