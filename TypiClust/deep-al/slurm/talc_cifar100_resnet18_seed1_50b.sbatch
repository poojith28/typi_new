#!/usr/bin/env bash
#SBATCH --job-name=talc-c100-r18-s1
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/talc-c100-r18-s1-%j.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/talc-c100-r18-s1-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-v100|gpu-l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --mail-type=END,FAIL

# TALC pilot:
#   CIFAR-100 / ResNet-18 / seed 1 / acquisition batch 50
#   automatic label-free base radius, dynamic LID curriculum,
#   multiscale H0 component representation, and late margin uncertainty.

set -euo pipefail

PROJECT_ROOT=/vast/s219110279
DEEP_AL="$PROJECT_ROOT/TypiClust/deep-al"
TRAIN_SCRIPT="$DEEP_AL/tools/train_al.py"
OUTPUT_ROOT="$PROJECT_ROOT/TypiClust/output"
LOG_ROOT="$OUTPUT_ROOT/_logs"
LOCK_ROOT="$OUTPUT_ROOT/_submission_locks/talc-pilot"
CACHE_ROOT=/scratch/s219110279/talc_cache/resnet18/cifar100

DATASET=CIFAR100
BACKBONE=resnet18
SEED=1
BUDGET=50
SAMPLER=talc_auto_delta
EXP_NAME=TALC_AUTODELTA_C100_R18_S1_B050_20260827
OUT_DIR="$OUTPUT_ROOT/$DATASET/$BACKBONE/$EXP_NAME"
CFG="$DEEP_AL/configs/cifar100/al/RESNET18.yaml"

print_run() {
  echo "dataset=$DATASET backbone=$BACKBONE seed=$SEED sampler=$SAMPLER"
  echo "budget=$BUDGET initial_size=0 max_iter=100 expected_final_budget=5050"
  echo "auto_delta=median_50th_neighbour_distance"
  echo "talc=alpha_max:1.0 coverage_target:0.9 coverage_epsilon:0.05 topology_weight:0.5"
  echo "exp_name=$EXP_NAME"
  echo "output=$OUT_DIR"
  echo "cache=$CACHE_ROOT"
}

# Local preflight without Slurm or training:
#   DRY_RUN=1 bash slurm/talc_cifar100_resnet18_seed1_50b.sbatch
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  print_run
  exit 0
fi

if [[ ! -f "$CFG" ]]; then
  echo "ERROR: missing configuration: $CFG" >&2
  exit 2
fi
if [[ ! -f "$DEEP_AL/pycls/al/talc.py" ]]; then
  echo "ERROR: missing TALC implementation: $DEEP_AL/pycls/al/talc.py" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT" "$LOCK_ROOT" "$CACHE_ROOT"

module purge
module load Anaconda3
set +u
source activate
conda activate typiclust
set -u

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="$SEED"
export TYPI_FEATURES_ROOT="$PROJECT_ROOT/results/results"
export TYPI_FEATURE_BACKBONE="$BACKBONE"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-talc}"
mkdir -p "$MPLCONFIGDIR"

if nvidia-smi -L | grep -Eiq 'Blackwell|RTX PRO 6000'; then
  echo "ERROR: job landed on an unsupported Blackwell GPU." >&2
  exit 75
fi

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
    records = summary.get("episode_records", [])
    valid = (
        summary.get("dataset") == dataset
        and str(summary.get("model", "")).lower() == backbone
        and str(summary.get("sampling_fn", "")).lower() == sampler
        and int(summary.get("seed", -1)) == seed
        and int(summary.get("budget_per_round", -1)) == 50
        and int(summary.get("num_rounds_completed", -1)) == 101
        and len(records) == 101
        and np.load(root_lset, allow_pickle=True).size == 0
        and np.load(final_lset, allow_pickle=True).size == 5050
        and math.isfinite(float(summary.get("final_test_accuracy")))
        and math.isfinite(float(summary.get("test_auc")))
        and provenance.get("rule") == "median_50th_neighbour_distance"
        and int(provenance.get("k", -1)) == 50
        and float(provenance.get("quantile", -1.0)) == 0.5
        and provenance.get("labels_used") is False
        and provenance.get("validation_or_test_accuracy_used") is False
        and math.isfinite(float(provenance.get("delta_auto")))
        and float(provenance.get("delta_auto")) > 0.0
        and all(
            record.get("sampling_metadata", {}).get("strategy")
            in {
                "trajectory_adaptive_lid_cover",
                "trajectory_adaptive_lid_cover_auto_delta",
                None,
            }
            for record in records
        )
    )
except Exception:
    valid = False

raise SystemExit(0 if valid else 1)
PY
}

if [[ -e "$OUT_DIR" ]]; then
  if validate_complete; then
    echo "Validated COMPLETE; skipping existing TALC run: $OUT_DIR"
    exit 0
  fi
  echo "ERROR: non-complete output exists; refusing to overwrite: $OUT_DIR" >&2
  exit 3
fi

cmd=(
  python -u "$TRAIN_SCRIPT"
  --cfg "$CFG"
  --exp-name "$EXP_NAME"
  --al "$SAMPLER"
  --budget "$BUDGET"
  --initial_size 0
  --seed "$SEED"
  --initial_delta 0.0
  --auto_delta_k 50
  --auto_delta_quantile 0.5
  --auto_delta_cache_root "$CACHE_ROOT"
  --idpc_k_id 50
  --idpc_k_knn 50
  --talc_alpha_max 1.0
  --talc_coverage_target 0.9
  --talc_coverage_epsilon 0.05
  --talc_topology_weight 0.5
  --talc_min_component_size 5
  --talc_use_topology true
  --talc_use_uncertainty true
  --talc_cache_root "$CACHE_ROOT"
)

# Preserve train_al.py's historical relative output/data resolution.
cd "$DEEP_AL/tools"
print_run
nvidia-smi -L
printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

if ! validate_complete; then
  echo "ERROR: training returned successfully, but TALC validation failed: $OUT_DIR" >&2
  exit 4
fi

echo "Validated COMPLETE: $OUT_DIR"
