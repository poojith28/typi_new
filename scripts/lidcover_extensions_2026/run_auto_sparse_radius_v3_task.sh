#!/usr/bin/env bash
# Fail-closed execution of one immutable v3 automatic sparse-radius run.
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 ARRAY_INDEX" >&2
  exit 2
fi

TASK_ID="$1"
PROJECT_ROOT=/vast/s219110279
MANIFEST="$PROJECT_ROOT/evidence/lidcover_extensions_2026/auto_sparse_radius_v3_manifest.csv"
ENTRYPOINT="$PROJECT_ROOT/experiments/lidcover_extensions_2026/extension_entrypoint.py"
FAMILY=auto_sparse_radius_v3

IFS=$'\t' read -r EXP_ID DATASET BACKBONE METHOD SEED BUDGET MAX_ITER DELTA BASE_CFG OUTPUT_ROOT OUTPUT_DIR < <(
  python - "$MANIFEST" "$TASK_ID" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
matches = [row for row in rows if int(row["array_index"]) == int(sys.argv[2])]
if len(matches) != 1:
    raise SystemExit(f"expected one manifest row, found {len(matches)}")
row = matches[0]
fields = ["experiment_id", "dataset", "backbone", "method", "seed",
          "acquisition_batch_size", "max_iter", "configured_delta0", "base_config",
          "output_root", "output_dir"]
print("\t".join(row[field] for field in fields))
PY
)

case "$METHOD" in
  probcover_auto_sparse_radius_v3|lidcover_auto_sparse_radius_v3) ;;
  *) echo "unexpected method in v3 manifest: $METHOD" >&2; exit 2 ;;
esac
EXPECTED="$OUTPUT_ROOT/$DATASET/$BACKBONE/$EXP_ID"
if [[ -z "$EXP_ID" || ! -f "$BASE_CFG" || ! -f "$ENTRYPOINT" || "$OUTPUT_DIR" != "$EXPECTED" ]]; then
  echo "invalid v3 manifest row or missing source/config" >&2
  exit 2
fi
if [[ "$DELTA" != AUTO_SPARSE_V3 ]]; then
  echo "unexpected configured delta marker: $DELTA" >&2
  exit 2
fi

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "DRY_RUN PASS family=$FAMILY array_index=$TASK_ID experiment_id=$EXP_ID"
  echo "dataset=$DATASET backbone=$BACKBONE method=$METHOD seed=$SEED"
  echo "budget=$BUDGET max_iter=$MAX_ITER configured_delta0=$DELTA runtime_delta=0.0"
  echo "base_config=$BASE_CFG"
  echo "output=$OUTPUT_DIR"
  exit 0
fi

LOCK_ROOT="$PROJECT_ROOT/experiments/lidcover_extensions_2026/locks/$FAMILY"
AUTO_CACHE="$PROJECT_ROOT/experiments/lidcover_extensions_2026/cache/$FAMILY/auto_delta"
IDPC_CACHE="$PROJECT_ROOT/experiments/lidcover_extensions_2026/cache/$FAMILY/idpc/$BACKBONE"
mkdir -p "$LOCK_ROOT" "$AUTO_CACHE" "$IDPC_CACHE" "$OUTPUT_ROOT"
exec 9>"$LOCK_ROOT/$EXP_ID.lock"
if ! flock -n 9; then
  echo "another task owns the run-local lock: $EXP_ID"
  exit 0
fi

module purge
module load Anaconda3
set --
set +u
source activate
conda activate typiclust
set -u

export PYTHONUNBUFFERED=1
export TYPI_FEATURES_ROOT="$PROJECT_ROOT/results/results"
export TYPI_FEATURE_BACKBONE="$BACKBONE"
export MPLCONFIGDIR="$PROJECT_ROOT/experiments/lidcover_extensions_2026/matplotlib-cache"
mkdir -p "$MPLCONFIGDIR"

validate_complete() {
  python - "$OUTPUT_DIR" "$METHOD" "$SEED" "$BUDGET" "$MAX_ITER" <<'PY'
import json, math, sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1]); method = sys.argv[2]; seed = int(sys.argv[3])
budget = int(sys.argv[4]); max_iter = int(sys.argv[5])
summary_path = out / "benchmark_summary.json"
final_path = out / f"episode_{max_iter}" / "lSet.npy"
root_lset = out / "lSet.npy"
provenance_path = out / "auto_delta_provenance.json"
if not all(path.is_file() for path in (summary_path, final_path, root_lset, provenance_path)):
    raise SystemExit(1)
try:
    summary = json.loads(summary_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    valid = (
        str(summary.get("sampling_fn", "")).lower() == method
        and int(summary.get("seed", -1)) == seed
        and int(summary.get("budget_per_round", -1)) == budget
        and int(summary.get("num_rounds_completed", -1)) == max_iter + 1
        and len(summary.get("episode_records", [])) == max_iter + 1
        and np.load(str(root_lset), allow_pickle=True).size == 0
        and np.load(str(final_path), allow_pickle=True).size == budget * (max_iter + 1)
        and math.isfinite(float(summary.get("final_test_accuracy")))
        and provenance.get("schema") == "lidcover_auto_sparse_radius_v3"
        and provenance.get("rule") == "candidate_lid_adaptive_target_mean_degree_v3"
        and provenance.get("labels_used") is False
        and provenance.get("validation_or_test_accuracy_used") is False
        and provenance.get("held_out_validation_or_test_included") is False
        and provenance.get("knn_exact") is True
        and int(provenance.get("k", -1)) == 50
        and float(provenance.get("target_mean_nonself_out_degree", -1)) == 2.0
        and float(provenance.get("realised_mean_nonself_out_degree", -1)) >= 2.0
        and 0.0 < float(provenance.get("delta_auto", 0.0)) < 1.0
        and len(str(provenance.get("representation_sha256", ""))) == 64
        and len(str(provenance.get("candidate_index_sha256", ""))) == 64
    )
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

if [[ -e "$OUTPUT_DIR" ]]; then
  if validate_complete; then
    echo "validated COMPLETE; excluding existing run: $OUTPUT_DIR"
    exit 0
  fi
  echo "ERROR: output exists but is not strictly complete; refusing reuse: $OUTPUT_DIR" >&2
  exit 3
fi

RUN_CFG="$BASE_CFG"
TEMP_CFG=""
cleanup() {
  if [[ -n "$TEMP_CFG" && -f "$TEMP_CFG" ]]; then
    rm -f -- "$TEMP_CFG"
  fi
}
trap cleanup EXIT
if [[ "$DATASET" == TINYIMAGENET ]]; then
  TEMP_CFG="$(mktemp "/tmp/lidext_${FAMILY}_${SEED}_XXXXXX.yaml")"
  sed 's/NUM_WORKERS: 4/NUM_WORKERS: 0/' "$BASE_CFG" > "$TEMP_CFG"
  RUN_CFG="$TEMP_CFG"
fi

cmd=(
  python -u "$ENTRYPOINT"
  --cfg "$RUN_CFG"
  --exp-name "$EXP_ID"
  --al "$METHOD"
  --budget "$BUDGET"
  --initial_size 0
  --seed "$SEED"
  --initial_delta 0.0
  --extension-output-root "$OUTPUT_ROOT"
  --extension-max-iter "$MAX_ITER"
  --idpc_alpha 1.0
  --idpc_mode high_id_more_centers
  --idpc_k_id 50
  --idpc_k_knn 50
  --idpc_cache_root "$IDPC_CACHE"
  --idpc_log_csv id_probcover_rounds.csv
  --auto_delta_k 50
  --auto_delta_quantile 0.5
  --auto_delta_cache_root "$AUTO_CACHE"
)

cd "$PROJECT_ROOT/TypiClust/deep-al/tools"
echo "family=$FAMILY array_index=$TASK_ID experiment_id=$EXP_ID"
echo "dataset=$DATASET backbone=$BACKBONE method=$METHOD seed=$SEED budget=$BUDGET max_iter=$MAX_ITER"
echo "output=$OUTPUT_DIR"
printf 'command:'; printf ' %q' "${cmd[@]}"; printf '\n'
nvidia-smi -L
"${cmd[@]}"

if ! validate_complete; then
  echo "ERROR: command returned but strict completion validation failed: $OUTPUT_DIR" >&2
  exit 4
fi
echo "validated COMPLETE: $OUTPUT_DIR"
