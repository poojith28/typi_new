#!/usr/bin/env bash

# Submit phase-2 automatic-radius evidence only after the historical matrix is
# complete. The sbatch file limits execution to fifteen simultaneous tasks.

set -euo pipefail

PROJECT_ROOT=/vast/s219110279
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="$SCRIPT_DIR/lidcover_auto_delta_30.sbatch"
AUDIT_SCRIPT="$PROJECT_ROOT/analysis/audit_lidcover_evidence.py"
MISSING_CSV="$PROJECT_ROOT/evidence_pack/lidcover_missing_jobs.csv"
PYTHON_BIN=/sw/software/Anaconda3/2024.02/bin/python
JOB_NAME=lidcover-autodelta

if [[ ! -f "$SBATCH_FILE" || ! -f "$AUDIT_SCRIPT" || ! -x "$PYTHON_BIN" ]]; then
  echo "Missing sbatch file, audit script, or typiclust Python interpreter." >&2
  exit 2
fi

echo "Refreshing the historical completion audit before phase-2 submission."
"$PYTHON_BIN" "$AUDIT_SCRIPT"

missing_count="$("$PYTHON_BIN" - "$MISSING_CSV" <<'PY'
import csv
import sys

with open(sys.argv[1], newline='') as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)"

if [[ "$missing_count" != 0 ]]; then
  echo "Phase-2 submission blocked: $missing_count historical runs are still missing." >&2
  exit 5
fi

if squeue -h -u "${USER:?USER is required}" -o '%j' | grep -Fqx "$JOB_NAME"; then
  echo "Refusing duplicate submission: an active $JOB_NAME array already exists." >&2
  squeue -u "$USER" -n "$JOB_NAME"
  exit 3
fi

if squeue -h -u "$USER" -o '%j' | grep -Eq '^(lidcover-hist|lidcover-hist-repair)$'; then
  echo "Historical LIDCover completion jobs are still active; phase 2 remains gated." >&2
  exit 4
fi

echo "Submitting 30 automatic-radius runs with at most fifteen running at once."
echo "File: $SBATCH_FILE"
sbatch "$SBATCH_FILE"
