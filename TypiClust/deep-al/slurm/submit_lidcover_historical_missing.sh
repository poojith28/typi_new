#!/usr/bin/env bash

# Re-audit and submit only genuinely missing historical tasks. The sparse Slurm
# array is capped at 15 simultaneous tasks.

set -euo pipefail

PROJECT_ROOT=/vast/s219110279
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="$SCRIPT_DIR/lidcover_historical_missing_40.sbatch"
AUDIT_SCRIPT="$PROJECT_ROOT/analysis/audit_lidcover_evidence.py"
MISSING_CSV="$PROJECT_ROOT/evidence_pack/lidcover_missing_jobs.csv"
PYTHON_BIN=/sw/software/Anaconda3/2024.02/bin/python
JOB_NAME=lidcover-hist-repair

if [[ ! -f "$SBATCH_FILE" || ! -f "$AUDIT_SCRIPT" || ! -x "$PYTHON_BIN" ]]; then
  echo "Missing sbatch file, audit script, or audit Python interpreter." >&2
  exit 2
fi

echo "Refreshing the historical completion audit."
"$PYTHON_BIN" "$AUDIT_SCRIPT"

task_ids="$("$PYTHON_BIN" - "$MISSING_CSV" <<'PY'
import csv
import sys

blocks = [
    ("CIFAR100", "alexnet", "LIDCover", "0.60"),
    ("CIFAR100", "alexnet", "ProbCover", "0.25"),
    ("CIFAR100", "resnet18", "LIDCover", "0.25"),
    ("CIFAR100", "resnet18", "LIDCover", "0.60"),
    ("CIFAR100", "resnet50", "LIDCover", "0.60"),
    ("CIFAR100", "resnet50", "ProbCover", "0.25"),
    ("CIFAR10", "resnet18", "LIDCover", "0.60"),
    ("TINYIMAGENET", "resnet18", "LIDCover", "0.60"),
]
block_index = {block: index for index, block in enumerate(blocks)}
task_ids = []
with open(sys.argv[1], newline="") as handle:
    for row in csv.DictReader(handle):
        radius = row["delta"] or row["delta0"]
        key = (row["dataset"], row["backbone"], row["method"], f"{float(radius):.2f}")
        if key not in block_index:
            raise SystemExit(f"missing row cannot be mapped to the audited sbatch: {key}")
        seed = int(row["seed"])
        if seed not in range(1, 6):
            raise SystemExit(f"invalid canonical seed: {seed}")
        task_ids.append(block_index[key] * 5 + seed - 1)

if len(task_ids) != len(set(task_ids)):
    raise SystemExit("duplicate Slurm task IDs derived from the missing-job audit")
print(",".join(str(task) for task in sorted(task_ids)))
PY
)"

if [[ -z "$task_ids" ]]; then
  echo "Historical matrix is already complete; nothing to submit."
  exit 0
fi

task_count="$(awk -F, 'NR > 1 {count++} END {print count+0}' "$MISSING_CSV")"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "Dry run: would submit $task_count missing historical runs with a %15 cap."
  echo "Task IDs: $task_ids"
  exit 0
fi

if ! queue_names="$(squeue -h -u "${USER:?USER is required}" -o '%j')"; then
  echo "Unable to contact the Slurm controller; refusing an unverified submission." >&2
  exit 3
fi

if grep -Eq '^(lidcover-hist|lidcover-hist-repair)$' <<<"$queue_names"; then
  echo "Refusing duplicate submission: a historical LIDCover array is active." >&2
  squeue -u "$USER" -n lidcover-hist,lidcover-hist-repair
  exit 3
fi

echo "Submitting $task_count missing historical runs with at most 15 running at once."
echo "Task IDs: $task_ids"
echo "File: $SBATCH_FILE"
sbatch --job-name="$JOB_NAME" --array="${task_ids}%15" "$SBATCH_FILE"
