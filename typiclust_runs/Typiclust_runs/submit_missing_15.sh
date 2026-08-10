#!/bin/bash
# Requeue unfinished TypiClust AL runs with at most 15 active Slurm tasks.
#
# Usage:
#   bash /vast/s219110279/typiclust_runs/Typiclust_runs/submit_missing_15.sh
#
# The script rebuilds missing_runs.tsv from current benchmark_summary.json files,
# then submits one Slurm array where each task is exactly one missing seed/run.

set -euo pipefail

BASE=/vast/s219110279/typiclust_runs/Typiclust_runs
MANIFEST="$BASE/missing_runs.tsv"
SBATCH_FILE="$BASE/run_missing_15.sbatch"

cd "$BASE"

python "$BASE/build_missing_manifest.py" --out "$MANIFEST"

n_missing=$(( $(wc -l < "$MANIFEST") - 1 ))
if (( n_missing <= 0 )); then
  echo "Nothing missing. No jobs submitted."
  exit 0
fi

last=$((n_missing - 1))
echo "Submitting $n_missing missing runs with concurrency cap 15"
echo "Manifest: $MANIFEST"

sbatch \
  --array=0-"$last"%15 \
  --export=ALL,MANIFEST="$MANIFEST" \
  "$SBATCH_FILE"

echo "Submitted. Check with: squeue -u \$USER"
echo "Logs: /vast/s219110279/TypiClust/output/_logs/missing_<job>_<task>.out"
