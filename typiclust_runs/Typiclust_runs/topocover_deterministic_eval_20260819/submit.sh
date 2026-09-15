#!/bin/bash
set -euo pipefail
campaign=/vast/s219110279/typiclust_runs/Typiclust_runs/topocover_deterministic_eval_20260819
snapshot=$(mktemp)
trap 'rm -f "$snapshot"' EXIT
timeout 30s squeue -h -u s219110279 -o '%i|%T|%j|%o' > "$snapshot"
if grep -Eiq 'tc_deteval|topocover_deterministic_eval_20260819' "$snapshot"; then
  echo "Matching deterministic TopoCover jobs are already queued" >&2
  exit 76
fi
sbatch --parsable "$campaign/run_array.sbatch"
