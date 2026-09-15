#!/usr/bin/env bash

# Submit the ICONIP reviewer arrays independently. Each array is capped at
# fifteen tasks; Slurm/user/QoS limits determine the campaign-wide total.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT=/vast/s219110279/TypiClust/output/_logs
mkdir -p "$LOG_ROOT"

FILES=(
  lidcover_review_fallback_30.sbatch
  lidcover_review_probcover_radius_25.sbatch
  lidcover_review_signal_tuning_50.sbatch
  lidcover_review_maxherding_15.sbatch
  lidcover_review_k_disentangle_20.sbatch
  lidcover_auto_delta_30.sbatch
)

for file in "${FILES[@]}"; do
  [[ -f "$SCRIPT_DIR/$file" ]] || { echo "missing: $SCRIPT_DIR/$file" >&2; exit 2; }
done

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "Independent submission order (each array capped at fifteen tasks):"
  for file in "${FILES[@]}"; do
    echo "  sbatch $SCRIPT_DIR/$file"
  done
  echo "Total: 170 runs = 140 reviewer-specific + 30 matched automatic-radius runs."
  exit 0
fi

ACTIVE_NAMES='^(lidrev-fallback|lidrev-pcradius|lidrev-signals|lidrev-maxherd|lidrev-kgrid|lidcover-autodelta)$'
if squeue -h -u "${USER:?USER is required}" -o '%j' | grep -Eq "$ACTIVE_NAMES"; then
  echo "Refusing duplicate campaign submission; at least one campaign job is already active." >&2
  squeue -u "$USER" -o '%.18i %.24j %.10T %.10M %R' | grep -E "$ACTIVE_NAMES" || true
  exit 3
fi

for file in "${FILES[@]}"; do
  job_id="$(sbatch --parsable "$SCRIPT_DIR/$file")"
  job_id="${job_id%%;*}"
  [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "unexpected sbatch result: $job_id" >&2; exit 4; }
  echo "$file -> independent job $job_id"
done

echo "Submitted all 170 runs as independent arrays."
echo "Monitor: squeue -u $USER -o '%.18i %.24j %.10T %.10M %R'"
