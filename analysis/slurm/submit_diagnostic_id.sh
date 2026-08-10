#!/bin/bash
# Submit ID-diagnostic jobs (recommended: one job per backbone on CPU).
#
# Usage:
#   bash submit_diagnostic_id.sh            # 3 CPU jobs (resnet18/50/alexnet)
#   MODE=gpu bash submit_diagnostic_id.sh  # GPU partition instead
#   bash submit_diagnostic_id.sh resnet18  # single backbone

set -euo pipefail
cd "$(dirname "$0")"

MODE="${MODE:-cpu}"
if [[ "$MODE" == "gpu" ]]; then
  SBATCH=run_diagnostic_id_gpu.sbatch
else
  SBATCH=run_diagnostic_id.sbatch
fi

if [[ $# -gt 0 ]]; then
  BACKBONE_LIST=("$@")
else
  BACKBONE_LIST=(resnet18 resnet50 alexnet)
fi

for bb in "${BACKBONE_LIST[@]}"; do
  echo "Submitting diagnostic ID job: backbone=$bb mode=$MODE"
  sbatch --job-name="diag_id_${bb}" \
    --export=ALL,BACKBONES="$bb",SKIP_FIGURES=0 \
    "$SBATCH"
done

echo "Check with: squeue --me"
echo "Logs land in: $(pwd)/diag_id_*.out"
echo "Report: /vast/s219110279/outputs/diagnostic_id/diagnostic_id_generation_report.md"
