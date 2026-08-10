#!/bin/bash
# Submit all alexnet Slurm job scripts (parallel folders to the resnet50 suites).
#
#   bash /vast/s219110279/typiclust_runs/Typiclust_runs/submit_all_alexnet.sh
#   bash .../submit_all_alexnet.sh cifar10          # one dataset only

set -euo pipefail

BASE=/vast/s219110279/typiclust_runs/Typiclust_runs
FILTER="${1:-all}"

submit_dir() {
  local dir="$1"
  cd "$dir"
  for f in *.sh; do
    [[ -f "$f" ]] || continue
    [[ "$f" == submit* ]] && continue
    echo "sbatch $dir/$f"
    sbatch "$f"
  done
}

case "$FILTER" in
  all)
    DIRS=(
      cifar10_full_50b_alexnet
      cifar100_full_50b_alexnet
      cifar100_ablation_50b_alexnet
      cifar100_budget_ablation_alexnet
      tinyimagenet_full_50b_alexnet
    )
    ;;
  cifar10)  DIRS=(cifar10_full_50b_alexnet) ;;
  cifar100) DIRS=(cifar100_full_50b_alexnet cifar100_ablation_50b_alexnet cifar100_budget_ablation_alexnet) ;;
  tiny|tinyimagenet) DIRS=(tinyimagenet_full_50b_alexnet) ;;
  *)
    echo "Usage: $0 [all|cifar10|cifar100|tiny]" >&2
    exit 1
    ;;
esac

for d in "${DIRS[@]}"; do
  echo "======== $d ========"
  submit_dir "$BASE/$d"
done

echo "Done. Logs: TypiClust/output/_logs/ and *.out next to each script."
