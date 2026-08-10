#!/bin/bash
# Queue the full CIFAR10 AL suite for resnet50 and alexnet only (resnet18 skipped).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for backbone in resnet50 alexnet; do
  echo "=== Submitting all methods for $backbone ==="
  BACKBONE="$backbone" bash submit_cifar10_all_methods.sh
done

echo "Done: 24 Slurm jobs queued (12 methods x 2 backbones)."
