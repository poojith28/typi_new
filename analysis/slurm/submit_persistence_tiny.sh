#!/bin/bash
# Submit TinyImageNet H0 persistence (one Slurm job per backbone).
#
# Usage (from login node):
#   bash submit_persistence_tiny.sh
#   bash submit_persistence_tiny.sh resnet18    # single backbone only
#
# Logs: /vast/s219110279/TypiClust/output/_logs/pers-tiny-*

set -euo pipefail
cd "$(dirname "$0")"

submit_one() {
    local sbatch_file="$1"
    local jid
    jid=$(sbatch --parsable "$sbatch_file")
    echo "  submitted $sbatch_file -> job $jid"
}

target="${1:-all}"

case "$target" in
    all)
        submit_one persistence_tiny_alexnet.sbatch
        submit_one persistence_tiny_resnet18.sbatch
        submit_one persistence_tiny_resnet50.sbatch
        ;;
    alexnet|alex)
        submit_one persistence_tiny_alexnet.sbatch
        ;;
    resnet18|r18)
        submit_one persistence_tiny_resnet18.sbatch
        ;;
    resnet50|r50)
        submit_one persistence_tiny_resnet50.sbatch
        ;;
    *)
        echo "Unknown target: $target" >&2
        echo "Use: all | alexnet | resnet18 | resnet50" >&2
        exit 1
        ;;
esac

echo "Done. Check: squeue -u $USER"
