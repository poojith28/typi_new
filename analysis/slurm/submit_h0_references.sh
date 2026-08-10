#!/bin/bash
# Submit full-pool H0 reference PD jobs — one Slurm array per dataset.
# Each array covers alexnet, resnet18, resnet50 (CPU only).
#
# Usage (login node):
#   bash analysis/slurm/submit_h0_references.sh            # all three datasets
#   bash analysis/slurm/submit_h0_references.sh cifar10
#   bash analysis/slurm/submit_h0_references.sh cifar100
#   bash analysis/slurm/submit_h0_references.sh tiny
#
# TinyImageNet may already be running as h0ref_tiny — skip if so.

set -euo pipefail
cd "$(dirname "$0")"

submit_one() {
    local f="$1"
    local jid
    jid=$(sbatch --parsable "$f")
    echo "  submitted $f -> job $jid   (array: alexnet, resnet18, resnet50)"
}

target="${1:-all}"

echo "H0 reference PDs — full pool, CPU, all backbones"
case "$target" in
    all)
        submit_one h0_reference_cifar10.sbatch
        submit_one h0_reference_cifar100.sbatch
        # Skip Tiny if an h0ref_tiny job is already in the queue
        if squeue -u "$USER" -h -n h0ref_tiny 2>/dev/null | grep -q .; then
            echo "  skip h0_reference_tiny.sbatch — h0ref_tiny already queued/running"
            squeue -u "$USER" -n h0ref_tiny
        else
            submit_one h0_reference_tiny.sbatch
        fi
        ;;
    cifar10|c10)
        submit_one h0_reference_cifar10.sbatch
        ;;
    cifar100|c100)
        submit_one h0_reference_cifar100.sbatch
        ;;
    tiny|tinyimagenet)
        submit_one h0_reference_tiny.sbatch
        ;;
    *)
        echo "Unknown target: $target" >&2
        echo "Use: all | cifar10 | cifar100 | tiny" >&2
        exit 1
        ;;
esac

echo "Done. Check: squeue -u $USER"
echo "Logs: /vast/s219110279/TypiClust/output/_logs/h0ref_*"
echo "Outputs staged to: /vast/s219110279/outputs/diagnostic_h0/reference/"
