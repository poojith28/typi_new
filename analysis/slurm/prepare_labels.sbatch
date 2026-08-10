#!/bin/bash
#SBATCH --job-name=prep-labels
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/prep-labels-%j.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/prep-labels-%j.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# One-off: dump CIFAR10/100 train+test labels next to the SimCLR/SCAN embeddings.

source /vast/s219110279/analysis/slurm/env_setup.sh
mkdir -p /vast/s219110279/TypiClust/output/_logs

cd "$ANALYSIS_DIR"

$PY prepare_labels.py \
    --embeddings-root "$EMBED_ROOT" \
    --datasets cifar-10 cifar-100 \
    --backbones alexnet resnet18 resnet50

echo "done at $(date)"
