#!/bin/bash
#SBATCH --job-name=rem_01_res
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/remaining11_01_%j.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/remaining11_01_%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s219110279@deakin.edu.au

set -eo pipefail

module purge
module load Anaconda3
source activate
conda activate typiclust
set -u

export TYPI_FEATURES_ROOT=/vast/s219110279/results/results
export TYPI_FEATURE_BACKBONE=resnet50

cd /scratch/s219110279/TypiClust/deep-al/tools

summary=/vast/s219110279/TypiClust/output/CIFAR10/resnet50/coreset_5_50b/benchmark_summary.json
if [[ -f "$summary" ]]; then
  echo "Skip: already complete: $summary"
  exit 0
fi

echo "folder=cifar10_full_50b script=coreset.sh dataset=CIFAR10 backbone=resnet50 exp=coreset_5_50b"
echo "start=$(date) node=$(hostname)"
nvidia-smi -L || true
echo "Command: python train_al.py --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar10/al/RESNET50.yaml --al=coreset --exp-name=coreset_5_50b --budget=50 --initial_size=50 --seed 5"
python train_al.py --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar10/al/RESNET50.yaml --al=coreset --exp-name=coreset_5_50b --budget=50 --initial_size=50 --seed 5
echo "done=$(date)"
