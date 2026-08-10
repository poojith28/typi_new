#!/bin/bash
#SBATCH --job-name=redo_c10_dbal_s4
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/redo_c10_resnet50_dbal_s4_b050_%j.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/redo_c10_resnet50_dbal_s4_b050_%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --constraint=gpu-v100|gpu-l40s
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

cd /vast/s219110279/TypiClust/deep-al/tools

nvidia-smi -L
if nvidia-smi -L | grep -Eiq 'Blackwell|RTX PRO 6000'; then
  echo "ERROR: Job landed on an unsupported Blackwell GPU." >&2
  exit 2
fi

exp_name="dbal_4_50b"
summary="/vast/s219110279/TypiClust/output/CIFAR10/resnet50/${exp_name}/benchmark_summary.json"
if [[ -f "${summary}" ]]; then
  echo "Skip: already complete: ${summary}"
  exit 0
fi

echo "recovery dataset=CIFAR10 backbone=resnet50 strategy=dbal seed=4 budget=50"
echo "start=$(date) node=$(hostname)"

python train_al.py \
  --cfg=/vast/s219110279/TypiClust/deep-al/configs/cifar10/al/RESNET50.yaml \
  --al=dbal \
  --exp-name="${exp_name}" \
  --budget=50 \
  --initial_size=50 \
  --seed=4

echo "done=$(date)"

