#!/bin/bash
#SBATCH --job-name=topo_c100_r18_d070_s3_b100
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/budget_c100_r18_d070_s3_b100_%j.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/budget_c100_r18_d070_s3_b100_%j.err
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
export TYPI_FEATURE_BACKBONE=resnet18

cd /vast/s219110279/TypiClust/deep-al/tools

nvidia-smi -L
if nvidia-smi -L | grep -Eiq 'Blackwell|RTX PRO 6000'; then
  echo "ERROR: Job landed on an unsupported Blackwell GPU." >&2
  exit 2
fi

exp_name="TOPOCOVER_PAPER_D070_3_100b"
summary="/vast/s219110279/TypiClust/output/CIFAR100/resnet18/${exp_name}/benchmark_summary.json"
if [[ -f "${summary}" ]]; then
  echo "Skip: already complete: ${summary}"
  exit 0
fi

echo "dataset=CIFAR100 backbone=resnet18 seed=3 delta=0.70 batch_budget=100"
echo "start=$(date) node=$(hostname)"

python train_al.py \
  --cfg=/vast/s219110279/TypiClust/deep-al/configs/cifar100/al/RESNET18.yaml \
  --al=topocover \
  --exp-name="${exp_name}" \
  --budget=100 \
  --initial_size=0 \
  --seed=3 \
  --initial_delta=0.70 \
  --topocover_k_knn=50 \
  --topocover_cache_root=/scratch/s219110279/topocover_cache_paper/resnet18/cifar100 \
  --topocover_recompute_components=True

echo "done=$(date)"

