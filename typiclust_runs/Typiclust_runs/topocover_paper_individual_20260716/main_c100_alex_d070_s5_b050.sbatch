#!/bin/bash
#SBATCH --job-name=topo_c100_alex_d070_s5_b50
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/main_c100_alex_d070_s5_b050_%j.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/main_c100_alex_d070_s5_b050_%j.err
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
export TYPI_FEATURE_BACKBONE=alexnet

cd /vast/s219110279/TypiClust/deep-al/tools

nvidia-smi -L
if nvidia-smi -L | grep -Eiq 'Blackwell|RTX PRO 6000'; then
  echo "ERROR: Job landed on an unsupported Blackwell GPU." >&2
  exit 2
fi

exp_name="TOPOCOVER_PAPER_D070_5_50b"
summary="/vast/s219110279/TypiClust/output/CIFAR100/alexnet/${exp_name}/benchmark_summary.json"
if [[ -f "${summary}" ]]; then
  echo "Skip: already complete: ${summary}"
  exit 0
fi

echo "dataset=CIFAR100 backbone=alexnet seed=5 delta=0.70 batch_budget=50"
echo "start=$(date) node=$(hostname)"

python train_al.py \
  --cfg=/vast/s219110279/TypiClust/deep-al/configs/cifar100/al/ALEXNET.yaml \
  --al=topocover \
  --exp-name="${exp_name}" \
  --budget=50 \
  --initial_size=0 \
  --seed=5 \
  --initial_delta=0.70 \
  --topocover_k_knn=50 \
  --topocover_cache_root=/scratch/s219110279/topocover_cache_paper/alexnet/cifar100 \
  --topocover_recompute_components=True

echo "done=$(date)"

