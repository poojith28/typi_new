#!/bin/bash
#SBATCH --job-name=rerun_tin_alx5
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/rerun_tin_alx5_%j.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/rerun_tin_alx5_%j.err
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

exp_name="TOPOCOVER_PAPER_D070_5_50b"
exp_dir="/vast/s219110279/TypiClust/output/TINYIMAGENET/alexnet/${exp_name}"
summary="${exp_dir}/benchmark_summary.json"

if [[ -f "$summary" ]]; then
  echo "Already complete: $summary"
  exit 0
fi

if [[ -d "$exp_dir" ]]; then
  echo "Removing incomplete run dir: $exp_dir"
  rm -rf "$exp_dir"
fi

nvidia-smi -L || true
if nvidia-smi -L | grep -Eiq 'Blackwell|RTX PRO 6000'; then
  echo "ERROR: Job landed on Blackwell GPU. Requeue on V100/L40S." >&2
  exit 2
fi

python train_al.py \
  --cfg=/vast/s219110279/TypiClust/deep-al/configs/tinyimagenet/al/ALEXNET.yaml \
  --al=topocover \
  --exp-name="$exp_name" \
  --budget=50 \
  --initial_size=0 \
  --seed 5 \
  --initial_delta=0.70 \
  --topocover_k_knn=50 \
  --topocover_cache_root=/scratch/s219110279/topocover_cache_d070/alexnet/tinyimagenet \
  --topocover_recompute_components=True

echo "done=$(date)"
