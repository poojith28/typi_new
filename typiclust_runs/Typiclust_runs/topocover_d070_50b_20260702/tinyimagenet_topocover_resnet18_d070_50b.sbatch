#!/bin/bash
#SBATCH --job-name=topo70_tin_r18
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/topo70_tin_r18_%j.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/topo70_tin_r18_%j.err
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

nvidia-smi -L || true
if nvidia-smi -L | grep -Eiq 'Blackwell|RTX PRO 6000'; then
  echo "ERROR: Job landed on Blackwell GPU. Requeue on V100/L40S." >&2
  exit 2
fi

for seed in 1 2 3 4 5; do
  exp_name="TOPOCOVER_PAPER_D070_${seed}_50b"
  summary="/vast/s219110279/TypiClust/output/TINYIMAGENET/resnet18/${exp_name}/benchmark_summary.json"
  if [[ -f "$summary" ]]; then
    echo "Skip: already complete: $summary"
    continue
  fi

  echo "============================================================"
  echo "dataset=TINYIMAGENET backbone=resnet18 seed=$seed exp=$exp_name delta=0.70"
  echo "start=$(date) node=$(hostname)"
  echo "============================================================"

  python train_al.py \
    --cfg=/vast/s219110279/TypiClust/deep-al/configs/tinyimagenet/al/RESNET18.yaml \
    --al=topocover \
    --exp-name="$exp_name" \
    --budget=50 \
    --initial_size=0 \
    --seed "$seed" \
    --initial_delta=0.70 \
    --topocover_k_knn=50 \
    --topocover_cache_root=/scratch/s219110279/topocover_cache_d070/resnet18/tinyimagenet \
    --topocover_recompute_components=True
done

echo "done=$(date)"
