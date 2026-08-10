#!/bin/bash
#SBATCH --job-name=rerun_tin_lid3
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/rerun_tin_lid3_%j.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/rerun_tin_lid3_%j.err
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

exp_name="LIDCOVER_3_50b"
exp_dir="/vast/s219110279/TypiClust/output/TINYIMAGENET/resnet50/${exp_name}"
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
  --cfg=/vast/s219110279/TypiClust/deep-al/configs/tinyimagenet/al/RESNET50.yaml \
  --al=idprobcover \
  --exp-name="$exp_name" \
  --budget=50 \
  --initial_size=0 \
  --initial_delta=0.25 \
  --idpc_alpha=1.0 \
  --idpc_mode=high_id_more_centers \
  --idpc_k_id=50 \
  --idpc_k_knn=50 \
  --idpc_cache_root=/scratch/s219110279/idpc_cache/resnet50 \
  --seed 3

echo "done=$(date)"
