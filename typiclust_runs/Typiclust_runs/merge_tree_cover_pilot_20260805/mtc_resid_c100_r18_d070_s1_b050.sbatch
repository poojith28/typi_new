#!/bin/bash
#SBATCH --job-name=mtc_resid_c100_s1
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/mtc_resid_c100_r18_d070_s1_b050_%j.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/mtc_resid_c100_r18_d070_s1_b050_%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --constraint=gpu-v100|gpu-l40s
#SBATCH --exclude=v100-f-07
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
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
  echo "ERROR: Blackwell GPU" >&2
  exit 2
fi

exp_name="MERGE_TREE_COVER_RESIDUAL_D070_1_50b"
summary="/vast/s219110279/TypiClust/output/CIFAR100/resnet18/${exp_name}/benchmark_summary.json"
if [[ -f "$summary" ]]; then
  echo "Skip: already complete: $summary"
  exit 0
fi

echo "al=merge_tree_cover residual rebuild-every-pick point_lambda=0 fallback=OFF exp=$exp_name"
echo "start=$(date) node=$(hostname)"

python train_al.py \
  --cfg=/vast/s219110279/TypiClust/deep-al/configs/cifar100/al/RESNET18.yaml \
  --al=merge_tree_cover \
  --exp-name="$exp_name" \
  --budget=50 \
  --initial_size=0 \
  --seed=1 \
  --initial_delta=0.70 \
  --mtc_cache_root=/scratch/s219110279/merge_tree_cover_cache/resnet18/cifar100 \
  --mtc_k_knn=50 \
  --mtc_min_component_size=5 \
  --mtc_point_lambda=0.0 \
  --mtc_radius_min=0.05 \
  --mtc_radius_max=0.70 \
  --mtc_probcover_fallback=False

echo "done=$(date)"
