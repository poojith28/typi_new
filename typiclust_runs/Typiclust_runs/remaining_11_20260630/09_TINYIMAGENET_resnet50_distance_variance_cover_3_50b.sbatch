#!/bin/bash
#SBATCH --job-name=rem_09_res
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/remaining11_09_%j.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/remaining11_09_%j.err
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

summary=/vast/s219110279/TypiClust/output/TINYIMAGENET/resnet50/distance_variance_cover_3_50b/benchmark_summary.json
if [[ -f "$summary" ]]; then
  echo "Skip: already complete: $summary"
  exit 0
fi

echo "folder=tinyimagenet_full_50b script=distance_variance_cover.sh dataset=TINYIMAGENET backbone=resnet50 exp=distance_variance_cover_3_50b"
echo "start=$(date) node=$(hostname)"
nvidia-smi -L || true
echo "Command: python train_al.py --cfg=/scratch/s219110279/TypiClust/deep-al/configs/tinyimagenet/al/RESNET50.yaml --al=distance_variance_cover --exp-name=distance_variance_cover_3_50b --budget=50 --initial_size=0 --initial_delta=0.25 --arc_alpha=1.0 --arc_k_signal=50 --arc_k_knn=50 --arc_cache_root=/scratch/s219110279/idpc_cache/resnet50/adaptive_cover --seed 3"
python train_al.py --cfg=/scratch/s219110279/TypiClust/deep-al/configs/tinyimagenet/al/RESNET50.yaml --al=distance_variance_cover --exp-name=distance_variance_cover_3_50b --budget=50 --initial_size=0 --initial_delta=0.25 --arc_alpha=1.0 --arc_k_signal=50 --arc_k_knn=50 --arc_cache_root=/scratch/s219110279/idpc_cache/resnet50/adaptive_cover --seed 3
echo "done=$(date)"
