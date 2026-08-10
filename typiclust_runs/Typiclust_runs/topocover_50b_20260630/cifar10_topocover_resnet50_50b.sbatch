#!/bin/bash
#SBATCH --job-name=topo_cif_r50
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/topo_cif_r50_%j.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/topo_cif_r50_%j.err
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

cd /vast/s219110279/TypiClust/deep-al/tools

nvidia-smi -L || true

for seed in 1 2 3 4 5; do
  exp_name="TOPOCOVER_PAPER_${seed}_50b"
  summary="/vast/s219110279/TypiClust/output/CIFAR10/resnet50/${exp_name}/benchmark_summary.json"
  if [[ -f "$summary" ]]; then
    echo "Skip: already complete: $summary"
    continue
  fi

  echo "============================================================"
  echo "dataset=CIFAR10 backbone=resnet50 seed=$seed exp=$exp_name"
  echo "start=$(date) node=$(hostname)"
  echo "============================================================"

  python train_al.py \
    --cfg=/vast/s219110279/TypiClust/deep-al/configs/cifar10/al/RESNET50.yaml \
    --al=topocover \
    --exp-name="$exp_name" \
    --budget=50 \
    --initial_size=0 \
    --seed "$seed" \
    --initial_delta=0.25 \
    --topocover_k_knn=50 \
    --topocover_cache_root=/scratch/s219110279/topocover_cache/resnet50/cifar10 \
    --topocover_recompute_components=True
done

echo "done=$(date)"
