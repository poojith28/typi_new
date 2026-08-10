#!/bin/bash
#SBATCH --job-name=wtopo_c100_r18_s1
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/wtopo_c100_r18_d070_s1_b050_%j.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/wtopo_c100_r18_d070_s1_b050_%j.err
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
  echo "ERROR: unsupported Blackwell GPU" >&2
  exit 2
fi

exp_name="WEIGHTED_TOPOCOVER_D070_1_50b"
summary="/vast/s219110279/TypiClust/output/CIFAR100/resnet18/${exp_name}/benchmark_summary.json"
if [[ -f "${summary}" ]]; then
  echo "Skip: already complete: ${summary}"
  exit 0
fi

echo "al=weighted_topocover exp=${exp_name} seed=1 delta=0.70"
echo "start=$(date) node=$(hostname)"

extra_args=()
if [[ "weighted_topocover" == "phased_topocover" ]]; then
  extra_args+=(--phased_topo_switch_labeled=1000)
fi

python train_al.py \
  --cfg=/vast/s219110279/TypiClust/deep-al/configs/cifar100/al/RESNET18.yaml \
  --al=weighted_topocover \
  --exp-name="${exp_name}" \
  --budget=50 \
  --initial_size=0 \
  --seed=1 \
  --initial_delta=0.70 \
  --topocover_k_knn=50 \
  --topocover_cache_root=/scratch/s219110279/topocover_cache_variants/resnet18/cifar100 \
  --topocover_recompute_components=True \
  "${extra_args[@]}"

echo "done=$(date)"
