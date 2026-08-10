#!/bin/bash
#SBATCH --job-name=ma000_tin_r18_c060_s4_b50
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/scale_fine_tin_r18_c060_f040_a000_s4_b050_%j.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/scale_fine_tin_r18_c060_f040_a000_s4_b050_%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --constraint=gpu-v100|gpu-l40s
#SBATCH --exclude=v100-f-07
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
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

exp_name="MSTC_C060_F040_A000_4_50b"
summary="/vast/s219110279/TypiClust/output/TINYIMAGENET/resnet18/${exp_name}/benchmark_summary.json"
if [[ -f "${summary}" ]]; then
  echo "Skip: already complete: ${summary}"
  exit 0
fi

echo "kind=scale_fine dataset=TINYIMAGENET backbone=resnet18 seed=4"
echo "delta_coarse=0.6 delta_fine=0.4 alpha=0.0 budget=50"
echo "note: alpha=1 => coarse-only; alpha=0 => fine-only"
echo "start=$(date) node=$(hostname)"

python train_al.py \
  --cfg=/vast/s219110279/TypiClust/deep-al/configs/tinyimagenet/al/RESNET18.yaml \
  --al=multiscale_topocover \
  --exp-name="${exp_name}" \
  --budget=50 \
  --initial_size=0 \
  --seed=4 \
  --initial_delta=0.6 \
  --mstc_delta_coarse=0.6 \
  --mstc_delta_fine=0.4 \
  --mstc_alpha=0.0 \
  --mstc_graph_batch_size=500 \
  --mstc_device=cuda

echo "done=$(date)"
