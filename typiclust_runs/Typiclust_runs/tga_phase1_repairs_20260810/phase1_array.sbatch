#!/bin/bash
#SBATCH --job-name=tga_p1_repair
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/tga_p1_repair_%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/tga_p1_repair_%A_%a.err
#SBATCH --array=0-3%20
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

set -eo pipefail

campaign_root=/vast/s219110279/typiclust_runs/Typiclust_runs/tga_phase1_repairs_20260810
jobs_file=${campaign_root}/jobs.tsv
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
line_number=$((task_id + 2))
job_line=$(sed -n "${line_number}p" "${jobs_file}")

if [[ -z "${job_line}" ]]; then
  echo "ERROR: no manifest row for task ${task_id}" >&2
  exit 2
fi

IFS=$'\t' read -r manifest_task experiment_id dataset backbone seed alpha delta_coarse delta_fine active_control preserved_source <<< "${job_line}"
if [[ "${manifest_task}" != "${task_id}" ]]; then
  echo "ERROR: manifest task ${manifest_task} does not match array task ${task_id}" >&2
  exit 2
fi

output_root=/vast/s219110279/TypiClust/output
target_dir=${output_root}/${dataset}/${backbone}/${experiment_id}
if [[ -e "${target_dir}" ]]; then
  echo "ERROR: unique target already exists; refusing reuse: ${target_dir}" >&2
  exit 3
fi

echo "experiment_id=${experiment_id}"
echo "dataset=${dataset} backbone=${backbone} seed=${seed} batch=50"
echo "alpha=${alpha} delta_coarse=${delta_coarse} delta_fine=${delta_fine}"
echo "scientifically_active_control=${active_control}"
echo "preserved_partial_source=${preserved_source}"
echo "target_dir=${target_dir}"
echo "start=$(date --iso-8601=seconds) node=$(hostname) slurm_job=${SLURM_ARRAY_JOB_ID}_${task_id}"

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
  echo "ERROR: unsupported Blackwell GPU; infrastructure failure, safe to retry once under a new experiment ID" >&2
  exit 75
fi

python train_al.py \
  --cfg=/vast/s219110279/TypiClust/deep-al/configs/tinyimagenet/al/RESNET18.yaml \
  --al=multiscale_topocover \
  --exp-name="${experiment_id}" \
  --budget=50 \
  --initial_size=0 \
  --seed="${seed}" \
  --initial_delta="${delta_coarse}" \
  --mstc_delta_coarse="${delta_coarse}" \
  --mstc_delta_fine="${delta_fine}" \
  --mstc_alpha="${alpha}" \
  --mstc_graph_batch_size=500 \
  --mstc_device=cuda

echo "done=$(date --iso-8601=seconds)"
