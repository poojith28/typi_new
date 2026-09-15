#!/bin/bash
#SBATCH --job-name=thesis_matched_start
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/thesis_matched_start_%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/thesis_matched_start_%A_%a.err
#SBATCH --array=0-14%15
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --constraint=gpu-v100|gpu-l40s
#SBATCH --exclude=v100-f-07
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --mail-type=END,FAIL

set -eo pipefail

manifest=/vast/s219110279/analysis/slurm/thesis_appendix_matched_start_15.tsv
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
line_number=$((task_id + 2))
job_line=$(sed -n "${line_number}p" "${manifest}")
if [[ -z "${job_line}" ]]; then
  echo "ERROR: no manifest row for array task ${task_id}" >&2
  exit 2
fi

IFS=$'\t' read -r manifest_task method experiment_id seed init_lset expected_sha256 <<< "${job_line}"
if [[ "${manifest_task}" != "${task_id}" ]]; then
  echo "ERROR: manifest task ${manifest_task} does not match array task ${task_id}" >&2
  exit 2
fi

target_dir=/vast/s219110279/TypiClust/output/CIFAR100/resnet18/${experiment_id}
if [[ -e "${target_dir}" ]]; then
  echo "ERROR: refusing to reuse or resume existing target: ${target_dir}" >&2
  exit 3
fi
if [[ ! -f "${init_lset}" ]]; then
  echo "ERROR: initial labelled-set file absent: ${init_lset}" >&2
  exit 4
fi
actual_sha256=$(sha256sum "${init_lset}" | awk '{print $1}')
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
  echo "ERROR: initial labelled-set SHA256 mismatch: expected=${expected_sha256} actual=${actual_sha256}" >&2
  exit 5
fi

case "${method}" in
  ProbCover)
    sampling_fn=probcover
    method_args=(--initial_delta=0.60)
    ;;
  LIDCover)
    sampling_fn=idprobcover
    method_args=(
      --initial_delta=0.25
      --idpc_alpha=1.0
      --idpc_mode=high_id_more_centers
      --idpc_k_id=50
      --idpc_k_knn=50
      --idpc_eps=1e-8
    )
    ;;
  MSTC)
    sampling_fn=multiscale_topocover
    method_args=(
      --initial_delta=0.70
      --mstc_delta_coarse=0.70
      --mstc_delta_fine=0.50
      --mstc_alpha=0.70
      --mstc_graph_batch_size=500
      --mstc_device=cuda
    )
    ;;
  *)
    echo "ERROR: unsupported manifest method: ${method}" >&2
    exit 6
    ;;
esac

echo "experiment_id=${experiment_id} method=${method} seed=${seed}"
echo "initialisation=seed_specific_shared_random_50 source=${init_lset} sha256=${actual_sha256}"
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
  echo "ERROR: unsupported Blackwell GPU; no experiment was intentionally started" >&2
  exit 75
fi

sha256sum train_al.py ../pycls/datasets/initial_lset.py ../pycls/al/IDProbCover.py ../pycls/al/prob_cover.py ../pycls/al/multiscale_topocover.py

python train_al.py \
  --cfg=/vast/s219110279/TypiClust/deep-al/configs/cifar100/al/RESNET18.yaml \
  --al="${sampling_fn}" \
  --exp-name="${experiment_id}" \
  --budget=50 \
  --initial_size=0 \
  --init_lset_path="${init_lset}" \
  --seed="${seed}" \
  "${method_args[@]}"

echo "done=$(date --iso-8601=seconds)"
