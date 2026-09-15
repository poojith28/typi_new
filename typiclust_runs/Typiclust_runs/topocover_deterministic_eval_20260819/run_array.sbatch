#!/bin/bash
#SBATCH --job-name=tc_deteval
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/tc_deteval_%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/tc_deteval_%A_%a.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --constraint=gpu-v100|gpu-l40s
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --array=0-44%20

set -eo pipefail
module purge
module load Anaconda3
source activate
conda activate typiclust
set -u

campaign=/vast/s219110279/typiclust_runs/Typiclust_runs/topocover_deterministic_eval_20260819
row=$(awk -F '\t' -v id="$SLURM_ARRAY_TASK_ID" 'NR>1 && $1==id {print; exit}' "$campaign/jobs.tsv")
if [[ -z "$row" ]]; then
  echo "ERROR: no manifest row for task $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi
IFS=$'\t' read -r task_id dataset dataset_dir config_dir backbone model_config seed <<< "$row"

export TYPI_FEATURES_ROOT=/vast/s219110279/results/results
export TYPI_FEATURE_BACKBONE="$backbone"
export PYTHONHASHSEED="$seed"

if nvidia-smi -L | grep -Eiq 'Blackwell|RTX PRO 6000'; then
  echo "ERROR: unsupported Blackwell GPU" >&2
  exit 75
fi

exp_name="TOPOCOVER_DETEVAL_D070_${seed}_50b"
summary="/vast/s219110279/TypiClust/output/${dataset}/${backbone}/${exp_name}/benchmark_summary.json"
if [[ -f "$summary" ]]; then
  echo "Skip complete $summary"
  exit 0
fi

cd /vast/s219110279/TypiClust/deep-al/tools
python train_al.py \
  --cfg="/vast/s219110279/TypiClust/deep-al/configs/${config_dir}/al/${model_config}.yaml" \
  --al=topocover \
  --exp-name="$exp_name" \
  --budget=50 \
  --initial_size=0 \
  --seed="$seed" \
  --initial_delta=0.70 \
  --topocover_k_knn=50 \
  --topocover_cache_root="/scratch/s219110279/topocover_cache_deteval/${backbone}/${dataset_dir}" \
  --topocover_recompute_components=True
