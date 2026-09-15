#!/usr/bin/env bash
#SBATCH --job-name=lidext-asr-v3
#SBATCH --comment=lidcover_extensions_2026:auto_sparse_radius_v3
#SBATCH --output=/vast/s219110279/experiments/lidcover_extensions_2026/logs/asr-v3-%A_%a.out
#SBATCH --error=/vast/s219110279/experiments/lidcover_extensions_2026/logs/asr-v3-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-v100|gpu-l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --array=0-29%10
#SBATCH --mail-type=FAIL
set -euo pipefail
bash /vast/s219110279/scripts/lidcover_extensions_2026/run_auto_sparse_radius_v3_task.sh "${SLURM_ARRAY_TASK_ID:?}"
