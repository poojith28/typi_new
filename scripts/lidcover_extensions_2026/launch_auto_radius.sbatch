#!/usr/bin/env bash
#SBATCH --job-name=lidext-auto-v2
#SBATCH --comment=lidcover_extensions_2026:auto_radius
#SBATCH --output=/vast/s219110279/experiments/lidcover_extensions_2026/logs/auto-%A_%a.out
#SBATCH --error=/vast/s219110279/experiments/lidcover_extensions_2026/logs/auto-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-v100|gpu-l40s
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --array=0-29%15
#SBATCH --mail-type=FAIL
set -euo pipefail
bash /vast/s219110279/scripts/lidcover_extensions_2026/run_manifest_task.sh auto_radius "${SLURM_ARRAY_TASK_ID:?}"
