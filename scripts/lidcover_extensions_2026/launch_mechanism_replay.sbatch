#!/usr/bin/env bash
#SBATCH --job-name=lidext-mech-replay
#SBATCH --comment=lidcover_extensions_2026:mechanism_replay
#SBATCH --output=/vast/s219110279/experiments/lidcover_extensions_2026/logs/mechanism-replay-%A_%a.out
#SBATCH --error=/vast/s219110279/experiments/lidcover_extensions_2026/logs/mechanism-replay-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-v100|gpu-l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --qos=batch-short
#SBATCH --array=0-4%5
#SBATCH --mail-type=FAIL
set -euo pipefail
module purge
module load Anaconda3
set +u
source activate
conda activate typiclust
set -u
export MPLCONFIGDIR=/vast/s219110279/experiments/lidcover_extensions_2026/matplotlib-cache
python /vast/s219110279/analysis/lidcover_extensions_2026/mechanism_per_seed.py --seed "$((SLURM_ARRAY_TASK_ID + 1))"
