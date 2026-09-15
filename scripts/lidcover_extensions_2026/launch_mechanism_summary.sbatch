#!/usr/bin/env bash
#SBATCH --job-name=lidext-mech-summary
#SBATCH --comment=lidcover_extensions_2026:mechanism_summary
#SBATCH --output=/vast/s219110279/experiments/lidcover_extensions_2026/logs/mechanism-summary-%j.out
#SBATCH --error=/vast/s219110279/experiments/lidcover_extensions_2026/logs/mechanism-summary-%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --qos=batch-short
#SBATCH --mail-type=FAIL
set -euo pipefail
module purge
module load Anaconda3
set +u
source activate
conda activate typiclust
set -u
export MPLCONFIGDIR=/vast/s219110279/experiments/lidcover_extensions_2026/matplotlib-cache
python /vast/s219110279/analysis/lidcover_extensions_2026/summarize_mechanism.py
