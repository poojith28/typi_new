#!/bin/bash
#SBATCH --job-name=typi-missing
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/missing_%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/missing_%A_%a.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --constraint=gpu-v100|gpu-l40s
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s219110279@deakin.edu.au

set -eo pipefail

MANIFEST="${MANIFEST:-/vast/s219110279/typiclust_runs/Typiclust_runs/missing_runs.tsv}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

module purge
module load Anaconda3
source activate
# Some conda activation hooks read unset variables; keep nounset off until
# typiclust is active, then restore it for the actual job body.
conda activate typiclust
set -u

export TYPI_FEATURES_ROOT="${TYPI_FEATURES_ROOT:-/vast/s219110279/results/results}"

cd /scratch/s219110279/TypiClust/deep-al/tools

line="$(awk -v n="$((TASK_ID + 2))" 'NR == n { print; exit }' "$MANIFEST")"
if [[ -z "$line" ]]; then
  echo "No manifest row for array task $TASK_ID"
  exit 0
fi

IFS=$'\t' read -r folder script dataset backbone exp_name command <<< "$line"

export TYPI_FEATURE_BACKBONE="$backbone"

echo "============================================================"
echo "task     : $TASK_ID"
echo "folder   : $folder"
echo "script   : $script"
echo "dataset  : $dataset"
echo "backbone : $backbone"
echo "exp_name : $exp_name"
echo "features : $TYPI_FEATURES_ROOT/$backbone"
echo "node     : $(hostname)"
echo "start    : $(date)"
echo "============================================================"

nvidia-smi -L || true
if nvidia-smi -L | grep -Eiq 'Blackwell|RTX PRO 6000'; then
  echo "ERROR: Job landed on Blackwell GPU. Requeue on V100/L40S." >&2
  exit 2
fi

summary="/vast/s219110279/TypiClust/output/$dataset/$backbone/$exp_name/benchmark_summary.json"
if [[ -f "$summary" ]]; then
  echo "Skip: already complete: $summary"
  exit 0
fi

echo "Command: $command"
eval "$command"

echo "done at $(date)"
