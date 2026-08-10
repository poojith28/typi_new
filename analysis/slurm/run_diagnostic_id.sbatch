#!/bin/bash
#SBATCH --job-name=diag_id
#SBATCH --output=diag_id_%x_%j.out
#SBATCH --error=diag_id_%x_%j.err
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s219110279@deakin.edu.au

# Intrinsic-dimensionality diagnostics (FULL pool / FULL L_t, exact kNN).
#
# Env overrides:
#   BACKBONES="resnet18"                 # default: all three
#   SKIP_FIGURES=1                       # data/CSV only
#   REFRESH_CACHE=1                      # delete/rebuild ID caches (final thesis run)
#
# Examples:
#   sbatch --export=ALL,BACKBONES=resnet18 run_diagnostic_id.sbatch
#   sbatch --export=ALL,BACKBONES=resnet18,REFRESH_CACHE=1 run_diagnostic_id.sbatch
#   sbatch --export=ALL,BACKBONES="resnet18 resnet50 alexnet" run_diagnostic_id.sbatch
#   sbatch --job-name=diag_id_r18 --export=ALL,BACKBONES=resnet18 run_diagnostic_id.sbatch

set -euo pipefail

PROJECT_ROOT=/vast/s219110279
BACKBONES="${BACKBONES:-resnet18 resnet50 alexnet}"
SKIP_FIGURES="${SKIP_FIGURES:-0}"
REFRESH_CACHE="${REFRESH_CACHE:-0}"

module purge
module load Anaconda3
source activate
# Prefer the diagnostics venv (has faiss + matplotlib); fall back to typiclust
if [[ -x "$PROJECT_ROOT/.venv_diag/bin/python" ]]; then
  PYTHON="$PROJECT_ROOT/.venv_diag/bin/python"
else
  conda activate typiclust
  PYTHON=python
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"
echo "Host=$(hostname)  CPUs=$OMP_NUM_THREADS  BACKBONES=$BACKBONES  SKIP_FIGURES=$SKIP_FIGURES  REFRESH_CACHE=$REFRESH_CACHE"
echo "Python=$PYTHON"
"$PYTHON" -c "import faiss,numpy,scipy,sklearn,matplotlib,pandas; print('deps ok', faiss.__version__)"

EXTRA=()
if [[ "$SKIP_FIGURES" == "1" ]]; then
  EXTRA+=(--skip-figures)
fi
if [[ "$REFRESH_CACHE" == "1" ]]; then
  EXTRA+=(--refresh-cache)
fi

# shellcheck disable=SC2086
"$PYTHON" analysis/generate_diagnostic_id_figures.py --backbones ${BACKBONES} "${EXTRA[@]}"

echo "Done. See outputs/diagnostic_id/diagnostic_id_generation_report.md"
