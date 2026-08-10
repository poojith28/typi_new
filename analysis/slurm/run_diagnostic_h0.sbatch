#!/bin/bash
#SBATCH --job-name=diag_h0
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/diag_h0_%x_%j.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/diag_h0_%x_%j.err
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --qos=batch-short
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s219110279@deakin.edu.au

# H0 touched-pair diagnostics (reference PD + birth-simplex mapping).
#
# Env overrides:
#   BACKBONES="resnet18"
#   SKIP_FIGURES=1
#   REFRESH_CACHE=1
#
# Example:
#   sbatch --export=ALL,BACKBONES=resnet18 analysis/slurm/run_diagnostic_h0.sbatch

set -euo pipefail

PROJECT_ROOT=/vast/s219110279
BACKBONES="${BACKBONES:-resnet18}"
SKIP_FIGURES="${SKIP_FIGURES:-0}"
REFRESH_CACHE="${REFRESH_CACHE:-0}"

module purge
module load Anaconda3
source activate
if [[ -x "$PROJECT_ROOT/.venv_diag/bin/python" ]]; then
  PYTHON="$PROJECT_ROOT/.venv_diag/bin/python"
else
  conda activate gudhi-py39
  PYTHON=python
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/${USER:-user}-mpl-cache"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$PROJECT_ROOT/TypiClust/output/_logs"

cd "$PROJECT_ROOT"
echo "Host=$(hostname) BACKBONES=$BACKBONES SKIP_FIGURES=$SKIP_FIGURES REFRESH_CACHE=$REFRESH_CACHE"
echo "Python=$PYTHON"
"$PYTHON" -c "import numpy,scipy,sklearn,matplotlib,pandas; print('deps ok')"

EXTRA=()
if [[ "$SKIP_FIGURES" == "1" ]]; then EXTRA+=(--skip-figures); fi
if [[ "$REFRESH_CACHE" == "1" ]]; then EXTRA+=(--refresh-cache); fi

# shellcheck disable=SC2086
"$PYTHON" analysis/generate_diagnostic_h0_figures.py --backbones ${BACKBONES} "${EXTRA[@]}"

echo "Done. See outputs/diagnostic_h0/diagnostic_h0_generation_report.md"
