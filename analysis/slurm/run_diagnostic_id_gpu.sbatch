#!/bin/bash
#SBATCH --job-name=diag_id_gpu
#SBATCH --output=diag_id_gpu_%j.out
#SBATCH --error=diag_id_gpu_%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --constraint=gpu-v100|gpu-l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s219110279@deakin.edu.au

# GPU partition variant. Current .venv_diag has CPU FAISS only — the GPU is
# mostly for node access / memory; kNN still runs on CPU (exact Flat L2).
# Prefer the CPU script unless the CPU partition is saturated.
#
#   sbatch --export=ALL,BACKBONES=resnet18 run_diagnostic_id_gpu.sbatch

set -euo pipefail

PROJECT_ROOT=/vast/s219110279
BACKBONES="${BACKBONES:-resnet18 resnet50 alexnet}"
SKIP_FIGURES="${SKIP_FIGURES:-0}"

module purge
module load Anaconda3
source activate

if [[ -x "$PROJECT_ROOT/.venv_diag/bin/python" ]]; then
  PYTHON="$PROJECT_ROOT/.venv_diag/bin/python"
else
  conda activate typiclust
  PYTHON=python
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"
nvidia-smi -L || true
echo "Host=$(hostname)  BACKBONES=$BACKBONES"

EXTRA=()
if [[ "$SKIP_FIGURES" == "1" ]]; then
  EXTRA+=(--skip-figures)
fi

# shellcheck disable=SC2086
"$PYTHON" analysis/generate_diagnostic_id_figures.py --backbones ${BACKBONES} "${EXTRA[@]}"

echo "Done. See outputs/diagnostic_id/diagnostic_id_generation_report.md"
