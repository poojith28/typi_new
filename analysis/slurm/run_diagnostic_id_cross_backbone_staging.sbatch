#!/bin/bash
#SBATCH --job-name=diag_id_xbb
#SBATCH --output=/vast/s219110279/outputs/diagnostic_id_cross_backbone_staging/_logs/diag-id-%A_%a.out
#SBATCH --error=/vast/s219110279/outputs/diagnostic_id_cross_backbone_staging/_logs/diag-id-%A_%a.err
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --array=0-1%2
#SBATCH --mail-type=END,FAIL

# Exact full-pool/full-L_t intrinsic-dimension diagnostics only.
# No active-learning or classifier training is performed.
# Array mapping: 0=alexnet, 1=resnet50.

set -euo pipefail

PROJECT_ROOT=/vast/s219110279
RUNNER="$PROJECT_ROOT/analysis/run_diagnostic_id_cross_backbone_staging.py"
PYTHON="$PROJECT_ROOT/.venv_diag/bin/python"
BACKBONES=(alexnet resnet50)
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

if (( TASK_ID < 0 || TASK_ID >= ${#BACKBONES[@]} )); then
  echo "Invalid array task: $TASK_ID" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" || ! -f "$RUNNER" ]]; then
  echo "Required Python environment or runner is missing" >&2
  exit 3
fi

backbone="${BACKBONES[$TASK_ID]}"

module purge
module load Anaconda3
set +u
source activate
set -u

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PROJECT_ROOT/outputs/diagnostic_id_cross_backbone_staging/_mplconfig"

mkdir -p "$MPLCONFIGDIR"
cd "$PROJECT_ROOT"

echo "Host=$(hostname) job=${SLURM_JOB_ID:-unknown} task=$TASK_ID backbone=$backbone CPUs=$OMP_NUM_THREADS"
"$PYTHON" -c "import faiss,numpy,scipy,sklearn,matplotlib,pandas; print('deps ok; faiss', faiss.__version__)"
"$PYTHON" "$RUNNER" --backbone "$backbone"

echo "Completed: $backbone"
echo "Outputs: $PROJECT_ROOT/outputs/diagnostic_id_cross_backbone_staging/by_backbone/$backbone"
echo "Figures: $PROJECT_ROOT/figures/diagnostic_id_cross_backbone_staging/by_backbone/$backbone"
