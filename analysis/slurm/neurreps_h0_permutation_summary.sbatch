#!/bin/bash
#SBATCH --job-name=nr_h0sum
#SBATCH --output=/vast/s219110279/evidence_pack/neurreps_revision_20260821/slurm_h0summary_%j.out
#SBATCH --error=/vast/s219110279/evidence_pack/neurreps_revision_20260821/slurm_h0summary_%j.err
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --qos=batch-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL

# Submit this with an afterok dependency on neurreps_h0_permutation_array.sbatch.

set -euo pipefail

PROJECT_ROOT=/vast/s219110279
OUT_DIR="$PROJECT_ROOT/evidence_pack/neurreps_revision_20260821"
PYTHON="$PROJECT_ROOT/.conda/envs/typiclust/bin/python"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/${USER:-user}-nr-h0summary-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$OUT_DIR"

cd "$PROJECT_ROOT"
echo "Summarizing ten validated permutations on $(hostname) at $(date --iso-8601=seconds)"
"$PYTHON" analysis/neurreps_revision.py permutation-summary --out-dir "$OUT_DIR"
echo "Done at $(date --iso-8601=seconds)"
