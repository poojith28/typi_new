#!/bin/bash
#SBATCH --job-name=nr_h0perm
#SBATCH --output=/vast/s219110279/evidence_pack/neurreps_revision_20260821/slurm_h0perm_%A_%a.out
#SBATCH --error=/vast/s219110279/evidence_pack/neurreps_revision_20260821/slurm_h0perm_%A_%a.err
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --qos=batch-short
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=1-10%2
#SBATCH --mail-type=END,FAIL

# Exact CIFAR-100/ResNet-18 H0 relabeling robustness experiment.
#
# Ten fixed permutation seeds are used.  The %2 array throttle is deliberate:
# the exact dense Prim calculation is BLAS-bound and runs faster when two jobs,
# rather than many oversubscribed jobs, share a node.  This does not train a
# classifier or launch an active-learning campaign.

set -euo pipefail

PROJECT_ROOT=/vast/s219110279
OUT_DIR="$PROJECT_ROOT/evidence_pack/neurreps_revision_20260821"
PYTHON="$PROJECT_ROOT/.conda/envs/typiclust/bin/python"
PERMUTATION_SEED=$((2026082100 + SLURM_ARRAY_TASK_ID))
CSV="$OUT_DIR/h0_permutation_reference_seed${PERMUTATION_SEED}.csv"
META="$OUT_DIR/h0_permutation_reference_seed${PERMUTATION_SEED}.json"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/${USER:-user}-nr-h0perm-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$OUT_DIR"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python environment not found: $PYTHON" >&2
    exit 2
fi

# A completed result is reusable only if its metadata confirms that the exact
# MST death multiset is unchanged.  No existing file is silently overwritten.
if [[ -s "$CSV" && -s "$META" ]]; then
    if "$PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); ok=m.get("death_count_matches_original") and m.get("death_multiset_allclose_atol_1e-6"); raise SystemExit(0 if ok else 1)' "$META"; then
        echo "Validated result already exists; skipping seed $PERMUTATION_SEED"
        exit 0
    fi
fi

if [[ -e "$CSV" || -e "$META" ]]; then
    echo "ERROR: incomplete or unvalidated output exists for seed $PERMUTATION_SEED." >&2
    echo "Quarantine it manually before resubmitting; this job will not overwrite it." >&2
    exit 3
fi

cd "$PROJECT_ROOT"
echo "Start seed=$PERMUTATION_SEED host=$(hostname) cpus=$SLURM_CPUS_PER_TASK time=$(date --iso-8601=seconds)"
"$PYTHON" analysis/neurreps_revision.py permutation \
    --out-dir "$OUT_DIR" \
    --permutation-seed "$PERMUTATION_SEED"
echo "Done seed=$PERMUTATION_SEED time=$(date --iso-8601=seconds)"
