#!/bin/bash
# Submit exact post-hoc pool-reference arrays. No AL retraining is launched.
set -euo pipefail
ROOT=/vast/s219110279
SLURM_DIR="$ROOT/analysis/slurm"
LOG_DIR="$ROOT/outputs/thesis_appendix/pool_reference_robustness/reference/logs/slurm"
mkdir -p "$LOG_DIR"

for name in thesis_pool_h0 thesis_id_full thesis_id_pool; do
    if squeue -u "$USER" -h -n "$name" 2>/dev/null | grep -q .; then
        echo "Refusing duplicate submission: active job name $name" >&2
        exit 1
    fi
done

H0_JOB=$(sbatch --parsable "$SLURM_DIR/thesis_appendix_pool_h0_45.sbatch")
ID_FULL_JOB=$(sbatch --parsable "$SLURM_DIR/thesis_appendix_pool_id_full_9.sbatch")
ID_POOL_JOB=$(sbatch --parsable --dependency="afterok:${ID_FULL_JOB}" "$SLURM_DIR/thesis_appendix_pool_id_seeded_45.sbatch")

echo "Submitted exact post-hoc references:"
echo "  H0 pool-only (45):       $H0_JOB"
echo "  ID full-train (9):       $ID_FULL_JOB"
echo "  ID pool-only (45):       $ID_POOL_JOB (afterok:$ID_FULL_JOB)"
echo "Monitor: squeue -u $USER -n thesis_pool_h0,thesis_id_full,thesis_id_pool"
echo "Logs: $LOG_DIR"
