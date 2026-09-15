#!/bin/bash
#SBATCH --job-name=aulc_correct
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/aulc-correct-%j.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/aulc-correct-%j.err
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --qos=batch-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_ROOT=/vast/s219110279
PYTHON="$PROJECT_ROOT/.conda/envs/typiclust/bin/python"
INPUT_ROOT="${INPUT_ROOT:-$PROJECT_ROOT/TypiClust/output}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/aulc_trapezoidal_corrected}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$PROJECT_ROOT"
echo "Starting corrected AULC generation on $(hostname) at $(date --iso-8601=seconds)"
echo "Input:  $INPUT_ROOT"
echo "Output: $OUTPUT_DIR"

"$PYTHON" analysis/generate_corrected_aulc_tables.py \
  --input-root "$INPUT_ROOT" \
  --output-dir "$OUTPUT_DIR"

echo "Finished at $(date --iso-8601=seconds)"
