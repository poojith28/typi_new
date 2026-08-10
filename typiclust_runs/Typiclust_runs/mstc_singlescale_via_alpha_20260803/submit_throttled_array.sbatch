#!/bin/bash
#SBATCH --job-name=mstc_a_pool
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/mstc_alpha_array_%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/mstc_alpha_array_%A_%a.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --constraint=gpu-v100|gpu-l40s
#SBATCH --exclude=v100-f-07
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=s219110279@deakin.edu.au
# Full grid; at most 20 concurrent. Next task starts on finish OR fail.
#SBATCH --array=0-159%20

set -eo pipefail

module purge
module load Anaconda3
source activate
conda activate typiclust
set -u

ROOT="/vast/s219110279/typiclust_runs/Typiclust_runs/mstc_singlescale_via_alpha_20260803"
INDEX="$ROOT/job_index.txt"
SCRIPT_NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$INDEX")
SCRIPT_PATH="$ROOT/$SCRIPT_NAME"

echo "array_task=$SLURM_ARRAY_TASK_ID / 159 script=$SCRIPT_NAME"
echo "start=$(date) node=$(hostname)"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "ERROR: missing $SCRIPT_PATH" >&2
  exit 2
fi

bash "$SCRIPT_PATH"
status=$?
echo "done=$(date) status=$status"
exit $status
