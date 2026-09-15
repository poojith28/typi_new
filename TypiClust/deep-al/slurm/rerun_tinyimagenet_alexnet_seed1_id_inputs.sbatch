#!/bin/bash
#SBATCH --job-name=ti_alex_s1_repair
#SBATCH --output=/vast/s219110279/TypiClust/output/_logs/ti-alex-s1-repair-%A_%a.out
#SBATCH --error=/vast/s219110279/TypiClust/output/_logs/ti-alex-s1-repair-%A_%a.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --array=0-2%3
#SBATCH --mail-type=END,FAIL

# Rerun only the three invalid TinyImageNet/AlexNet/seed-1 trajectories.
# Array mapping: 0=uncertainty, 1=entropy, 2=margin.

set -euo pipefail

PROJECT_ROOT=/vast/s219110279
DEEP_AL="$PROJECT_ROOT/TypiClust/deep-al"
TRAIN_SCRIPT="$DEEP_AL/tools/train_al.py"
CFG="$DEEP_AL/configs/tinyimagenet/al/ALEXNET.yaml"
VALIDATOR="$PROJECT_ROOT/analysis/validate_al_trajectory.py"
OUT_ROOT="$PROJECT_ROOT/TypiClust/output/TINYIMAGENET/alexnet"

METHODS=(uncertainty entropy margin)
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( TASK_ID < 0 || TASK_ID >= ${#METHODS[@]} )); then
  echo "Invalid array task: $TASK_ID" >&2
  exit 2
fi

method="${METHODS[$TASK_ID]}"
exp_name="${method}_1_50b"
target="$OUT_ROOT/$exp_name"

# Never merge a replacement run into old episode files.
if [[ -e "$target" ]]; then
  echo "Refusing to reuse existing target: $target" >&2
  exit 3
fi
if [[ ! -f "$CFG" || ! -f "$TRAIN_SCRIPT" || ! -f "$VALIDATOR" ]]; then
  echo "Required config/script is missing" >&2
  exit 4
fi

module purge
module load Anaconda3
# Conda activation hooks may probe unset MKL variables; suspend nounset only
# while the environment is being activated, then restore it immediately.
set +u
source activate
conda activate typiclust
set -u

export TYPI_FEATURES_ROOT="$PROJECT_ROOT/results/results"
export TYPI_FEATURE_BACKBONE=alexnet
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"

cd "$DEEP_AL/tools"

echo "Host=$(hostname) job=${SLURM_JOB_ID:-unknown} task=$TASK_ID method=$method"
echo "Target=$target"
nvidia-smi -L

python "$TRAIN_SCRIPT" \
  --cfg "$CFG" \
  --al "$method" \
  --exp-name "$exp_name" \
  --budget 50 \
  --initial_size 50 \
  --seed 1

python "$VALIDATOR" "$target" \
  --pool-size 100000 \
  --initial-size 50 \
  --batch-size 50 \
  --last-episode 100 \
  --validation-size 5000

echo "Completed and validated: $target"
