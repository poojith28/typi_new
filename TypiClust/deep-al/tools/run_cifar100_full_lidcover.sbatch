#!/bin/bash

#SBATCH --job-name=c100_al
#SBATCH --output=/scratch/s219110279/TypiClust/slurm-cifar100-%A_%a.out
#SBATCH --error=/scratch/s219110279/TypiClust/slurm-cifar100-%A_%a.err
#SBATCH --partition=gpu-large
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s219110279@deakin.edu.au
#SBATCH --array=0-59%2

set -euo pipefail

ROOT="/scratch/s219110279/TypiClust/deep-al"
TRAIN_SCRIPT="$ROOT/tools/train_al.py"
CFG="$ROOT/configs/cifar100/al/RESNET18.yaml"
OUT_ROOT="/scratch/s219110279/TypiClust/output/CIFAR100/resnet18"

SEEDS=(1 2 3 4 5)
METHODS=(
  random
  uncertainty
  entropy
  margin
  coreset
  dbal
  probcover
  knn_distance_cover
  density_cover
  distance_variance_cover
  distance_cv_cover
  idprobcover
)

BUDGET="${BUDGET:-50}"
INITIAL_SIZE="${INITIAL_SIZE:-50}"
INITIAL_DELTA="${INITIAL_DELTA:-0.25}"
IDPC_CACHE_ROOT="${IDPC_CACHE_ROOT:-/scratch/s219110279/idpc_cache}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
NUM_METHODS="${#METHODS[@]}"
SEED_INDEX="$((TASK_ID / NUM_METHODS))"
METHOD_INDEX="$((TASK_ID % NUM_METHODS))"

seed="${SEEDS[$SEED_INDEX]}"
method="${METHODS[$METHOD_INDEX]}"
exp_method="$method"
if [[ "$method" == "idprobcover" ]]; then
  exp_method="LIDCOVER"
fi
exp_name="cifar100_${exp_method}_s${seed}"

summary_path="$OUT_ROOT/$exp_name/benchmark_summary.json"
if [[ -f "$summary_path" ]]; then
  echo "Skipping completed run: $exp_name"
  echo "Found: $summary_path"
  exit 0
fi

module purge
module load Anaconda3
source activate
conda activate typiclust

cd "$ROOT"

cmd=(
  python "$TRAIN_SCRIPT"
  --cfg "$CFG"
  --exp-name "$exp_name"
  --al "$method"
  --budget "$BUDGET"
  --initial_size "$INITIAL_SIZE"
  --seed "$seed"
)

if [[ "$method" == "probcover" || "$method" == "idprobcover" || "$method" == "knn_distance_cover" || "$method" == "density_cover" || "$method" == "distance_variance_cover" || "$method" == "distance_cv_cover" ]]; then
  cmd+=(--initial_delta "$INITIAL_DELTA")
fi

if [[ "$method" == "idprobcover" ]]; then
  cmd+=(
    --idpc_alpha 1.0
    --idpc_mode high_id_more_centers
    --idpc_k_id 50
    --idpc_k_knn 50
    --idpc_cache_root "$IDPC_CACHE_ROOT"
  )
fi

if [[ "$method" == "knn_distance_cover" || "$method" == "density_cover" || "$method" == "distance_variance_cover" || "$method" == "distance_cv_cover" ]]; then
  cmd+=(
    --arc_alpha 1.0
    --arc_k_signal 50
    --arc_k_knn 50
    --arc_cache_root "$IDPC_CACHE_ROOT/adaptive_cover"
  )
fi

echo "Running $exp_name"
echo "Method: $method"
echo "Seed: $seed"
echo "Command: ${cmd[*]}"
"${cmd[@]}"
