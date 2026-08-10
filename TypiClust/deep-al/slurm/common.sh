# Shared setup for CIFAR10 active-learning Slurm jobs.
# Sourced by every sbatch script in this directory.

set -euo pipefail

# Prefer /vast if the project lives there (same tree as /scratch on this cluster).
if [[ -d /vast/s219110279/TypiClust/deep-al ]]; then
  PROJECT_ROOT=/vast/s219110279
else
  PROJECT_ROOT=/scratch/s219110279
fi

DEEP_AL="$PROJECT_ROOT/TypiClust/deep-al"
TOOLS="$DEEP_AL/tools"
LOG_DIR="$PROJECT_ROOT/TypiClust/output/_logs"
mkdir -p "$LOG_DIR"

module purge
module load Anaconda3
source activate
conda activate typiclust

export TYPI_FEATURES_ROOT="${TYPI_FEATURES_ROOT:-$PROJECT_ROOT/results/results}"

run_train_al() {
  local backbone="$1"
  local method="$2"
  local seed="$3"
  local exp_name="$4"
  shift 4

  export TYPI_FEATURE_BACKBONE="$backbone"

  local cfg="$DEEP_AL/configs/cifar10/al/$(echo "$backbone" | tr '[:lower:]' '[:upper:]').yaml"
  local idpc_cache="${IDPC_CACHE_ROOT:-$PROJECT_ROOT/idpc_cache/$backbone}"

  local cmd=(
    python "$TOOLS/train_al.py"
    --cfg "$cfg"
    --al "$method"
    --exp-name "$exp_name"
    --budget "${BUDGET:-50}"
    --initial_size "${INITIAL_SIZE:-50}"
    --seed "$seed"
  )

  case "$method" in
    probcover|idprobcover|knn_distance_cover|density_cover|distance_variance_cover|distance_cv_cover)
      cmd+=(--initial_delta "${INITIAL_DELTA:-0.25}")
      ;;
  esac

  if [[ "$method" == "idprobcover" ]]; then
    cmd+=(
      --idpc_alpha 1.0
      --idpc_mode high_id_more_centers
      --idpc_k_id 50
      --idpc_k_knn 50
      --idpc_cache_root "$idpc_cache"
    )
  fi

  if [[ "$method" == "knn_distance_cover" || "$method" == "density_cover" || "$method" == "distance_variance_cover" || "$method" == "distance_cv_cover" ]]; then
    cmd+=(
      --arc_alpha 1.0
      --arc_k_signal 50
      --arc_k_knn 50
      --arc_cache_root "$idpc_cache/adaptive_cover"
    )
  fi

  echo ">>> $exp_name  (backbone=$backbone, method=$method, seed=$seed)"
  "${cmd[@]}"
}
