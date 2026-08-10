#!/bin/bash
# Interactive smoke test — same style as your resnet18 one-liners.
#
#   module purge && module load Anaconda3 && source activate && conda activate typiclust
#   cd /scratch/s219110279/TypiClust/deep-al/tools   # or /vast/.../tools
#   bash /vast/s219110279/analysis/slurm/run_al_interactive_test.sh
#
# Optional: only one backbone or dataset
#   BACKBONE=resnet50 DATASETS=cifar10 bash .../run_al_interactive_test.sh

set -euo pipefail

if [[ -d /vast/s219110279/TypiClust/deep-al ]]; then
  ROOT=/vast/s219110279/TypiClust/deep-al
else
  ROOT=/scratch/s219110279/TypiClust/deep-al
fi

TOOLS="$ROOT/tools"
METHOD="${METHOD:-probcover}"
BUDGET="${BUDGET:-50}"
INITIAL_SIZE="${INITIAL_SIZE:-50}"
SEED="${SEED:-1}"
BACKBONES="${BACKBONES:-resnet50 alexnet}"
DATASETS="${DATASETS:-cifar10 cifar100}"

module purge
module load Anaconda3
source activate
conda activate typiclust

export TYPI_FEATURES_ROOT="${TYPI_FEATURES_ROOT:-/vast/s219110279/results/results}"

cd "$TOOLS"
echo "PWD=$PWD  backbones=$BACKBONES  method=$METHOD  seed=$SEED"
nvidia-smi -L || true

for BACKBONE in $BACKBONES; do
export TYPI_FEATURE_BACKBONE="$BACKBONE"

for dataset in $DATASETS; do
  case "$dataset" in
    cifar10)
      CFG="$ROOT/configs/cifar10/al/$(echo "$BACKBONE" | tr '[:lower:]' '[:upper:]').yaml"
      INITIAL_DELTA="${INITIAL_DELTA:-0.25}"
      ;;
    cifar100)
      CFG="$ROOT/configs/cifar100/al/$(echo "$BACKBONE" | tr '[:lower:]' '[:upper:]').yaml"
      INITIAL_DELTA="${INITIAL_DELTA:-0.25}"
      ;;
    tinyimagenet)
      CFG="$ROOT/configs/tinyimagenet/al/$(echo "$BACKBONE" | tr '[:lower:]' '[:upper:]').yaml"
      INITIAL_DELTA="${INITIAL_DELTA:-0.30}"
      ;;
    *)
      echo "Unknown dataset: $dataset" >&2
      exit 1
      ;;
  esac

  exp_name="${METHOD}_${SEED}_${BUDGET}b"
  idpc_cache="/vast/s219110279/idpc_cache/${BACKBONE}/${dataset}"

  echo "======== $dataset / $BACKBONE / $exp_name ========"

  cmd=(
    python train_al.py
    --cfg="$CFG"
    --al="$METHOD"
    --exp-name="$exp_name"
    --budget="$BUDGET"
    --initial_size="$INITIAL_SIZE"
    --seed "$SEED"
    --initial_delta "$INITIAL_DELTA"
  )

  if [[ "$METHOD" == "idprobcover" ]]; then
    exp_name="LIDCOVER_${SEED}_${BUDGET}b"
    cmd[5]="--exp-name=$exp_name"
    cmd+=(
      --idpc_alpha 1.0
      --idpc_mode high_id_more_centers
      --idpc_k_id 50
      --idpc_k_knn 50
      --idpc_cache_root "$idpc_cache"
    )
  fi

  printf '%q ' "${cmd[@]}"; echo
  "${cmd[@]}"
done
done

echo "Done at $(date)"
