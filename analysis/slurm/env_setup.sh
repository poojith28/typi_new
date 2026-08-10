# Sourced by every analysis sbatch file.
# Cluster setup (matches the Deakin/A2I2 Slurm recipe):
#   module purge
#   module load Anaconda3
#   source activate
#   conda activate gudhi-py39

set -euo pipefail

module purge
module load Anaconda3
source activate
conda activate gudhi-py39

# Resolve env root after activation (home and /vast/... point to the same env).
export CONDA_PREFIX="${CONDA_PREFIX:-/home/s219110279/.conda/envs/gudhi-py39}"
export PY="${CONDA_PREFIX}/bin/python"

# GUDHI / shared-library paths (needed for the pip wheel's bundled libs on compute nodes).
export GUDHI_PREFIX="${CONDA_PREFIX}"
export GUDHI_PYTHON="${CONDA_PREFIX}/lib/python3.9/site-packages/gudhi"
_gudhi_libs="${CONDA_PREFIX}/lib/python3.9/site-packages/gudhi.libs"
if [[ -d "$_gudhi_libs" ]]; then
    export LD_LIBRARY_PATH="${_gudhi_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export MPLCONFIGDIR="/tmp/${USER:-user}-mpl-cache"
mkdir -p "$MPLCONFIGDIR"

# Match Slurm's CPU allocation so openTSNE / UMAP don't oversubscribe.
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
    export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
    export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
    export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
fi

# Project paths.
export ANALYSIS_DIR=/vast/s219110279/analysis
export EMBED_ROOT=/vast/s219110279/results/results
export OUTPUT_ROOT=/vast/s219110279/TypiClust/output
export RAW_DATA_ROOT=/vast/s219110279/TypiClust/scan/datasets

# Quick sanity check — fails the job early if the env is broken.
"$PY" -c "import gudhi, umap, openTSNE; print('env ok:', gudhi.__version__, umap.__version__, openTSNE.__version__)"

# Backbones x datasets enumeration used by job arrays. Keep this list in sync
# everywhere; the array indexing assumes 3*3 = 9 combinations in this order.
BACKBONES=(alexnet resnet18 resnet50)
DATASETS=(cifar-10 cifar-100 tiny-imagenet)

declare -A DATASET_TAG=(
    [cifar-10]=CIFAR10
    [cifar-100]=CIFAR100
    [tiny-imagenet]=TINYIMAGENET
)
declare -A DATASET_OUTDIR=(
    [cifar-10]=CIFAR10
    [cifar-100]=CIFAR100
    [tiny-imagenet]=TINYIMAGENET
)

decode_combo() {
    local i="$1"
    local n_datasets=${#DATASETS[@]}
    BACKBONE="${BACKBONES[$((i / n_datasets))]}"
    DATASET="${DATASETS[$((i % n_datasets))]}"
    DATASET_TAG_VAL="${DATASET_TAG[$DATASET]}"
    OUTDIR_NAME="${DATASET_OUTDIR[$DATASET]}"
}

print_combo() {
    echo "================================================================"
    echo "  job        : ${SLURM_JOB_NAME:-?} (id=${SLURM_JOB_ID:-?})"
    echo "  array      : ${SLURM_ARRAY_TASK_ID:-?} / ${SLURM_ARRAY_TASK_MAX:-?}"
    echo "  backbone   : ${BACKBONE:-?}"
    echo "  dataset    : ${DATASET:-?} -> ${DATASET_TAG_VAL:-?}"
    echo "  python     : $PY"
    echo "  gudhi      : ${GUDHI_PYTHON}"
    echo "  cpus       : ${SLURM_CPUS_PER_TASK:-?}"
    echo "  node       : $(hostname)"
    echo "  start      : $(date)"
    echo "================================================================"
}
