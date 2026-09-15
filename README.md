# TypiClust Research Extensions

This repository contains image active-learning research code built around
TypiClust, SCAN representations, and the `deep-al` training toolkit. The main
additions are geometry-aware coverage methods, intrinsic-dimension-aware
sampling, topology-based variants, deterministic evaluation support, and the
analysis code used to audit experimental trajectories.

## Repository layout

- `TypiClust/deep-al/` contains the active-learning training code and tests.
- `TypiClust/scan/` contains the SCAN representation-learning pipeline.
- `analysis/` contains post-hoc analysis and validation programs.
- `configs/` contains experiment-matrix and diagnostic configuration files.
- `experiments/` contains standalone experiment extensions.
- `scripts/` contains manifest builders, validators, and SLURM launchers.

Datasets, embeddings, checkpoints, caches, logs, locks, and generated result
directories are intentionally excluded from version control.

## Requirements

The training stack was developed for Linux with NVIDIA CUDA. The pinned
`deep-al` environment uses Python 3.7, PyTorch 1.7.1, and torchvision 0.8.2.
Those versions may require an older CUDA-compatible package index or Conda
environment.

For the active-learning code:

```bash
git clone https://github.com/poojith28/typi_new.git
cd typi_new

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r TypiClust/deep-al/requirements.txt
```

Some analysis programs additionally require `pandas`, `scipy`, `PyYAML`,
`scikit-learn`, `gudhi`, and FAISS. Install only the optional packages needed
by the analysis being run. SCAN has its own historical environment listing in
`TypiClust/scan/requirements.txt`.

## Data and representations

The default active-learning configs use `DATASET.ROOT_DIR: ./data`. Run the
training command from `TypiClust/deep-al/tools`; this resolves the dataset path
to `TypiClust/data`. CIFAR-10 and CIFAR-100 are downloaded automatically by
torchvision. TinyImageNet must be downloaded and extracted separately.

Coverage methods that use precomputed representations load them through
`TypiClust/deep-al/pycls/datasets/utils/features.py`. Set these variables when
the files are not in the historical default location:

```bash
export TYPI_FEATURES_ROOT=/path/to/representation/root
export TYPI_FEATURE_BACKBONE=resnet18
```

Do not commit local datasets, embeddings, model checkpoints, or credentials.

## Configuration

Start with one of the dataset configs under
`TypiClust/deep-al/configs/<dataset>/al/`. The command-line values for the
experiment name, method, acquisition budget, initial labelled-set size, and
seed override the corresponding YAML values.

The most commonly changed YAML fields are:

- `DATASET.NAME`, `DATASET.ROOT_DIR`, and `DATASET.VAL_RATIO`
- `MODEL.TYPE` and `MODEL.NUM_CLASSES`
- `OPTIM.MAX_EPOCH`, learning rate, and weight decay
- `TRAIN.BATCH_SIZE` and `DATA_LOADER.NUM_WORKERS`
- `ACTIVE_LEARNING.MAX_ITER` and method-specific parameters

All defaults, including the `IDPC_*`, `AUTO_DELTA_*`, `TALC_*`, and topology
settings, are defined in `TypiClust/deep-al/pycls/core/config.py`.

## Run an experiment

Run from the tools directory so relative data and output paths resolve as the
training driver expects:

```bash
cd TypiClust/deep-al/tools
CUDA_VISIBLE_DEVICES=0 python train_al.py \
  --cfg ../configs/cifar10/al/RESNET18.yaml \
  --exp-name cifar10_typiclust_seed1 \
  --al typiclust \
  --budget 50 \
  --initial_size 50 \
  --seed 1
```

For automatic-radius LIDCover with an empty cold start:

```bash
CUDA_VISIBLE_DEVICES=0 python train_al.py \
  --cfg ../configs/cifar100/al/RESNET18.yaml \
  --exp-name cifar100_lidcover_auto_seed1 \
  --al lidcover_auto_delta \
  --budget 50 \
  --initial_size 0 \
  --seed 1 \
  --auto_delta_k 50 \
  --auto_delta_quantile 0.5
```

Outputs are written below `TypiClust/output/<dataset>/<model>/<experiment>`.
Use a new experiment name for each run; reusing a directory can overwrite or
mix artifacts.

## Tests

```bash
cd TypiClust/deep-al
python -m pytest -q tests

cd ../..
python -m pytest -q analysis/tests
```

The first suite covers automatic-radius selection, deterministic evaluation,
fallback behavior, explicit initial labelled sets, and TALC. Analysis scripts
often require completed experiment directories and are documented separately
in `analysis/README.md` and `README_diagnostics.md`.

## Cluster runs

SLURM scripts are provided as reproducibility references. Before submitting
them on another cluster, replace site-specific partitions, module names,
notification addresses, environment names, and filesystem roots. The reusable
extension workflow is documented in `scripts/lidcover_extensions_2026/README.md`.

## Attribution and licences

This is a research fork of existing TypiClust, SCAN, and deep active-learning
implementations. Preserve their original notices and cite the upstream work
when using the code in research. The active-learning code is distributed under
the MIT licence in `TypiClust/deep-al/LICENSE`; the SCAN subtree retains its
separate CC BY-NC 4.0 licence in `TypiClust/scan/LICENSE`.

Method-specific notes and upstream citations are available in
`TypiClust/README.md`, `TypiClust/deep-al/README.md`, and
`TypiClust/deep-al/pycls/al/TALC.md`.
