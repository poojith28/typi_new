# TypiClust Research Workspace

This repository is a working research fork that combines two related codebases:

- `scan/`: unsupervised image classification and representation learning based on SCAN
- `deep-al/`: deep active learning experiments built on top of `pycls`

The current fork also includes local extensions for geometry-aware and intrinsic-dimension-aware active learning, along with experiment scripts used for CIFAR-10, CIFAR-100, and TinyImageNet studies.

## What This Repo Contains

### `scan/`

The `scan` directory contains the SCAN pipeline for unsupervised image classification:

- pretext training with SimCLR or MoCo-style setups
- clustering with SCAN
- self-label refinement

Useful entry points:

- `scan/simclr.py`
- `scan/scan.py`
- `scan/selflabel.py`
- `scan/eval.py`

See [scan/README.md](/scratch/s219110279/TypiClust/scan/README.md) for the original SCAN usage details.

### `deep-al/`

The `deep-al` directory contains the active learning codebase and experiment runner:

- standard uncertainty and diversity baselines
- custom ProbCover variants
- reviewer and ablation scripts
- dataset/config support for CIFAR and TinyImageNet

Useful entry points:

- `deep-al/tools/train_al.py`
- `deep-al/tools/train.py`
- `deep-al/tools/analyze_id_stability.py`
- `deep-al/tools/analyze_reviewer_experiments.py`

See [deep-al/README.md](/scratch/s219110279/TypiClust/deep-al/README.md) for method-specific details.

## Custom Methods Added In This Fork

This fork includes additional research code under `deep-al/pycls/al/`, including:

- `IDProbCover.py`
- `geometry_auto_research.py`
- `adaptive_cover/`
- `idpc_tiebreak/`

These components extend the baseline active learning toolkit with:

- intrinsic-dimension-aware query selection
- geometry-aware scoring and neighborhood analysis
- adaptive cover strategies
- reviewer-oriented experiment helpers and analysis scripts

## Repository Layout

```text
TypiClust/
├── README.md
├── scan/
│   ├── configs/
│   ├── images/
│   ├── models/
│   ├── losses/
│   └── utils/
└── deep-al/
    ├── configs/
    ├── docs/
    ├── pycls/
    │   ├── al/
    │   ├── core/
    │   ├── datasets/
    │   └── models/
    └── tools/
```

## Environment Notes

The repository contains two subprojects with separate dependency sets:

- `scan/requirements.txt`
- `deep-al/requirements.txt`

Install dependencies for the part you want to run. In practice, it is usually easiest to create a dedicated environment and then install the relevant requirements from each subdirectory.

Example:

```bash
cd /scratch/s219110279/TypiClust

# for SCAN experiments
pip install -r scan/requirements.txt

# for active learning experiments
pip install -r deep-al/requirements.txt
```

## Quick Start

### Run SCAN

```bash
cd /scratch/s219110279/TypiClust/scan
python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_cifar10.yml
python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml
python selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_cifar10.yml
```

### Run Active Learning

```bash
cd /scratch/s219110279/TypiClust/deep-al
python tools/train_al.py \
  --cfg configs/cifar10/al/RESNET18.yaml \
  --exp-name cifar10_run \
  --al typiclust \
  --seed 1
```

### Run Custom IDProbCover Experiment

```bash
cd /scratch/s219110279/TypiClust/deep-al
python tools/train_al.py \
  --cfg configs/cifar100/al/RESNET18.yaml \
  --exp-name cifar100_idprobcover \
  --al id_prob_cover \
  --budget 50 \
  --initial_size 50 \
  --initial_delta 0.25 \
  --seed 1
```

## Data

Datasets are expected locally and are not meant to be versioned in git. Keep large downloaded data, extracted archives, logs, and generated experiment outputs outside commits or under ignored paths.

Before running experiments, check:

- dataset paths used by `scan/utils/mypath.py`
- environment/output settings in `scan/configs/env.yml`
- active learning dataset and config settings under `deep-al/configs/`

## Reproducibility

For cleaner experiment tracking, record at minimum:

- config file used
- random seed
- acquisition method
- budget / initial labeled size
- dataset split and feature extractor settings

Several helper scripts in `deep-al/tools/` are already included for reviewer runs, ablations, and post-hoc analysis.

## Git Backup

This repository is now connected to:

- `origin`: `https://github.com/poojith28/TypiClust.git`

Typical backup flow:

```bash
cd /scratch/s219110279/TypiClust
git status
git add -A
git commit -m "Describe your experiment changes"
git push origin main
```

## Acknowledgement

This workspace builds on the original SCAN and deep active learning repositories, with local research modifications for ongoing experimentation.
