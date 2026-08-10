# Thesis analysis toolkit

Location: `/vast/s219110279/analysis/` (scripts) and `/vast/s219110279/analysis/slurm/` (batch jobs).

Companion scripts for the active-learning experiments. Everything
here reads the same on-disk conventions as the rest of the codebase:

  * **features**: a `[N, D]` float `.npy` (e.g. `scan/results/cifar-10/pretext/features_seed1.npy`)
  * **labels**: a `[N]` int `.npy`
  * **AL outputs**: `output/<DATASET>/<MODEL>/<STRATEGY>_<SEED>_<BUDGET>/episode_<r>/{lSet,uSet,activeSet}.npy`

The toolkit lets you build the same figures and tables for two regimes:

  1. **raw** pixel space (run `extract_raw_features.py` first)
  2. **embedding** space (use SimCLR / SCAN features directly)

so the thesis can argue *why* the embedding helps, with side-by-side evidence.

## Data paths

| what | path |
|------|------|
| **SimCLR / SCAN embeddings** | `/vast/s219110279/results/results/<backbone>/<dataset>/pretext/features_seed1.npy` |
| **raw pixel datasets (SCAN layout)** | `/vast/s219110279/TypiClust/scan/datasets/` |
| CIFAR10 raw | `.../scan/datasets/cifar-10/cifar-10-batches-py/` |
| CIFAR100 raw | `.../scan/datasets/cifar-100/cifar-100-python/` |
| TinyImageNet raw | `.../scan/datasets/TinyImageNet/tiny-imagenet-200/` |
| **fallback raw (legacy)** | `/vast/s219110279/TypiClust/data/cifar-10-batches-py/` etc. |
| **analysis outputs** | `/vast/s219110279/TypiClust/output/<DATASET>/analysis/...` |
| **Slurm logs** | `/vast/s219110279/TypiClust/output/_logs/` |

The raw scripts auto-detect: they try `scan/datasets/cifar-10` first, then fall
back to `TypiClust/data` if the SCAN copy is not present yet.

## Environment

Use the `gudhi-py39` conda env via the cluster module system:

```bash
module purge
module load Anaconda3
source activate
conda activate gudhi-py39
```

Every sbatch file sources `slurm/env_setup.sh`, which does the above and also exports:

```
GUDHI_PREFIX   -> $CONDA_PREFIX
GUDHI_PYTHON   -> .../site-packages/gudhi
LD_LIBRARY_PATH -> .../site-packages/gudhi.libs (bundled GMP/MPFR libs)
```

Run scripts manually with:

```bash
cd /vast/s219110279/analysis
/vast/s219110279/.conda/envs/gudhi-py39/bin/python compute_tsne.py ...
# or after: module load Anaconda3 && source activate && conda activate gudhi-py39
python compute_tsne.py ...
```

## Output convention

Every script writes three files next to a common prefix `--out PATH`:

  * `PATH.npy`        — numerical artefact (coords, persistence triples, ...)
  * `PATH.csv`        — tidy CSV ready for `pandas.read_csv()` and seaborn plots
  * `PATH_meta.json`  — exact parameters, runtime, library versions

This keeps thesis figures reproducible end-to-end.

## Scripts

### 1. `extract_raw_features.py` — raw-pixel baseline

Flattens CIFAR10/CIFAR100 images to full 3072-d raw pixel vectors (no PCA).
Optionally L2-normalizes rows.

```bash
python extract_raw_features.py \
    --dataset CIFAR10 --split train \
    --data-root /vast/s219110279/TypiClust/scan/datasets/cifar-10 \
    --out  /vast/s219110279/TypiClust/output/CIFAR10/analysis/raw/cifar10_train_raw
```

(`--data-root` is optional — omit it to use the auto-detect order above.)

### 2. `compute_tsne.py` — t-SNE

```bash
# embedding regime
python compute_tsne.py \
    --features /vast/.../scan/results/cifar-10/pretext/features_seed1.npy \
    --labels   /vast/.../scan/results/cifar-10/pretext/labels_seed1.npy \
    --dataset  CIFAR10 \
    --out      /vast/.../output/CIFAR10/analysis/tsne/cifar10_simclr \
        --backend  openTSNE --perplexity 30 --n-iter 1000 --pca-dim 0

# raw regime — point --features at the file produced in step 1
python compute_tsne.py \
    --features /vast/.../analysis/raw/cifar10_train_raw.npy \
    --labels   /vast/.../analysis/raw/cifar10_train_raw_labels.npy \
    --dataset  CIFAR10 \
    --out      /vast/.../analysis/tsne/cifar10_raw \
    --backend  openTSNE --pca-dim 0 --perplexity 30 --n-iter 1000
```

### 3. `compute_umap.py` — UMAP

```bash
python compute_umap.py \
    --features /vast/.../scan/results/cifar-10/pretext/features_seed1.npy \
    --labels   /vast/.../scan/results/cifar-10/pretext/labels_seed1.npy \
    --dataset  CIFAR10 \
    --out      /vast/.../analysis/umap/cifar10_simclr \
    --n-neighbors 15 --min-dist 0.1
```

Optional flags:

  * `--densmap`     — keep local density (useful chapter figure)
  * `--supervised`  — feed labels in (sanity-check upper bound)

### 4. `compute_persistence.py` — H0/H1 persistence diagram

Drop-in upgrade for `generate_h0_persistence_csv.py`:

```bash
python compute_persistence.py \
    --features /vast/.../scan/results/cifar-10/pretext/features_seed1.npy \
    --labels   /vast/.../scan/results/cifar-10/pretext/labels_seed1.npy \
    --dataset  CIFAR10 \
    --out      /vast/.../analysis/persistence/cifar10_simclr_h1 \
    --max-dim 0 --normalize --per-class
```

  * `--max-dim 1`  enables H1 (loops). Default also writes H0.
  * `--sparse 0.3` uses a sparse Rips for memory-bound runs.
  * `--per-class`  also writes persistence per class — feeds great into a thesis table.
  * Writes a summary block (total persistence, entropy, max) into `_meta.json`.

### 5. `compute_persistence_per_round.py` — TDA × AL strategy

The thesis-relevant one. Walks an experiment's `episode_*` folders, computes
persistence of the current selection per round, and writes a per-round summary
that can be plotted against test accuracy:

```bash
python compute_persistence_per_round.py \
    --experiment-dir /vast/.../output/CIFAR10/resnet18/LIDCOVER_1_50b \
    --features       /vast/.../scan/results/cifar-10/pretext/features_seed1.npy \
    --out            /vast/.../analysis/persistence_rounds/cifar10_lidcover_s1 \
    --scope labeled --max-dim 1 --normalize --max-points 2000
```

The `_summary.csv` it writes has columns like `round, test_accuracy,
h0_total_persistence, h0_persistence_entropy, h1_max_persistence, ...`. Drop
it into a notebook and plot trajectories per strategy.

### 6. `plot_embedding.py` — thesis figure renderer

```bash
python plot_embedding.py \
    --embedding-csv  /vast/.../analysis/umap/cifar10_simclr.csv \
    --highlight-npy  /vast/.../output/CIFAR10/.../episode_0/activeSet.npy \
                     /vast/.../output/CIFAR10/.../episode_5/activeSet.npy \
    --highlight-name "round 0" "round 5" \
    --title "CIFAR10 SimCLR UMAP — IDProbCover selections" \
    --out   /vast/.../analysis/umap/cifar10_simclr_idpc.png
```

## Running on Slurm (CPU)

Every script also has a matching sbatch wrapper under `slurm/`. They use job
arrays so a single `sbatch tsne.sbatch` queues 9 independent tasks (3
backbones x 3 datasets), and you can have multiple sbatch files in flight at
once. All jobs are CPU-only (`--partition=cpu --qos=cpu`).

The path layout the sbatch files assume:

```
/vast/s219110279/results/results/<backbone>/<dataset>/pretext/features_seed1.npy
/vast/s219110279/TypiClust/output/<DATASET>/analysis/<task>/...   <- outputs
/vast/s219110279/TypiClust/output/_logs/                          <- stdout/err
```

Files:

| sbatch file                  | what it does                                            | array size |
| ---------------------------- | -------------------------------------------------------- | ---------- |
| `prepare_labels.sbatch`      | dumps CIFAR10/100 train+test `labels_seed1.npy` next to each backbone's embeddings | 1 |
| `extract_raw.sbatch`         | flatten CIFAR10/100 -> 3072-d raw pixels (no PCA) | 2 (one per dataset) |
| `tsne.sbatch`                | openTSNE on every (backbone, dataset) embedding          | 9 |
| `tsne_raw.sbatch`            | openTSNE on full raw-pixel CIFAR10/100 features        | 2 |
| `umap.sbatch`                | UMAP + densMAP on every embedding                        | 9 |
| `umap_raw.sbatch`            | UMAP on raw-pixel baselines                              | 2 |
| `persistence.sbatch`         | H0 GUDHI persistence (full pool, per-class on CIFAR) on every embedding | 9 |
| `persistence_rounds.sbatch`  | per-round H0+H1 of the labeled set across every strategy x seed on CIFAR10 / resnet18 | 65 |
| `al_method.sbatch`             | GPU AL (typiclust): one dataset x backbone x method, **seeds 1-5** | 1 per submit |
| `submit_al.sh`                 | queues **72** GPU jobs: 3 datasets x 2 backbones x 12 methods; resnet18 skipped | — |

### Active learning (GPU): 3 datasets, 5 seeds, resnet50 + alexnet

Datasets: **cifar10**, **cifar100**, **tinyimagenet**. Each job runs **seeds 1-5**.
resnet18 is already complete on all three — not resubmitted.

From **any** directory:

```bash
bash /vast/s219110279/analysis/slurm/submit_al.sh
```

Logs: `/vast/s219110279/TypiClust/output/_logs/al_*.out`. Results:
`TypiClust/output/<CIFAR10|CIFAR100|TINYIMAGENET>/{resnet50,alexnet}/<method>_<seed>_50b/`.

Either submit one at a time:

```bash
cd /vast/s219110279/analysis/slurm
sbatch prepare_labels.sbatch    # once, before anything that uses labels
sbatch tsne.sbatch              # 9 parallel array tasks
sbatch umap.sbatch              # 9 more
sbatch persistence.sbatch       # 9 more
sbatch persistence_rounds.sbatch  # 65 small tasks, one per strategy x seed
```

Or fan out everything with sensible dependencies via the convenience launcher:

```bash
bash submit_all.sh              # queue the entire pipeline
bash submit_all.sh embedding    # only embedding-side jobs
bash submit_all.sh rounds       # only per-round persistence
bash submit_all.sh raw          # only the raw-pixel baselines
```

Resources requested by each job (tunable inside each sbatch file):

  * t-SNE / t-SNE raw: 16 CPU / 32 GB / **1 day**
  * UMAP / UMAP raw:   8 CPU / 32 GB / **1 day**
  * Persistence (single shot, full pool H0): 4 CPU / **500 GB** / **1 day**
  * Persistence per round: 4 CPU / 24 GB / **1 day**
  * Raw extraction: 8 CPU / 16 GB / 4 h
  * `prepare_labels`: 2 CPU / 8 GB / 30 min

A single failing array task does not bring the rest down. To re-run just one
combo, e.g. resnet18 + cifar-100 (array index 4) of t-SNE:

```bash
sbatch --array=4 tsne.sbatch
```

To pick a different dataset/backbone for the per-round persistence sweep,
edit `DATASET`, `DATASET_LOWER` and `BACKBONE` at the top of
`persistence_rounds.sbatch` (everything else auto-derives).

## Suggested figure inventory for the chapter

  1. Raw-pixel UMAP vs SimCLR UMAP — class separability before/after
     self-supervised pre-training.
  2. UMAP overlay of each strategy's first/last round selection.
  3. H0/H1 persistence diagrams of the full pool (per class) — motivates
     "what does the manifold look like".
  4. Per-round `total_persistence` / `persistence_entropy` curves of the
     labeled set, one line per strategy — geometric story behind the accuracy
     curve.
  5. Optional table: TDA summary statistics × strategy at the budget where
     each one achieves its target accuracy.

Feel free to ping me when the embeddings land — only the `--features` path
needs to change in any of the commands above.
