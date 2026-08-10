"""
Compute a 2D (or 3D) t-SNE projection of a feature matrix and save it as both
an .npy file and a CSV ready to be loaded into pandas / matplotlib / seaborn.

Two backends are supported:

  * openTSNE (preferred when N >= ~5k or D >= ~64; multithreaded, faster,
    Barnes-Hut + FFT interpolation).
  * sklearn.manifold.TSNE as a fallback.

Both raw image features (use extract_raw_features.py first) and self-supervised
embeddings can be projected with the exact same invocation.

Typical thesis use:

    python compute_tsne.py \\
        --features /vast/.../features_seed1.npy \\
        --labels   /vast/.../labels_train.npy \\
        --out      /vast/.../analysis/tsne/cifar10_simclr \\
        --backend  openTSNE --perplexity 30 --pca-dim 50

Outputs:
    <out_prefix>.npy        float32 [N, n_components]
    <out_prefix>.csv        columns: index, x, y[, z], label, label_name
    <out_prefix>_meta.json  parameters / runtime / library versions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from _common import (
    CIFAR10_CLASSES,
    RunMeta,
    Timer,
    collect_library_versions,
    load_features,
    maybe_load_labels,
    out_prefix_paths,
    save_meta,
    subsample_indices,
)


def _preprocess(X: np.ndarray, pca_dim: Optional[int], seed: int):
    """Optionally L2-normalize + PCA-reduce features (the standard t-SNE recipe)."""
    explained = 1.0
    if pca_dim is not None and pca_dim > 0 and pca_dim < X.shape[1]:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=int(pca_dim), random_state=seed)
        X = pca.fit_transform(X).astype(np.float32)
        explained = float(np.sum(pca.explained_variance_ratio_))
    return X, explained


def _run_opentsne(X, n_components, perplexity, learning_rate, n_iter, seed, metric, exaggeration, dof):
    from openTSNE import TSNE as OpenTSNE

    n = X.shape[0]
    lr = learning_rate
    if isinstance(lr, str) and lr == "auto":
        lr = max(200.0, n / 12.0)

    tsne = OpenTSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=lr,
        n_iter=n_iter,
        metric=metric,
        early_exaggeration=exaggeration,
        dof=dof,
        random_state=seed,
        n_jobs=-1,
        verbose=True,
    )
    Y = tsne.fit(X)
    return np.asarray(Y, dtype=np.float32)


def _run_sklearn(X, n_components, perplexity, learning_rate, n_iter, seed, metric):
    from sklearn.manifold import TSNE

    n = X.shape[0]
    lr = learning_rate
    if isinstance(lr, str) and lr == "auto":
        lr = max(200.0, n / 12.0)
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=lr,
        n_iter=n_iter,
        metric=metric,
        init="pca",
        random_state=seed,
        verbose=1,
    )
    return tsne.fit_transform(X).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True, help="Path to NxD feature matrix .npy")
    parser.add_argument("--labels", default=None, help="Optional path to Nx class-id labels .npy")
    parser.add_argument("--indices", default=None,
                        help="Optional path to Nx dataset-index .npy (preserved in CSV)")
    parser.add_argument("--label-names", default=None,
                        help="Optional comma-separated class names (overrides CIFAR10 default)")
    parser.add_argument("--dataset", choices=["CIFAR10", "CIFAR100", "TINYIMAGENET", "OTHER"], default="OTHER",
                        help="Used only to auto-pick class names for the CSV")
    parser.add_argument("--out", required=True, help="Output path prefix (no extension)")
    parser.add_argument("--backend", choices=["openTSNE", "sklearn"], default="openTSNE")
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--learning-rate", default="auto",
                        help='"auto" or a number; openTSNE uses N/12 when auto')
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--exaggeration", type=float, default=12.0,
                        help="Early exaggeration (openTSNE).")
    parser.add_argument("--dof", type=float, default=1.0,
                        help="Degrees of freedom of the Student-t kernel (openTSNE).")
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize feature rows first")
    parser.add_argument("--pca-dim", type=int, default=0,
                        help="Optional PCA pre-reduction before t-SNE (0 = disabled, default)")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Optional class-stratified subsample size (uses labels if provided)")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    paths = out_prefix_paths(args.out)

    X = load_features(args.features, normalize=args.normalize)
    labels = maybe_load_labels(args.labels)
    if args.indices is not None:
        ds_indices = maybe_load_labels(args.indices)  # same loader works
    else:
        ds_indices = np.arange(X.shape[0], dtype=np.int64)

    sub_idx = subsample_indices(X.shape[0], args.subsample, seed=args.seed, stratify_labels=labels)
    if sub_idx.shape[0] != X.shape[0]:
        X = X[sub_idx]
        if labels is not None:
            labels = labels[sub_idx]
        ds_indices = ds_indices[sub_idx]
        print(f"Sub-sampled from {sub_idx.max() + 1} -> {X.shape[0]} points")

    pca_dim = None if args.pca_dim is None or args.pca_dim <= 0 else args.pca_dim
    X_proc, explained = _preprocess(X, pca_dim, args.seed)

    try:
        lr_arg = float(args.learning_rate)
    except (TypeError, ValueError):
        lr_arg = args.learning_rate

    with Timer() as timer:
        if args.backend == "openTSNE":
            try:
                Y = _run_opentsne(
                    X_proc,
                    n_components=args.n_components,
                    perplexity=args.perplexity,
                    learning_rate=lr_arg,
                    n_iter=args.n_iter,
                    seed=args.seed,
                    metric=args.metric,
                    exaggeration=args.exaggeration,
                    dof=args.dof,
                )
            except ImportError:
                print("openTSNE not available -> falling back to sklearn.manifold.TSNE")
                Y = _run_sklearn(
                    X_proc,
                    n_components=args.n_components,
                    perplexity=args.perplexity,
                    learning_rate=lr_arg,
                    n_iter=args.n_iter,
                    seed=args.seed,
                    metric=args.metric,
                )
        else:
            Y = _run_sklearn(
                X_proc,
                n_components=args.n_components,
                perplexity=args.perplexity,
                learning_rate=lr_arg,
                n_iter=args.n_iter,
                seed=args.seed,
                metric=args.metric,
            )

    np.save(paths["npy"], Y)

    df = pd.DataFrame({"index": ds_indices})
    for d in range(Y.shape[1]):
        df[["x", "y", "z"][d] if d < 3 else f"dim{d}"] = Y[:, d]
    if labels is not None:
        df["label"] = labels
        names = None
        if args.label_names:
            names = [s.strip() for s in args.label_names.split(",")]
        elif args.dataset == "CIFAR10":
            names = list(CIFAR10_CLASSES)
        if names is not None and labels.max() < len(names):
            df["label_name"] = [names[int(l)] for l in labels]
    df.to_csv(paths["csv"], index=False)

    meta = RunMeta(
        script=Path(__file__).name,
        params=vars(args),
        inputs={
            "features": str(args.features),
            "labels": str(args.labels) if args.labels else None,
            "indices": str(args.indices) if args.indices else None,
        },
        outputs=[str(paths["npy"]), str(paths["csv"])],
        runtime_sec=timer.elapsed,
        library_versions=collect_library_versions(["numpy", "pandas", "sklearn", "openTSNE"]),
        notes={
            "n": int(Y.shape[0]),
            "input_dim": int(X.shape[1]),
            "pca_explained_variance": explained,
            "backend_used": args.backend,
        },
    )
    save_meta(meta, paths["meta"])

    print(f"Saved t-SNE coords: {paths['npy']}  shape={Y.shape}")
    print(f"Saved CSV:          {paths['csv']}  rows={len(df)}")
    print(f"Saved meta:         {paths['meta']}")
    print(f"Elapsed: {timer.elapsed:.1f}s")


if __name__ == "__main__":
    main()
