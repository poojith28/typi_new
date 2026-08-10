"""
Compute a 2D (or 3D) UMAP projection of a feature matrix.

Same interface as compute_tsne.py: takes a feature .npy + optional labels +
optional dataset-index mapping, writes:

    <out_prefix>.npy   float32 [N, n_components]
    <out_prefix>.csv   index, x, y[, z], label, label_name
    <out_prefix>_meta.json

Typical thesis use:

    python compute_umap.py \\
        --features /vast/.../features_seed1.npy \\
        --labels   /vast/.../labels_train.npy \\
        --out      /vast/.../analysis/umap/cifar10_simclr \\
        --n-neighbors 15 --min-dist 0.1
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--indices", default=None)
    parser.add_argument("--label-names", default=None)
    parser.add_argument("--dataset", choices=["CIFAR10", "CIFAR100", "TINYIMAGENET", "OTHER"], default="OTHER")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--spread", type=float, default=1.0)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="UMAP picks 200/500 by default depending on N")
    parser.add_argument("--init", choices=["spectral", "random", "pca"], default="spectral")
    parser.add_argument("--densmap", action="store_true",
                        help="Enable densMAP (preserves local density; nice for thesis figures)")
    parser.add_argument("--supervised", action="store_true",
                        help="Use labels as supervision (--labels required)")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--pca-dim", type=int, default=0,
                        help="Optional PCA pre-reduction (0 disables; rarely needed for UMAP)")
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    paths = out_prefix_paths(args.out)

    X = load_features(args.features, normalize=args.normalize)
    labels = maybe_load_labels(args.labels)
    if args.indices is not None:
        ds_indices = maybe_load_labels(args.indices)
    else:
        ds_indices = np.arange(X.shape[0], dtype=np.int64)

    sub_idx = subsample_indices(X.shape[0], args.subsample, seed=args.seed, stratify_labels=labels)
    if sub_idx.shape[0] != X.shape[0]:
        X = X[sub_idx]
        if labels is not None:
            labels = labels[sub_idx]
        ds_indices = ds_indices[sub_idx]
        print(f"Sub-sampled to {X.shape[0]} points")

    explained = 1.0
    if args.pca_dim and 0 < args.pca_dim < X.shape[1]:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=int(args.pca_dim), random_state=args.seed)
        X = pca.fit_transform(X).astype(np.float32)
        explained = float(np.sum(pca.explained_variance_ratio_))

    import umap  # local import so the script can be inspected without UMAP installed

    reducer_kwargs = dict(
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        spread=args.spread,
        metric=args.metric,
        init=args.init,
        random_state=args.seed,
        verbose=True,
    )
    if args.n_epochs is not None:
        reducer_kwargs["n_epochs"] = int(args.n_epochs)
    if args.densmap:
        reducer_kwargs["densmap"] = True

    reducer = umap.UMAP(**reducer_kwargs)

    with Timer() as timer:
        if args.supervised:
            if labels is None:
                raise SystemExit("--supervised requires --labels")
            Y = reducer.fit_transform(X, y=labels).astype(np.float32)
        else:
            Y = reducer.fit_transform(X).astype(np.float32)

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
        library_versions=collect_library_versions(["numpy", "pandas", "sklearn", "umap"]),
        notes={
            "n": int(Y.shape[0]),
            "input_dim": int(X.shape[1]),
            "pca_explained_variance": explained,
            "densmap": bool(args.densmap),
            "supervised": bool(args.supervised),
        },
    )
    save_meta(meta, paths["meta"])

    print(f"Saved UMAP coords:  {paths['npy']}  shape={Y.shape}")
    print(f"Saved CSV:          {paths['csv']}  rows={len(df)}")
    print(f"Saved meta:         {paths['meta']}")
    print(f"Elapsed: {timer.elapsed:.1f}s")


if __name__ == "__main__":
    main()
