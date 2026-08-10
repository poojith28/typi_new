#!/usr/bin/env python3
"""Generate 2D embedding CSVs for AL geometry/topology visualizations.

Outputs columns: sample_id,x,y. Uses pretrained TypiClust/SCAN features by
default, matching the ID diagnostic feature space.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/scratch/s219110279")
DEEP_AL_ROOT = ROOT / "TypiClust/deep-al"
if str(DEEP_AL_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEP_AL_ROOT))

import pycls.datasets.utils as ds_utils  # noqa: E402


def load_features(dataset: str) -> np.ndarray:
    x = ds_utils.load_features(dataset, seed=1, train=True, normalized=True)
    return np.asarray(x, dtype=np.float32)


def compute_embedding(x: np.ndarray, method: str, seed: int, n_neighbors: int, min_dist: float) -> tuple[np.ndarray, str]:
    method = method.lower()
    if method == "auto":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=int(n_neighbors),
                min_dist=float(min_dist),
                metric="euclidean",
                random_state=int(seed),
                low_memory=True,
            )
            return reducer.fit_transform(x), "umap"
        except Exception as exc:
            print(f"[WARN] UMAP unavailable, falling back to PCA: {exc}")
            method = "pca"

    if method == "umap":
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
            metric="euclidean",
            random_state=int(seed),
            low_memory=True,
        )
        return reducer.fit_transform(x), "umap"

    if method == "tsne":
        from sklearn.manifold import TSNE

        emb = TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            random_state=int(seed),
        ).fit_transform(x)
        return emb, "tsne"

    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=int(seed)).fit_transform(x), "pca"

    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["CIFAR10", "CIFAR100", "TINYIMAGENET"])
    parser.add_argument("--method", default="auto", choices=["auto", "umap", "tsne", "pca"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--max-points", type=int, default=0, help="Optional deterministic subset size; 0 uses all points.")
    args = parser.parse_args()

    x = load_features(args.dataset)
    sample_id = np.arange(len(x), dtype=np.int64)
    if args.max_points and args.max_points < len(x):
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(sample_id, size=int(args.max_points), replace=False))
        sample_id = keep
        x = x[keep]

    emb, used_method = compute_embedding(x, args.method, args.seed, args.n_neighbors, args.min_dist)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sample_id": sample_id.astype(int),
        "x": emb[:, 0].astype(float),
        "y": emb[:, 1].astype(float),
    }).to_csv(out, index=False)
    print(f"[done] wrote {used_method} embedding: {out} rows={len(sample_id)}")


if __name__ == "__main__":
    main()
