"""
Flatten raw CIFAR images into a feature matrix that the rest of the analysis
pipeline (t-SNE / UMAP / persistence) can consume.

This is the "raw" counterpart of SimCLR/SCAN embeddings, so that any figure
in the thesis can be reproduced in two regimes:

    1) raw pixel space (high-D, noisy)
    2) self-supervised feature space (the embeddings .npy)

Default behaviour:
    * load CIFAR10/CIFAR100 train images from SCAN dataset root
      (``TypiClust/scan/datasets/cifar-10/cifar-10-batches-py/`` etc.)
    * fallback to ``TypiClust/data/`` if the SCAN copy is missing
    * convert to float32, scale to [0, 1]
    * flatten 32x32x3 -> 3072 (no PCA — full raw pixel vectors)
    * optionally L2-normalize each row

Outputs:
    <out_prefix>.npy        float32 [N, D] feature matrix
    <out_prefix>_labels.npy int64 [N] class labels
    <out_prefix>_meta.json  metadata
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from _common import (
    CIFAR10_CLASSES,
    RunMeta,
    Timer,
    collect_library_versions,
    load_cifar_raw,
    out_prefix_paths,
    save_meta,
    subsample_indices,
)


def build_features(
    dataset: str,
    train: bool = True,
    normalize: bool = False,
    pca_dim: Optional[int] = None,
    subsample: Optional[int] = None,
    seed: int = 1,
    data_root: Optional[str] = None,
):
    images, labels, classes = load_cifar_raw(dataset, train=train, data_root=data_root)
    n = images.shape[0]

    idx = subsample_indices(n, subsample, seed=seed, stratify_labels=labels)
    images = images[idx]
    labels = labels[idx]

    X = images.astype(np.float32).reshape(images.shape[0], -1) / 255.0

    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms

    if pca_dim is not None and pca_dim > 0 and pca_dim < X.shape[1]:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=int(pca_dim), random_state=seed)
        X = pca.fit_transform(X).astype(np.float32)
        explained = float(np.sum(pca.explained_variance_ratio_))
    else:
        explained = 1.0

    return X, labels.astype(np.int64), classes, idx, explained


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["CIFAR10", "CIFAR100"], default="CIFAR10")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--out", required=True, help="Output prefix (no extension)")
    parser.add_argument("--data-root", default=None,
                        help="Parent of cifar-10-batches-py / cifar-100-python "
                             "(default: scan/datasets/cifar-10 or cifar-100, "
                             "fallback TypiClust/data)")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize each row")
    parser.add_argument("--pca-dim", type=int, default=0,
                        help="Optional PCA target dimensionality (0 = disabled, default)")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Optional class-stratified subsample size")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    paths = out_prefix_paths(args.out)
    labels_path = paths["prefix"].with_name(paths["prefix"].name + "_labels.npy")

    with Timer() as timer:
        X, labels, classes, idx, explained = build_features(
            dataset=args.dataset,
            train=args.split == "train",
            normalize=args.normalize,
            pca_dim=None if args.pca_dim is None or args.pca_dim <= 0 else args.pca_dim,
            subsample=args.subsample,
            seed=args.seed,
            data_root=args.data_root,
        )

    np.save(paths["npy"], X)
    np.save(labels_path, labels)
    np.save(paths["prefix"].with_name(paths["prefix"].name + "_indices.npy"), idx)

    meta = RunMeta(
        script=Path(__file__).name,
        params=vars(args),
        inputs={"dataset": args.dataset, "split": args.split},
        outputs=[str(paths["npy"]), str(labels_path)],
        runtime_sec=timer.elapsed,
        library_versions=collect_library_versions(["numpy", "sklearn", "torchvision"]),
        notes={
            "n": int(X.shape[0]),
            "d": int(X.shape[1]),
            "pca_explained_variance": explained,
            "classes": classes,
        },
    )
    save_meta(meta, paths["meta"])

    print(f"Saved features: {paths['npy']}  shape={X.shape}  dtype={X.dtype}")
    print(f"Saved labels:   {labels_path}    shape={labels.shape}")
    if explained < 1.0:
        print(f"PCA explained variance ratio: {explained:.3f}")
    print(f"Elapsed: {timer.elapsed:.1f}s")


if __name__ == "__main__":
    main()
