"""
Dump CIFAR10 / CIFAR100 train+test labels as .npy files next to the embeddings.

The downstream analysis scripts colour scatter plots by these labels, and
the persistence script can use them for class-stratified subsampling and
per-class persistence diagrams.

Run once. Outputs go next to features_seed1.npy by default:

    /vast/.../results/results/<backbone>/<dataset>/pretext/labels_seed1.npy
    /vast/.../results/results/<backbone>/<dataset>/pretext/test_labels_seed1.npy

For tiny-imagenet we leave labels unset because raw tiny-imagenet isn't
pre-staged on the cluster; the analysis scripts work fine without labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import load_cifar_raw


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings-root", default="/vast/s219110279/results/results",
                        help="Root of the embeddings tree (contains <backbone>/<dataset>/pretext/...)")
    parser.add_argument("--data-root", default=None,
                        help="Folder with cifar-10-batches-py / cifar-100-python (default: TypiClust/data)")
    parser.add_argument("--datasets", nargs="+",
                        default=["cifar-10", "cifar-100"],
                        help="Datasets to dump labels for")
    parser.add_argument("--backbones", nargs="+",
                        default=["alexnet", "resnet18", "resnet50"],
                        help="Backbone subfolders to place labels under")
    args = parser.parse_args()

    embeddings_root = Path(args.embeddings_root).expanduser().resolve()

    for ds in args.datasets:
        if ds.lower() in ("cifar-10", "cifar10"):
            ds_key = "CIFAR10"
            ds_dir = "cifar-10"
        elif ds.lower() in ("cifar-100", "cifar100"):
            ds_key = "CIFAR100"
            ds_dir = "cifar-100"
        else:
            print(f"  skipping {ds!r}: only cifar-10 / cifar-100 supported")
            continue

        train_imgs, train_lbls, classes = load_cifar_raw(ds_key, train=True, data_root=args.data_root)
        test_imgs, test_lbls, _ = load_cifar_raw(ds_key, train=False, data_root=args.data_root)

        for bb in args.backbones:
            target = embeddings_root / bb / ds_dir / "pretext"
            if not target.exists():
                print(f"  {bb}/{ds_dir}/pretext does not exist, skipping")
                continue
            tr_path = target / "labels_seed1.npy"
            te_path = target / "test_labels_seed1.npy"
            np.save(tr_path, train_lbls.astype(np.int64))
            np.save(te_path, test_lbls.astype(np.int64))
            print(f"  wrote {tr_path}  ({train_lbls.shape}) and {te_path}  ({test_lbls.shape})")


if __name__ == "__main__":
    main()
