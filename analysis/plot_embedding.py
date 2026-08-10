"""
Render a t-SNE / UMAP scatter plot for the thesis.

It takes the CSV produced by compute_tsne.py or compute_umap.py and (optionally)
overlays one or more active-learning selection sets (lSet.npy, activeSet.npy,
etc.) so a reader can immediately see *where* a strategy is picking samples in
the feature manifold.

Example:
    python plot_embedding.py \\
        --embedding-csv /vast/.../analysis/umap/cifar10_simclr.csv \\
        --highlight-npy /vast/.../output/CIFAR10/.../episode_0/activeSet.npy \\
        --highlight-name "round 0 selection" \\
        --out /vast/.../analysis/umap/cifar10_simclr_round0.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import load_index_array


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedding-csv", required=True,
                        help="CSV from compute_tsne.py / compute_umap.py (columns: index, x, y, [label, label_name])")
    parser.add_argument("--out", required=True, help="Output PNG/PDF path")
    parser.add_argument("--title", default=None)
    parser.add_argument("--no-labels", action="store_true",
                        help="Render points in a single colour, ignoring any label column")
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--size", type=float, default=4.0)
    parser.add_argument("--cmap", default="tab10")
    parser.add_argument("--highlight-npy", nargs="*", default=[],
                        help="Optional list of .npy index files to overlay (e.g. activeSet.npy lSet.npy)")
    parser.add_argument("--highlight-name", nargs="*", default=[],
                        help="Names matching --highlight-npy entries")
    parser.add_argument("--highlight-color", nargs="*", default=[],
                        help="Optional colours matching --highlight-npy entries")
    parser.add_argument("--figsize", nargs=2, type=float, default=(8.0, 7.0))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    csv_path = Path(args.embedding_csv).expanduser().resolve()
    df = pd.read_csv(csv_path)

    if "x" not in df.columns or "y" not in df.columns:
        raise SystemExit(f"CSV {csv_path} is missing required 'x'/'y' columns")

    fig, ax = plt.subplots(figsize=tuple(args.figsize), dpi=args.dpi)

    label_col = "label" if "label" in df.columns and not args.no_labels else None
    if label_col is not None:
        unique_labels = sorted(df[label_col].dropna().unique().tolist())
        cmap = plt.get_cmap(args.cmap, max(10, len(unique_labels)))
        for i, l in enumerate(unique_labels):
            mask = df[label_col] == l
            name = df.loc[mask, "label_name"].iloc[0] if "label_name" in df.columns else str(l)
            ax.scatter(df.loc[mask, "x"], df.loc[mask, "y"],
                       s=args.size, alpha=args.alpha,
                       color=cmap(i % cmap.N),
                       label=str(name), linewidths=0)
    else:
        ax.scatter(df["x"], df["y"], s=args.size, alpha=args.alpha, color="#7f7f7f",
                   linewidths=0, label="pool")

    names = args.highlight_name if args.highlight_name else [Path(p).stem for p in args.highlight_npy]
    colors: List[str] = list(args.highlight_color) if args.highlight_color else []
    default_colors = ["red", "black", "orange", "magenta", "cyan", "green"]
    for j, hp in enumerate(args.highlight_npy):
        idx = load_index_array(hp)
        sub = df[df["index"].isin(idx.tolist())]
        c = colors[j] if j < len(colors) else default_colors[j % len(default_colors)]
        name = names[j] if j < len(names) else Path(hp).stem
        ax.scatter(sub["x"], sub["y"], s=args.size * 6, marker="x", color=c,
                   linewidths=1.5, label=f"{name} (n={len(sub)})")

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if args.title:
        ax.set_title(args.title)
    ax.legend(loc="best", fontsize=8, markerscale=2, frameon=False)
    fig.tight_layout()

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
