#!/usr/bin/env python3
"""
H0 storytelling panels: t-SNE embedding next to the reference persistence diagram.

Left  = geometry of the embedding world (all points + selected highlighted)
Right = H0 connectivity events (all pairs + pairs touched by the selection)

Designed so a thesis caption can narrate:
  "In the embedding (left), method M selects these points;
   those selections touch these H0 bars in the reference diagram (right)."

Usage:
  python analysis/generate_diagnostic_h0_story_panels.py
  python analysis/generate_diagnostic_h0_story_panels.py --dataset CIFAR10 --backbone resnet18
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/vast/s219110279")
OUTPUT_ROOT = REPO / "TypiClust" / "output"
REF_DIR = REPO / "outputs" / "diagnostic_h0" / "reference"
FIG_DIR = REPO / "figures" / "diagnostic_h0" / "story"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASET_LABELS = {
    "CIFAR10": "CIFAR-10",
    "CIFAR100": "CIFAR-100",
    "TINYIMAGENET": "TinyImageNet",
}
BACKBONE_LABELS = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "alexnet": "AlexNet",
}
METHOD_DIR = {
    "random": "random",
    "uncertainty": "uncertainty",
    "entropy": "entropy",
    "margin": "margin",
    "dbal": "dbal",
    "coreset": "coreset",
    "probcover": "probcover",
    "typiclust": "typiclust",
    "maxherding": "maxherding",
}
METHOD_DISPLAY = {
    "random": "Random",
    "uncertainty": "Uncertainty (LC)",
    "entropy": "Entropy",
    "margin": "Margin",
    "dbal": "DBAL (BALD)",
    "coreset": "CoreSet",
    "probcover": "ProbCover",
    "typiclust": "TypiClust",
    "maxherding": "MaxHerding",
}
METHOD_COLOR = {
    "random": "#000000",
    "uncertainty": "#E69F00",
    "entropy": "#56B4E9",
    "margin": "#009E73",
    "dbal": "#B0A800",
    "coreset": "#0072B2",
    "probcover": "#D55E00",
    "typiclust": "#CC79A7",
    "maxherding": "#666666",
}

# Story casts
STORY_METHODS = ["random", "probcover", "typiclust", "entropy", "coreset"]
STORY_EPISODES_CUM = [10, 30, 60, 90]  # cumulative L_t (E100 ok but busy)
COMPARE_EPISODE = 30


def configure():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
    })


def save(fig, stem: str):
    pdf = FIG_DIR / f"{stem}.pdf"
    png = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=400)
    fig.savefig(png, bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f"[SAVE] {pdf}")
    return pdf, png


def tsne_path(dataset: str, backbone: str) -> Path:
    tag = {"CIFAR10": "cifar10", "CIFAR100": "cifar100", "TINYIMAGENET": "tinyimagenet"}[dataset]
    return OUTPUT_ROOT / dataset / "analysis" / "tsne" / f"{backbone}_{tag}_simclr_tsne.csv"


def ref_path(dataset: str, backbone: str) -> Path:
    return REF_DIR / f"{dataset}_{backbone}_h0_reference.csv"


def load_tsne(dataset, backbone) -> pd.DataFrame:
    p = tsne_path(dataset, backbone)
    df = pd.read_csv(p)
    # unify id column
    if "index" in df.columns:
        df = df.rename(columns={"index": "sample_id"})
    df["sample_id"] = df["sample_id"].astype(int)
    return df


def load_reference(dataset, backbone):
    df = pd.read_csv(ref_path(dataset, backbone))
    if "scope" in df.columns:
        df = df[df["scope"] == "all"]
    if "dim" in df.columns:
        df = df[df["dim"] == 0]
    df = df[np.isfinite(df["death"])].reset_index(drop=True)
    births = []
    for s in df["birth_simplex"]:
        v = ast.literal_eval(s) if isinstance(s, str) else s
        births.append(int(v[0]))
    birth_v = np.asarray(births, dtype=np.int64)
    n_pool = int(birth_v.max()) + 1
    pair_of = -np.ones(n_pool, dtype=np.int32)
    pair_of[birth_v] = np.arange(len(df), dtype=np.int32)
    return df, birth_v, pair_of


def load_lset(dataset, backbone, method, seed, episode) -> np.ndarray:
    run = OUTPUT_ROOT / dataset / backbone / f"{METHOD_DIR[method]}_{seed}_50b"
    p = run / f"episode_{episode}" / "lSet.npy"
    if not p.exists():
        return np.array([], dtype=np.int64)
    arr = np.load(p, allow_pickle=True)
    if arr.dtype == object:
        arr = np.asarray(arr.tolist())
    return np.asarray(arr, dtype=np.int64).reshape(-1)


def load_batch(dataset, backbone, method, seed, episode) -> np.ndarray:
    """Newly acquired batch via active_set_ids if present, else lSet diff."""
    import json
    run = OUTPUT_ROOT / dataset / backbone / f"{METHOD_DIR[method]}_{seed}_50b"
    summary = run / "benchmark_summary.json"
    if summary.exists():
        data = json.loads(summary.read_text())
        for ep in data.get("episode_records") or []:
            if int(ep["episode"]) != int(episode):
                continue
            batch = ep.get("active_set_ids")
            if batch is not None:
                return np.sort(np.asarray(batch, dtype=np.int64).reshape(-1))
    # fallback: diff
    cur = load_lset(dataset, backbone, method, seed, episode)
    if episode == 0:
        return cur
    prev = load_lset(dataset, backbone, method, seed, episode - 1)
    if len(prev) == 0:
        return cur
    return np.setdiff1d(cur, prev)


def touched_mask(pair_of: np.ndarray, indices: np.ndarray, n_pairs: int) -> np.ndarray:
    mask = np.zeros(n_pairs, dtype=bool)
    if len(indices) == 0:
        return mask
    idx = indices[(indices >= 0) & (indices < len(pair_of))]
    pids = pair_of[idx]
    pids = pids[pids >= 0]
    mask[pids] = True
    return mask


def class_colors(n: int):
    cmap = plt.cm.get_cmap("tab20", max(n, 20))
    return [cmap(i % cmap.N) for i in range(n)]


def plot_tsne(ax, embed: pd.DataFrame, selected: set, title: str, accent: str,
              show_legend: bool = False):
    # background: all points, light class colours
    n_cls = int(embed["label"].nunique()) if "label" in embed.columns else 1
    palette = class_colors(n_cls)
    if "label" in embed.columns:
        for cls, g in embed.groupby("label"):
            ax.scatter(
                g["x"], g["y"], s=3, alpha=0.18, color=palette[int(cls) % len(palette)],
                linewidths=0, rasterized=True,
            )
    else:
        ax.scatter(embed["x"], embed["y"], s=3, alpha=0.15, c="#bbbbbb",
                   linewidths=0, rasterized=True)

    if selected:
        fg = embed[embed["sample_id"].isin(selected)]
        if len(fg):
            ax.scatter(
                fg["x"], fg["y"], s=28, alpha=0.95,
                facecolors="none", edgecolors=accent, linewidths=1.1,
                label=f"Selected (n={len(fg)})", zorder=5,
            )
    ax.set_title(title, pad=6)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_alpha(0.3)
    if show_legend and selected:
        ax.legend(loc="best", frameon=True, fontsize=8)


def plot_pd(ax, ref_df: pd.DataFrame, mask: np.ndarray, title: str, accent: str):
    b = ref_df["birth"].to_numpy(dtype=float)
    d = ref_df["death"].to_numpy(dtype=float)
    ax.scatter(b, d, s=4, alpha=0.10, c="#888888", linewidths=0,
               rasterized=True, label="All H0 pairs")
    if mask.any():
        ax.scatter(
            b[mask], d[mask], s=18, alpha=0.9,
            facecolors="none", edgecolors=accent, linewidths=1.0,
            label=f"Touched (n={int(mask.sum())})", zorder=5,
        )
    lo = float(min(np.nanmin(b), np.nanmin(d)))
    hi = float(max(np.nanmax(b), np.nanmax(d)))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    ax.set_title(title, pad=6)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", frameon=True, fontsize=8)


def story_pair_figure(embed, ref_df, pair_of, indices, method, episode, mode, dataset, backbone):
    """One t-SNE | PD pair for a single method/episode."""
    selected = set(map(int, indices))
    mask = touched_mask(pair_of, np.asarray(list(selected), dtype=np.int64), len(ref_df))
    accent = METHOD_COLOR[method]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    mode_lab = "cumulative $L_t$" if mode == "cumulative" else r"new batch $\Delta L_t$"
    plot_tsne(
        axes[0], embed, selected,
        f"t-SNE — {METHOD_DISPLAY[method]} — E{episode} ({mode_lab})",
        accent, show_legend=True,
    )
    plot_pd(
        axes[1], ref_df, mask,
        f"Reference H0 PD — pairs touched by {mode_lab}",
        accent,
    )
    fig.suptitle(
        f"{DATASET_LABELS[dataset]} / {BACKBONE_LABELS[backbone]} — "
        f"embedding world ↔ H0 connectivity",
        fontsize=13, fontweight="bold",
    )
    # caption strip
    fig.text(
        0.5, -0.02,
        "Left: selected samples in the fixed embedding.  "
        "Right: reference H0 pairs whose birth vertex lies in that selection.",
        ha="center", fontsize=9, style="italic",
    )
    stem = (
        f"story_{backbone}_{dataset.lower()}_{method}_E{episode}_{mode}"
        .replace("tinyimagenet", "tinyimagenet")
    )
    if dataset == "CIFAR10":
        stem = f"story_{backbone}_cifar10_{method}_E{episode}_{mode}"
    elif dataset == "CIFAR100":
        stem = f"story_{backbone}_cifar100_{method}_E{episode}_{mode}"
    else:
        stem = f"story_{backbone}_tinyimagenet_{method}_E{episode}_{mode}"
    return save(fig, stem)


def story_compare_methods(embed, ref_df, pair_of, dataset, backbone, seed, episode, methods):
    """Rows = methods, cols = t-SNE | PD. One episode, cumulative."""
    n = len(methods)
    fig, axes = plt.subplots(n, 2, figsize=(11.2, 3.2 * n), constrained_layout=True)
    if n == 1:
        axes = np.array([axes])
    for i, method in enumerate(methods):
        indices = load_lset(dataset, backbone, method, seed, episode)
        selected = set(map(int, indices))
        mask = touched_mask(pair_of, indices, len(ref_df))
        accent = METHOD_COLOR[method]
        plot_tsne(
            axes[i, 0], embed, selected,
            f"{METHOD_DISPLAY[method]} — t-SNE (E{episode}, cumulative)",
            accent, show_legend=(i == 0),
        )
        plot_pd(
            axes[i, 1], ref_df, mask,
            f"{METHOD_DISPLAY[method]} — touched H0 pairs",
            accent,
        )
        # side label
        axes[i, 0].annotate(
            METHOD_DISPLAY[method], xy=(-0.08, 0.5), xycoords="axes fraction",
            rotation=90, va="center", ha="center", fontsize=11, fontweight="bold",
            color=accent,
        )
    fig.suptitle(
        f"Storyboard @ E{episode}: who is selected, which H0 bars they touch\n"
        f"{DATASET_LABELS[dataset]} / {BACKBONE_LABELS[backbone]} (seed {seed})",
        fontsize=14, fontweight="bold",
    )
    fig.text(
        0.5, -0.01,
        "Same reference PD for every method. Differences come only from which birth vertices enter L_t.",
        ha="center", fontsize=9, style="italic",
    )
    tag = {"CIFAR10": "cifar10", "CIFAR100": "cifar100", "TINYIMAGENET": "tinyimagenet"}[dataset]
    return save(fig, f"storyboard_{backbone}_{tag}_E{episode}_cumulative")


def story_trajectory(embed, ref_df, pair_of, dataset, backbone, seed, method, episodes):
    """One method across episodes: growing L_t in t-SNE | PD."""
    n = len(episodes)
    fig, axes = plt.subplots(n, 2, figsize=(11.2, 3.15 * n), constrained_layout=True)
    if n == 1:
        axes = np.array([axes])
    accent = METHOD_COLOR[method]
    for i, ep in enumerate(episodes):
        indices = load_lset(dataset, backbone, method, seed, ep)
        selected = set(map(int, indices))
        mask = touched_mask(pair_of, indices, len(ref_df))
        budget = 50 * (ep + 1)
        plot_tsne(
            axes[i, 0], embed, selected,
            f"E{ep} (budget≈{budget}) — labelled set in t-SNE",
            accent, show_legend=(i == 0),
        )
        plot_pd(
            axes[i, 1], ref_df, mask,
            f"E{ep} — cumulative touched H0 pairs (n={int(mask.sum())})",
            accent,
        )
    fig.suptitle(
        f"Trajectory story: {METHOD_DISPLAY[method]} exposing H0 connectivity over time\n"
        f"{DATASET_LABELS[dataset]} / {BACKBONE_LABELS[backbone]} (seed {seed})",
        fontsize=14, fontweight="bold",
    )
    fig.text(
        0.5, -0.01,
        "As labelling grows (left), more birth vertices enter L_t and more reference bars light up (right).",
        ha="center", fontsize=9, style="italic",
    )
    tag = {"CIFAR10": "cifar10", "CIFAR100": "cifar100", "TINYIMAGENET": "tinyimagenet"}[dataset]
    return save(fig, f"trajectory_{backbone}_{tag}_{method}_cumulative")


def story_excl_vs_cum(embed, ref_df, pair_of, dataset, backbone, seed, method, episode):
    """2×2: exclusive vs cumulative for one method/episode."""
    batch = load_batch(dataset, backbone, method, seed, episode)
    cum = load_lset(dataset, backbone, method, seed, episode)
    accent = METHOD_COLOR[method]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.0), constrained_layout=True)
    # exclusive
    plot_tsne(axes[0, 0], embed, set(map(int, batch)),
              fr"t-SNE — new batch $\Delta L$ @ E{episode} (n={len(batch)})", accent, True)
    plot_pd(axes[0, 1], ref_df, touched_mask(pair_of, batch, len(ref_df)),
            fr"H0 pairs touched by $\Delta L$ only", accent)
    # cumulative
    plot_tsne(axes[1, 0], embed, set(map(int, cum)),
              fr"t-SNE — cumulative $L_t$ @ E{episode} (n={len(cum)})", accent, True)
    plot_pd(axes[1, 1], ref_df, touched_mask(pair_of, cum, len(ref_df)),
            fr"H0 pairs touched by full $L_t$", accent)
    fig.suptitle(
        f"Exclusive vs cumulative — {METHOD_DISPLAY[method]} @ E{episode}\n"
        f"{DATASET_LABELS[dataset]} / {BACKBONE_LABELS[backbone]} (seed {seed})",
        fontsize=14, fontweight="bold",
    )
    fig.text(
        0.5, -0.01,
        "Top row: what this round newly acquired. Bottom row: the labelled set so far.",
        ha="center", fontsize=9, style="italic",
    )
    tag = {"CIFAR10": "cifar10", "CIFAR100": "cifar100", "TINYIMAGENET": "tinyimagenet"}[dataset]
    return save(fig, f"excl_vs_cum_{backbone}_{tag}_{method}_E{episode}")


def write_story_guide(dataset, backbone, paths):
    p = FIG_DIR / "STORY_GUIDE.md"
    lines = [
        "# H0 story panels — how to read them",
        "",
        "These figures are for **narration** in the thesis (and talks), not for seed-averaged claims.",
        "Quantitative comparisons remain in the main `figures/diagnostic_h0/` curves.",
        "",
        "## The visual grammar",
        "",
        "| Panel | Data | Message |",
        "|---|---|---|",
        "| **Left: t-SNE** | Fixed SimCLR embedding of the full training pool | Where the method is looking in the world |",
        "| **Right: H0 PD** | One reference persistence diagram for the same pool | Which connectivity scales those samples expose |",
        "",
        "- Grey / faded = everything (pool or all reference pairs).",
        "- Coloured rings = **selected samples** (left) or **touched H0 pairs** (right).",
        "- A pair is touched iff its **birth vertex** is in the selection.",
        "",
        "## Story types",
        "",
        "1. **Storyboard** (`storyboard_*_E30_cumulative`): several methods at the same episode — compare who selects where and which bars they light up.",
        "2. **Trajectory** (`trajectory_*`): one method across E10→E90 — watch coverage grow.",
        "3. **Exclusive vs cumulative** (`excl_vs_cum_*`): this round’s batch vs the whole labelled set.",
        "4. **Single pairs** (`story_*_E*_cumulative`): one method, one episode, for a caption figure.",
        "",
        f"## This run",
        "",
        f"- Dataset: **{DATASET_LABELS[dataset]}**",
        f"- Backbone: **{BACKBONE_LABELS[backbone]}**",
        f"- Seed: **1** (illustrative; not averaged)",
        "",
        "## Suggested caption skeleton",
        "",
        "> Embedding geometry (left) and reference H0 persistence (right) for method *M* ",
        "> at episode *t*. Highlighted points are the cumulative labelled set $L_t$; ",
        "> highlighted persistence pairs are those whose birth simplex intersects $L_t$. ",
        "> The reference diagram is fixed; only the highlighted subset changes with acquisition.",
        "",
        "## Files",
        "",
    ]
    for x in paths:
        lines.append(f"- `{x}`")
    p.write_text("\n".join(lines) + "\n")
    print(f"[SAVE] {p}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CIFAR10", choices=list(DATASET_LABELS))
    ap.add_argument("--backbone", default="resnet18", choices=list(BACKBONE_LABELS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--episode", type=int, default=COMPARE_EPISODE)
    ap.add_argument("--methods", nargs="+", default=STORY_METHODS)
    args = ap.parse_args()
    configure()

    print(f"[LOAD] t-SNE {args.dataset}/{args.backbone}")
    embed = load_tsne(args.dataset, args.backbone)
    print(f"[LOAD] reference PD")
    ref_df, _birth_v, pair_of = load_reference(args.dataset, args.backbone)
    print(f"  embed={len(embed)}  pairs={len(ref_df)}")

    paths = []

    # 1) Storyboard at fixed episode
    pdf, png = story_compare_methods(
        embed, ref_df, pair_of, args.dataset, args.backbone, args.seed,
        args.episode, args.methods,
    )
    paths += [pdf, png]

    # 2) Trajectories for Random, ProbCover, Entropy
    for method in ["random", "probcover", "entropy"]:
        if method not in args.methods and method not in STORY_METHODS:
            continue
        pdf, png = story_trajectory(
            embed, ref_df, pair_of, args.dataset, args.backbone, args.seed,
            method, STORY_EPISODES_CUM,
        )
        paths += [pdf, png]

    # 3) Exclusive vs cumulative for ProbCover and Entropy at story episode
    for method in ["probcover", "entropy", "random"]:
        pdf, png = story_excl_vs_cum(
            embed, ref_df, pair_of, args.dataset, args.backbone, args.seed,
            method, args.episode,
        )
        paths += [pdf, png]

    # 4) A few clean single pairs for captions
    for method in ["random", "probcover"]:
        for ep in [10, 60]:
            indices = load_lset(args.dataset, args.backbone, method, args.seed, ep)
            pdf, png = story_pair_figure(
                embed, ref_df, pair_of, indices, method, ep, "cumulative",
                args.dataset, args.backbone,
            )
            paths += [pdf, png]

    write_story_guide(args.dataset, args.backbone, paths)
    print("[DONE] story panels in", FIG_DIR)


if __name__ == "__main__":
    main()
