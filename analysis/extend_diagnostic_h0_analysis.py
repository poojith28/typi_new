#!/usr/bin/env python3
"""
Extended H0 analyses on top of generate_diagnostic_h0_figures.py outputs.

Adds:
  - method-family contrasts (coverage / uncertainty / random)
  - persistence bias vs reference
  - touch residual (touch fraction − labelled fraction)
  - method fingerprint heatmaps
  - persistence density log-ratio at key episodes
  - early-vs-late exclusive specialization
  - cross-backbone ranking consistency
  - enriched findings write-up

Usage:
  python analysis/extend_diagnostic_h0_analysis.py
  python analysis/extend_diagnostic_h0_analysis.py --backbones resnet18
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path("/vast/s219110279")
OUT = REPO / "outputs" / "diagnostic_h0"
FIG = REPO / "figures" / "diagnostic_h0"
REF = OUT / "reference"
CACHE = OUT / "h0_cache"

DATASETS = ["CIFAR10", "CIFAR100", "TINYIMAGENET"]
DATASET_LABELS = {
    "CIFAR10": "CIFAR-10",
    "CIFAR100": "CIFAR-100",
    "TINYIMAGENET": "TinyImageNet",
}
BACKBONES = ["resnet18", "resnet50", "alexnet"]
BACKBONE_LABELS = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "alexnet": "AlexNet",
}
METHOD_ORDER = [
    "random", "uncertainty", "entropy", "margin", "dbal",
    "coreset", "probcover", "typiclust", "maxherding",
]
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
METHOD_STYLE = {
    "random":      {"color": "#000000", "linestyle": "-",  "linewidth": 2.0},
    "uncertainty": {"color": "#E69F00", "linestyle": "-",  "linewidth": 2.0},
    "entropy":     {"color": "#56B4E9", "linestyle": "--", "linewidth": 2.0},
    "margin":      {"color": "#009E73", "linestyle": "-.", "linewidth": 2.0},
    "dbal":        {"color": "#F0E442", "linestyle": ":",  "linewidth": 2.4},
    "coreset":     {"color": "#0072B2", "linestyle": "-",  "linewidth": 2.0},
    "probcover":   {"color": "#D55E00", "linestyle": "--", "linewidth": 2.2},
    "typiclust":   {"color": "#CC79A7", "linestyle": "-.", "linewidth": 2.2},
    "maxherding":  {"color": "#999999", "linestyle": ":",  "linewidth": 2.4},
}
FAMILY = {
    "random": "Random",
    "uncertainty": "Uncertainty",
    "entropy": "Uncertainty",
    "margin": "Uncertainty",
    "dbal": "Uncertainty",
    "coreset": "Coverage",
    "probcover": "Coverage",
    "typiclust": "Coverage",
    "maxherding": "Coverage",
}
FAMILY_COLORS = {
    "Random": "#000000",
    "Uncertainty": "#E69F00",
    "Coverage": "#0072B2",
}
KEY_EARLY_LATE = [10, 90]
KEY_DENSITY = [10, 30, 60, 100]
BATCH_SIZE = 50
POOL_N = {"CIFAR10": 50000, "CIFAR100": 50000, "TINYIMAGENET": 100000}


def configure_matplotlib():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "pdf.fonttype": 42,
    })


def save_fig(fig, stem):
    pdf = FIG / f"{stem}.pdf"
    png = FIG / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=600)
    fig.savefig(png, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return pdf, png


def _budget_to_episode(budget):
    return np.asarray(budget, dtype=float) / BATCH_SIZE - 1.0


def _episode_to_budget(episode):
    return BATCH_SIZE * (np.asarray(episode, dtype=float) + 1.0)


def add_episode_axis(ax):
    ax.spines["top"].set_visible(True)
    sec = ax.secondary_xaxis("top", functions=(_budget_to_episode, _episode_to_budget))
    sec.set_xlabel("Episode", labelpad=4)


def load_tables():
    long = pd.read_csv(OUT / "diagnostic_h0_all_backbones_long.csv")
    summary = pd.read_csv(OUT / "diagnostic_h0_all_backbones_summary.csv")
    long["family"] = long["method"].map(FAMILY)
    summary["family"] = summary["method"].map(FAMILY)
    summary["labeled_fraction"] = summary.apply(
        lambda r: r["labeled_budget"] / POOL_N.get(r["dataset"], np.nan), axis=1
    )
    if "cumulative_touched_fraction_mean" in summary.columns:
        summary["touch_residual_mean"] = (
            summary["cumulative_touched_fraction_mean"] - summary["labeled_fraction"]
        )
    # persistence bias vs reference median (computed later per ds/bb)
    return long, summary


def ref_persistence_stats(dataset, backbone):
    p = REF / f"{dataset}_{backbone}_h0_reference.csv"
    df = pd.read_csv(p, usecols=["persistence", "death"])
    pers = df["persistence"].to_numpy(dtype=float)
    return {
        "mean": float(np.nanmean(pers)),
        "median": float(np.nanmedian(pers)),
        "values": pers,
    }


def make_family_curves(summary, backbone, metric_mean, metric_sem, ylabel, stem, title):
    """Mean±SEM within method family (Coverage / Uncertainty / Random)."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), sharey=False)
    for ax, ds, letter in zip(axes, DATASETS, ["(a)", "(b)", "(c)"]):
        panel = summary[(summary.backbone == backbone) & (summary.dataset == ds)]
        for fam, color in FAMILY_COLORS.items():
            sub = panel[panel.family == fam]
            if sub.empty:
                continue
            # average methods inside family at each round
            g = sub.groupby("labeled_budget").agg(
                y=(metric_mean, "mean"),
                # rough SEM: mean of method SEMs / sqrt(n_methods) — descriptive only
                sem=(metric_sem, "mean"),
                n=("method", "nunique"),
            ).reset_index().sort_values("labeled_budget")
            y = g["y"].to_numpy(dtype=float)
            sem = g["sem"].to_numpy(dtype=float) / np.sqrt(np.maximum(g["n"].to_numpy(), 1))
            x = g["labeled_budget"].to_numpy()
            m = np.isfinite(y)
            ax.plot(x[m], y[m], color=color, lw=2.2, label=fam)
            ax.fill_between(x[m], y[m] - sem[m], y[m] + sem[m], color=color, alpha=0.18, lw=0)
        ax.set_title(f"{letter} {DATASET_LABELS[ds]}", pad=28)
        ax.set_xlabel("Labelled budget")
        if ax is axes[0]:
            ax.set_ylabel(ylabel)
        ax.spines["right"].set_visible(False)
        add_episode_axis(ax)
    fig.suptitle(title, fontsize=15, y=1.04, fontweight="bold")
    handles = [mpl.lines.Line2D([0], [0], color=c, lw=2.2) for c in FAMILY_COLORS.values()]
    fig.legend(handles, list(FAMILY_COLORS.keys()), loc="lower center", ncol=3,
               frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.10, 1, 0.96])
    return save_fig(fig, stem)


def make_method_curves(summary, backbone, mean_col, sem_col, ylabel, stem, title):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), sharey=False)
    present = set()
    for ax, ds, letter in zip(axes, DATASETS, ["(a)", "(b)", "(c)"]):
        panel = summary[(summary.backbone == backbone) & (summary.dataset == ds)]
        for method in METHOD_ORDER:
            sub = panel[panel.method == method].sort_values("labeled_budget")
            if sub.empty or mean_col not in sub.columns:
                continue
            s = METHOD_STYLE[method]
            x = sub["labeled_budget"].to_numpy()
            y = sub[mean_col].to_numpy(dtype=float)
            sem = sub[sem_col].to_numpy(dtype=float) if sem_col in sub.columns else np.zeros_like(y)
            m = np.isfinite(y)
            if not m.any():
                continue
            ax.plot(x[m], y[m], color=s["color"], linestyle=s["linestyle"],
                    lw=s["linewidth"], label=METHOD_DISPLAY[method])
            if np.any(np.isfinite(sem[m]) & (sem[m] > 0)):
                ax.fill_between(x[m], y[m] - sem[m], y[m] + sem[m],
                                color=s["color"], alpha=0.15, lw=0)
            present.add(method)
        ax.set_title(f"{letter} {DATASET_LABELS[ds]}", pad=28)
        ax.set_xlabel("Labelled budget")
        if ax is axes[0]:
            ax.set_ylabel(ylabel)
        ax.spines["right"].set_visible(False)
        add_episode_axis(ax)
    handles, labels = [], []
    for m in METHOD_ORDER:
        if m not in present:
            continue
        s = METHOD_STYLE[m]
        handles.append(mpl.lines.Line2D([0], [0], color=s["color"], linestyle=s["linestyle"], lw=s["linewidth"]))
        labels.append(METHOD_DISPLAY[m])
    fig.suptitle(title, fontsize=15, y=1.04, fontweight="bold")
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    return save_fig(fig, stem)


def make_fingerprint(summary, backbone, stem):
    """Late-budget fingerprint: z-scored metrics within each dataset."""
    metrics = [
        ("cumulative_touched_fraction_mean", "TouchFrac"),
        ("touch_residual_mean", "TouchResidual"),
        ("cumulative_wasserstein_mean", "W1"),
        ("cumulative_ks_mean", "KS"),
        ("cumulative_l1_ecdf_mean", "L1ECDF"),
        ("beta0_auc_normalised_mean", "β0AUC"),
        ("epsilon_50_mean", "ε50"),
        ("pers_bias_mean_mean", "PersBias"),
        ("test_accuracy_mean", "Acc"),
    ]
    # late window: last 10 episodes present
    rows = []
    for ds in DATASETS:
        panel = summary[(summary.backbone == backbone) & (summary.dataset == ds)]
        if panel.empty:
            continue
        rmax = int(panel["round"].max())
        late = panel[panel["round"] >= max(0, rmax - 9)]
        for method in METHOD_ORDER:
            sub = late[late.method == method]
            if sub.empty:
                continue
            rec = {"dataset": ds, "method": method}
            for col, _ in metrics:
                if col in sub.columns:
                    rec[col] = float(sub[col].mean())
                else:
                    rec[col] = np.nan
            rows.append(rec)
    if not rows:
        return None, None
    df = pd.DataFrame(rows)
    # z-score within dataset
    for col, _ in metrics:
        if col not in df.columns:
            continue
        for ds in DATASETS:
            m = df.dataset == ds
            vals = df.loc[m, col].astype(float)
            mu, sd = vals.mean(), vals.std(ddof=0)
            df.loc[m, col] = (vals - mu) / sd if sd > 1e-12 else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.8), sharey=True)
    for ax, ds, letter in zip(axes, DATASETS, ["(a)", "(b)", "(c)"]):
        sub = df[df.dataset == ds].set_index("method").reindex(METHOD_ORDER)
        mat = sub[[c for c, _ in metrics if c in sub.columns]].to_numpy(dtype=float)
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_yticks(range(len(METHOD_ORDER)))
        ax.set_yticklabels([METHOD_DISPLAY[m] for m in METHOD_ORDER])
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([lab for _, lab in metrics], rotation=45, ha="right")
        ax.set_title(f"{letter} {DATASET_LABELS[ds]}")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="Within-dataset z-score")
    fig.suptitle(f"H0 method fingerprint (late rounds) — {BACKBONE_LABELS[backbone]}",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, stem)


def make_early_late_scatter(summary, backbone, stem):
    """Exclusive mean persistence: E10 vs E90 (batch specialization)."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0))
    for ax, ds, letter in zip(axes, DATASETS, ["(a)", "(b)", "(c)"]):
        panel = summary[(summary.backbone == backbone) & (summary.dataset == ds)]
        e10 = panel[panel["round"] == 10].set_index("method")
        e90 = panel[panel["round"] == 90].set_index("method")
        for method in METHOD_ORDER:
            if method not in e10.index or method not in e90.index:
                continue
            x = float(e10.loc[method, "exclusive_mean_persistence_mean"])
            y = float(e90.loc[method, "exclusive_mean_persistence_mean"])
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            s = METHOD_STYLE[method]
            ax.scatter([x], [y], color=s["color"], s=70, zorder=3)
            ax.text(x, y, METHOD_DISPLAY[method], fontsize=7, ha="left", va="bottom")
        # diagonal
        lims = ax.get_xlim(), ax.get_ylim()
        lo = min(lims[0][0], lims[1][0])
        hi = max(lims[0][1], lims[1][1])
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, lw=1)
        ax.set_title(f"{letter} {DATASET_LABELS[ds]}")
        ax.set_xlabel("Exclusive mean persistence @ E10")
        if ax is axes[0]:
            ax.set_ylabel("Exclusive mean persistence @ E90")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(
        f"Early vs late batch persistence specialization — {BACKBONE_LABELS[backbone]}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    return save_fig(fig, stem)


def make_density_logratio(backbone, stem_prefix):
    """2D-ish 1D persistence density log-ratio: selected vs reference (hist)."""
    generated = []
    for dataset in DATASETS:
        ref_p = REF / f"{dataset}_{backbone}_h0_reference.csv"
        if not ref_p.exists():
            continue
        ref = pd.read_csv(ref_p, usecols=["persistence"])["persistence"].to_numpy(dtype=float)
        ref = ref[np.isfinite(ref)]
        bins = np.linspace(np.nanmin(ref), np.nanmax(ref), 60)
        fig, axes = plt.subplots(1, len(KEY_DENSITY), figsize=(3.5 * len(KEY_DENSITY), 4.6), sharey=True)
        if len(KEY_DENSITY) == 1:
            axes = [axes]
        for ax, r in zip(axes, KEY_DENSITY):
            # reference density
            Href, _ = np.histogram(ref, bins=bins, density=True)
            for method in METHOD_ORDER:
                # pool seeds from cache
                vals = []
                for seed in range(1, 6):
                    cp = CACHE / backbone / dataset / method / f"seed{seed}_h0_metrics.npz"
                    if not cp.exists():
                        continue
                    z = np.load(cp, allow_pickle=True)
                    kr = z["key_round"].astype(int)
                    if r not in kr:
                        continue
                    i = int(np.where(kr == r)[0][0])
                    vals.append(np.asarray(z["key_cum_pers"][i], dtype=float))
                if not vals:
                    continue
                v = np.concatenate(vals)
                v = v[np.isfinite(v)]
                if len(v) < 10:
                    continue
                Hsel, _ = np.histogram(v, bins=bins, density=True)
                ratio = np.log((Hsel + 1e-6) / (Href + 1e-6))
                centers = 0.5 * (bins[:-1] + bins[1:])
                s = METHOD_STYLE[method]
                ax.plot(centers, ratio, color=s["color"], linestyle=s["linestyle"],
                        lw=1.6, label=METHOD_DISPLAY[method])
            ax.axhline(0, color="k", lw=1, alpha=0.5)
            ax.set_title(f"E{r}")
            ax.set_xlabel("Persistence")
            if ax is axes[0]:
                ax.set_ylabel("log density ratio (touched / reference)")
        fig.suptitle(
            f"Persistence density log-ratio — {BACKBONE_LABELS[backbone]} / {DATASET_LABELS[dataset]}",
            fontsize=13, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        tag = {"CIFAR10": "cifar10", "CIFAR100": "cifar100", "TINYIMAGENET": "tinyimagenet"}[dataset]
        pdf, png = save_fig(fig, f"{stem_prefix}_{tag}")
        generated += [pdf, png]
    return generated


def make_cross_backbone_ranks(summary, path_csv, stem):
    """Rank methods by late cumulative W1 (lower better) per dataset; compare across backbones."""
    rows = []
    for ds in DATASETS:
        for bb in BACKBONES:
            panel = summary[(summary.dataset == ds) & (summary.backbone == bb)]
            if panel.empty:
                continue
            rmax = int(panel["round"].max())
            late = panel[panel["round"] >= max(0, rmax - 9)]
            for method in METHOD_ORDER:
                sub = late[late.method == method]
                if sub.empty:
                    continue
                rows.append({
                    "dataset": ds, "backbone": bb, "method": method,
                    "method_display": METHOD_DISPLAY[method],
                    "family": FAMILY[method],
                    "w1_late": float(sub["cumulative_wasserstein_mean"].mean()),
                    "touch_residual_late": float(sub["touch_residual_mean"].mean())
                    if "touch_residual_mean" in sub else np.nan,
                    "acc_late": float(sub["test_accuracy_mean"].mean()),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df, None, None
    df["w1_rank"] = df.groupby(["dataset", "backbone"])["w1_late"].rank(method="average")
    df.to_csv(path_csv, index=False)

    # heatmap of ranks for resnet18 focus datasets stacked by backbone
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), sharey=True)
    for ax, ds, letter in zip(axes, DATASETS, ["(a)", "(b)", "(c)"]):
        mat = []
        for bb in BACKBONES:
            sub = df[(df.dataset == ds) & (df.backbone == bb)].set_index("method").reindex(METHOD_ORDER)
            mat.append(sub["w1_rank"].to_numpy(dtype=float))
        mat = np.vstack(mat)
        im = ax.imshow(mat, aspect="auto", cmap="viridis_r", vmin=1, vmax=len(METHOD_ORDER))
        ax.set_yticks(range(len(BACKBONES)))
        ax.set_yticklabels([BACKBONE_LABELS[b] for b in BACKBONES])
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels([METHOD_DISPLAY[m] for m in METHOD_ORDER], rotation=55, ha="right", fontsize=8)
        ax.set_title(f"{letter} {DATASET_LABELS[ds]}")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="Rank by late W1 (1=closest to ref)")
    fig.suptitle("Cross-backbone ranking consistency (late cumulative Wasserstein)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    pdf, png = save_fig(fig, stem)
    return df, pdf, png


def write_findings(summary, ranks, corr_path, out_path, backbone_focus="resnet18"):
    lines = [
        "# H0 extended findings (descriptive)",
        "",
        "Generated by `analysis/extend_diagnostic_h0_analysis.py`.",
        "All statements are associative, not causal.",
        "",
        f"## Focus backbone: {BACKBONE_LABELS.get(backbone_focus, backbone_focus)}",
        "",
    ]
    # Family W1 at mid/late
    for ds in DATASETS:
        panel = summary[(summary.backbone == backbone_focus) & (summary.dataset == ds)]
        if panel.empty:
            continue
        rmax = int(panel["round"].max())
        late = panel[panel["round"] >= max(0, rmax - 9)]
        fam = late.groupby("family")["cumulative_wasserstein_mean"].mean().sort_values()
        lines.append(f"### {DATASET_LABELS[ds]}")
        lines.append("")
        lines.append("Late-round mean cumulative W1 by family (lower = closer to reference persistence):")
        for f, v in fam.items():
            lines.append(f"- **{f}**: {v:.4f}")
        # best/worst methods
        by_m = late.groupby("method")["cumulative_wasserstein_mean"].mean().sort_values()
        if len(by_m):
            lines.append(
                f"- Closest to reference: **{METHOD_DISPLAY[by_m.index[0]]}** (W1={by_m.iloc[0]:.4f})"
            )
            lines.append(
                f"- Farthest from reference: **{METHOD_DISPLAY[by_m.index[-1]]}** (W1={by_m.iloc[-1]:.4f})"
            )
        # touch residual
        if "touch_residual_mean" in late.columns:
            tr = late.groupby("method")["touch_residual_mean"].mean().sort_values(ascending=False)
            lines.append(
                f"- Highest touch residual (above labelled fraction): "
                f"**{METHOD_DISPLAY[tr.index[0]]}** ({tr.iloc[0]:+.4f})"
            )
        lines.append("")

    if ranks is not None and len(ranks):
        lines += ["## Cross-backbone note", ""]
        for ds in DATASETS:
            sub = ranks[ranks.dataset == ds]
            if sub.empty:
                continue
            # Spearman of ranks between r18 and r50
            a = sub[sub.backbone == "resnet18"].set_index("method")["w1_rank"]
            b = sub[sub.backbone == "resnet50"].set_index("method")["w1_rank"]
            common = a.index.intersection(b.index)
            if len(common) >= 3:
                rho = spearmanr(a.loc[common], b.loc[common]).correlation
                lines.append(
                    f"- {DATASET_LABELS[ds]}: Spearman rank agreement ResNet-18 vs ResNet-50 "
                    f"on late W1 = {rho:.3f}"
                )
        lines.append("")

    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        lines += ["## Accuracy associations (|Spearman| on cumulative W1)", ""]
        if "spearman_cumulative_wasserstein" in corr.columns:
            tmp = corr[corr.backbone == backbone_focus].dropna(subset=["spearman_cumulative_wasserstein"]).copy()
            tmp["abs"] = tmp["spearman_cumulative_wasserstein"].abs()
            for _, r in tmp.nlargest(8, "abs").iterrows():
                lines.append(
                    f"- {DATASET_LABELS.get(r.dataset, r.dataset)} / {r.method_display}: "
                    f"{r.spearman_cumulative_wasserstein:.3f}"
                )
        lines.append("")

    lines += [
        "## Interpretation caveats",
        "",
        "- Touch fraction closely tracks labelled fraction because each finite H0 pair has a unique birth vertex;",
        "  prefer **touch residual**, **W1/KS/L1**, and **β₀ shape** for method differences.",
        "- Density log-ratio and early/late exclusive persistence show *which scales* are preferred,",
        "  not whether those choices cause accuracy gains.",
        "",
        "## New figures",
        "",
        "- `diagnostic_h0_*_family_wasserstein_combined.*`",
        "- `diagnostic_h0_*_touch_residual_combined.*`",
        "- `diagnostic_h0_*_pers_bias_combined.*`",
        "- `diagnostic_h0_*_fingerprint_heatmap.*`",
        "- `diagnostic_h0_*_early_late_persistence.*`",
        "- `diagnostic_h0_*_density_logratio_*.*`",
        "- `diagnostic_h0_cross_backbone_w1_ranks.*`",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def extend_summary_with_bias(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    bias_mean = []
    bias_med = []
    for _, r in summary.iterrows():
        try:
            st = ref_persistence_stats(r["dataset"], r["backbone"])
        except Exception:
            bias_mean.append(np.nan)
            bias_med.append(np.nan)
            continue
        bias_mean.append(float(r.get("cumulative_mean_persistence_mean", np.nan)) - st["mean"])
        bias_med.append(float(r.get("cumulative_median_persistence_mean", np.nan)) - st["median"])
    summary["pers_bias_mean_mean"] = bias_mean
    summary["pers_bias_median_mean"] = bias_med
    summary["pers_bias_mean_sem"] = 0.0
    summary["pers_bias_median_sem"] = 0.0
    # fake sem columns for residual plotting
    summary["touch_residual_sem"] = summary.get("cumulative_touched_fraction_sem", 0.0)
    return summary


def update_figure_guide():
    guide = OUT / "FIGURE_GUIDE.md"
    extra = """

---

## Extended analyses (added)

### Family Wasserstein
**File:** `diagnostic_h0_<backbone>_family_wasserstein_combined.pdf`

- Averages methods into **Coverage / Uncertainty / Random** families.
- **Shows:** whether coverage-style strategies expose reference persistence scales differently from uncertainty-style ones.

### Touch residual
**File:** `diagnostic_h0_<backbone>_touch_residual_combined.pdf`

- **Y:** cumulative touched fraction − labelled fraction (|L_t|/N).
- **Why:** raw touch fraction mostly tracks budget (unique birth vertices); residual highlights over/under-exposure of H0 births.

### Persistence bias
**File:** `diagnostic_h0_<backbone>_pers_bias_combined.pdf`

- **Y:** mean touched persistence − reference mean persistence.
- **Shows:** bias toward shorter or longer H0 bars than the pool average.

### Method fingerprint
**File:** `diagnostic_h0_<backbone>_fingerprint_heatmap.pdf`

- Late-round z-scores of H0 metrics (+ accuracy) within each dataset.
- **Shows:** compact structural signature of each method.

### Early vs late exclusive persistence
**File:** `diagnostic_h0_<backbone>_early_late_persistence.pdf`

- Exclusive mean persistence at **E10 vs E90**.
- **Shows:** whether batch complexity preference shifts over the AL trajectory.

### Persistence density log-ratio
**Files:** `diagnostic_h0_<backbone>_density_logratio_{dataset}.pdf`

- log(density_touched / density_reference) over persistence bins at E10/30/60/100.
- **Shows:** which persistence scales are over- or under-sampled relative to the full pool.

### Cross-backbone W1 ranks
**File:** `diagnostic_h0_cross_backbone_w1_ranks.pdf`

- Method ranks by late cumulative W1 across ResNet-18/50/AlexNet.
- **Shows:** whether H0 exposure rankings are backbone-stable.

Findings write-up: `outputs/diagnostic_h0/diagnostic_h0_extended_findings.md`
"""
    text = guide.read_text() if guide.exists() else "# H0 FIGURE GUIDE\n"
    if "Extended analyses" not in text:
        guide.write_text(text.rstrip() + "\n" + extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["resnet18"], choices=BACKBONES)
    args = ap.parse_args()
    configure_matplotlib()
    FIG.mkdir(parents=True, exist_ok=True)

    long, summary = load_tables()
    summary = extend_summary_with_bias(summary)
    # persist enriched summary
    enriched = OUT / "diagnostic_h0_all_backbones_summary_enriched.csv"
    summary.to_csv(enriched, index=False)
    print(f"Wrote {enriched}")

    generated = [enriched]
    for bb in args.backbones:
        print(f"\n===== extended figures: {bb} =====")
        pdf, png = make_family_curves(
            summary, bb,
            "cumulative_wasserstein_mean", "cumulative_wasserstein_sem",
            "Family-mean cumulative W1",
            f"diagnostic_h0_{bb}_family_wasserstein_combined",
            f"Method-family H0 Wasserstein — {BACKBONE_LABELS[bb]}",
        )
        generated += [pdf, png]

        pdf, png = make_method_curves(
            summary, bb,
            "touch_residual_mean", "touch_residual_sem",
            "Touch residual (touch frac − labelled frac)",
            f"diagnostic_h0_{bb}_touch_residual_combined",
            f"H0 touch residual — {BACKBONE_LABELS[bb]}",
        )
        generated += [pdf, png]

        pdf, png = make_method_curves(
            summary, bb,
            "pers_bias_mean_mean", "pers_bias_mean_sem",
            "Persistence bias (touched mean − ref mean)",
            f"diagnostic_h0_{bb}_pers_bias_combined",
            f"Touched persistence bias vs reference — {BACKBONE_LABELS[bb]}",
        )
        generated += [pdf, png]

        out = make_fingerprint(summary, bb, f"diagnostic_h0_{bb}_fingerprint_heatmap")
        if out[0] is not None:
            generated += list(out)

        pdf, png = make_early_late_scatter(
            summary, bb, f"diagnostic_h0_{bb}_early_late_persistence"
        )
        generated += [pdf, png]

        generated += make_density_logratio(bb, f"diagnostic_h0_{bb}_density_logratio")

    ranks, pdf, png = make_cross_backbone_ranks(
        summary,
        OUT / "diagnostic_h0_cross_backbone_w1_ranks.csv",
        "diagnostic_h0_cross_backbone_w1_ranks",
    )
    if pdf is not None:
        generated += [pdf, png, OUT / "diagnostic_h0_cross_backbone_w1_ranks.csv"]

    write_findings(
        summary, ranks,
        OUT / "diagnostic_h0_accuracy_relationship.csv",
        OUT / "diagnostic_h0_extended_findings.md",
        backbone_focus=args.backbones[0],
    )
    generated.append(OUT / "diagnostic_h0_extended_findings.md")
    update_figure_guide()

    print("\nGenerated:")
    for g in generated:
        print(" ", g)


if __name__ == "__main__":
    main()
