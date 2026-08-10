#!/usr/bin/env python3
"""
Intrinsic-dimensionality diagnostics for the thesis chapter
"Diagnostic Analysis of Active Learning".

FULL-DATASET POLICY (no sparsifying):
  - Local ID: Levina–Bickel MLE on the FULL fixed training embedding pool.
  - Exact FAISS IndexFlatL2 / sklearn kNN (never IVF / approximate).
  - Global ID: recomputed on the FULL cumulative labelled set L_t
    (neighbours restricted to L_t; no point subsampling).
  - Missing runs are reported, never invented.

Usage:
  python analysis/generate_diagnostic_id_figures.py
  python analysis/generate_diagnostic_id_figures.py --backbones resnet18
  python analysis/generate_diagnostic_id_figures.py --backbones resnet18 resnet50 alexnet
  python analysis/generate_diagnostic_id_figures.py --skip-figures   # data only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Make posthoc_graph_diagnostics importable
REPO_ROOT = Path("/vast/s219110279")
sys.path.insert(0, str(REPO_ROOT))

from posthoc_graph_diagnostics.id_diagnostics import (  # noqa: E402
    _levina_bickel_from_sorted_distances,
    compute_global_mle_id,
)
from posthoc_graph_diagnostics.neighbors import knn_query  # noqa: E402

# ---------------------------------------------------------------------------
# Paths / scope
# ---------------------------------------------------------------------------
OUTPUT_ROOT = REPO_ROOT / "TypiClust" / "output"
FEATURES_ROOT = REPO_ROOT / "results" / "results"
FIGURES_DIR = REPO_ROOT / "figures" / "diagnostic_id"
OUTPUTS_DIR = REPO_ROOT / "outputs" / "diagnostic_id"
ID_CACHE_ROOT = OUTPUTS_DIR / "id_cache"  # Levina–Bickel caches (full pool / full L_t)
ACCURACY_OUTPUTS_DIR = REPO_ROOT / "outputs" / "diagnostic_accuracy"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
ID_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = ["CIFAR10", "CIFAR100", "TINYIMAGENET"]
DATASET_LABELS = {
    "CIFAR10": "CIFAR-10",
    "CIFAR100": "CIFAR-100",
    "TINYIMAGENET": "TinyImageNet",
}
FEATURE_DATASET_DIR = {
    "CIFAR10": "cifar-10",
    "CIFAR100": "cifar-100",
    "TINYIMAGENET": "tiny-imagenet",
}
BACKBONES = ["resnet18", "resnet50", "alexnet"]
BACKBONE_LABELS = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "alexnet": "AlexNet",
}
SEEDS = [1, 2, 3, 4, 5]
BATCH_SIZE = 50
# Cumulative / trajectory snapshots (global ID, accuracy relations, tables).
# E100 is valid here: labelled set L_t exists and has accuracy.
KEY_ROUNDS = [10, 25, 50, 100]
# Batch-level only (regime bars, ID distributions). E100 has no ΔL (final
# episode does not acquire), so use the last episode that still selects a batch.
BATCH_KEY_ROUNDS = [10, 25, 50, 75]
DIST_ROUNDS = [10, 30, 60, 90]
MAIN_K = 50
SENS_K = [20, 50, 75]

METHOD_DIR_PREFIX = {
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
METHOD_ORDER = list(METHOD_DIR_PREFIX.keys())
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
# Same colours/styles as predictive-accuracy figures
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
ACCURACY_ZONE = {
    "CIFAR10": (0.0, 96.0),
    "CIFAR100": (0.0, 80.0),
    "TINYIMAGENET": (0.0, 70.0),
}


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
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _budget_to_episode(budget):
    return np.asarray(budget, dtype=float) / BATCH_SIZE - 1.0


def _episode_to_budget(episode):
    return BATCH_SIZE * (np.asarray(episode, dtype=float) + 1.0)


def _add_episode_axis(ax):
    ax.spines["top"].set_visible(True)
    sec = ax.secondary_xaxis("top", functions=(_budget_to_episode, _episode_to_budget))
    sec.set_xlabel("Episode", labelpad=4)
    return sec


# ---------------------------------------------------------------------------
# Features / local ID (FULL pool, exact kNN)
# ---------------------------------------------------------------------------
def feature_path(dataset: str, backbone: str) -> Path:
    # Fixed embeddings are stored once as features_seed1.npy for each backbone/dataset
    return FEATURES_ROOT / backbone / FEATURE_DATASET_DIR[dataset] / "pretext" / "features_seed1.npy"


def l2_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(n, 1e-12)).astype(np.float32)


def load_features(dataset: str, backbone: str) -> np.ndarray:
    p = feature_path(dataset, backbone)
    if not p.exists():
        raise FileNotFoundError(f"Missing features: {p}")
    X = np.load(p).astype(np.float32)
    return l2_normalize(X)


def local_id_cache_path(dataset: str, backbone: str, k: int) -> Path:
    d = ID_CACHE_ROOT / backbone / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d / f"levina_bickel_local_k{k}_fullpool_l2.npy"


def compute_or_load_local_ids(X: np.ndarray, dataset: str, backbone: str, k_values,
                              refresh_cache: bool = False) -> dict:
    """Exact Levina–Bickel local ID on the FULL pool for each k. Cached to disk."""
    out = {}
    if refresh_cache:
        for k in k_values:
            path = local_id_cache_path(dataset, backbone, k)
            if path.exists():
                path.unlink()
                print(f"[local ID] deleted stale cache {path}")

    missing = [k for k in k_values if not local_id_cache_path(dataset, backbone, k).exists()]
    if missing:
        k_max = max(k_values)
        print(f"[local ID] exact kNN FULL pool {dataset}/{backbone}: N={X.shape[0]}, k_max={k_max}")
        dist, _idx = knn_query(
            X, k=k_max, metric="euclidean", use_faiss=True, self_query=True, approx=False,
        )
        # Sanity: self-neighbour must already be removed
        d0 = dist[:, 0]
        print(
            f"[local ID] first-neighbour distance: min={float(d0.min()):.4g} "
            f"median={float(np.median(d0)):.4g} frac_zero={float(np.mean(d0 == 0)):.4g}"
        )
        if float(np.mean(d0 == 0)) > 0.01:
            raise RuntimeError(
                "kNN first distance is often zero — self-neighbour was not removed. "
                "Local ID would be invalid."
            )
        for k in k_values:
            path = local_id_cache_path(dataset, backbone, k)
            if path.exists():
                out[k] = np.load(path).astype(np.float32)
                continue
            lid = _levina_bickel_from_sorted_distances(dist, int(k)).astype(np.float32)
            med = np.nanmedian(lid)
            lid = np.nan_to_num(lid, nan=med if np.isfinite(med) else 0.0).astype(np.float32)
            np.save(path, lid)
            print(f"[local ID] saved {path}")
            out[k] = lid
    for k in k_values:
        if k not in out:
            out[k] = np.load(local_id_cache_path(dataset, backbone, k)).astype(np.float32)
            print(f"[local ID] loaded {local_id_cache_path(dataset, backbone, k)}")
    return out


# ---------------------------------------------------------------------------
# Selection / accuracy loaders
# ---------------------------------------------------------------------------
def _plausible_acc(acc, dataset):
    try:
        acc = float(acc)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(acc) or acc < 0 or acc > 100 + 1e-6:
        return False
    lo, hi = ACCURACY_ZONE.get(dataset, (0.0, 100.0))
    return lo <= acc <= hi


def _load_index_npy(path: Path) -> np.ndarray:
    """Load index arrays that may be object-dtype (allow_pickle required)."""
    arr = np.load(path, allow_pickle=True)
    if arr.dtype == object:
        arr = np.asarray(arr.tolist())
    return np.asarray(arr, dtype=np.int64).reshape(-1)


def load_run_rounds(dataset, backbone, method, seed):
    """Return list of dicts with round, batch, cumulative, test_accuracy — or None.

    Convention verified on TypiClust logs:
      - `active_set_ids` = newly acquired batch ΔL (size ≈ budget, usually 50)
      - `lSet.npy` / labeled_count_before_sampling = labelled set used for
        training+accuracy at this episode (aligned with each other)
      - Final episode may have active_set_ids=None when no further acquisition
    """
    prefix = METHOD_DIR_PREFIX[method]
    run_dir = OUTPUT_ROOT / dataset / backbone / f"{prefix}_{seed}_50b"
    summary_path = run_dir / "benchmark_summary.json"
    if not summary_path.exists():
        return None, f"missing:{summary_path}"

    with open(summary_path) as f:
        data = json.load(f)
    episodes = data.get("episode_records") or []
    if not episodes:
        return None, f"empty:{summary_path}"

    rows = []
    for ep in episodes:
        r = int(ep["episode"])
        acc = ep.get("test_accuracy")
        if not _plausible_acc(acc, dataset):
            continue

        labeled_before = ep.get("labeled_count_before_sampling")
        labeled_after = ep.get("labeled_count_after_sampling")
        if labeled_before is None:
            labeled_before = BATCH_SIZE + r * BATCH_SIZE

        batch = ep.get("active_set_ids")
        cum = None
        lset_p = run_dir / f"episode_{r}" / "lSet.npy"
        if lset_p.exists():
            cum = np.sort(_load_index_npy(lset_p))
            if len(cum) != int(labeled_before):
                print(
                    f"[WARN] size mismatch {dataset}/{backbone}/{method}/seed{seed}/E{r}: "
                    f"len(lSet)={len(cum)} vs labeled_before={labeled_before}"
                )

        if batch is not None:
            batch_arr = np.sort(np.asarray(batch, dtype=np.int64).reshape(-1))
        else:
            # Final round often has no acquisition (before == after).
            if labeled_after is not None and int(labeled_after) == int(labeled_before):
                batch_arr = np.array([], dtype=np.int64)
            elif cum is None:
                continue
            elif r == 0:
                batch_arr = cum.copy()
            else:
                prev_p = run_dir / f"episode_{r-1}" / "lSet.npy"
                if prev_p.exists():
                    prev = np.sort(_load_index_npy(prev_p))
                    batch_arr = np.setdiff1d(cum, prev, assume_unique=False)
                else:
                    batch_arr = np.array([], dtype=np.int64)

        rows.append({
            "round": r,
            "batch": batch_arr.astype(np.int64),
            "cumulative": None if cum is None else cum.astype(np.int64),
            "test_accuracy": float(acc),
            "labeled_budget": int(labeled_before),
        })

    # Fill cumulative if missing by progressive union of batches
    # (only used when lSet.npy absent; prefer lSet when present).
    if rows and any(r["cumulative"] is None for r in rows):
        running = np.array([], dtype=np.int64)
        for row in rows:
            if row["cumulative"] is None:
                running = np.union1d(running, row["batch"])
                row["cumulative"] = running.copy()
            else:
                running = row["cumulative"]

    if not rows:
        return None, f"no_valid_rounds:{summary_path}"
    return rows, None


# ---------------------------------------------------------------------------
# Per-run ID metrics (FULL L_t for global ID)
# ---------------------------------------------------------------------------
def global_id_cache_path(dataset, backbone, method, seed):
    d = ID_CACHE_ROOT / backbone / dataset / "global_id" / method
    d.mkdir(parents=True, exist_ok=True)
    return d / f"seed{seed}_k{MAIN_K}_fullLt.npz"


def compute_run_id_metrics(X, local_ids_k, rounds, dataset, backbone, method, seed,
                           refresh_cache: bool = False):
    """Compute exclusive/cumulative local ID + global ID for every round.

    Global ID uses the FULL cumulative labelled set (exact kNN inside L_t).
    Results cached per run. On cache hit, batch_local_ids are still recomputed
    from the current batch indices so distribution plots stay correct.
    """
    cache_p = global_id_cache_path(dataset, backbone, method, seed)
    if refresh_cache and cache_p.exists():
        cache_p.unlink()
        print(f"[global ID] deleted stale cache {cache_p}")

    lid = local_ids_k[MAIN_K]
    q25, q75 = np.quantile(lid, [0.25, 0.75])

    if cache_p.exists():
        z = np.load(cache_p, allow_pickle=True)
        cached_rounds = z["rounds"].astype(int)
        if set(cached_rounds.tolist()) >= {r["round"] for r in rounds}:
            by_r = {int(r): i for i, r in enumerate(cached_rounds)}
            out = []
            for row in rounds:
                i = by_r.get(row["round"])
                if i is None:
                    continue
                batch = row["batch"]
                batch = batch[(batch >= 0) & (batch < len(lid))]
                batch_lids = lid[batch].astype(np.float32)
                out.append({
                    "dataset": dataset,
                    "backbone": backbone,
                    "method": method,
                    "method_display": METHOD_DISPLAY[method],
                    "seed": seed,
                    "round": row["round"],
                    "labeled_budget": row["labeled_budget"],
                    "exclusive_local_id_mean": float(z["exclusive"][i]),
                    "cumulative_local_id_mean": float(z["cumulative"][i]),
                    "global_id": float(z["global_id"][i]),
                    "test_accuracy": row["test_accuracy"],
                    "n_selected_batch": int(z["n_batch"][i]),
                    "n_labelled_cumulative": int(z["n_cum"][i]),
                    "frac_low_id": float(z["frac_low"][i]),
                    "frac_mid_id": float(z["frac_mid"][i]),
                    "frac_high_id": float(z["frac_high"][i]),
                    # Always recompute from current batch indices (not cached)
                    "batch_local_ids": batch_lids,
                })
            print(f"[global ID] cache hit {cache_p} (batch_local_ids recomputed)")
            return out

    excl, cumu, gids, n_batch, n_cum = [], [], [], [], []
    frac_low, frac_mid, frac_high = [], [], []
    batch_ids_store = []
    round_ids = []
    out = []

    print(f"[ID metrics] {dataset}/{backbone}/{method}/seed{seed}: {len(rounds)} rounds (FULL L_t)")
    for row in rounds:
        batch = row["batch"]
        cum = row["cumulative"]
        # clamp indices
        batch = batch[(batch >= 0) & (batch < len(lid))]
        cum = cum[(cum >= 0) & (cum < len(lid))]

        batch_lids = lid[batch]
        excl_mean = float(np.mean(batch_lids)) if len(batch_lids) else float("nan")
        cum_mean = float(np.mean(lid[cum])) if len(cum) else float("nan")

        # FULL cumulative set for global ID — no sparsifying
        if len(cum) > MAIN_K + 2:
            gid = compute_global_mle_id(X[cum], k=MAIN_K, metric="euclidean", use_faiss=True)
        else:
            gid = float("nan")

        low = float(np.mean(batch_lids <= q25)) if len(batch_lids) else float("nan")
        high = float(np.mean(batch_lids >= q75)) if len(batch_lids) else float("nan")
        mid = float(np.mean((batch_lids > q25) & (batch_lids < q75))) if len(batch_lids) else float("nan")

        excl.append(excl_mean)
        cumu.append(cum_mean)
        gids.append(gid)
        n_batch.append(len(batch))
        n_cum.append(len(cum))
        frac_low.append(low)
        frac_mid.append(mid)
        frac_high.append(high)
        batch_ids_store.append(batch_lids.astype(np.float32))
        round_ids.append(row["round"])

        out.append({
            "dataset": dataset,
            "backbone": backbone,
            "method": method,
            "method_display": METHOD_DISPLAY[method],
            "seed": seed,
            "round": row["round"],
            "labeled_budget": row["labeled_budget"],
            "exclusive_local_id_mean": excl_mean,
            "cumulative_local_id_mean": cum_mean,
            "global_id": gid,
            "test_accuracy": row["test_accuracy"],
            "n_selected_batch": len(batch),
            "n_labelled_cumulative": len(cum),
            "frac_low_id": low,
            "frac_mid_id": mid,
            "frac_high_id": high,
            "batch_local_ids": batch_lids.astype(np.float32),
        })

    # cache numeric series (not ragged batch_ids)
    np.savez_compressed(
        cache_p,
        rounds=np.asarray(round_ids, dtype=np.int32),
        exclusive=np.asarray(excl, dtype=np.float32),
        cumulative=np.asarray(cumu, dtype=np.float32),
        global_id=np.asarray(gids, dtype=np.float32),
        n_batch=np.asarray(n_batch, dtype=np.int32),
        n_cum=np.asarray(n_cum, dtype=np.int32),
        frac_low=np.asarray(frac_low, dtype=np.float32),
        frac_mid=np.asarray(frac_mid, dtype=np.float32),
        frac_high=np.asarray(frac_high, dtype=np.float32),
        q25=np.float32(q25),
        q75=np.float32(q75),
    )
    print(f"[global ID] saved {cache_p}")
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def summarise(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "exclusive_local_id_mean",
        "cumulative_local_id_mean",
        "global_id",
        "test_accuracy",
        "frac_low_id",
        "frac_mid_id",
        "frac_high_id",
    ]
    rows = []
    for keys, g in df.groupby(["dataset", "backbone", "method", "method_display", "round"]):
        rec = {
            "dataset": keys[0],
            "backbone": keys[1],
            "method": keys[2],
            "method_display": keys[3],
            "round": keys[4],
            "labeled_budget": int(g["labeled_budget"].median()),
            "n_seeds": int(g["seed"].nunique()),
        }
        for m in metrics:
            vals = g[m].astype(float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                rec[f"{m}_mean"] = np.nan
                rec[f"{m}_sem"] = np.nan
            else:
                rec[f"{m}_mean"] = float(vals.mean())
                rec[f"{m}_sem"] = float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["dataset", "backbone", "method", "round"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def save_fig(fig, stem):
    pdf = FIGURES_DIR / f"{stem}.pdf"
    png = FIGURES_DIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=600)
    fig.savefig(png, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return pdf, png


def _plot_metric_panel(ax, summary, dataset, backbone, mean_col, sem_col, ylabel, title):
    panel = summary[(summary.dataset == dataset) & (summary.backbone == backbone)]
    present = []
    for method in METHOD_ORDER:
        sub = panel[panel.method == method].sort_values("labeled_budget")
        if sub.empty:
            continue
        style = METHOD_STYLE[method]
        x = sub["labeled_budget"].to_numpy()
        y = sub[mean_col].to_numpy(dtype=float)
        sem = sub[sem_col].to_numpy(dtype=float)
        mask = np.isfinite(y)
        if not mask.any():
            continue
        ax.plot(x[mask], y[mask], color=style["color"], linestyle=style["linestyle"],
                linewidth=style["linewidth"], label=METHOD_DISPLAY[method], zorder=3)
        if np.any(np.isfinite(sem[mask]) & (sem[mask] > 0)):
            ax.fill_between(x[mask], y[mask] - sem[mask], y[mask] + sem[mask],
                            color=style["color"], alpha=0.18, linewidth=0, zorder=2)
        present.append(method)
    ax.set_title(title, pad=28)
    ax.set_xlabel("Labelled budget")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    ax.spines["right"].set_visible(False)
    _add_episode_axis(ax)
    return present


def _shared_legend(fig, methods, y=0.0):
    handles, labels = [], []
    for m in METHOD_ORDER:
        if m not in methods:
            continue
        s = METHOD_STYLE[m]
        handles.append(mpl.lines.Line2D([0], [0], color=s["color"], linestyle=s["linestyle"],
                                        linewidth=s["linewidth"]))
        labels.append(METHOD_DISPLAY[m])
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(handles)),
                   frameon=True, fancybox=False, edgecolor="#cccccc",
                   bbox_to_anchor=(0.5, y), handlelength=2.8)


def make_combined_metric_figure(summary, backbone, mean_col, sem_col, ylabel, stem, suptitle):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), sharey=False)
    all_m = set()
    for ax, ds, letter in zip(axes, DATASETS, ["(a)", "(b)", "(c)"]):
        present = _plot_metric_panel(
            ax, summary, ds, backbone, mean_col, sem_col, ylabel if ax is axes[0] else "",
            f"{letter} {DATASET_LABELS[ds]}",
        )
        all_m.update(present)
        if not present:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle(suptitle, fontsize=15, y=1.04, fontweight="bold")
    _shared_legend(fig, all_m, y=0.0)
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    return save_fig(fig, stem)


def make_per_dataset_metric_figure(summary, backbone, dataset, mean_col, sem_col, ylabel, stem, title):
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
    present = _plot_metric_panel(ax, summary, dataset, backbone, mean_col, sem_col, ylabel, title)
    if present:
        handles, labels = [], []
        for m in METHOD_ORDER:
            if m not in present:
                continue
            s = METHOD_STYLE[m]
            handles.append(mpl.lines.Line2D([0], [0], color=s["color"], linestyle=s["linestyle"],
                                            linewidth=s["linewidth"]))
            labels.append(METHOD_DISPLAY[m])
        ax.legend(handles, labels, loc="best", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    return save_fig(fig, stem)


def make_id_vs_accuracy_trajectories(summary, backbone, id_mean_col, stem, xlabel, title_prefix):
    """Selected rounds as trajectories: ID vs accuracy."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0), sharey=False)
    markers = {10: "o", 25: "s", 50: "D", 100: "^"}
    all_m = set()
    for ax, ds, letter in zip(axes, DATASETS, ["(a)", "(b)", "(c)"]):
        panel = summary[(summary.backbone == backbone) & (summary.dataset == ds)
                        & (summary["round"].isin(KEY_ROUNDS))]
        for method in METHOD_ORDER:
            sub = panel[panel.method == method].sort_values("round")
            if sub.empty:
                continue
            s = METHOD_STYLE[method]
            x = sub[id_mean_col].to_numpy(dtype=float)
            y = sub["test_accuracy_mean"].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 1:
                continue
            ax.plot(x[mask], y[mask], color=s["color"], linestyle=s["linestyle"],
                    linewidth=s["linewidth"], label=METHOD_DISPLAY[method], zorder=3)
            for _, r in sub.iterrows():
                if not (np.isfinite(r[id_mean_col]) and np.isfinite(r["test_accuracy_mean"])):
                    continue
                ax.scatter(r[id_mean_col], r["test_accuracy_mean"],
                           color=s["color"], marker=markers.get(int(r["round"]), "o"),
                           s=40, zorder=4, edgecolors="white", linewidths=0.4)
            all_m.add(method)
        ax.set_title(f"{letter} {DATASET_LABELS[ds]}", pad=8)
        ax.set_xlabel(xlabel)
        if ax is axes[0]:
            ax.set_ylabel("Test accuracy (%)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    # round marker legend
    round_handles = [
        mpl.lines.Line2D([0], [0], marker=markers[r], color="gray", linestyle="None",
                         markersize=7, label=f"E{r}")
        for r in KEY_ROUNDS
    ]
    fig.suptitle(title_prefix, fontsize=15, y=1.02, fontweight="bold")
    _shared_legend(fig, all_m, y=0.02)
    fig.legend(handles=round_handles, loc="lower right", title="Episode",
               fontsize=8, frameon=True, bbox_to_anchor=(0.98, 0.12))
    fig.tight_layout(rect=[0, 0.14, 1, 0.96])
    return save_fig(fig, stem)


def make_batch_id_vs_gain(df_long, backbone, stem):
    """Scatter of exclusive local ID vs next-round accuracy gain (seed-averaged)."""
    rows = []
    for (ds, method, seed), g in df_long[df_long.backbone == backbone].groupby(
            ["dataset", "method", "seed"]):
        g = g.sort_values("round")
        acc = g["test_accuracy"].to_numpy()
        excl = g["exclusive_local_id_mean"].to_numpy()
        rounds = g["round"].to_numpy()
        for i in range(len(g) - 1):
            rows.append({
                "dataset": ds, "method": method, "seed": seed,
                "round": int(rounds[i]),
                "exclusive_local_id_mean": float(excl[i]),
                "accuracy_gain": float(acc[i + 1] - acc[i]),
            })
    gain_df = pd.DataFrame(rows)
    # seed-average
    agg = (gain_df.groupby(["dataset", "method", "round"], as_index=False)
           .agg(exclusive_local_id_mean=("exclusive_local_id_mean", "mean"),
                accuracy_gain=("accuracy_gain", "mean"),
                n_seeds=("seed", "nunique")))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0))
    all_m = set()
    for ax, ds, letter in zip(axes, DATASETS, ["(a)", "(b)", "(c)"]):
        panel = agg[agg.dataset == ds]
        for method in METHOD_ORDER:
            sub = panel[panel.method == method]
            if sub.empty:
                continue
            s = METHOD_STYLE[method]
            ax.scatter(sub["exclusive_local_id_mean"], sub["accuracy_gain"],
                       color=s["color"], s=12, alpha=0.55, label=METHOD_DISPLAY[method])
            all_m.add(method)
        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.set_title(f"{letter} {DATASET_LABELS[ds]}")
        ax.set_xlabel("Exclusive local ID (batch)")
        if ax is axes[0]:
            ax.set_ylabel("Next-round Δaccuracy (%)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"Batch local ID vs next-round accuracy gain — {BACKBONE_LABELS[backbone]}",
                 fontsize=15, y=1.02, fontweight="bold")
    _shared_legend(fig, all_m, y=0.0)
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    pdf, png = save_fig(fig, stem)
    return pdf, png, gain_df, agg


def make_k_sensitivity_figure(sens_df, backbone, stem):
    """Grouped bars: Spearman(k50,k20) and Spearman(k50,k75) per dataset.

    Values are full-pool ranking correlations computed once per dataset/backbone
    (embeddings shared across AL seeds). No seed SEM — error bars omitted.
    """
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    datasets = DATASETS
    x = np.arange(len(datasets))
    width = 0.35
    means20, means75 = [], []
    for ds in datasets:
        sub = sens_df[(sens_df.backbone == backbone) & (sens_df.dataset == ds)]
        if sub.empty:
            means20.append(np.nan)
            means75.append(np.nan)
            continue
        # Prefer a single row; if legacy multi-seed copies exist, take unique value
        means20.append(float(sub["spearman_k50_k20"].astype(float).iloc[0]))
        means75.append(float(sub["spearman_k50_k75"].astype(float).iloc[0]))
    ax.bar(x - width / 2, means20, width, label="Spearman(k=50, k=20)", color="#0072B2")
    ax.bar(x + width / 2, means75, width, label="Spearman(k=50, k=75)", color="#D55E00")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets])
    ax.set_ylabel("Spearman rank correlation")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        f"Local-ID ranking robustness across k — {BACKBONE_LABELS[backbone]}\n"
        "(full-pool correlation; computed once per dataset)"
    )
    ax.legend(frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_fig(fig, stem)


def make_regime_allocation(summary, backbone, stem):
    """High-ID fraction of the newly acquired batch ΔL_t (batch-level, not cumulative)."""
    return make_combined_metric_figure(
        summary, backbone,
        "frac_high_id_mean", "frac_high_id_sem",
        "High-ID selection fraction (new batch)",
        stem,
        f"High-ID fraction of newly acquired batches — {BACKBONE_LABELS[backbone]}",
    )


def make_regime_stacked_keyrounds(summary, backbone, dataset, stem):
    """Stacked bar of low/mid/high ID fractions of the newly acquired batch at key episodes.

    Uses BATCH_KEY_ROUNDS (not E100): the final episode has no further acquisition.
    """
    rounds = BATCH_KEY_ROUNDS
    panel = summary[(summary.backbone == backbone) & (summary.dataset == dataset)
                    & (summary["round"].isin(rounds))]
    methods = [m for m in METHOD_ORDER if m in set(panel.method)]
    if not methods:
        return None, None
    fig, axes = plt.subplots(1, len(rounds), figsize=(3.2 * len(rounds), 4.8), sharey=True)
    if len(rounds) == 1:
        axes = [axes]
    colors = {"low": "#0072B2", "mid": "#999999", "high": "#D55E00"}
    for ax, r in zip(axes, rounds):
        sub = panel[panel["round"] == r]
        x = np.arange(len(methods))
        low = [float(sub[sub.method == m]["frac_low_id_mean"].values[0])
               if len(sub[sub.method == m]) else 0 for m in methods]
        mid = [float(sub[sub.method == m]["frac_mid_id_mean"].values[0])
               if len(sub[sub.method == m]) else 0 for m in methods]
        high = [float(sub[sub.method == m]["frac_high_id_mean"].values[0])
                if len(sub[sub.method == m]) else 0 for m in methods]
        ax.bar(x, low, color=colors["low"], label="Low ID (≤Q25)")
        ax.bar(x, mid, bottom=low, color=colors["mid"], label="Mid ID")
        bottom2 = np.asarray(low) + np.asarray(mid)
        ax.bar(x, high, bottom=bottom2, color=colors["high"], label="High ID (≥Q75)")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_DISPLAY[m] for m in methods], rotation=55, ha="right", fontsize=7)
        ax.set_title(f"E{r} / B{_episode_to_budget(r):.0f}")
        ax.set_ylim(0, 1)
        if ax is axes[0]:
            ax.set_ylabel("Fraction of new batch ΔL")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"ID-regime allocation of newly acquired batches — "
        f"{BACKBONE_LABELS[backbone]} / {DATASET_LABELS[dataset]}",
        y=1.08, fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    return save_fig(fig, stem)


def make_selected_id_distribution(df_long, backbone, dataset, stem):
    """Boxplots of batch local IDs at DIST_ROUNDS (excludes E100: no ΔL there)."""
    sub = df_long[(df_long.backbone == backbone) & (df_long.dataset == dataset)
                  & (df_long["round"].isin(DIST_ROUNDS))]
    if sub.empty:
        return None, None
    fig, axes = plt.subplots(1, len(DIST_ROUNDS), figsize=(3.6 * len(DIST_ROUNDS), 4.8), sharey=True)
    if len(DIST_ROUNDS) == 1:
        axes = [axes]
    methods = [m for m in METHOD_ORDER if m in set(sub.method)]
    for ax, r in zip(axes, DIST_ROUNDS):
        data, colors, labels = [], [], []
        for m in methods:
            # concatenate batch IDs across seeds if stored; else use exclusive mean as fallback
            vals = []
            for _, row in sub[(sub.method == m) & (sub["round"] == r)].iterrows():
                bids = row.get("batch_local_ids")
                if bids is not None and len(bids):
                    vals.append(np.asarray(bids, dtype=float))
                else:
                    vals.append(np.array([row["exclusive_local_id_mean"]], dtype=float))
            if not vals:
                continue
            data.append(np.concatenate(vals))
            colors.append(METHOD_STYLE[m]["color"])
            labels.append(METHOD_DISPLAY[m])
        if not data:
            continue
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax.set_title(f"E{r}")
        if ax is axes[0]:
            ax.set_ylabel("Local ID (new batch)")
    fig.suptitle(
        f"Selected-batch local-ID distribution — {BACKBONE_LABELS[backbone]} / {DATASET_LABELS[dataset]}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    return save_fig(fig, stem)


def make_global_id_change(summary, backbone, stem):
    """delta global ID over rounds."""
    rows = []
    for (ds, method), g in summary[summary.backbone == backbone].groupby(["dataset", "method"]):
        g = g.sort_values("round")
        gid = g["global_id_mean"].to_numpy(dtype=float)
        sem = g["global_id_sem"].to_numpy(dtype=float)
        rounds = g["round"].to_numpy()
        budgets = g["labeled_budget"].to_numpy()
        for i in range(1, len(g)):
            if np.isfinite(gid[i]) and np.isfinite(gid[i - 1]):
                rows.append({
                    "dataset": ds, "method": method, "round": int(rounds[i]),
                    "labeled_budget": int(budgets[i]),
                    "delta_global_id_mean": float(gid[i] - gid[i - 1]),
                    # rough SEM propagation
                    "delta_global_id_sem": float(np.sqrt(sem[i] ** 2 + sem[i - 1] ** 2)),
                })
    ddf = pd.DataFrame(rows)
    # inject into a fake summary-like frame for plotting
    ddf["method_display"] = ddf["method"].map(METHOD_DISPLAY)
    ddf["backbone"] = backbone
    # rename for plot helper
    plot_df = ddf.rename(columns={
        "delta_global_id_mean": "delta_global_id_mean",
        "delta_global_id_sem": "delta_global_id_sem",
    })
    # manual plot
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2))
    all_m = set()
    for ax, ds, letter in zip(axes, DATASETS, ["(a)", "(b)", "(c)"]):
        panel = plot_df[plot_df.dataset == ds]
        for method in METHOD_ORDER:
            sub = panel[panel.method == method].sort_values("labeled_budget")
            if sub.empty:
                continue
            s = METHOD_STYLE[method]
            x = sub["labeled_budget"].to_numpy()
            y = sub["delta_global_id_mean"].to_numpy()
            sem = sub["delta_global_id_sem"].to_numpy()
            ax.plot(x, y, color=s["color"], linestyle=s["linestyle"],
                    linewidth=s["linewidth"], label=METHOD_DISPLAY[method])
            if np.any(sem > 0):
                ax.fill_between(x, y - sem, y + sem, color=s["color"], alpha=0.15, linewidth=0)
            all_m.add(method)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_title(f"{letter} {DATASET_LABELS[ds]}", pad=28)
        ax.set_xlabel("Labelled budget")
        if ax is axes[0]:
            ax.set_ylabel(r"$\Delta$ global ID")
        _add_episode_axis(ax)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"Per-round change in global ID — {BACKBONE_LABELS[backbone]}",
                 fontsize=15, y=1.04, fontweight="bold")
    _shared_legend(fig, all_m, y=0.0)
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    return save_fig(fig, stem), ddf


def make_fingerprint_heatmap(summary, key_acc, backbone, stem):
    """Normalised method fingerprint heatmap per dataset (3 panels)."""
    metrics = [
        ("aulc", "AULC"),
        ("final_acc", "Final acc."),
        ("avg_excl", "Avg excl. local ID"),
        ("avg_cum", "Avg cum. local ID"),
        ("final_gid", "Final global ID"),
        ("avg_high", "High-ID frac."),
        ("avg_low", "Low-ID frac."),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.8))
    for ax, ds, letter in zip(axes, DATASETS, ["(a)", "(b)", "(c)"]):
        panel = summary[(summary.backbone == backbone) & (summary.dataset == ds)]
        methods = [m for m in METHOD_ORDER if m in set(panel.method)]
        mat = []
        for m in methods:
            sub = panel[panel.method == m].sort_values("round")
            acc_sub = key_acc[(key_acc.dataset == ds) & (key_acc.backbone == backbone)
                              & (key_acc.method == m)]
            aulc = float(acc_sub["aulc_mean"].iloc[0]) if len(acc_sub) else float(sub["test_accuracy_mean"].mean())
            final_acc = float(sub.iloc[-1]["test_accuracy_mean"]) if len(sub) else np.nan
            row = [
                aulc,
                final_acc,
                float(sub["exclusive_local_id_mean_mean"].mean()),
                float(sub["cumulative_local_id_mean_mean"].mean()),
                float(sub.iloc[-1]["global_id_mean"]) if len(sub) else np.nan,
                float(sub["frac_high_id_mean"].mean()),
                float(sub["frac_low_id_mean"].mean()),
            ]
            mat.append(row)
        mat = np.asarray(mat, dtype=float)
        # column-wise z-score within dataset for visual comparison
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(mat, axis=0)
            sd = np.nanstd(mat, axis=0)
            sd[sd < 1e-12] = 1.0
            z = (mat - mu) / sd
        im = ax.imshow(z, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([METHOD_DISPLAY[m] for m in methods], fontsize=8)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([lab for _, lab in metrics], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{letter} {DATASET_LABELS[ds]}")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="Within-dataset z-score")
    fig.suptitle(f"Method structural fingerprint — {BACKBONE_LABELS[backbone]}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, stem)


# ---------------------------------------------------------------------------
# Tables / summaries
# ---------------------------------------------------------------------------
def write_key_rounds(summary, path_csv, path_tex):
    rows = []
    for (ds, bb, method), g in summary.groupby(["dataset", "backbone", "method"]):
        rec = {
            "dataset": ds, "backbone": bb, "method": method,
            "method_display": METHOD_DISPLAY[method],
            "n_seeds_max": int(g["n_seeds"].max()),
        }
        for r in KEY_ROUNDS:
            sub = g[g["round"] == r]
            for base in ["exclusive_local_id_mean", "cumulative_local_id_mean",
                         "global_id", "test_accuracy"]:
                if len(sub) == 1:
                    rec[f"{base}_E{r}"] = float(sub[f"{base}_mean"].iloc[0])
                    rec[f"{base}_sem_E{r}"] = float(sub[f"{base}_sem"].iloc[0])
                    rec[f"budget_E{r}"] = int(sub["labeled_budget"].iloc[0])
                else:
                    rec[f"{base}_E{r}"] = np.nan
                    rec[f"{base}_sem_E{r}"] = np.nan
                    rec[f"budget_E{r}"] = int(_episode_to_budget(r))
        rows.append(rec)
    key = pd.DataFrame(rows)
    key.to_csv(path_csv, index=False)

    def fmt(m, s):
        if pd.isna(m):
            return "---"
        if pd.isna(s) or s == 0:
            return f"{m:.2f}"
        return f"{m:.2f}$\\pm${s:.2f}"

    lines = [
        "% Auto-generated ID key rounds",
        "% Requires booktabs",
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Intrinsic-dimensionality diagnostics at selected episodes "
        f"(mean $\\pm$ SEM). Headers: episode $E$ / labelled budget $B$ "
        f"(budget $= {BATCH_SIZE}+E\\times{BATCH_SIZE}$)." + "}",
        "\\label{tab:diagnostic_id_key_rounds}",
        "\\begin{tabular}{lllcccc}",
        "\\toprule",
        "Dataset & Backbone & Method & "
        + " & ".join([f"ExclID E{r}" for r in KEY_ROUNDS]) + " \\\\",
        "\\midrule",
    ]
    for _, r in key.iterrows():
        cells = [fmt(r[f"exclusive_local_id_mean_E{ep}"], r[f"exclusive_local_id_mean_sem_E{ep}"])
                 for ep in KEY_ROUNDS]
        lines.append(
            f"{DATASET_LABELS[r['dataset']]} & {BACKBONE_LABELS[r['backbone']]} & "
            f"{r['method_display']} & " + " & ".join(cells) + " \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path_tex.write_text("\n".join(lines))
    return key


def compute_correlations(df_long, gain_df):
    rows = []
    for (ds, bb, method), g in df_long.groupby(["dataset", "backbone", "method"]):
        g = g.sort_values(["seed", "round"])
        # pool seed-level round means first
        gm = g.groupby("round", as_index=False).agg(
            global_id=("global_id", "mean"),
            cumulative_local_id_mean=("cumulative_local_id_mean", "mean"),
            exclusive_local_id_mean=("exclusive_local_id_mean", "mean"),
            test_accuracy=("test_accuracy", "mean"),
        )

        def corr_pair(x, y):
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 5:
                return np.nan, np.nan, int(mask.sum())
            pr = pearsonr(x[mask], y[mask])[0]
            sr = spearmanr(x[mask], y[mask])[0]
            return float(pr), float(sr), int(mask.sum())

        p_g, s_g, n_g = corr_pair(gm["global_id"].to_numpy(), gm["test_accuracy"].to_numpy())
        p_c, s_c, n_c = corr_pair(gm["cumulative_local_id_mean"].to_numpy(), gm["test_accuracy"].to_numpy())

        gg = gain_df[(gain_df.dataset == ds) & (gain_df.method == method)]
        if "backbone" in gain_df.columns:
            gg = gain_df[(gain_df.dataset == ds) & (gain_df.backbone == bb) & (gain_df.method == method)]
        # gain_df from make_batch may not have backbone — handle
        if bb and "backbone" not in gain_df.columns:
            # filter via long join: already method/dataset; ok for single-bb calls
            pass
        gga = gg.groupby("round", as_index=False).agg(
            exclusive_local_id_mean=("exclusive_local_id_mean", "mean"),
            accuracy_gain=("accuracy_gain", "mean"),
        ) if len(gg) else pd.DataFrame()
        if len(gga):
            p_b, s_b, n_b = corr_pair(
                gga["exclusive_local_id_mean"].to_numpy(),
                gga["accuracy_gain"].to_numpy(),
            )
        else:
            p_b = s_b = np.nan
            n_b = 0

        rows.append({
            "dataset": ds, "backbone": bb, "method": method,
            "method_display": METHOD_DISPLAY[method],
            "pearson_global_id_vs_acc": p_g,
            "spearman_global_id_vs_acc": s_g,
            "n_rounds_global_id_vs_acc": n_g,
            "pearson_cum_local_id_vs_acc": p_c,
            "spearman_cum_local_id_vs_acc": s_c,
            "n_rounds_cum_local_id_vs_acc": n_c,
            "pearson_batch_id_vs_acc_gain": p_b,
            "spearman_batch_id_vs_acc_gain": s_b,
            "n_rounds_batch_id_vs_acc_gain": n_b,
            "note": "descriptive only; not causal",
        })
    return pd.DataFrame(rows)


def write_trend_summary(summary, corr, missing, path):
    lines = [
        "# Intrinsic-dimensionality diagnostics — trend summary",
        "",
        "Descriptive observations only for *Diagnostic Analysis of Active Learning*. "
        "ID is a diagnostic of geometric complexity regimes, **not** a causal predictor of accuracy.",
        "",
        "## Setup",
        "",
        "- Local ID: Levina–Bickel MLE, k=50, L2-normalised **full** fixed embedding pool, exact kNN.",
        "- Global ID: same estimator on the **full** cumulative labelled set L_t (no sparsifying).",
        "- Sensitivity: Spearman of local-ID rankings for k=50 vs k=20 and k=50 vs k=75.",
        "",
        "## Which methods select higher / lower local-ID samples?",
        "",
    ]
    for ds in DATASETS:
        for bb in summary.backbone.unique():
            sub = summary[(summary.dataset == ds) & (summary.backbone == bb)]
            if sub.empty:
                continue
            avg = sub.groupby("method")["exclusive_local_id_mean_mean"].mean().sort_values(ascending=False)
            if avg.empty:
                continue
            lines.append(
                f"- **{DATASET_LABELS[ds]} / {BACKBONE_LABELS[bb]}**: "
                f"highest excl. local ID ≈ {METHOD_DISPLAY[avg.index[0]]} ({avg.iloc[0]:.2f}); "
                f"lowest ≈ {METHOD_DISPLAY[avg.index[-1]]} ({avg.iloc[-1]:.2f})."
            )
    lines += [
        "",
        "## Global ID of labelled sets",
        "",
        "Methods that keep lower/stabler global ID often emphasise coverage of simpler regions; "
        "uncertainty methods may push into higher-ID neighbourhoods. Compare ProbCover/TypiClust "
        "against Entropy/Margin/DBAL on CIFAR-100 and TinyImageNet especially.",
        "",
        "## ID vs accuracy (descriptive)",
        "",
        "High ID does **not** always mean better accuracy; low ID does **not** always mean better. "
        "Correlations in `diagnostic_id_accuracy_relationship_correlations.csv` are associative only.",
        "",
        "## Missing data / caveats",
        "",
    ]
    if missing:
        for m in missing[:50]:
            lines.append(f"- {m}")
        if len(missing) > 50:
            lines.append(f"- ... and {len(missing) - 50} more")
    else:
        lines.append("- None reported.")
    lines += [
        "",
        "### Caveats",
        "",
        "- Features are fixed `features_seed1.npy` per backbone/dataset. "
        "The fixed embedding matrix is shared across active learning seeds; "
        "seed variation comes from the initial labelled set and acquisition trajectory.",
        "- ID-regime allocation is **batch-level**: at episode t it is the composition of "
        "the newly acquired batch ΔL_t, not the cumulative labelled set.",
        "- Global ID undefined (NaN) when |L_t| ≤ k+2.",
        "- Exact FAISS Flat L2 / sklearn only — no approximate IVF, no point subsampling.",
        "- Use `--refresh-cache` (or delete `outputs/diagnostic_id/id_cache`) before final thesis figures.",
        "",
    ]
    path.write_text("\n".join(lines))


def write_latex_snippets(backbone, out_dir):
    snippets = {
        f"diagnostic_id_{backbone}_local_exclusive_latex_figure.txt": (
            f"diagnostic_id_{backbone}_local_exclusive_combined.pdf",
            "Local intrinsic dimensionality of newly acquired samples using "
            f"{BACKBONE_LABELS[backbone]} fixed embeddings. Curves show the mean local ID of the "
            "newly selected batch at each acquisition round, averaged across seeds, with shaded "
            "regions showing SEM.",
            f"fig:diagnostic_id_{backbone}_local_exclusive",
        ),
        f"diagnostic_id_{backbone}_global_cumulative_latex_figure.txt": (
            f"diagnostic_id_{backbone}_global_cumulative_combined.pdf",
            "Global intrinsic dimensionality of the cumulative labelled set using "
            f"{BACKBONE_LABELS[backbone]} fixed embeddings. Curves show mean global ID across seeds, "
            "with shaded regions showing SEM.",
            f"fig:diagnostic_id_{backbone}_global_cumulative",
        ),
        f"diagnostic_id_{backbone}_local_cumulative_latex_figure.txt": (
            f"diagnostic_id_{backbone}_local_cumulative_combined.pdf",
            "Mean local intrinsic dimensionality of the cumulative labelled set using "
            f"{BACKBONE_LABELS[backbone]} fixed embeddings. Curves show mean cumulative local ID "
            "across seeds, with shaded regions showing SEM.",
            f"fig:diagnostic_id_{backbone}_local_cumulative",
        ),
        f"diagnostic_id_{backbone}_global_id_vs_accuracy_latex_figure.txt": (
            f"diagnostic_id_{backbone}_global_id_vs_accuracy_combined.pdf",
            "Relationship between global labelled-set intrinsic dimensionality and test accuracy "
            f"using {BACKBONE_LABELS[backbone]} fixed embeddings. Points show selected acquisition "
            "rounds averaged across seeds, and lines connect each method’s trajectory. "
            "The relationship is interpreted descriptively and does not imply causality.",
            f"fig:diagnostic_id_{backbone}_global_id_vs_accuracy",
        ),
        f"diagnostic_id_{backbone}_cumulative_local_id_vs_accuracy_latex_figure.txt": (
            f"diagnostic_id_{backbone}_cumulative_local_id_vs_accuracy_combined.pdf",
            "Relationship between cumulative local intrinsic dimensionality and test accuracy "
            f"using {BACKBONE_LABELS[backbone]} fixed embeddings. Points show selected acquisition "
            "rounds averaged across seeds, and lines connect each method’s trajectory.",
            f"fig:diagnostic_id_{backbone}_cum_local_id_vs_accuracy",
        ),
        f"diagnostic_id_{backbone}_batch_local_id_vs_accuracy_gain_latex_figure.txt": (
            f"diagnostic_id_{backbone}_batch_local_id_vs_accuracy_gain_combined.pdf",
            "Relationship between the mean local ID of newly acquired samples and next-round "
            "accuracy gain. Each point corresponds to an acquisition method and round, averaged "
            "across seeds. The plot is used as a descriptive diagnostic of whether selected-batch "
            "complexity is associated with immediate performance change.",
            f"fig:diagnostic_id_{backbone}_batch_id_vs_gain",
        ),
        f"diagnostic_id_{backbone}_k_sensitivity_latex_figure.txt": (
            f"diagnostic_id_{backbone}_k_sensitivity.pdf",
            "Robustness of local intrinsic-dimensionality rankings across neighbourhood sizes. "
            "Bars show the Spearman rank correlation of full-pool local-ID rankings with the "
            "reference setting $k{=}50$, computed once per dataset (fixed embeddings shared "
            "across active-learning seeds).",
            f"fig:diagnostic_id_{backbone}_k_sensitivity",
        ),
        f"diagnostic_id_{backbone}_regime_allocation_latex_figure.txt": (
            f"diagnostic_id_{backbone}_regime_allocation_combined.pdf",
            "High-ID selection fraction of newly acquired batches. At each episode $t$, "
            "the plotted value is the fraction of the episode-specific query batch "
            "$\\Delta\\mathcal{L}_t$ drawn from the high-ID region (pool local ID $\\ge$ Q75), "
            "not the composition of the cumulative labelled set. Curves are seed means with SEM.",
            f"fig:diagnostic_id_{backbone}_regime_allocation",
        ),
    }
    paths = []
    for fname, (pdf, caption, label) in snippets.items():
        text = (
            "% Auto-generated\n"
            "\\begin{figure}[t]\n"
            "  \\centering\n"
            f"  \\includegraphics[width=\\textwidth]{{figures/diagnostic_id/{pdf}}}\n"
            f"  \\caption{{{caption}}}\n"
            f"  \\label{{{label}}}\n"
            "\\end{figure}\n"
        )
        p = out_dir / fname
        p.write_text(text)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_for_backbones(backbones, skip_figures=False, refresh_cache=False):
    configure_matplotlib()
    warnings.filterwarnings("ignore", category=UserWarning)

    all_long = []
    missing = []
    sens_rows = []
    generated = []

    # Load accuracy key rounds if present (for fingerprint AULC)
    acc_key_path = ACCURACY_OUTPUTS_DIR / "diagnostic_accuracy_key_rounds.csv"
    if not acc_key_path.exists():
        # fallback for older flat layout
        acc_key_path = REPO_ROOT / "outputs" / "diagnostic_accuracy_key_rounds.csv"
    key_acc = pd.read_csv(acc_key_path) if acc_key_path.exists() else pd.DataFrame()

    for backbone in backbones:
        for dataset in DATASETS:
            print(f"\n===== {dataset} / {backbone} =====")
            try:
                X = load_features(dataset, backbone)
            except FileNotFoundError as e:
                missing.append(str(e))
                print("SKIP:", e)
                continue

            local_ids = compute_or_load_local_ids(
                X, dataset, backbone, SENS_K, refresh_cache=refresh_cache,
            )

            # k-sensitivity: full-pool ranking correlation once per dataset/backbone.
            # Embeddings are shared (features_seed1); do NOT duplicate across AL seeds
            # or the SEM would be artificially zero.
            lid50 = local_ids[50]
            rho20 = spearmanr(lid50, local_ids[20]).correlation
            rho75 = spearmanr(lid50, local_ids[75]).correlation
            sens_rows.append({
                "dataset": dataset,
                "backbone": backbone,
                "spearman_k50_k20": float(rho20),
                "spearman_k50_k75": float(rho75),
                "n_points": int(len(lid50)),
                "note": (
                    "full-pool ranking correlation, computed once per dataset/backbone; "
                    "features_seed1 shared across AL seeds"
                ),
            })

            for method in METHOD_ORDER:
                for seed in SEEDS:
                    rounds, err = load_run_rounds(dataset, backbone, method, seed)
                    if err:
                        missing.append(f"{dataset}/{backbone}/{method}/s{seed}: {err}")
                        continue
                    rows = compute_run_id_metrics(
                        X, local_ids, rounds, dataset, backbone, method, seed,
                        refresh_cache=refresh_cache,
                    )
                    all_long.extend(rows)

    if not all_long:
        raise SystemExit("No ID rows computed — aborting.")

    # Drop heavy batch_local_ids from CSV long (keep in memory for distribution plots)
    df = pd.DataFrame(all_long)
    df_csv = df.drop(columns=["batch_local_ids"], errors="ignore")
    long_path = OUTPUTS_DIR / "diagnostic_id_all_backbones_long.csv"
    df_csv.to_csv(long_path, index=False)
    generated.append(str(long_path))
    print(f"Wrote {long_path} ({len(df_csv)} rows)")

    summary = summarise(df_csv)
    sum_path = OUTPUTS_DIR / "diagnostic_id_all_backbones_summary.csv"
    summary.to_csv(sum_path, index=False)
    generated.append(str(sum_path))

    sens_df = pd.DataFrame(sens_rows)
    sens_path = OUTPUTS_DIR / "diagnostic_id_k_sensitivity.csv"
    sens_df.to_csv(sens_path, index=False)
    generated.append(str(sens_path))

    key = write_key_rounds(
        summary,
        OUTPUTS_DIR / "diagnostic_id_key_rounds.csv",
        OUTPUTS_DIR / "diagnostic_id_key_rounds.tex",
    )
    generated += [
        str(OUTPUTS_DIR / "diagnostic_id_key_rounds.csv"),
        str(OUTPUTS_DIR / "diagnostic_id_key_rounds.tex"),
    ]

    # Regime allocation exports
    regime_cols = [
        "dataset", "backbone", "method", "seed", "round", "labeled_budget",
        "frac_low_id", "frac_mid_id", "frac_high_id",
    ]
    regime_path = OUTPUTS_DIR / "diagnostic_id_regime_allocation.csv"
    df_csv[regime_cols].to_csv(regime_path, index=False)
    regime_sum = summary[[
        "dataset", "backbone", "method", "round", "labeled_budget", "n_seeds",
        "frac_low_id_mean", "frac_low_id_sem",
        "frac_mid_id_mean", "frac_mid_id_sem",
        "frac_high_id_mean", "frac_high_id_sem",
    ]]
    regime_sum_path = OUTPUTS_DIR / "diagnostic_id_regime_allocation_summary.csv"
    regime_sum.to_csv(regime_sum_path, index=False)
    generated += [str(regime_path), str(regime_sum_path)]

    if skip_figures:
        print("Skipping figures (--skip-figures).")
    else:
        for backbone in backbones:
            if summary[summary.backbone == backbone].empty:
                continue
            print(f"\n--- Figures: {backbone} ---")
            # Fig 1–3 (combined + per-dataset)
            for mean_col, sem_col, tag, ylabel, title in [
                ("exclusive_local_id_mean_mean", "exclusive_local_id_mean_sem",
                 "local_exclusive", "Mean local ID (new batch)",
                 f"Local ID of newly acquired samples — {BACKBONE_LABELS[backbone]}"),
                ("global_id_mean", "global_id_sem",
                 "global_cumulative", "Global ID of labelled set",
                 f"Global ID of cumulative labelled set — {BACKBONE_LABELS[backbone]}"),
                ("cumulative_local_id_mean_mean", "cumulative_local_id_mean_sem",
                 "local_cumulative", "Mean local ID (cumulative)",
                 f"Cumulative local ID of labelled set — {BACKBONE_LABELS[backbone]}"),
            ]:
                pdf, png = make_combined_metric_figure(
                    summary, backbone, mean_col, sem_col, ylabel,
                    f"diagnostic_id_{backbone}_{tag}_combined", title)
                generated += [str(pdf), str(png)]
                for ds in DATASETS:
                    pdf, png = make_per_dataset_metric_figure(
                        summary, backbone, ds, mean_col, sem_col, ylabel,
                        f"diagnostic_id_{backbone}_{tag}_{ds.lower()}",
                        f"{DATASET_LABELS[ds]} — {BACKBONE_LABELS[backbone]}",
                    )
                    generated += [str(pdf), str(png)]

            # ID vs accuracy trajectories
            pdf, png = make_id_vs_accuracy_trajectories(
                summary, backbone, "global_id_mean",
                f"diagnostic_id_{backbone}_global_id_vs_accuracy_combined",
                "Global ID of cumulative labelled set",
                f"Global ID vs accuracy — {BACKBONE_LABELS[backbone]}",
            )
            generated += [str(pdf), str(png)]
            # also save alias accuracy_relation
            for ext in ("pdf", "png"):
                src = FIGURES_DIR / f"diagnostic_id_{backbone}_global_id_vs_accuracy_combined.{ext}"
                dst = FIGURES_DIR / f"diagnostic_id_{backbone}_accuracy_relation_combined.{ext}"
                if src.exists():
                    dst.write_bytes(src.read_bytes())
                    generated.append(str(dst))

            pdf, png = make_id_vs_accuracy_trajectories(
                summary, backbone, "cumulative_local_id_mean_mean",
                f"diagnostic_id_{backbone}_cumulative_local_id_vs_accuracy_combined",
                "Cumulative local ID",
                f"Cumulative local ID vs accuracy — {BACKBONE_LABELS[backbone]}",
            )
            generated += [str(pdf), str(png)]

            # Export selected-round relation CSVs
            for idcol, sem_id, fname in [
                ("global_id_mean", "global_id_sem",
                 f"diagnostic_id_{backbone}_global_id_vs_accuracy.csv"),
                ("cumulative_local_id_mean_mean", "cumulative_local_id_mean_sem",
                 f"diagnostic_id_{backbone}_cumulative_local_id_vs_accuracy.csv"),
            ]:
                rel = summary[(summary.backbone == backbone) & (summary["round"].isin(KEY_ROUNDS))][
                    ["dataset", "backbone", "method", "method_display", "round", "labeled_budget",
                     idcol, sem_id, "test_accuracy_mean", "test_accuracy_sem", "n_seeds"]
                ]
                p = OUTPUTS_DIR / fname
                rel.to_csv(p, index=False)
                generated.append(str(p))

            # Batch ID vs gain
            pdf, png, gain_df, gain_agg = make_batch_id_vs_gain(
                df_csv, backbone,
                f"diagnostic_id_{backbone}_batch_local_id_vs_accuracy_gain_combined",
            )
            generated += [str(pdf), str(png)]
            # aliases
            for ext in ("pdf", "png"):
                src = FIGURES_DIR / f"diagnostic_id_{backbone}_batch_local_id_vs_accuracy_gain_combined.{ext}"
                dst = FIGURES_DIR / f"diagnostic_id_{backbone}_batch_id_vs_accuracy_gain_combined.{ext}"
                if src.exists():
                    dst.write_bytes(src.read_bytes())
                    generated.append(str(dst))
            gain_df.assign(backbone=backbone).to_csv(
                OUTPUTS_DIR / f"diagnostic_id_{backbone}_batch_local_id_vs_accuracy_gain.csv",
                index=False,
            )
            generated.append(str(OUTPUTS_DIR / f"diagnostic_id_{backbone}_batch_local_id_vs_accuracy_gain.csv"))

            # k sensitivity
            pdf, png = make_k_sensitivity_figure(
                sens_df, backbone, f"diagnostic_id_{backbone}_k_sensitivity")
            generated += [str(pdf), str(png)]

            # regime
            pdf, png = make_regime_allocation(
                summary, backbone, f"diagnostic_id_{backbone}_regime_allocation_combined")
            generated += [str(pdf), str(png)]
            for ds in DATASETS:
                out = make_regime_stacked_keyrounds(
                    summary, backbone, ds,
                    f"diagnostic_id_{backbone}_regime_allocation_{ds.lower()}")
                if out[0]:
                    generated += [str(out[0]), str(out[1])]

            # selected ID distributions (uses in-memory batch ids when available)
            for ds in DATASETS:
                out = make_selected_id_distribution(
                    df, backbone, ds,
                    f"diagnostic_id_{backbone}_selected_id_distribution_{ds.lower()}")
                if out[0]:
                    generated += [str(out[0]), str(out[1])]

            # delta global ID
            (pdf, png), _ddf = make_global_id_change(
                summary, backbone, f"diagnostic_id_{backbone}_global_id_change_combined")
            generated += [str(pdf), str(png)]

            # fingerprint
            pdf, png = make_fingerprint_heatmap(
                summary, key_acc if len(key_acc) else summary, backbone,
                f"diagnostic_id_{backbone}_method_fingerprint_heatmap")
            generated += [str(pdf), str(png)]

            # latex snippets
            for p in write_latex_snippets(backbone, OUTPUTS_DIR):
                generated.append(str(p))

    # Correlations (all backbones)
    # Build gain_df for all
    gain_all = []
    for (ds, bb, method, seed), g in df_csv.groupby(["dataset", "backbone", "method", "seed"]):
        g = g.sort_values("round")
        acc = g["test_accuracy"].to_numpy()
        excl = g["exclusive_local_id_mean"].to_numpy()
        rounds = g["round"].to_numpy()
        for i in range(len(g) - 1):
            gain_all.append({
                "dataset": ds, "backbone": bb, "method": method, "seed": seed,
                "round": int(rounds[i]),
                "exclusive_local_id_mean": float(excl[i]),
                "accuracy_gain": float(acc[i + 1] - acc[i]),
            })
    gain_all_df = pd.DataFrame(gain_all)
    corr = compute_correlations(df_csv, gain_all_df)
    corr_path = OUTPUTS_DIR / "diagnostic_id_accuracy_correlations.csv"
    corr_path2 = OUTPUTS_DIR / "diagnostic_id_accuracy_relationship_correlations.csv"
    corr.to_csv(corr_path, index=False)
    corr.to_csv(corr_path2, index=False)
    generated += [str(corr_path), str(corr_path2)]

    write_trend_summary(summary, corr, missing, OUTPUTS_DIR / "diagnostic_id_trend_summary.md")
    generated.append(str(OUTPUTS_DIR / "diagnostic_id_trend_summary.md"))

    # accuracy-relationship summary
    rel_md = OUTPUTS_DIR / "diagnostic_id_accuracy_relationship_summary.md"
    rel_lines = [
        "# ID–accuracy relationship summary",
        "",
        "Descriptive only — ID is a diagnostic, not a standalone predictor of performance.",
        "",
        "## Observations to check in the figures/tables",
        "",
        "- Whether high accuracy coincides with lower, higher, or more stable global ID.",
        "- Whether ProbCover reaches high accuracy at lower/stabler ID than uncertainty methods.",
        "- Whether uncertainty methods occupy higher-ID regions without matching accuracy gains.",
        "- Whether batch local ID predicts next-round accuracy gain, or the link is weak/noisy.",
        "- Whether patterns differ across CIFAR-10, CIFAR-100, and TinyImageNet.",
        "",
        "See `diagnostic_id_accuracy_relationship_correlations.csv` for Pearson/Spearman values.",
        "",
        "## Caveats",
        "",
        "- No causal claims.",
        "- Fixed embeddings (`features_seed1.npy`) are shared across AL seeds.",
        "- ID-regime allocation is batch-level (ΔL_t), not cumulative L_t.",
        "- Global ID NaN when |L_t| ≤ k+2.",
        "- Full pool / full L_t only (no sparsifying).",
        "- Delete `id_cache` or pass `--refresh-cache` before final thesis figures.",
        "",
    ]
    # auto fill strongest correlations
    rel_lines.append("## Strongest |Spearman| (global ID vs accuracy)")
    rel_lines.append("")
    if len(corr):
        tmp = corr.dropna(subset=["spearman_global_id_vs_acc"]).copy()
        tmp["abs"] = tmp["spearman_global_id_vs_acc"].abs()
        for _, r in tmp.nlargest(10, "abs").iterrows():
            rel_lines.append(
                f"- {DATASET_LABELS[r.dataset]} / {BACKBONE_LABELS[r.backbone]} / "
                f"{r.method_display}: Spearman={r.spearman_global_id_vs_acc:.3f} (n={r.n_rounds_global_id_vs_acc})"
            )
    rel_md.write_text("\n".join(rel_lines) + "\n")
    generated.append(str(rel_md))

    # Final report
    report = OUTPUTS_DIR / "diagnostic_id_generation_report.md"
    report.write_text(
        "# Diagnostic ID generation — final report\n\n"
        "## Policy\n\n"
        "- FULL embedding pool for local ID (exact kNN, no IVF approx).\n"
        "- FULL cumulative labelled set L_t for global ID (no point subsampling).\n"
        "- `active_set_ids` = newly acquired batch; `lSet.npy` aligned with "
        "`labeled_count_before_sampling` (accuracy pairing).\n"
        "- ID-regime allocation is batch-level (ΔL_t), not cumulative.\n"
        "- Missing runs listed; values never invented.\n\n"
        f"## Rows\n\n- long: {len(df_csv)}\n- summary: {len(summary)}\n"
        f"- missing entries: {len(missing)}\n\n"
        "## Generated files\n\n"
        + "\n".join(f"- `{g}`" for g in generated)
        + "\n\n## Missing / skipped\n\n"
        + ("\n".join(f"- {m}" for m in missing) if missing else "- None")
        + "\n"
    )
    generated.append(str(report))
    print("\n========== FINAL REPORT ==========")
    print(report.read_text())
    return generated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=BACKBONES,
                    choices=BACKBONES, help="Backbones to process")
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument(
        "--refresh-cache",
        action="store_true",
        help=(
            "Delete and recompute local/global ID caches before generating figures. "
            "Use for final thesis output so stale caches cannot be reused."
        ),
    )
    args = ap.parse_args()
    run_for_backbones(
        args.backbones,
        skip_figures=args.skip_figures,
        refresh_cache=args.refresh_cache,
    )


if __name__ == "__main__":
    main()
