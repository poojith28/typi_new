#!/usr/bin/env python3
"""Generate full-train versus pool-only ID/H0 robustness evidence.

No active-learning training is performed.  Recorded selected-ID trajectories
are mapped to exact, provenance-recorded structural references.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr, pearsonr, wasserstein_distance


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "TypiClust/output"
OUT = ROOT / "outputs/thesis_appendix/pool_reference_robustness"
REF = OUT / "reference"
FIG = ROOT / "figures/thesis_appendix/pool_reference_robustness"
TABLE = ROOT / "tables/thesis_appendix"
DATASETS = ("CIFAR10", "CIFAR100", "TINYIMAGENET")
BACKBONES = ("resnet18", "resnet50", "alexnet")
SEEDS = range(1, 6)
KEY_EPISODES = (10, 25, 50, 75, 100)
N_EPS = 80
FEATURE_DATASET = {"CIFAR10": "cifar-10", "CIFAR100": "cifar-100", "TINYIMAGENET": "tiny-imagenet"}
METHOD_PREFIX = {
    "Random": "random", "Least Confidence": "uncertainty", "Entropy": "entropy",
    "Margin": "margin", "DBAL": "dbal", "CoreSet": "coreset",
    "ProbCover": "probcover", "TypiClust": "typiclust", "MaxHerding": "maxherding",
}
METHOD_ORDER = list(METHOD_PREFIX) + ["MSTC", "LIDCover"]
UNCERTAINTY = {"Least Confidence", "Entropy", "Margin", "DBAL"}
COVERAGE = {"ProbCover", "TypiClust"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_corr(function, left, right) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    keep = np.isfinite(left) & np.isfinite(right)
    if keep.sum() < 3 or np.ptp(left[keep]) == 0 or np.ptp(right[keep]) == 0:
        return float("nan")
    return float(function(left[keep], right[keep]).statistic)


def sem(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else (0.0 if len(values) else float("nan"))


def load_ids(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=True)
    if values.dtype == object:
        values = np.asarray(values.tolist())
    return np.asarray(values, dtype=np.int64).reshape(-1)


def run_catalog() -> dict[tuple[str, str, int, str], Path]:
    catalog: dict[tuple[str, str, int, str], Path] = {}
    for dataset in DATASETS:
        for backbone in BACKBONES:
            for seed in SEEDS:
                for method, prefix in METHOD_PREFIX.items():
                    path = RUN_ROOT / dataset / backbone / f"{prefix}_{seed}_50b"
                    if (path / "benchmark_summary.json").exists():
                        catalog[(dataset, backbone, seed, method)] = path
                mstc = RUN_ROOT / dataset / backbone / f"MSTC_C070_F050_A070_{seed}_50b"
                if (mstc / "benchmark_summary.json").exists():
                    catalog[(dataset, backbone, seed, "MSTC")] = mstc

    lid_matrix = ROOT / "thesis_appendix_audit/_regenerated_lidcover/lidcover_completion_matrix.csv"
    if lid_matrix.exists():
        with lid_matrix.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if (
                    row["status"] == "COMPLETE" and "candidate" not in row["experiment_id"]
                    and row["method"] == "LIDCover" and abs(float(row["delta0"]) - 0.25) < 1e-9
                ):
                    path = Path(row["source_path"])
                    catalog[(row["dataset"], row["backbone"], int(row["seed"]), "LIDCover")] = path
    return catalog


@lru_cache(maxsize=None)
def load_trajectory(path: Path) -> list[dict[str, Any]]:
    summary = json.loads((path / "benchmark_summary.json").read_text())
    records = {int(row["episode"]): row for row in (summary.get("episode_records") or [])}
    rows = []
    for episode_dir in path.glob("episode_*"):
        try:
            episode = int(episode_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        lset_path = episode_dir / "lSet.npy"
        if not lset_path.exists():
            continue
        cumulative = load_ids(lset_path)
        record = records.get(episode, {})
        active = record.get("active_set_ids")
        if active is not None:
            batch = np.asarray(active, dtype=np.int64).reshape(-1)
        elif (episode_dir / "activeSet.npy").exists():
            batch = load_ids(episode_dir / "activeSet.npy")
        else:
            batch = np.array([], dtype=np.int64)
        rows.append({
            "episode": episode, "budget": len(cumulative), "batch": batch,
            "cumulative": cumulative, "test_accuracy": record.get("test_accuracy"),
        })
    return sorted(rows, key=lambda row: row["episode"])


def id_paths(dataset: str, backbone: str, seed: int) -> tuple[Path, Path, Path]:
    full = REF / "id/full" / f"{dataset}_{backbone}_full_train_local_id_k50.npy"
    pool = REF / "id/pool" / f"{dataset}_{backbone}_seed{seed}_pool_only_local_id_k50.npy"
    eligible = REF / "eligible_indices" / f"{dataset}_{backbone}_seed{seed}_eligible.npy"
    return full, pool, eligible


def analyse_id(catalog, allow_partial: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pointwise_rows, trajectory_rows, regime_rows, stability_rows = [], [], [], []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            method_episode_rows = []
            reference_keys = []
            for seed in SEEDS:
                full_path, pool_path, eligible_path = id_paths(dataset, backbone, seed)
                if not (full_path.exists() and pool_path.exists() and eligible_path.exists()):
                    if allow_partial:
                        continue
                    raise FileNotFoundError(f"missing ID reference for {dataset}/{backbone}/seed{seed}")
                full = np.load(full_path).astype(float)
                pool = np.load(pool_path).astype(float)
                eligible = load_ids(eligible_path)
                full_pool = full[eligible]
                absolute = np.abs(full_pool - pool)
                pointwise_rows.append({
                    "dataset": dataset, "backbone": backbone, "seed": seed,
                    "n_points": len(eligible),
                    "spearman": finite_corr(spearmanr, full_pool, pool),
                    "pearson": finite_corr(pearsonr, full_pool, pool),
                    "median_absolute_difference": float(np.median(absolute)),
                    "p95_absolute_difference": float(np.quantile(absolute, 0.95)),
                    "full_reference_path": str(full_path), "pool_reference_path": str(pool_path),
                    "eligible_indices_path": str(eligible_path),
                })
                pool_by_original = np.full(len(full), np.nan, dtype=float)
                pool_by_original[eligible] = pool
                full_q25, full_q75 = np.quantile(full, [0.25, 0.75])
                pool_q25, pool_q75 = np.quantile(pool, [0.25, 0.75])
                reference_keys.append(seed)
                for method in METHOD_ORDER:
                    run = catalog.get((dataset, backbone, seed, method))
                    if run is None:
                        continue
                    per_episode = []
                    for row in load_trajectory(run):
                        batch = row["batch"]
                        cumulative = row["cumulative"]
                        full_batch = full[batch] if len(batch) else np.array([])
                        pool_batch = pool_by_original[batch] if len(batch) else np.array([])
                        full_cum = full[cumulative]
                        pool_cum = pool_by_original[cumulative]
                        rec = {
                            "record_type": "episode", "dataset": dataset, "backbone": backbone,
                            "method": method, "seed": seed, "episode": row["episode"],
                            "labelled_budget": row["budget"], "raw_path": str(run),
                            "full_batch_mean_local_id": float(np.mean(full_batch)) if len(full_batch) else np.nan,
                            "pool_batch_mean_local_id": float(np.nanmean(pool_batch)) if len(pool_batch) else np.nan,
                            "full_cumulative_mean_local_id": float(np.mean(full_cum)),
                            "pool_cumulative_mean_local_id": float(np.nanmean(pool_cum)),
                        }
                        per_episode.append(rec)
                        method_episode_rows.append(rec)
                        def fractions(values, low, high):
                            if len(values) == 0:
                                return (np.nan, np.nan, np.nan)
                            return (
                                float(np.mean(values <= low)),
                                float(np.mean((values > low) & (values < high))),
                                float(np.mean(values >= high)),
                            )
                        full_frac = fractions(full_batch, full_q25, full_q75)
                        pool_frac = fractions(pool_batch[np.isfinite(pool_batch)], pool_q25, pool_q75)
                        regime_rows.append({
                            "dataset": dataset, "backbone": backbone, "method": method, "seed": seed,
                            "episode": row["episode"], "labelled_budget": row["budget"],
                            "full_low_fraction": full_frac[0], "full_mid_fraction": full_frac[1],
                            "full_high_fraction": full_frac[2], "pool_low_fraction": pool_frac[0],
                            "pool_mid_fraction": pool_frac[1], "pool_high_fraction": pool_frac[2],
                            "change_high_fraction": pool_frac[2] - full_frac[2],
                        })
                    frame = pd.DataFrame(per_episode)
                    if frame.empty:
                        continue
                    trajectory_rows.extend(per_episode)
                    trajectory_rows.append({
                        "record_type": "trajectory_summary", "dataset": dataset, "backbone": backbone,
                        "method": method, "seed": seed, "episode": np.nan, "labelled_budget": np.nan,
                        "raw_path": str(run),
                        "batch_trajectory_spearman": finite_corr(
                            spearmanr, frame.full_batch_mean_local_id, frame.pool_batch_mean_local_id
                        ),
                        "cumulative_trajectory_spearman": finite_corr(
                            spearmanr, frame.full_cumulative_mean_local_id, frame.pool_cumulative_mean_local_id
                        ),
                        "mean_change_high_fraction": float(np.nanmean([
                            row["change_high_fraction"] for row in regime_rows
                            if row["dataset"] == dataset and row["backbone"] == backbone
                            and row["seed"] == seed and row["method"] == method
                        ])),
                    })

            method_frame = pd.DataFrame(method_episode_rows)
            point_frame = pd.DataFrame([row for row in pointwise_rows if row["dataset"] == dataset and row["backbone"] == backbone])
            for episode in KEY_EPISODES:
                current = method_frame[method_frame.episode == episode] if not method_frame.empty else pd.DataFrame()
                ranks = current.groupby("method")[["full_cumulative_mean_local_id", "pool_cumulative_mean_local_id"]].mean()
                ordering = finite_corr(spearmanr, ranks.iloc[:, 0], ranks.iloc[:, 1]) if len(ranks) else np.nan
                uncertainty = current[current.method.isin(UNCERTAINTY)]
                coverage = current[current.method.isin(COVERAGE)]
                full_gap = float(uncertainty.full_cumulative_mean_local_id.mean() - coverage.full_cumulative_mean_local_id.mean())
                pool_gap = float(uncertainty.pool_cumulative_mean_local_id.mean() - coverage.pool_cumulative_mean_local_id.mean())
                stable = bool(np.isfinite(full_gap) and np.isfinite(pool_gap) and np.sign(full_gap) == np.sign(pool_gap))
                stability_rows.append({
                    "diagnostic": "ID", "dataset": dataset, "backbone": backbone, "episode": episode,
                    "reference_seed_count": len(reference_keys),
                    "pointwise_spearman_mean": float(point_frame.spearman.mean()) if len(point_frame) else np.nan,
                    "method_order_spearman": ordering,
                    "uncertainty_minus_coverage_full": full_gap,
                    "uncertainty_minus_coverage_pool": pool_gap,
                    "qualitative_family_separation_stable": stable,
                    "assessment": "SUPPORTED" if stable and ordering >= 0.8 else "PARTLY CHANGED",
                })
    return map(pd.DataFrame, (pointwise_rows, trajectory_rows, regime_rows, stability_rows))


class H0Reference:
    def __init__(self, path: Path):
        frame = pd.read_csv(path)
        frame = frame[np.isfinite(frame.death)].reset_index(drop=True)
        self.path = path
        self.persistence = frame.persistence.to_numpy(float)
        self.persistence_sorted = np.sort(self.persistence)
        self.death = frame.death.to_numpy(float)
        self.birth_vertex = np.asarray([int(ast.literal_eval(str(value))[0]) for value in frame.birth_simplex], dtype=np.int64)
        self.lookup = {int(vertex): i for i, vertex in enumerate(self.birth_vertex)}
        self.epsilon = np.linspace(float(self.death.min()), float(self.death.max()), N_EPS)

        # Exact 1-D empirical-Wasserstein queries can be answered in
        # O(n log m), rather than sorting/scanning all m reference values for
        # every one of the thousands of recorded trajectory states.
        unique, counts = np.unique(self.persistence_sorted, return_counts=True)
        self._w1_unique = unique
        if len(unique) > 1:
            self._w1_width = np.diff(unique)
            self._w1_level = np.cumsum(counts)[:-1] / len(self.persistence_sorted)
            self._w1_prefix_width = np.r_[0.0, np.cumsum(self._w1_width)]
            self._w1_prefix_weighted = np.r_[0.0, np.cumsum(self._w1_width * self._w1_level)]
        else:
            self._w1_width = np.array([], dtype=float)
            self._w1_level = np.array([], dtype=float)
            self._w1_prefix_width = np.array([0.0])
            self._w1_prefix_weighted = np.array([0.0])

    def values(self, ids: np.ndarray):
        pair_ids = sorted({self.lookup[int(value)] for value in ids if int(value) in self.lookup})
        return self.persistence[pair_ids], self.death[pair_ids]

    def _absolute_cdf_primitive(self, q: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Integral of |F_ref(t)-q| from min(reference) to x."""
        q = np.asarray(q, dtype=float)
        x = np.asarray(x, dtype=float)
        lower, upper = self._w1_unique[0], self._w1_unique[-1]
        result = np.empty(np.broadcast(q, x).shape, dtype=float)
        q, x = np.broadcast_arrays(q, x)
        below = x <= lower
        above = x >= upper
        middle = ~(below | above)
        result[below] = -(lower - x[below]) * np.abs(q[below])

        def segment_sum(q_values: np.ndarray, count: np.ndarray) -> np.ndarray:
            split = np.searchsorted(self._w1_level, q_values, side="right")
            split = np.minimum(split, count)
            low_w = self._w1_prefix_width[split]
            low_wp = self._w1_prefix_weighted[split]
            all_w = self._w1_prefix_width[count]
            all_wp = self._w1_prefix_weighted[count]
            return q_values * low_w - low_wp + (all_wp - low_wp) - q_values * (all_w - low_w)

        total_count = np.full(q[above].shape, len(self._w1_level), dtype=int)
        total = segment_sum(q[above], total_count)
        result[above] = total + (x[above] - upper) * np.abs(1.0 - q[above])
        if middle.any():
            indices = np.searchsorted(self._w1_unique, x[middle], side="right") - 1
            base = segment_sum(q[middle], indices)
            partial = (x[middle] - self._w1_unique[indices]) * np.abs(
                self._w1_level[indices] - q[middle]
            )
            result[middle] = base + partial
        return result

    def wasserstein_1(self, values: np.ndarray) -> float:
        """Exact W1 between an empirical sample and this cached reference."""
        sample = np.sort(np.asarray(values, dtype=float))
        if not len(sample):
            return float("nan")
        start = min(sample[0], self._w1_unique[0])
        end = max(sample[-1], self._w1_unique[-1])
        left = np.r_[start, sample]
        right = np.r_[sample, end]
        levels = np.arange(len(sample) + 1, dtype=float) / len(sample)
        return float(np.sum(
            self._absolute_cdf_primitive(levels, right)
            - self._absolute_cdf_primitive(levels, left)
        ))

    def ks(self, values: np.ndarray) -> float:
        """Exact two-sample KS statistic without rescanning the reference."""
        sample = np.sort(np.asarray(values, dtype=float))
        if not len(sample):
            return float("nan")
        locations = np.unique(sample)
        sample_left = np.searchsorted(sample, locations, side="left") / len(sample)
        sample_right = np.searchsorted(sample, locations, side="right") / len(sample)
        ref_left = np.searchsorted(self.persistence_sorted, locations, side="left") / len(self.persistence_sorted)
        ref_right = np.searchsorted(self.persistence_sorted, locations, side="right") / len(self.persistence_sorted)
        return float(max(
            np.max(sample_right - ref_right),
            np.max(ref_left - sample_left),
        ))


def h0_metrics(reference: H0Reference, ids: np.ndarray) -> dict[str, float]:
    persistence, deaths = reference.values(ids)
    if not len(persistence):
        return {key: np.nan for key in (
            "exposed_count", "mean_persistence", "median_persistence", "wasserstein",
            "ks", "l1_ecdf", "survival_auc", "epsilon10", "epsilon50"
        )}
    wasserstein = reference.wasserstein_1(persistence)
    sorted_death = np.sort(deaths)
    survival = (len(sorted_death) - np.searchsorted(sorted_death, reference.epsilon, side="right")) / len(sorted_death)
    def epsilon_at(level):
        where = np.where(survival <= level)[0]
        return float(reference.epsilon[where[0]]) if len(where) else float(reference.epsilon[-1])
    return {
        "exposed_count": len(persistence), "mean_persistence": float(np.mean(persistence)),
        "median_persistence": float(np.median(persistence)),
        "wasserstein": wasserstein,
        "ks": reference.ks(persistence),
        # In one dimension, W1 equals the integral of the absolute ECDF
        # difference; retain both historical column names explicitly.
        "l1_ecdf": wasserstein,
        "survival_auc": float(np.trapezoid(survival, reference.epsilon)),
        "epsilon10": epsilon_at(0.1), "epsilon50": epsilon_at(0.5),
    }


def h0_paths(dataset, backbone, seed):
    full = ROOT / "outputs/diagnostic_h0/reference" / f"{dataset}_{backbone}_h0_reference.csv"
    pool = REF / "h0" / f"{dataset}_{backbone}_seed{seed}_pool_only_h0_reference.csv"
    return full, pool


def analyse_h0(catalog, allow_partial):
    reference_rows, trajectory_rows, stability_rows = [], [], []
    reference_cache: dict[Path, H0Reference] = {}
    for dataset in DATASETS:
        for backbone in BACKBONES:
            episode_rows = []
            available_seeds = []
            for seed in SEEDS:
                full_path, pool_path = h0_paths(dataset, backbone, seed)
                if not (full_path.exists() and pool_path.exists()):
                    if allow_partial:
                        continue
                    raise FileNotFoundError(f"missing H0 reference for {dataset}/{backbone}/seed{seed}")
                full_ref = reference_cache.setdefault(full_path, H0Reference(full_path)) if full_path not in reference_cache else reference_cache[full_path]
                pool_ref = reference_cache.setdefault(pool_path, H0Reference(pool_path)) if pool_path not in reference_cache else reference_cache[pool_path]
                available_seeds.append(seed)
                for kind, reference in (("FULL_TRAIN_REFERENCE", full_ref), ("POOL_ONLY_REFERENCE", pool_ref)):
                    meta_path = (
                        ROOT / "outputs/diagnostic_h0/reference" / f"{dataset}_{backbone}_h0_reference_meta.json"
                        if kind == "FULL_TRAIN_REFERENCE" else Path(str(pool_path).replace(".csv", "_meta.json"))
                    )
                    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                    reference_rows.append({
                        "dataset": dataset, "backbone": backbone, "seed": seed, "reference_type": kind,
                        "n_reference_vertices": meta.get("n_reference_vertices", meta.get("notes", {}).get("n")),
                        "n_finite_h0_pairs": len(reference.persistence), "persistence_min": float(reference.persistence.min()),
                        "persistence_median": float(np.median(reference.persistence)),
                        "persistence_mean": float(reference.persistence.mean()), "persistence_max": float(reference.persistence.max()),
                        "feature_sha256": meta.get("feature_sha256", ""),
                        "eligible_index_file_sha256": meta.get("eligible_index_file_sha256", ""),
                        "code_sha256": meta.get("code_sha256", ""), "reference_path": str(reference.path),
                    })
                for method in METHOD_ORDER:
                    run = catalog.get((dataset, backbone, seed, method))
                    if run is None:
                        continue
                    per_episode = []
                    for row in load_trajectory(run):
                        full_metrics = h0_metrics(full_ref, row["cumulative"])
                        pool_metrics = h0_metrics(pool_ref, row["cumulative"])
                        full_batch = h0_metrics(full_ref, row["batch"])
                        pool_batch = h0_metrics(pool_ref, row["batch"])
                        record = {
                            "record_type": "episode", "dataset": dataset, "backbone": backbone,
                            "method": method, "seed": seed, "episode": row["episode"],
                            "labelled_budget": row["budget"], "raw_path": str(run),
                        }
                        for prefix, metrics in (
                            ("full_cumulative", full_metrics), ("pool_cumulative", pool_metrics),
                            ("full_batch", full_batch), ("pool_batch", pool_batch),
                        ):
                            record.update({f"{prefix}_{key}": value for key, value in metrics.items()})
                        per_episode.append(record)
                        episode_rows.append(record)
                    frame = pd.DataFrame(per_episode)
                    trajectory_rows.extend(per_episode)
                    summary = {
                        "record_type": "trajectory_summary", "dataset": dataset, "backbone": backbone,
                        "method": method, "seed": seed, "episode": np.nan, "labelled_budget": np.nan,
                        "raw_path": str(run),
                    }
                    for metric in ("wasserstein", "ks", "l1_ecdf", "survival_auc", "epsilon10", "epsilon50"):
                        summary[f"cumulative_{metric}_trajectory_spearman"] = finite_corr(
                            spearmanr, frame[f"full_cumulative_{metric}"], frame[f"pool_cumulative_{metric}"]
                        )
                    trajectory_rows.append(summary)
            frame = pd.DataFrame(episode_rows)
            for episode in KEY_EPISODES:
                current = frame[frame.episode == episode] if not frame.empty else pd.DataFrame()
                ranks = current.groupby("method")[["full_cumulative_wasserstein", "pool_cumulative_wasserstein"]].mean()
                ordering = finite_corr(spearmanr, ranks.iloc[:, 0], ranks.iloc[:, 1]) if len(ranks) else np.nan
                stability_rows.append({
                    "diagnostic": "H0", "dataset": dataset, "backbone": backbone, "episode": episode,
                    "reference_seed_count": len(available_seeds), "method_order_spearman": ordering,
                    "median_method_seed_wasserstein_trajectory_spearman": float(pd.DataFrame(trajectory_rows).query(
                        "record_type == 'trajectory_summary' and dataset == @dataset and backbone == @backbone"
                    ).cumulative_wasserstein_trajectory_spearman.median()) if trajectory_rows else np.nan,
                    "assessment": "SUPPORTED" if ordering >= 0.8 else "PARTLY CHANGED",
                })
    references = pd.DataFrame(reference_rows).drop_duplicates(["dataset", "backbone", "seed", "reference_type"])
    return references, pd.DataFrame(trajectory_rows), pd.DataFrame(stability_rows)


def configure_plot():
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
        "pdf.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white",
    })


def plot_combined(trajectory, diagnostic: str, path: Path):
    configure_plot()
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharex=False)
    if diagnostic == "ID":
        value = "pool_cumulative_mean_local_id"
        ylabel = "Cumulative mean pool-local ID"
    else:
        value = "pool_cumulative_wasserstein"
        ylabel = "Pool-reference Wasserstein-1"
    episode = trajectory[trajectory.record_type == "episode"]
    episode = episode[episode.backbone == "resnet18"]
    for axis, dataset in zip(axes, DATASETS):
        panel = episode[episode.dataset == dataset]
        for method in METHOD_ORDER:
            method_frame = panel[panel.method == method]
            if method_frame.empty:
                continue
            curve = method_frame.groupby("labelled_budget")[value].mean().sort_index()
            axis.plot(curve.index, curve.values, label=method, linewidth=1.4)
        axis.set_title(dataset)
        axis.set_xlabel("Labelled budget")
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.08))
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_correlations(pointwise, h0_trajectory):
    configure_plot()
    point_summary = pointwise.groupby(["dataset", "backbone"]).spearman.agg(["mean", sem]).reset_index()
    figure, axis = plt.subplots(figsize=(9, 4))
    labels = [f"{row.dataset}\n{row.backbone}" for row in point_summary.itertuples()]
    axis.bar(range(len(point_summary)), point_summary["mean"], yerr=point_summary["sem"], color="#4477AA")
    axis.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Pointwise Spearman correlation")
    figure.tight_layout()
    figure.savefig(FIG / "full_vs_pool_id_rank_correlation.pdf", bbox_inches="tight")
    plt.close(figure)

    summaries = h0_trajectory[h0_trajectory.record_type == "trajectory_summary"]
    grouped = summaries.groupby(["dataset", "backbone"]).cumulative_wasserstein_trajectory_spearman.agg(["mean", sem]).reset_index()
    figure, axis = plt.subplots(figsize=(9, 4))
    labels = [f"{row.dataset}\n{row.backbone}" for row in grouped.itertuples()]
    axis.bar(range(len(grouped)), grouped["mean"], yerr=grouped["sem"], color="#CC6677")
    axis.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    axis.set_ylim(-1, 1.02)
    axis.set_ylabel("Wasserstein trajectory Spearman correlation")
    figure.tight_layout()
    figure.savefig(FIG / "full_vs_pool_h0_trajectory_correlation.pdf", bbox_inches="tight")
    plt.close(figure)


def latex_tables(pointwise, h0_trajectory):
    id_rows = []
    for (dataset, backbone), group in pointwise.groupby(["dataset", "backbone"]):
        id_rows.append((dataset, backbone, group.spearman.mean(), sem(group.spearman), group.pearson.mean(), sem(group.pearson), group.median_absolute_difference.mean(), len(group)))
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule", "Dataset & Backbone & Spearman & Pearson & Median $|\\Delta|$ & $n$ \\\\", r"\midrule"]
    for dataset, backbone, sr, sr_sem, pr, pr_sem, mad, n in id_rows:
        lines.append(f"{dataset} & {backbone} & {sr:.3f} $\\pm$ {sr_sem:.3f} & {pr:.3f} $\\pm$ {pr_sem:.3f} & {mad:.3f} & {n} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLE / "pool_reference_id_summary.tex").write_text("\n".join(lines) + "\n")

    summaries = h0_trajectory[h0_trajectory.record_type == "trajectory_summary"]
    lines = [r"\begin{tabular}{llrr}", r"\toprule", "Dataset & Backbone & W1 trajectory Spearman & $n$ \\\\", r"\midrule"]
    for (dataset, backbone), group in summaries.groupby(["dataset", "backbone"]):
        values = group.cumulative_wasserstein_trajectory_spearman
        lines.append(f"{dataset} & {backbone} & {values.mean():.3f} $\\pm$ {sem(values):.3f} & {values.notna().sum()} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLE / "pool_reference_h0_summary.tex").write_text("\n".join(lines) + "\n")


def write_readme(stability, pointwise, h0_trajectory):
    id_stable = float(stability.query("diagnostic == 'ID'").qualitative_family_separation_stable.mean())
    id_rank = float(pointwise.spearman.mean())
    h0_summary = h0_trajectory[h0_trajectory.record_type == "trajectory_summary"]
    h0_corr = float(h0_summary.cumulative_wasserstein_trajectory_spearman.median())
    if id_stable == 1 and id_rank >= 0.9 and h0_corr >= 0.8:
        assessment = "SUPPORTED"
    elif id_stable >= 0.8 and id_rank >= 0.7 and h0_corr >= 0.5:
        assessment = "PARTLY CHANGED"
    else:
        assessment = "NOT ROBUST"
    text = f"""# Pool-only structural-reference robustness

Overall assessment: **{assessment}**.

This package compares the historical `FULL_TRAIN_REFERENCE` against exact seed-specific `POOL_ONLY_REFERENCE` calculations. No active-learning trajectory or classifier was rerun. Validation exclusions differ by seed, so references are keyed by dataset, backbone, and seed.

- Mean pointwise local-ID Spearman correlation: {id_rank:.3f}.
- Fraction of dataset/backbone/key-episode cells retaining the direction of the uncertainty-versus-coverage local-ID separation: {id_stable:.3f}.
- Median method/seed Wasserstein-trajectory Spearman correlation: {h0_corr:.3f}.

High local ID is interpreted only as estimated local dimensional structure. H0 Wasserstein distance and merge-scale survival are descriptive; smaller distance or faster merging is not labelled better acquisition. Any low-correlation or direction-changing cells remain in `qualitative_stability_summary.csv`.

Raw inputs are the run paths recorded in the trajectory CSVs, the historical references under `outputs/diagnostic_h0/reference`, and the new exact references under `reference/`. Feature, eligible-index, and code hashes are stored in each reference metadata JSON.
"""
    (OUT / "README.md").write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    catalog = run_catalog()
    pointwise, id_trajectory, regimes, id_stability = analyse_id(catalog, args.allow_partial)
    h0_references, h0_trajectory, h0_stability = analyse_h0(catalog, args.allow_partial)
    stability = pd.concat([id_stability, h0_stability], ignore_index=True, sort=False)
    pointwise.to_csv(OUT / "id_pointwise_reference_comparison.csv", index=False)
    id_trajectory.to_csv(OUT / "id_method_trajectory_comparison.csv", index=False)
    regimes.to_csv(OUT / "id_regime_comparison.csv", index=False)
    h0_references.to_csv(OUT / "h0_reference_statistics.csv", index=False)
    h0_trajectory.to_csv(OUT / "h0_method_trajectory_comparison.csv", index=False)
    stability.to_csv(OUT / "qualitative_stability_summary.csv", index=False)
    plot_combined(id_trajectory, "ID", FIG / "pool_only_id_resnet18_combined.pdf")
    plot_combined(h0_trajectory, "H0", FIG / "pool_only_h0_wasserstein_resnet18_combined.pdf")
    plot_correlations(pointwise, h0_trajectory)
    latex_tables(pointwise, h0_trajectory)
    write_readme(stability, pointwise, h0_trajectory)
    print(json.dumps({
        "pointwise_rows": len(pointwise), "id_trajectory_rows": len(id_trajectory),
        "h0_reference_rows": len(h0_references), "h0_trajectory_rows": len(h0_trajectory),
    }, indent=2))


if __name__ == "__main__":
    main()
