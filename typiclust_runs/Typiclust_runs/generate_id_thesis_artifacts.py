#!/usr/bin/env python3
"""Generate thesis-grade intrinsic-dimension diagnostics for AL runs.

The script is intentionally analysis-only: it reads completed TypiClust runs,
precomputed SimCLR/pretext features, and per-episode AL artifacts. By default it
excludes LIDCover-family methods and summarizes only established AL baselines.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-codex-{os.getuid()}")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree


ROOT = Path("/scratch/s219110279")
DEEP_AL_ROOT = ROOT / "TypiClust/deep-al"
if str(DEEP_AL_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEP_AL_ROOT))

import pycls.datasets.utils as ds_utils  # noqa: E402


METHOD_LABELS = {
    "random": "Random",
    "uncertainty": "Uncertainty",
    "entropy": "Entropy",
    "margin": "Margin",
    "dbal": "DBAL",
    "coreset": "CoreSet",
    "probcover": "ProbCover",
}

METHOD_COLORS = {
    "random": "#7f7f7f",
    "uncertainty": "#d62728",
    "entropy": "#ff7f0e",
    "margin": "#9467bd",
    "dbal": "#8c564b",
    "coreset": "#1f77b4",
    "probcover": "#2ca02c",
}

DATASET_LABELS = {
    "CIFAR10": "CIFAR-10",
    "CIFAR100": "CIFAR-100",
    "TINYIMAGENET": "TinyImageNet",
}


@dataclass(frozen=True)
class Run:
    dataset: str
    backbone: str
    method: str
    seed: int
    budget: int
    exp_name: str
    run_dir: Path
    summary_path: Path
    summary: dict


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(value, default=float("nan")) -> float:
    try:
        if value in (None, "not_available", "NA"):
            return default
        return float(value)
    except Exception:
        return default


def sem(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if len(vals) <= 1:
        return 0.0 if vals else float("nan")
    return stdev(vals) / math.sqrt(len(vals))


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return mean(vals) if vals else float("nan")


def safe_corr(a: Iterable[float], b: Iterable[float], kind: str) -> float:
    x = np.asarray(list(a), dtype=np.float64)
    y = np.asarray(list(b), dtype=np.float64)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    if kind == "pearson":
        return float(stats.pearsonr(x, y).statistic)
    return float(stats.spearmanr(x, y).statistic)


def parse_exp_name(name: str) -> tuple[str, int | None, int | None]:
    match = re.match(r"(.+)_([0-9]+)_([0-9]+)b$", name)
    if not match:
        return name, None, None
    return match.group(1), int(match.group(2)), int(match.group(3))


def base_method(method: str) -> str:
    return method.split("_rerun")[0]


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, data: dict) -> None:
    with path.open("w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def write_tex_table(path: Path, columns: list[str], rows: list[dict], caption: str, label: str) -> None:
    with path.open("w") as handle:
        handle.write("\\begin{table}[t]\n\\centering\n")
        handle.write(f"\\caption{{{caption}}}\n")
        handle.write(f"\\label{{{label}}}\n")
        handle.write("\\small\n")
        handle.write("\\begin{tabular}{" + "l" * len(columns) + "}\n")
        handle.write("\\toprule\n")
        handle.write(" & ".join(columns) + " \\\\\n")
        handle.write("\\midrule\n")
        for row in rows:
            handle.write(" & ".join(str(row.get(col, "")) for col in columns) + " \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def discover_backbones(output_root: Path, datasets: list[str], requested: list[str]) -> list[str]:
    if requested != ["auto"]:
        return requested
    found = set()
    for dataset in datasets:
        dataset_dir = output_root / dataset
        if not dataset_dir.exists():
            continue
        for path in dataset_dir.iterdir():
            if path.is_dir():
                found.add(path.name)
    return sorted(found)


def discover_runs(args) -> tuple[list[Run], list[dict]]:
    output_root = Path(args.output_root)
    backbones = discover_backbones(output_root, args.datasets, args.backbones)
    allowed = set(args.methods)
    runs: list[Run] = []
    skipped: list[dict] = []
    per_method_count: Counter[tuple[str, str, str]] = Counter()

    for dataset in args.datasets:
        for backbone in backbones:
            root = output_root / dataset / backbone
            if not root.exists():
                skipped.append({"dataset": dataset, "backbone": backbone, "reason": "missing_backbone_dir"})
                continue
            for summary_path in sorted(root.glob("*/benchmark_summary.json")):
                summary = read_json(summary_path)
                exp_name = str(summary.get("exp_name") or summary_path.parent.name)
                method_key, seed_from_name, budget_from_name = parse_exp_name(exp_name)
                method_key = base_method(method_key)
                reason = None
                if method_key.startswith("LIDCOVER"):
                    reason = "excluded_lidcover"
                elif method_key not in allowed:
                    reason = "method_not_requested"
                elif int(summary.get("num_rounds_completed") or 0) < int(args.min_rounds) and not args.allow_incomplete:
                    reason = "incomplete"
                if reason:
                    skipped.append({
                        "dataset": dataset,
                        "backbone": backbone,
                        "exp_name": exp_name,
                        "method": method_key,
                        "reason": reason,
                    })
                    continue
                budget = int(summary.get("budget_per_round") or budget_from_name or 0)
                seed = int(summary.get("seed") or seed_from_name or -1)
                allowed_budgets = None if args.budgets == ["all"] else {int(v) for v in args.budgets}
                if allowed_budgets is not None and budget not in allowed_budgets:
                    skipped.append({
                        "dataset": dataset,
                        "backbone": backbone,
                        "exp_name": exp_name,
                        "method": method_key,
                        "budget": budget,
                        "reason": "budget_not_requested",
                    })
                    continue
                key = (dataset, backbone, method_key)
                if args.max_runs_per_method and per_method_count[key] >= args.max_runs_per_method:
                    skipped.append({
                        "dataset": dataset,
                        "backbone": backbone,
                        "exp_name": exp_name,
                        "method": method_key,
                        "reason": "max_runs_per_method",
                    })
                    continue
                per_method_count[key] += 1
                runs.append(Run(
                    dataset=dataset,
                    backbone=backbone,
                    method=method_key,
                    seed=seed,
                    budget=budget,
                    exp_name=exp_name,
                    run_dir=summary_path.parent,
                    summary_path=summary_path,
                    summary=summary,
                ))
    return runs, skipped


def load_features(dataset: str) -> np.ndarray:
    features = ds_utils.load_features(dataset, seed=1, train=True, normalized=True)
    return np.asarray(features, dtype=np.float32)


def knn_distances(x: np.ndarray, k: int) -> np.ndarray:
    n = x.shape[0]
    k_eff = max(2, min(int(k), n - 1))
    try:
        import faiss  # type: ignore

        x32 = np.ascontiguousarray(x.astype(np.float32, copy=False))
        if n > 60000:
            index = faiss.IndexHNSWFlat(x32.shape[1], 32)
            index.hnsw.efConstruction = 80
            index.hnsw.efSearch = max(96, k_eff + 16)
        else:
            index = faiss.IndexFlatL2(x32.shape[1])
        index.add(x32)
        d2, _ = index.search(x32, k_eff + 1)
        return np.sqrt(np.maximum(d2[:, 1:], 0.0)).astype(np.float64)
    except Exception:
        pass
    tree = cKDTree(x)
    dists, _ = tree.query(x, k=k_eff + 1, workers=-1)
    return np.asarray(dists[:, 1:], dtype=np.float64)


def local_mle_id(x: np.ndarray, k: int, eps: float = 1e-12) -> np.ndarray:
    dists = knn_distances(x, k)
    return mle_from_neighbor_distances(dists, eps=eps)


def mle_from_neighbor_distances(dists: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    dists = np.asarray(dists, dtype=np.float64)
    if dists.shape[1] < 2:
        return np.full(dists.shape[0], np.nan, dtype=np.float32)
    rk = np.maximum(dists[:, -1], eps)
    prev = np.maximum(dists[:, :-1], eps)
    denom = np.mean(np.log(rk[:, None] / prev), axis=1)
    return (1.0 / np.maximum(denom, eps)).astype(np.float32)


def local_twonn_id(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    dists = knn_distances(x, 2)
    r1 = np.maximum(dists[:, 0], eps)
    r2 = np.maximum(dists[:, 1], r1 + eps)
    return (1.0 / np.maximum(np.log(r2 / r1), eps)).astype(np.float32)


def cache_key(*parts: object) -> str:
    raw = "::".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def full_id_cache_path(cache_dir: Path, dataset: str, backbone: str, k: int, estimator: str) -> Path:
    return cache_dir / f"full_{dataset}_{backbone}_simclr_{estimator}_k{k}.npy"


def get_full_ids(cache_dir: Path, dataset: str, backbone: str, features: np.ndarray, k: int, cache: bool) -> np.ndarray:
    path = full_id_cache_path(cache_dir, dataset, backbone, k, "mle")
    if cache and path.exists():
        print(f"[cache] full local ID loaded: {path}", flush=True)
        return np.load(path).astype(np.float32)

    external_root = ROOT / "idpc_cache" / dataset / "seed1"
    external_mle = external_root / f"mle_local_k{k}.npy"
    if external_mle.exists():
        ids = np.load(external_mle).astype(np.float32)
        if ids.shape[0] == features.shape[0]:
            print(f"[cache] external local ID loaded: {external_mle}", flush=True)
            if cache:
                np.save(path, ids)
            return ids

    for knn_path in sorted(external_root.glob("knn_k*.npz")):
        match = re.search(r"knn_k([0-9]+)\.npz$", knn_path.name)
        if not match or int(match.group(1)) < int(k):
            continue
        cached = np.load(knn_path)
        dists = cached["dist"].astype(np.float32)
        if dists.shape[0] != features.shape[0] or dists.shape[1] < int(k):
            continue
        ids = mle_from_neighbor_distances(dists[:, : int(k)]).astype(np.float32)
        print(f"[cache] derived local ID k={k} from external kNN: {knn_path}", flush=True)
        if cache:
            np.save(path, ids)
        return ids

    print(f"[compute] full local ID: dataset={dataset} backbone={backbone} k={k} n={len(features)}", flush=True)
    ids = local_mle_id(features, k)
    if cache:
        np.save(path, ids)
    return ids


def get_twonn_ids(cache_dir: Path, dataset: str, backbone: str, features: np.ndarray, cache: bool) -> np.ndarray:
    path = full_id_cache_path(cache_dir, dataset, backbone, 2, "twonn")
    if cache and path.exists():
        print(f"[cache] 2NN ID loaded: {path}", flush=True)
        return np.load(path).astype(np.float32)
    external_root = ROOT / "idpc_cache" / dataset / "seed1"
    for knn_path in sorted(external_root.glob("knn_k*.npz")):
        cached = np.load(knn_path)
        dists = cached["dist"].astype(np.float32)
        if dists.shape[0] == features.shape[0] and dists.shape[1] >= 2:
            ids = mle_from_neighbor_distances(dists[:, :2]).astype(np.float32)
            print(f"[cache] derived 2NN ID from external kNN: {knn_path}", flush=True)
            if cache:
                np.save(path, ids)
            return ids
    print(f"[compute] 2NN ID: dataset={dataset} backbone={backbone} n={len(features)}", flush=True)
    ids = local_twonn_id(features)
    if cache:
        np.save(path, ids)
    return ids


def episode_record_map(run: Run) -> dict[int, dict]:
    records = {}
    for rec in run.summary.get("episode_records", []):
        if "episode" in rec:
            records[int(rec["episode"])] = rec
    return records


def load_episode_indices(run: Run, episode: int, name: str) -> np.ndarray | None:
    path = run.run_dir / f"episode_{episode}" / f"{name}.npy"
    if path.exists():
        return np.load(path, allow_pickle=True).astype(np.int64)
    return None


def active_indices(run: Run, episode: int, rec: dict) -> np.ndarray:
    active = load_episode_indices(run, episode, "activeSet")
    if active is not None:
        return active
    values = rec.get("active_set_ids") or []
    return np.asarray(values, dtype=np.int64)


def summarize_values(values: np.ndarray, prefix: str) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_sem": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_q25": float("nan"),
            f"{prefix}_q75": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_sem": sem(values),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
    }


def global_ids_for_lset(cache_dir: Path, run: Run, episode: int, lset: np.ndarray, features: np.ndarray, k: int, cache: bool) -> np.ndarray:
    key = cache_key(run.dataset, run.backbone, run.exp_name, episode, k, len(lset), int(np.sum(lset[: min(20, len(lset))])) if len(lset) else 0)
    path = cache_dir / f"global_{key}.npy"
    if cache and path.exists():
        return np.load(path).astype(np.float32)
    lset = lset[(lset >= 0) & (lset < len(features))]
    if len(lset) <= 2:
        ids = np.full(len(lset), np.nan, dtype=np.float32)
    else:
        ids = local_mle_id(features[lset], min(k, len(lset) - 1))
    if cache:
        np.save(path, ids)
    return ids


def id_bin_edges(ids: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(ids, 1 / 3)), float(np.quantile(ids, 2 / 3))


def id_bin_counts(values: np.ndarray, low_edge: float, high_edge: float) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"selected_low_id_fraction": float("nan"), "selected_mid_id_fraction": float("nan"), "selected_high_id_fraction": float("nan")}
    return {
        "selected_low_id_fraction": float(np.mean(values <= low_edge)),
        "selected_mid_id_fraction": float(np.mean((values > low_edge) & (values <= high_edge))),
        "selected_high_id_fraction": float(np.mean(values > high_edge)),
    }


def load_labels(dataset: str) -> np.ndarray | None:
    try:
        from torchvision.datasets import CIFAR10, CIFAR100
        from pycls.datasets.tiny_imagenet import TinyImageNet

        if dataset == "CIFAR10":
            data = CIFAR10(root=str(ROOT / "TypiClust/data"), train=True, download=False)
        elif dataset == "CIFAR100":
            data = CIFAR100(root=str(ROOT / "TypiClust/data"), train=True, download=False)
        elif dataset == "TINYIMAGENET":
            data = TinyImageNet(root=str(ROOT / "TypiClust/scan/datasets/TinyImageNet/tiny-imagenet-200"), split="train")
        else:
            return None
        return np.asarray(data.targets, dtype=np.int64)
    except Exception as exc:
        print(f"[labels] torchvision path unavailable for {dataset}: {exc}")

    try:
        if dataset == "CIFAR10":
            labels = []
            base = ROOT / "TypiClust/data/cifar-10-batches-py"
            for idx in range(1, 6):
                with (base / f"data_batch_{idx}").open("rb") as handle:
                    labels.extend(pickle.load(handle, encoding="latin1")["labels"])
            return np.asarray(labels, dtype=np.int64)
        if dataset == "CIFAR100":
            with (ROOT / "TypiClust/data/cifar-100-python/train").open("rb") as handle:
                return np.asarray(pickle.load(handle, encoding="latin1")["fine_labels"], dtype=np.int64)
        if dataset == "TINYIMAGENET":
            with (ROOT / "TypiClust/scan/datasets/TinyImageNet/train.pkl").open("rb") as handle:
                obj = pickle.load(handle, encoding="latin1")
            return np.asarray(obj[1], dtype=np.int64)
    except Exception as exc:
        print(f"[labels] pickle fallback unavailable for {dataset}: {exc}")
    return None


def class_entropy(labels: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    counts = np.bincount(labels.astype(np.int64))
    probs = counts[counts > 0] / float(labels.size)
    return float(-(probs * np.log(probs)).sum())


def coverage_proxy(features: np.ndarray, lset: np.ndarray, uset: np.ndarray, max_pool: int, seed: int) -> dict:
    lset = lset[(lset >= 0) & (lset < len(features))]
    uset = uset[(uset >= 0) & (uset < len(features))]
    if len(lset) == 0 or len(uset) == 0:
        return {"coverage_nn_mean": float("nan"), "coverage_nn_median": float("nan"), "coverage_nn_q90": float("nan"), "coverage_used_unlabeled": 0}
    rng = np.random.default_rng(seed)
    if len(uset) > max_pool:
        uset = rng.choice(uset, size=max_pool, replace=False)
    tree = cKDTree(features[lset])
    dists, _ = tree.query(features[uset], k=1, workers=-1)
    return {
        "coverage_nn_mean": float(np.mean(dists)),
        "coverage_nn_median": float(np.median(dists)),
        "coverage_nn_q90": float(np.quantile(dists, 0.90)),
        "coverage_used_unlabeled": int(len(uset)),
    }


def build_rows(args, runs: list[Run], cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    round_rows: list[dict] = []
    dist_rows: list[dict] = []
    class_rows: list[dict] = []
    coverage_rows: list[dict] = []
    k_rows: list[dict] = []
    cache_paths: set[str] = set()
    features_by_dataset: dict[str, np.ndarray] = {}
    labels_by_dataset: dict[str, np.ndarray | None] = {}

    for dataset in sorted({r.dataset for r in runs}):
        features_by_dataset[dataset] = load_features(dataset)
        labels_by_dataset[dataset] = load_labels(dataset)

    for run in runs:
        print(f"[run] {run.dataset}/{run.backbone}/{run.exp_name}", flush=True)
        features = features_by_dataset[run.dataset]
        labels = labels_by_dataset[run.dataset]
        recs = episode_record_map(run)
        episodes = args.episodes if args.episodes else sorted(recs)
        if not episodes:
            episodes = sorted(int(p.name.split("_")[-1]) for p in run.run_dir.glob("episode_*") if p.is_dir())

        ids_by_k = {}
        for k in args.k_values:
            ids_by_k[k] = get_full_ids(cache_dir, run.dataset, run.backbone, features, k, args.cache)
            cache_paths.add(str(full_id_cache_path(cache_dir, run.dataset, run.backbone, k, "mle")))
        twonn = get_twonn_ids(cache_dir, run.dataset, run.backbone, features, args.cache)
        cache_paths.add(str(full_id_cache_path(cache_dir, run.dataset, run.backbone, 2, "twonn")))

        ref_ids = ids_by_k[int(args.ref_k)]
        low_edge, high_edge = id_bin_edges(ref_ids)
        rank_percentile = stats.rankdata(ref_ids, method="average") / len(ref_ids)

        for episode in episodes:
            rec = recs.get(int(episode), {})
            active = active_indices(run, int(episode), rec)
            active = active[(active >= 0) & (active < len(features))]
            lset = load_episode_indices(run, int(episode), "lSet")
            if lset is None:
                lset = np.asarray([], dtype=np.int64)
            uset = load_episode_indices(run, int(episode), "uSet")
            if uset is None:
                uset = np.asarray([], dtype=np.int64)

            for k, full_ids in ids_by_k.items():
                selected_ids = full_ids[active] if len(active) else np.asarray([], dtype=np.float32)
                global_ids = global_ids_for_lset(cache_dir, run, int(episode), lset, features, int(k), args.cache)
                row = {
                    "dataset": run.dataset,
                    "backbone": run.backbone,
                    "method": run.method,
                    "method_label": METHOD_LABELS.get(run.method, run.method),
                    "seed": run.seed,
                    "budget": run.budget,
                    "exp_name": run.exp_name,
                    "episode": int(episode),
                    "k": int(k),
                    "test_accuracy": safe_float(rec.get("test_accuracy")),
                    "best_val_accuracy": safe_float(rec.get("best_val_accuracy")),
                    "labeled_count_before_sampling": safe_float(rec.get("labeled_count_before_sampling")),
                    "labeled_count_after_sampling": safe_float(rec.get("labeled_count_after_sampling")),
                    "selected_count": int(len(active)),
                    "labeled_set_size": int(len(lset)),
                    **summarize_values(selected_ids, "selected_local_id"),
                    **summarize_values(global_ids, "global_labeled_id"),
                }
                if int(k) == int(args.ref_k):
                    row.update(id_bin_counts(selected_ids, low_edge, high_edge))
                    row["selected_id_rank_percentile_mean"] = float(np.mean(rank_percentile[active])) if len(active) else float("nan")
                    row["selected_twonn_mean"] = float(np.mean(twonn[active])) if len(active) else float("nan")
                round_rows.append(row)

            if len(active):
                vals = ref_ids[active]
                for ep_group in episode_bucket(int(episode), args.episodes):
                    for value in vals:
                        dist_rows.append({
                            "dataset": run.dataset,
                            "backbone": run.backbone,
                            "method": run.method,
                            "seed": run.seed,
                            "episode": int(episode),
                            "episode_bucket": ep_group,
                            "selected_local_id": float(value),
                        })

            if labels is not None and len(lset):
                valid = lset[(lset >= 0) & (lset < len(labels))]
                label_values = labels[valid]
                counts = Counter(label_values.tolist())
                class_rows.append({
                    "dataset": run.dataset,
                    "backbone": run.backbone,
                    "method": run.method,
                    "seed": run.seed,
                    "exp_name": run.exp_name,
                    "episode": int(episode),
                    "labeled_set_size": int(len(valid)),
                    "num_classes_seen": int(len(counts)),
                    "class_entropy": class_entropy(label_values),
                    "class_max_fraction": float(max(counts.values()) / len(valid)) if len(valid) else float("nan"),
                    "class_min_count": int(min(counts.values())) if counts else 0,
                })
                if len(active):
                    valid_active = active[(active >= 0) & (active < len(labels))]
                    for cls in np.unique(labels[valid_active]):
                        cls_idx = valid_active[labels[valid_active] == cls]
                        class_rows.append({
                            "dataset": run.dataset,
                            "backbone": run.backbone,
                            "method": run.method,
                            "seed": run.seed,
                            "exp_name": run.exp_name,
                            "episode": int(episode),
                            "class_id": int(cls),
                            "selected_class_count": int(len(cls_idx)),
                            "selected_class_id_mean": float(np.mean(ref_ids[cls_idx])) if len(cls_idx) else float("nan"),
                            "row_type": "selected_per_class",
                        })

            if int(episode) in set(args.coverage_episodes) and len(lset) and len(uset):
                coverage_rows.append({
                    "dataset": run.dataset,
                    "backbone": run.backbone,
                    "method": run.method,
                    "seed": run.seed,
                    "exp_name": run.exp_name,
                    "episode": int(episode),
                    **coverage_proxy(features, lset, uset, args.coverage_max_unlabeled, run.seed * 1000 + int(episode)),
                })

        for k in args.k_values:
            if int(k) == int(args.ref_k):
                continue
            rho = safe_corr(ids_by_k[int(args.ref_k)], ids_by_k[int(k)], "spearman")
            pear = safe_corr(ids_by_k[int(args.ref_k)], ids_by_k[int(k)], "pearson")
            k_rows.append({
                "dataset": run.dataset,
                "backbone": run.backbone,
                "method": run.method,
                "seed": run.seed,
                "comparison": f"k={args.ref_k} vs k={k}",
                "spearman": rho,
                "pearson": pear,
            })
        k_rows.append({
            "dataset": run.dataset,
            "backbone": run.backbone,
            "method": run.method,
            "seed": run.seed,
            "comparison": f"MLE k={args.ref_k} vs 2NN",
            "spearman": safe_corr(ids_by_k[int(args.ref_k)], twonn, "spearman"),
            "pearson": safe_corr(ids_by_k[int(args.ref_k)], twonn, "pearson"),
        })

    return (
        pd.DataFrame(round_rows),
        pd.DataFrame(dist_rows),
        pd.DataFrame(class_rows),
        pd.DataFrame(coverage_rows),
        pd.DataFrame(k_rows),
        {"cache_paths": sorted(cache_paths)},
    )


def episode_bucket(episode: int, episodes: list[int]) -> list[str]:
    if not episodes:
        return [str(episode)]
    sorted_eps = sorted(episodes)
    picks = {sorted_eps[0]: "early", sorted_eps[len(sorted_eps) // 2]: "mid", sorted_eps[-1]: "late"}
    return [picks[episode]] if episode in picks else []


def aggregate_rounds(round_df: pd.DataFrame, ref_k: int) -> pd.DataFrame:
    numeric = [
        "test_accuracy",
        "selected_local_id_mean",
        "global_labeled_id_mean",
        "selected_low_id_fraction",
        "selected_mid_id_fraction",
        "selected_high_id_fraction",
        "selected_id_rank_percentile_mean",
    ]
    rows = []
    subset = round_df[round_df["k"] == int(ref_k)].copy()
    for keys, group in subset.groupby(["dataset", "backbone", "method", "episode"], dropna=False):
        row = dict(zip(["dataset", "backbone", "method", "episode"], keys))
        row["n_seeds"] = int(group["seed"].nunique())
        for col in numeric:
            row[f"{col}_mean"] = safe_mean(group[col])
            row[f"{col}_sem"] = sem(group[col])
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_summary(round_df: pd.DataFrame, ref_k: int) -> pd.DataFrame:
    rows = []
    subset = round_df[round_df["k"] == int(ref_k)].copy()
    for keys, group in subset.groupby(["dataset", "backbone", "method"], dropna=False):
        group = group.sort_values(["seed", "episode"])
        gain_values = []
        id_values = []
        for _, seed_group in group.groupby("seed"):
            seed_group = seed_group.sort_values("episode")
            acc = seed_group["test_accuracy"].to_numpy(dtype=float)
            ids = seed_group["selected_local_id_mean"].to_numpy(dtype=float)
            if len(acc) >= 2:
                gain_values.extend(np.diff(acc).tolist())
                id_values.extend(ids[1:].tolist())
        rows.append({
            "dataset": keys[0],
            "backbone": keys[1],
            "method": keys[2],
            "pearson_acc_vs_global_id": safe_corr(group["test_accuracy"], group["global_labeled_id_mean"], "pearson"),
            "spearman_acc_vs_global_id": safe_corr(group["test_accuracy"], group["global_labeled_id_mean"], "spearman"),
            "pearson_gain_vs_selected_id": safe_corr(gain_values, id_values, "pearson"),
            "spearman_gain_vs_selected_id": safe_corr(gain_values, id_values, "spearman"),
            "n_points": int(len(group)),
        })
    return pd.DataFrame(rows)


def set_plot_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_fig(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()


def plot_main_id_dynamics(agg: pd.DataFrame, fig_dir: Path) -> list[str]:
    paths = []
    for (dataset, backbone), group in agg.groupby(["dataset", "backbone"]):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharex=True)
        metrics = [
            ("test_accuracy_mean", "test_accuracy_sem", "Test accuracy (%)"),
            ("selected_local_id_mean_mean", "selected_local_id_mean_sem", "Selected local ID"),
            ("global_labeled_id_mean_mean", "global_labeled_id_mean_sem", "Global labeled ID"),
        ]
        for ax, (mean_col, sem_col, ylabel) in zip(axes, metrics):
            for method, method_group in group.groupby("method"):
                method_group = method_group.sort_values("episode")
                x = method_group["episode"].to_numpy()
                y = method_group[mean_col].to_numpy(dtype=float)
                e = method_group[sem_col].to_numpy(dtype=float)
                color = METHOD_COLORS.get(method)
                ax.plot(x, y, label=METHOD_LABELS.get(method, method), color=color, linewidth=1.8)
                ax.fill_between(x, y - e, y + e, color=color, alpha=0.14)
            ax.set_xlabel("Episode")
            ax.set_ylabel(ylabel)
        axes[0].set_title(f"{DATASET_LABELS.get(dataset, dataset)} / {backbone}")
        axes[-1].legend(fontsize=7, frameon=False)
        out = fig_dir / f"main_id_dynamics_{dataset}_{backbone}.png"
        save_fig(out)
        paths.append(str(out))
    return paths


def plot_distribution(dist_df: pd.DataFrame, fig_dir: Path) -> list[str]:
    paths = []
    if dist_df.empty:
        return paths
    for (dataset, backbone), group in dist_df.groupby(["dataset", "backbone"]):
        buckets = ["early", "mid", "late"]
        methods = [m for m in METHOD_LABELS if m in set(group["method"])]
        fig, axes = plt.subplots(1, len(buckets), figsize=(14, 3.8), sharey=True)
        if len(buckets) == 1:
            axes = [axes]
        for ax, bucket in zip(axes, buckets):
            vals = []
            labels = []
            colors = []
            for method in methods:
                arr = group[(group["method"] == method) & (group["episode_bucket"] == bucket)]["selected_local_id"].dropna().to_numpy()
                if len(arr):
                    vals.append(arr)
                    labels.append(METHOD_LABELS.get(method, method))
                    colors.append(METHOD_COLORS.get(method, "#333333"))
            if vals:
                parts = ax.violinplot(vals, showmeans=True, showextrema=False)
                for body, color in zip(parts["bodies"], colors):
                    body.set_facecolor(color)
                    body.set_alpha(0.45)
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, rotation=35, ha="right")
            ax.set_title(bucket.capitalize())
            ax.set_ylabel("Selected local ID")
        fig.suptitle(f"Selected local ID distribution: {DATASET_LABELS.get(dataset, dataset)} / {backbone}")
        out = fig_dir / f"local_id_distribution_{dataset}_{backbone}.png"
        save_fig(out)
        paths.append(str(out))
    return paths


def plot_accuracy_vs_id(round_df: pd.DataFrame, fig_dir: Path, ref_k: int) -> list[str]:
    paths = []
    subset = round_df[round_df["k"] == int(ref_k)].copy()
    for (dataset, backbone), group in subset.groupby(["dataset", "backbone"]):
        final = group.sort_values("episode").groupby(["method", "seed"], as_index=False).tail(1)
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        for method, method_group in final.groupby("method"):
            ax.scatter(
                method_group["global_labeled_id_mean"],
                method_group["test_accuracy"],
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method),
                s=38,
                alpha=0.85,
            )
        ax.set_xlabel("Final global labeled ID")
        ax.set_ylabel("Final test accuracy (%)")
        ax.set_title(f"Accuracy vs geometry: {DATASET_LABELS.get(dataset, dataset)} / {backbone}")
        ax.legend(fontsize=7, frameon=False)
        out = fig_dir / f"accuracy_vs_id_{dataset}_{backbone}.png"
        save_fig(out)
        paths.append(str(out))
    return paths


def plot_rank_profile(agg: pd.DataFrame, fig_dir: Path) -> list[str]:
    paths = []
    for (dataset, backbone), group in agg.groupby(["dataset", "backbone"]):
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        for method, method_group in group.groupby("method"):
            method_group = method_group.sort_values("episode")
            y = method_group["selected_id_rank_percentile_mean_mean"].to_numpy(dtype=float)
            e = method_group["selected_id_rank_percentile_mean_sem"].to_numpy(dtype=float)
            x = method_group["episode"].to_numpy()
            color = METHOD_COLORS.get(method)
            ax.plot(x, y, label=METHOD_LABELS.get(method, method), color=color)
            ax.fill_between(x, y - e, y + e, color=color, alpha=0.13)
        ax.axhline(0.5, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean selected ID rank percentile")
        ax.set_title(f"ID rank profile: {DATASET_LABELS.get(dataset, dataset)} / {backbone}")
        ax.legend(fontsize=7, frameon=False)
        out = fig_dir / f"id_rank_profile_{dataset}_{backbone}.png"
        save_fig(out)
        paths.append(str(out))
    return paths


def plot_class_balance(class_df: pd.DataFrame, fig_dir: Path) -> list[str]:
    paths = []
    if class_df.empty or "class_entropy" not in class_df:
        return paths
    summary_rows = class_df[class_df.get("row_type", "") != "selected_per_class"].copy()
    if summary_rows.empty:
        return paths
    for (dataset, backbone), group in summary_rows.groupby(["dataset", "backbone"]):
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharex=True)
        for method, method_group in group.groupby("method"):
            agg = method_group.groupby("episode").agg(
                class_entropy_mean=("class_entropy", "mean"),
                class_entropy_sem=("class_entropy", sem),
                class_max_fraction_mean=("class_max_fraction", "mean"),
                class_max_fraction_sem=("class_max_fraction", sem),
            ).reset_index()
            color = METHOD_COLORS.get(method)
            axes[0].plot(agg["episode"], agg["class_entropy_mean"], color=color, label=METHOD_LABELS.get(method, method))
            axes[1].plot(agg["episode"], agg["class_max_fraction_mean"], color=color, label=METHOD_LABELS.get(method, method))
        axes[0].set_ylabel("Class entropy")
        axes[1].set_ylabel("Largest class fraction")
        for ax in axes:
            ax.set_xlabel("Episode")
        axes[0].set_title(f"Class balance: {DATASET_LABELS.get(dataset, dataset)} / {backbone}")
        axes[1].legend(fontsize=7, frameon=False)
        out = fig_dir / f"class_balance_{dataset}_{backbone}.png"
        save_fig(out)
        paths.append(str(out))
    return paths


def plot_coverage(coverage_df: pd.DataFrame, fig_dir: Path) -> list[str]:
    paths = []
    if coverage_df.empty:
        return paths
    for (dataset, backbone), group in coverage_df.groupby(["dataset", "backbone"]):
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        for method, method_group in group.groupby("method"):
            agg = method_group.groupby("episode").agg(
                mean=("coverage_nn_mean", "mean"),
                err=("coverage_nn_mean", sem),
            ).reset_index()
            color = METHOD_COLORS.get(method)
            ax.plot(agg["episode"], agg["mean"], color=color, label=METHOD_LABELS.get(method, method))
            ax.fill_between(agg["episode"], agg["mean"] - agg["err"], agg["mean"] + agg["err"], color=color, alpha=0.13)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Unlabeled-to-labeled NN distance")
        ax.set_title(f"Coverage proxy: {DATASET_LABELS.get(dataset, dataset)} / {backbone}")
        ax.legend(fontsize=7, frameon=False)
        out = fig_dir / f"coverage_proxy_{dataset}_{backbone}.png"
        save_fig(out)
        paths.append(str(out))
    return paths


def plot_heatmaps(agg: pd.DataFrame, fig_dir: Path) -> list[str]:
    paths = []
    for (dataset, backbone), group in agg.groupby(["dataset", "backbone"]):
        pivot = group.pivot_table(index="method", columns="episode", values="selected_local_id_mean_mean", aggfunc="mean")
        if pivot.empty:
            continue
        order = [m for m in METHOD_LABELS if m in pivot.index]
        pivot = pivot.loc[order]
        fig, ax = plt.subplots(figsize=(9, 3.8))
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in pivot.index])
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45)
        ax.set_xlabel("Episode")
        ax.set_title(f"Selected local ID heatmap: {DATASET_LABELS.get(dataset, dataset)} / {backbone}")
        plt.colorbar(im, ax=ax, label="Mean selected local ID")
        out = fig_dir / f"selected_id_heatmap_{dataset}_{backbone}.png"
        save_fig(out)
        paths.append(str(out))
    return paths


def embedding_plot(args, runs: list[Run], round_df: pd.DataFrame, cache_dir: Path, fig_dir: Path) -> list[str]:
    paths = []
    if args.skip_embedding_plots:
        return paths
    try:
        from sklearn.decomposition import PCA
    except Exception as exc:
        print(f"[embedding] skipped: sklearn PCA unavailable: {exc}")
        return paths

    by_dataset_backbone = defaultdict(list)
    for run in runs:
        by_dataset_backbone[(run.dataset, run.backbone)].append(run)

    for (dataset, backbone), group_runs in by_dataset_backbone.items():
        features = load_features(dataset)
        ref_ids_path = full_id_cache_path(cache_dir, dataset, backbone, int(args.ref_k), "mle")
        if not ref_ids_path.exists():
            continue
        ref_ids = np.load(ref_ids_path)
        rng = np.random.default_rng(args.embedding_seed)
        n = min(args.embedding_points, len(features))
        base_idx = np.sort(rng.choice(np.arange(len(features)), size=n, replace=False))
        emb_path = cache_dir / f"embedding_pca_{dataset}_{backbone}_{n}.npy"
        idx_path = cache_dir / f"embedding_pca_{dataset}_{backbone}_{n}_idx.npy"
        if args.cache and emb_path.exists() and idx_path.exists():
            xy = np.load(emb_path)
            base_idx = np.load(idx_path)
        else:
            xy = PCA(n_components=2, random_state=args.embedding_seed).fit_transform(features[base_idx])
            if args.cache:
                np.save(emb_path, xy)
                np.save(idx_path, base_idx)
        idx_pos = {int(v): i for i, v in enumerate(base_idx)}

        for method in sorted({r.method for r in group_runs}):
            method_runs = [r for r in group_runs if r.method == method]
            if not method_runs:
                continue
            selected = set()
            for run in method_runs:
                recs = episode_record_map(run)
                for episode in [args.episodes[0], args.episodes[len(args.episodes) // 2], args.episodes[-1]]:
                    active = active_indices(run, int(episode), recs.get(int(episode), {}))
                    selected.update(int(v) for v in active)
            selected_pos = [idx_pos[v] for v in selected if v in idx_pos]
            fig, ax = plt.subplots(figsize=(5.4, 4.8))
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=ref_ids[base_idx], cmap="viridis", s=7, alpha=0.45, linewidths=0)
            if selected_pos:
                ax.scatter(xy[selected_pos, 0], xy[selected_pos, 1], facecolors="none", edgecolors=METHOD_COLORS.get(method, "red"), s=32, linewidths=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"PCA geometry: {DATASET_LABELS.get(dataset, dataset)} / {backbone} / {METHOD_LABELS.get(method, method)}")
            plt.colorbar(sc, ax=ax, label=f"Local ID k={args.ref_k}")
            out = fig_dir / f"umap_selected_{dataset}_{backbone}_{method}.png"
            save_fig(out)
            paths.append(str(out))
    return paths


def make_tables(report_dir: Path, corr_df: pd.DataFrame, k_df: pd.DataFrame, class_df: pd.DataFrame) -> list[str]:
    table_dir = report_dir / "tables"
    ensure_dir(table_dir)
    outputs = []
    if not corr_df.empty:
        rows = []
        for _, row in corr_df.iterrows():
            rows.append({
                "Dataset": DATASET_LABELS.get(row["dataset"], row["dataset"]),
                "Backbone": row["backbone"],
                "Method": METHOD_LABELS.get(row["method"], row["method"]),
                "Acc--Global ID": f"{row['spearman_acc_vs_global_id']:.3f}",
                "Gain--Selected ID": f"{row['spearman_gain_vs_selected_id']:.3f}",
            })
        path = table_dir / "id_accuracy_correlation_table.tex"
        write_tex_table(path, list(rows[0].keys()), rows, "Spearman correlations between ID diagnostics and accuracy.", "tab:id_accuracy_correlation")
        outputs.append(str(path))
    if not k_df.empty:
        rows = []
        grouped = k_df.groupby(["dataset", "backbone", "comparison"])
        for keys, group in grouped:
            rows.append({
                "Dataset": DATASET_LABELS.get(keys[0], keys[0]),
                "Backbone": keys[1],
                "Comparison": keys[2],
                "Spearman": f"{safe_mean(group['spearman']):.3f} $\\pm$ {sem(group['spearman']):.3f}",
            })
        path = table_dir / "k_sensitivity_table.tex"
        write_tex_table(path, list(rows[0].keys()), rows, "Robustness of ID rankings across estimators and neighborhood sizes.", "tab:k_sensitivity")
        outputs.append(str(path))
    if not class_df.empty and "class_entropy" in class_df:
        summary_rows = class_df[class_df.get("row_type", "") != "selected_per_class"].copy()
        if not summary_rows.empty:
            rows = []
            final = summary_rows.sort_values("episode").groupby(["dataset", "backbone", "method", "seed"], as_index=False).tail(1)
            for keys, group in final.groupby(["dataset", "backbone", "method"]):
                rows.append({
                    "Dataset": DATASET_LABELS.get(keys[0], keys[0]),
                    "Backbone": keys[1],
                    "Method": METHOD_LABELS.get(keys[2], keys[2]),
                    "Class entropy": f"{safe_mean(group['class_entropy']):.3f} $\\pm$ {sem(group['class_entropy']):.3f}",
                    "Classes seen": f"{safe_mean(group['num_classes_seen']):.1f}",
                })
            if rows:
                path = table_dir / "class_balance_table.tex"
                write_tex_table(path, list(rows[0].keys()), rows, "Final labeled-set class balance diagnostics.", "tab:class_balance")
                outputs.append(str(path))
    return outputs


def write_figure_snippets(report_dir: Path, figure_paths: list[str]) -> str:
    path = report_dir / "figure_snippets.tex"
    with path.open("w") as handle:
        for fig_path in figure_paths:
            rel = Path(fig_path).relative_to(report_dir)
            label = rel.stem.replace("_", ":")
            caption = rel.stem.replace("_", " ")
            handle.write("\\begin{figure}[t]\n\\centering\n")
            handle.write(f"\\includegraphics[width=0.95\\linewidth]{{{rel}}}\n")
            handle.write(f"\\caption{{{caption}.}}\n")
            handle.write(f"\\label{{fig:{label}}}\n")
            handle.write("\\end{figure}\n\n")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="/scratch/s219110279/TypiClust/output")
    parser.add_argument("--report-dir", default="/scratch/s219110279/paper_results/id_thesis")
    parser.add_argument("--datasets", nargs="+", default=["CIFAR10", "CIFAR100", "TINYIMAGENET"])
    parser.add_argument("--backbones", nargs="+", default=["auto"])
    parser.add_argument("--methods", nargs="+", default=["random", "uncertainty", "entropy", "margin", "dbal", "coreset", "probcover"])
    parser.add_argument("--budgets", nargs="+", default=["50"], help="Budget-per-round values to include, or 'all'. Default: 50.")
    parser.add_argument("--k-values", nargs="+", type=int, default=[25, 50, 75])
    parser.add_argument("--ref-k", type=int, default=50)
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 25, 50, 75, 100])
    parser.add_argument("--feature-space", choices=["simclr"], default="simclr")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--min-rounds", type=int, default=100)
    parser.add_argument("--max-runs-per-method", type=int, default=0)
    parser.add_argument("--coverage-episodes", nargs="+", type=int, default=[0, 25, 50, 75, 100])
    parser.add_argument("--coverage-max-unlabeled", type=int, default=5000)
    parser.add_argument("--embedding-points", type=int, default=3500)
    parser.add_argument("--embedding-seed", type=int, default=123)
    parser.add_argument("--skip-embedding-plots", action="store_true")
    args = parser.parse_args()

    if args.ref_k not in args.k_values:
        args.k_values = sorted(set(args.k_values + [args.ref_k]))

    report_dir = Path(args.report_dir)
    fig_dir = report_dir / "figures"
    cache_dir = report_dir / "cache"
    ensure_dir(report_dir)
    ensure_dir(fig_dir)
    ensure_dir(cache_dir)
    set_plot_style()

    runs, skipped = discover_runs(args)
    if not runs:
        raise SystemExit("No matching runs found. Check --output-root, --datasets, --backbones, and --methods.")

    run_index = pd.DataFrame([{
        "dataset": r.dataset,
        "backbone": r.backbone,
        "method": r.method,
        "method_label": METHOD_LABELS.get(r.method, r.method),
        "seed": r.seed,
        "budget": r.budget,
        "exp_name": r.exp_name,
        "run_dir": str(r.run_dir),
        "num_rounds_completed": r.summary.get("num_rounds_completed"),
        "final_test_accuracy": r.summary.get("final_test_accuracy"),
    } for r in runs])
    run_index.to_csv(report_dir / "run_index.csv", index=False)

    round_df, dist_df, class_df, coverage_df, k_df, cache_meta = build_rows(args, runs, cache_dir)
    round_df.to_csv(report_dir / "round_id_summary.csv", index=False)
    dist_df.to_csv(report_dir / "selected_id_distribution.csv", index=False)
    class_df.to_csv(report_dir / "class_balance_summary.csv", index=False)
    coverage_df.to_csv(report_dir / "coverage_proxy_summary.csv", index=False)
    k_df.to_csv(report_dir / "k_sensitivity_summary.csv", index=False)

    agg = aggregate_rounds(round_df, args.ref_k)
    agg.to_csv(report_dir / "round_id_aggregated.csv", index=False)
    corr_df = correlation_summary(round_df, args.ref_k)
    corr_df.to_csv(report_dir / "accuracy_id_correlation.csv", index=False)

    figures = []
    figures += plot_main_id_dynamics(agg, fig_dir)
    figures += plot_distribution(dist_df, fig_dir)
    figures += plot_accuracy_vs_id(round_df, fig_dir, args.ref_k)
    figures += plot_rank_profile(agg, fig_dir)
    figures += plot_class_balance(class_df, fig_dir)
    figures += plot_coverage(coverage_df, fig_dir)
    figures += plot_heatmaps(agg, fig_dir)
    figures += embedding_plot(args, runs, round_df, cache_dir, fig_dir)
    table_paths = make_tables(report_dir, corr_df, k_df, class_df)
    snippet_path = write_figure_snippets(report_dir, figures)

    skipped_path = report_dir / "skipped_runs.csv"
    pd.DataFrame(skipped).to_csv(skipped_path, index=False)
    manifest = {
        "script": str(Path(__file__).resolve()),
        "output_root": str(Path(args.output_root)),
        "report_dir": str(report_dir),
        "included_runs": int(len(runs)),
        "skipped_runs": int(len(skipped)),
        "datasets": args.datasets,
        "backbones": args.backbones,
        "methods": args.methods,
        "k_values": args.k_values,
        "ref_k": args.ref_k,
        "csvs": [
            "run_index.csv",
            "round_id_summary.csv",
            "round_id_aggregated.csv",
            "selected_id_distribution.csv",
            "accuracy_id_correlation.csv",
            "class_balance_summary.csv",
            "coverage_proxy_summary.csv",
            "k_sensitivity_summary.csv",
            "skipped_runs.csv",
        ],
        "figures": figures,
        "tables": table_paths,
        "figure_snippets": snippet_path,
        **cache_meta,
    }
    write_json(report_dir / "manifest.json", manifest)
    print(f"[done] report_dir={report_dir}")
    print(f"[done] included_runs={len(runs)} skipped_runs={len(skipped)} figures={len(figures)}")


if __name__ == "__main__":
    main()
