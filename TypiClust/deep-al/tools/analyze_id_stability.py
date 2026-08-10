#!/usr/bin/env python3

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import stats

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEEP_AL_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))

import sys

if _DEEP_AL_ROOT not in sys.path:
    sys.path.insert(0, _DEEP_AL_ROOT)

import pycls.datasets.utils as ds_utils


def _safe_spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def _iqr(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25))


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _sample_indices(pool_indices, max_points, seed):
    pool_indices = np.asarray(pool_indices, dtype=np.int64)
    if max_points is None or len(pool_indices) <= max_points:
        return np.sort(pool_indices)
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(pool_indices, size=int(max_points), replace=False)
    return np.sort(chosen.astype(np.int64))


def _knn_distances(features, k):
    n = features.shape[0]
    k_eff = max(2, min(int(k), n - 1))
    tree = cKDTree(features)
    dists, neigh = tree.query(features, k=k_eff + 1, workers=-1)
    dists = np.asarray(dists[:, 1:], dtype=np.float64)
    neigh = np.asarray(neigh[:, 1:], dtype=np.int64)
    return dists, neigh


def local_mle_id(features, k, eps=1e-12):
    dists, _ = _knn_distances(features, k)
    rk = np.maximum(dists[:, -1], eps)
    prev = np.maximum(dists[:, :-1], eps)
    logs = np.log(rk[:, None] / prev)
    denom = np.mean(logs, axis=1)
    denom = np.maximum(denom, eps)
    ids = 1.0 / denom
    return ids.astype(np.float64), dists


def local_twonn_ratio_id(features, eps=1e-12):
    dists, _ = _knn_distances(features, 2)
    r1 = np.maximum(dists[:, 0], eps)
    r2 = np.maximum(dists[:, 1], r1 + eps)
    ids = 1.0 / np.maximum(np.log(r2 / r1), eps)
    return ids.astype(np.float64), dists


def normalize_ratio(ids, eps=1e-12):
    ids = np.asarray(ids, dtype=np.float64)
    med = float(np.median(ids))
    return (ids + eps) / (med + eps)


def idprobcover_select(features, ids, budget, delta0, alpha, k_knn, eps=1e-12):
    n = features.shape[0]
    budget = min(int(budget), n)
    rho = normalize_ratio(ids, eps=eps)
    delta = float(delta0) * np.power(rho, -float(alpha))

    dists, neigh = _knn_distances(features, k_knn)
    out_neighbors = []
    in_sources = [[] for _ in range(n)]
    for i in range(n):
        nbrs = neigh[i][dists[i] < delta[i]].astype(np.int32)
        if nbrs.size == 0 or not np.any(nbrs == i):
            nbrs = np.concatenate([np.array([i], dtype=np.int32), nbrs])
        out_neighbors.append(nbrs)
        for j in nbrs:
            in_sources[int(j)].append(i)
    in_sources = [np.asarray(v, dtype=np.int32) for v in in_sources]

    covered = np.zeros(n, dtype=bool)
    degree = np.zeros(n, dtype=np.int32)
    for i in range(n):
        degree[i] = int(np.sum(~covered[out_neighbors[i]]))

    selected = []
    selected_mask = np.zeros(n, dtype=bool)
    for _ in range(budget):
        cand_deg = degree.copy()
        cand_deg[selected_mask] = -1
        best = int(cand_deg.max())
        if best < 0:
            break
        cands = np.where(cand_deg == best)[0]
        pick = int(cands[np.argmin(ids[cands])])
        selected.append(pick)
        selected_mask[pick] = True
        newly = out_neighbors[pick]
        newly = newly[~covered[newly]]
        if newly.size:
            covered[newly] = True
            for j in newly:
                srcs = in_sources[int(j)]
                if srcs.size:
                    degree[srcs] -= 1
        degree[pick] = 0
    return np.asarray(selected, dtype=np.int64)


def discover_runs(output_root, dataset, model, method):
    root = Path(output_root) / dataset / model
    runs = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "benchmark_summary.json"
        if not summary_path.exists():
            continue
        summary = _load_json(summary_path)
        if str(summary.get("sampling_fn", "")).lower() != method.lower():
            continue
        runs.append({
            "run_dir": str(run_dir),
            "exp_name": str(summary.get("exp_name", run_dir.name)),
            "seed": int(summary.get("seed", -1)),
        })
    return runs


def load_pool_indices(run_dir, episode):
    pool_path = Path(run_dir) / f"episode_{int(episode)}" / "uSet.npy"
    if not pool_path.exists():
        raise FileNotFoundError(f"Missing pool snapshot: {pool_path}")
    return np.load(str(pool_path), allow_pickle=True).astype(np.int64)


def summarize_ids(ids):
    ids = np.asarray(ids, dtype=np.float64)
    return {
        "mean": float(np.mean(ids)),
        "std": float(np.std(ids)),
        "median": float(np.median(ids)),
        "iqr": _iqr(ids),
        "min": float(np.min(ids)),
        "max": float(np.max(ids)),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze ID stability and estimator sensitivity on saved AL pool snapshots.")
    parser.add_argument("--output_root", type=str, default="/scratch/s219110279/TypiClust/output")
    parser.add_argument("--dataset", type=str, default="CIFAR100")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--method", type=str, default="idprobcover")
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 25, 50, 75, 100])
    parser.add_argument("--k_values", nargs="+", type=int, default=[20, 50, 75])
    parser.add_argument("--max_points", type=int, default=5000)
    parser.add_argument("--selection_budget", type=int, default=50)
    parser.add_argument("--delta0", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--k_knn", type=int, default=50)
    parser.add_argument("--report_dir", type=str, required=True)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    features = ds_utils.load_features(args.dataset, seed=1, train=True, normalized=True).astype(np.float64)
    runs = discover_runs(args.output_root, args.dataset, args.model, args.method)
    if not runs:
        raise SystemExit("No matching runs found.")

    snapshot_rows = []
    estimator_cmp_rows = []
    temporal_rows = []
    seed_rows = []
    selection_rows = []

    snapshot_cache = {}
    for run in runs:
        for episode in args.episodes:
            pool_indices = load_pool_indices(run["run_dir"], episode)
            sampled_pool = _sample_indices(pool_indices, args.max_points, seed=run["seed"] * 1000 + int(episode))
            x = features[sampled_pool]

            ids_by_name = {}
            rho_by_name = {}

            for k in args.k_values:
                ids, _ = local_mle_id(x, k=k)
                name = f"mle_k{k}"
                ids_by_name[name] = ids
                rho_by_name[name] = normalize_ratio(ids)
                stats_row = summarize_ids(ids)
                snapshot_rows.append({
                    "run_dir": run["run_dir"],
                    "exp_name": run["exp_name"],
                    "seed": run["seed"],
                    "episode": int(episode),
                    "pool_size": int(len(pool_indices)),
                    "used_points": int(len(sampled_pool)),
                    "estimator": name,
                    **stats_row,
                })

            ids_twonn, _ = local_twonn_ratio_id(x)
            ids_by_name["local_twonn"] = ids_twonn
            rho_by_name["local_twonn"] = normalize_ratio(ids_twonn)
            stats_row = summarize_ids(ids_twonn)
            snapshot_rows.append({
                "run_dir": run["run_dir"],
                "exp_name": run["exp_name"],
                "seed": run["seed"],
                "episode": int(episode),
                "pool_size": int(len(pool_indices)),
                "used_points": int(len(sampled_pool)),
                "estimator": "local_twonn",
                **stats_row,
            })

            ref_name = "mle_k50" if "mle_k50" in ids_by_name else sorted(ids_by_name.keys())[0]
            ref_sel = idprobcover_select(
                x,
                ids_by_name[ref_name],
                budget=args.selection_budget,
                delta0=args.delta0,
                alpha=args.alpha,
                k_knn=args.k_knn,
            )
            ref_sel_set = set(ref_sel.tolist())

            for name, ids in ids_by_name.items():
                estimator_cmp_rows.append({
                    "run_dir": run["run_dir"],
                    "exp_name": run["exp_name"],
                    "seed": run["seed"],
                    "episode": int(episode),
                    "reference_estimator": ref_name,
                    "estimator": name,
                    "id_spearman_vs_ref": _safe_spearman(ids_by_name[ref_name], ids),
                    "rho_spearman_vs_ref": _safe_spearman(rho_by_name[ref_name], rho_by_name[name]),
                })
                sel = idprobcover_select(
                    x,
                    ids,
                    budget=args.selection_budget,
                    delta0=args.delta0,
                    alpha=args.alpha,
                    k_knn=args.k_knn,
                )
                sel_set = set(sel.tolist())
                union = max(len(ref_sel_set | sel_set), 1)
                selection_rows.append({
                    "run_dir": run["run_dir"],
                    "exp_name": run["exp_name"],
                    "seed": run["seed"],
                    "episode": int(episode),
                    "reference_estimator": ref_name,
                    "estimator": name,
                    "selection_overlap_count": int(len(ref_sel_set & sel_set)),
                    "selection_jaccard": float(len(ref_sel_set & sel_set) / union),
                })

            snapshot_cache[(run["exp_name"], int(episode))] = {
                "pool_indices": sampled_pool,
                "ids_by_name": ids_by_name,
                "rho_by_name": rho_by_name,
            }

    for run in runs:
        for ep_a, ep_b in zip(args.episodes[:-1], args.episodes[1:]):
            left = snapshot_cache[(run["exp_name"], int(ep_a))]
            right = snapshot_cache[(run["exp_name"], int(ep_b))]
            common = np.intersect1d(left["pool_indices"], right["pool_indices"])
            if common.size == 0:
                continue
            common = _sample_indices(common, args.max_points, seed=run["seed"] * 10000 + ep_a + ep_b)
            left_pos = {int(v): i for i, v in enumerate(left["pool_indices"])}
            right_pos = {int(v): i for i, v in enumerate(right["pool_indices"])}
            left_idx = np.asarray([left_pos[int(v)] for v in common], dtype=np.int64)
            right_idx = np.asarray([right_pos[int(v)] for v in common], dtype=np.int64)
            for name in left["rho_by_name"].keys():
                if name not in right["rho_by_name"]:
                    continue
                temporal_rows.append({
                    "exp_name": run["exp_name"],
                    "seed": run["seed"],
                    "episode_a": int(ep_a),
                    "episode_b": int(ep_b),
                    "estimator": name,
                    "common_points": int(len(common)),
                    "rho_spearman": _safe_spearman(left["rho_by_name"][name][left_idx], right["rho_by_name"][name][right_idx]),
                    "id_spearman": _safe_spearman(left["ids_by_name"][name][left_idx], right["ids_by_name"][name][right_idx]),
                })

    for i, run_a in enumerate(runs):
        for run_b in runs[i + 1:]:
            for episode in args.episodes:
                left = snapshot_cache[(run_a["exp_name"], int(episode))]
                right = snapshot_cache[(run_b["exp_name"], int(episode))]
                common = np.intersect1d(left["pool_indices"], right["pool_indices"])
                if common.size == 0:
                    continue
                common = _sample_indices(common, args.max_points, seed=episode * 10000 + run_a["seed"] * 10 + run_b["seed"])
                left_pos = {int(v): idx for idx, v in enumerate(left["pool_indices"])}
                right_pos = {int(v): idx for idx, v in enumerate(right["pool_indices"])}
                left_idx = np.asarray([left_pos[int(v)] for v in common], dtype=np.int64)
                right_idx = np.asarray([right_pos[int(v)] for v in common], dtype=np.int64)
                for name in left["rho_by_name"].keys():
                    if name not in right["rho_by_name"]:
                        continue
                    seed_rows.append({
                        "episode": int(episode),
                        "seed_a": int(run_a["seed"]),
                        "seed_b": int(run_b["seed"]),
                        "estimator": name,
                        "common_points": int(len(common)),
                        "rho_spearman": _safe_spearman(left["rho_by_name"][name][left_idx], right["rho_by_name"][name][right_idx]),
                        "id_spearman": _safe_spearman(left["ids_by_name"][name][left_idx], right["ids_by_name"][name][right_idx]),
                    })

    pd.DataFrame(snapshot_rows).to_csv(report_dir / f"{args.dataset.lower()}_{args.method}_id_snapshot_summary.csv", index=False)
    pd.DataFrame(estimator_cmp_rows).to_csv(report_dir / f"{args.dataset.lower()}_{args.method}_estimator_comparison.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(report_dir / f"{args.dataset.lower()}_{args.method}_selection_overlap.csv", index=False)
    pd.DataFrame(temporal_rows).to_csv(report_dir / f"{args.dataset.lower()}_{args.method}_temporal_stability.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(report_dir / f"{args.dataset.lower()}_{args.method}_seed_stability.csv", index=False)

    summary = {
        "dataset": args.dataset,
        "method": args.method,
        "episodes": [int(x) for x in args.episodes],
        "k_values": [int(x) for x in args.k_values],
        "max_points": int(args.max_points),
        "selection_budget": int(args.selection_budget),
        "run_count": int(len(runs)),
        "run_names": [run["exp_name"] for run in runs],
    }
    with open(report_dir / f"{args.dataset.lower()}_{args.method}_id_analysis_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
