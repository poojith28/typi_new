#!/usr/bin/env python3
"""Recompute and strictly replay one canonical historical LIDCover seed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "thesis_appendix_audit/_regenerated_lidcover/lidcover_completion_matrix.csv"
STAGING = ROOT / "analysis/lidcover_extensions_2026/mechanism_staging"
FEATURES = ROOT / "results/results/resnet18/cifar-100/pretext/features_seed1.npy"
LABELS = ROOT / "results/results/resnet18/cifar-100/pretext/labels_seed1.npy"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode()); digest.update(np.asarray(values.shape, dtype=np.int64).tobytes()); digest.update(values.tobytes())
    return digest.hexdigest()


def load_ids(path):
    values = np.load(path, allow_pickle=True)
    if values.dtype == object:
        values = np.asarray(values.tolist())
    return np.asarray(values, dtype=np.int64).reshape(-1)


def canonical_run(seed):
    matches = []
    with MATRIX.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["status"] == "COMPLETE" and "candidate" not in row["experiment_id"]
                and row["dataset"] == "CIFAR100" and row["backbone"] == "resnet18"
                and row["method"] == "LIDCover" and row["delta0"] == "0.25"
                and row["alpha"] == "1" and row["k"] == "50" and row["batch_size"] == "50"
                and int(row["seed"]) == seed and int(row["observed_final_budget"]) == 5050
            ):
                matches.append(Path(row["source_path"]))
    if len(matches) != 1:
        raise RuntimeError(f"strict canonical selection found {len(matches)} runs for seed {seed}")
    return matches[0]


def exact_knn(features, k=50):
    try:
        import faiss  # type: ignore
        index = faiss.IndexFlatL2(features.shape[1]); backend = "faiss_indexflatl2_cpu_exact"
        try:
            resources = faiss.StandardGpuResources(); index = faiss.index_cpu_to_gpu(resources, 0, index)
            backend = "faiss_indexflatl2_gpu_exact"
        except Exception:
            pass
        index.add(features); d2, idx = index.search(features, k + 1)
        return idx[:, 1:].astype(np.int32), np.sqrt(np.maximum(d2[:, 1:], 0)).astype(np.float32), backend
    except ImportError:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="brute", n_jobs=-1).fit(features)
        distance, idx = nn.kneighbors(features)
        return idx[:, 1:].astype(np.int32), distance[:, 1:].astype(np.float32), "sklearn_brute_exact"


def geometry():
    raw = np.load(FEATURES).astype(np.float32)
    features = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
    from skdim.id import MLE
    lids = np.asarray(MLE().fit_transform_pw(features, n_neighbors=50), dtype=np.float32)
    median = np.nanmedian(lids)
    lids = np.nan_to_num(lids, nan=median, posinf=median, neginf=median).astype(np.float32)
    idx, distance, backend = exact_knn(np.ascontiguousarray(features), 50)
    return features, lids, idx, distance, backend


def event_inputs(run, event):
    if event == 0:
        lset, uset = load_ids(run / "lSet.npy"), load_ids(run / "uSet.npy")
        summary = json.loads((run / "initial_sampling_summary.json").read_text())
        selected = np.asarray(summary["active_set_ids"], dtype=np.int64)
        effective = 0.25 * 1.35
    else:
        episode = event - 1
        directory = run / f"episode_{episode}"
        lset, uset = load_ids(directory / "lSet.npy"), load_ids(directory / "uSet.npy")
        summary = json.loads((directory / "episode_summary.json").read_text())
        selected = np.asarray(summary["active_set_ids"], dtype=np.int64)
        effective = float(summary.get("sampling_metadata", {}).get("effective_delta", 0.25 * 0.85))
    return lset, uset, selected, effective


def neighbourhood(global_x, relative_x, g2r, knn_idx, knn_distance, radius, explicit_self=True):
    relative = g2r[knn_idx[global_x]]; distance = knn_distance[global_x]
    neighbours = relative[(relative >= 0) & (distance < radius)].astype(np.int32)
    if explicit_self and not np.any(neighbours == relative_x):
        neighbours = np.concatenate([np.asarray([relative_x], dtype=np.int32), neighbours])
    return neighbours


def replay_event(seed, event, lset, uset, selected, effective, lids, knn_idx, knn_distance, labels):
    if len(selected) != 50:
        raise RuntimeError(f"seed {seed} event {event}: expected 50 ordered selections, found {len(selected)}")
    relevant = np.concatenate([lset, uset]).astype(np.int64)
    if len(np.unique(relevant)) != len(relevant) or np.intersect1d(lset, uset).size:
        raise RuntimeError("invalid labelled/unlabelled partition")
    g2r = -np.ones(len(lids), dtype=np.int32); g2r[relevant] = np.arange(len(relevant), dtype=np.int32)
    rel_lids = lids[relevant]; pool_median = float(np.median(rel_lids)); rho = (rel_lids + 1e-12) / (pool_median + 1e-12)
    radii = effective * rho ** -1.0
    bounds = np.quantile(rel_lids, [0.25, 0.5, 0.75])
    out = [neighbourhood(int(g), r, g2r, knn_idx, knn_distance, float(radii[r])) for r, g in enumerate(relevant)]
    incoming = [[] for _ in relevant]
    for source, neighbours in enumerate(out):
        for target in neighbours: incoming[int(target)].append(source)
    incoming = [np.asarray(values, dtype=np.int32) for values in incoming]
    covered = np.zeros(len(relevant), dtype=bool)
    for relative_x in range(len(lset)): covered[out[relative_x]] = True
    degree = np.asarray([np.sum(~covered[n]) for n in out], dtype=np.int32)
    selected_mask = np.zeros(len(relevant), dtype=bool)
    pool_labels = labels[relevant]; class_counts = np.bincount(pool_labels, minlength=int(labels.max()) + 1)
    rows = []
    for position, global_x in enumerate(selected):
        relative_x = int(g2r[global_x])
        if relative_x < len(lset): raise RuntimeError("saved selection was not unlabelled")
        candidate_degree = degree.copy(); candidate_degree[:len(lset)] = -1; candidate_degree[selected_mask] = -1
        best = int(candidate_degree.max()); ties = np.where(candidate_degree == best)[0]
        if best <= 0:
            pool = np.where((~selected_mask) & (np.arange(len(relevant)) >= len(lset)))[0]
            expected = int(pool[np.argmin(rel_lids[pool])]); fallback = True
        else:
            expected = int(ties[np.argmin(rel_lids[ties])]); fallback = False
        if relative_x != expected:
            raise RuntimeError(
                f"strict replay mismatch seed={seed} event={event} position={position + 1}: "
                f"saved={global_x}, expected={relevant[expected]}"
            )
        adaptive_neighbours = out[relative_x]
        fixed_neighbours = neighbourhood(int(global_x), relative_x, g2r, knn_idx, knn_distance, effective)
        adaptive_gain = int(np.sum(~covered[adaptive_neighbours])); fixed_gain = int(np.sum(~covered[fixed_neighbours]))
        neighbour_global = knn_idx[global_x]
        local_same = labels[neighbour_global] == labels[global_x]
        rows.append({
            "dataset": "CIFAR100", "backbone": "resnet18", "method": "LIDCover", "seed": seed,
            "episode": event, "labelled_budget": int(len(lset)), "sample_index": int(global_x),
            "position_in_query_batch": position + 1, "pointwise_lid": float(lids[global_x]),
            "pool_median_lid": pool_median, "relative_lid_rho": float(rho[relative_x]),
            "lid_quartile": int(np.searchsorted(bounds, lids[global_x], side="left") + 1),
            "adaptive_radius": float(radii[relative_x]), "effective_base_radius": effective,
            "radius_ratio": float(radii[relative_x] / effective),
            "actual_adaptive_neighbourhood_size": int(len(adaptive_neighbours)),
            "actual_uncovered_gain_immediately_before_selection": adaptive_gain,
            "number_newly_covered": adaptive_gain, "maximum_gain_tied": bool(best > 0 and len(ties) > 1),
            "tie_set_size": int(len(ties)), "selected_through_zero_gain_fallback": fallback,
            "fixed_radius_counterfactual_neighbourhood_size": int(len(fixed_neighbours)),
            "fixed_radius_counterfactual_uncovered_gain": fixed_gain,
            "adaptive_minus_fixed_counterfactual_gain": adaptive_gain - fixed_gain,
            "local_class_purity_k50_posthoc": float(local_same.mean()),
            "same_class_neighbour_fraction_k50_posthoc": float(local_same.mean()),
            "class_frequency_in_pool_posthoc": int(class_counts[int(labels[global_x])]),
            "label_use": "POST_HOC_DIAGNOSTIC_ONLY",
        })
        newly = adaptive_neighbours[~covered[adaptive_neighbours]]
        if len(newly):
            covered[newly] = True
            for target in newly:
                sources = incoming[int(target)]
                if len(sources): degree[sources] -= 1
        degree[relative_x] = 0; selected_mask[relative_x] = True
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, choices=range(1, 6), required=True)
    args = parser.parse_args(); seed = args.seed; run = canonical_run(seed)
    root_lset, root_uset, valset = load_ids(run / "lSet.npy"), load_ids(run / "uSet.npy"), load_ids(run / "valSet.npy")
    final = load_ids(run / "episode_100/lSet.npy")
    if len(root_lset) != 0 or len(final) != 5050 or np.intersect1d(root_uset, valset).size:
        raise RuntimeError("run failed strict cold-start/final-budget/split validation")
    features, lids, knn_idx, knn_distance, backend = geometry(); labels = np.load(LABELS).reshape(-1)
    if len(features) != len(labels) or len(features) != len(lids): raise RuntimeError("feature/label/geometry row mismatch")
    rows = []
    for event in range(101):
        lset, uset, selected, effective = event_inputs(run, event)
        rows.extend(replay_event(seed, event, lset, uset, selected, effective, lids, knn_idx, knn_distance, labels))
    STAGING.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(STAGING / f"mechanism_seed{seed}.csv", index=False)
    provenance = {
        "status": "STRICT_REPLAY_PASS", "seed": seed, "canonical_run": str(run),
        "feature_path": str(FEATURES), "feature_sha256": file_hash(FEATURES),
        "normalised_feature_content_sha256": array_hash(features),
        "candidate_index_sha256": array_hash(root_uset), "validation_index_sha256": array_hash(valset),
        "labels_path": str(LABELS), "labels_sha256": file_hash(LABELS), "labels_use": "POST_HOC_ONLY",
        "knn": "exact_nonself_k50", "knn_backend": backend, "historical_geometry_cache_used": False,
        "rows": len(rows),
    }
    (STAGING / f"mechanism_seed{seed}_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__": main()
