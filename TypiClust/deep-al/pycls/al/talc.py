"""Trajectory-Adaptive LID Cover (TALC).

TALC keeps point coverage as the primary acquisition objective.  Local
intrinsic dimension controls the radius curriculum, while multiscale H0
component representation and model uncertainty only rank candidates whose
coverage gain is close to the current maximum.

The implementation deliberately uses component partitions at reference
filtration scales instead of per-vertex persistence attribution.  The latter
depends on persistence tie-breaking and is therefore unsuitable as a
pointwise acquisition score.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

import pycls.datasets.utils as ds_utils

from .IDProbCover import compute_or_load_ids_mle, compute_or_load_knn


def _safe_mkdir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def _summary_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std()),
    }


def percentile_ranks(values):
    """Return average-tie percentile ranks in [0, 1]."""
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        stop = start + 1
        while stop < n and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return (ranks / float(n - 1)).astype(np.float32)


def trajectory_progress(coverage_fraction, coverage_target):
    if not 0.0 < float(coverage_target) <= 1.0:
        raise ValueError("coverage_target must lie in (0, 1].")
    return float(np.clip(float(coverage_fraction) / float(coverage_target), 0.0, 1.0))


def coverage_shortlist(gains, candidates, epsilon):
    """Return candidates within epsilon of the current maximum gain."""
    gains = np.asarray(gains)
    candidates = np.asarray(candidates, dtype=np.int32)
    if candidates.size == 0:
        return candidates, -1
    if not 0.0 <= float(epsilon) < 1.0:
        raise ValueError("coverage epsilon must lie in [0, 1).")

    best = int(np.max(gains[candidates]))
    if best <= 0:
        return candidates, best
    threshold = max(1, int(np.ceil((1.0 - float(epsilon)) * best)))
    return candidates[gains[candidates] >= threshold], best


class _UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, value):
        value = int(value)
        parent = self.parent
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def _undirected_edges(knn_idx, knn_dist):
    knn_idx = np.asarray(knn_idx, dtype=np.int32)
    knn_dist = np.asarray(knn_dist, dtype=np.float32)
    if knn_idx.shape != knn_dist.shape or knn_idx.ndim != 2:
        raise ValueError("kNN indices and distances must be equal-shaped matrices.")

    n, k = knn_idx.shape
    src = np.repeat(np.arange(n, dtype=np.int32), k)
    dst = knn_idx.reshape(-1)
    weight = knn_dist.reshape(-1)
    valid = (dst >= 0) & (dst < n) & (src != dst) & np.isfinite(weight)
    src, dst, weight = src[valid], dst[valid], weight[valid]
    if src.size == 0:
        return src, dst, weight

    left = np.minimum(src, dst)
    right = np.maximum(src, dst)
    keys = left.astype(np.int64) * np.int64(n) + right.astype(np.int64)
    order = np.lexsort((weight, keys))
    keys = keys[order]
    first = np.r_[True, keys[1:] != keys[:-1]]
    chosen = order[first]
    return left[chosen], right[chosen], weight[chosen]


def multiscale_component_partitions(knn_idx, knn_dist, quantiles):
    """Approximate H0 partitions at quantiles of sparse-graph MST deaths."""
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if quantiles.ndim != 1 or quantiles.size == 0:
        raise ValueError("At least one topology quantile is required.")
    if np.any((quantiles <= 0.0) | (quantiles >= 1.0)):
        raise ValueError("Topology quantiles must lie strictly between 0 and 1.")

    n = int(np.asarray(knn_idx).shape[0])
    left, right, weight = _undirected_edges(knn_idx, knn_dist)
    edge_order = np.argsort(weight, kind="mergesort")
    left, right, weight = left[edge_order], right[edge_order], weight[edge_order]

    mst = _UnionFind(n)
    deaths = []
    for edge_left, edge_right, edge_weight in zip(left, right, weight):
        if mst.union(edge_left, edge_right):
            deaths.append(float(edge_weight))
    if not deaths:
        thresholds = np.zeros(len(quantiles), dtype=np.float32)
    else:
        thresholds = np.quantile(np.asarray(deaths), quantiles).astype(np.float32)

    threshold_order = np.argsort(thresholds, kind="mergesort")
    labels_by_scale = np.empty((n, len(thresholds)), dtype=np.int32)
    graph = _UnionFind(n)
    edge_cursor = 0
    for threshold_index in threshold_order:
        threshold = float(thresholds[threshold_index])
        while edge_cursor < len(weight) and float(weight[edge_cursor]) <= threshold:
            graph.union(left[edge_cursor], right[edge_cursor])
            edge_cursor += 1
        roots = np.asarray([graph.find(index) for index in range(n)], dtype=np.int32)
        _, labels = np.unique(roots, return_inverse=True)
        labels_by_scale[:, threshold_index] = labels.astype(np.int32)
    return thresholds, labels_by_scale


def _component_cache_path(cache_root, dataset, seed, k_knn, quantiles):
    quantile_key = "_".join("{:03d}".format(int(round(1000.0 * q))) for q in quantiles)
    base = os.path.join(cache_root, dataset, "seed{}".format(seed))
    _safe_mkdir(base)
    return os.path.join(base, "talc_h0_k{}_q{}.npz".format(k_knn, quantile_key))


def compute_or_load_components(knn_idx, knn_dist, cache_path, quantiles):
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        labels = cached["labels"].astype(np.int32)
        thresholds = cached["thresholds"].astype(np.float32)
        cached_quantiles = cached["quantiles"].astype(np.float64)
        if labels.shape[0] == len(knn_idx) and np.array_equal(cached_quantiles, np.asarray(quantiles)):
            return thresholds, labels, True

    thresholds, labels = multiscale_component_partitions(knn_idx, knn_dist, quantiles)
    _safe_mkdir(os.path.dirname(cache_path))
    np.savez_compressed(
        cache_path,
        thresholds=thresholds,
        labels=labels,
        quantiles=np.asarray(quantiles, dtype=np.float64),
    )
    return thresholds, labels, False


@torch.no_grad()
def margin_uncertainty_scores(data_obj, cfg, model, dataset, indices):
    """Return high-is-uncertain margin scores aligned with ``indices``."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return np.zeros(0, dtype=np.float32)

    batch_size = max(1, int(cfg.TRAIN.BATCH_SIZE / max(int(cfg.NUM_GPUS), 1)))
    loader = data_obj.getSequentialDataLoader(indexes=indices, batch_size=batch_size, data=dataset)
    old_no_aug = getattr(loader.dataset, "no_aug", None)
    if hasattr(loader.dataset, "no_aug"):
        loader.dataset.no_aug = True

    was_training = bool(model.training)
    had_penultimate = hasattr(model, "penultimate_active")
    old_penultimate = getattr(model, "penultimate_active", False)
    try:
        original_device = next(model.parameters()).device
    except StopIteration:
        original_device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = original_device
    model.to(device)
    if had_penultimate:
        model.penultimate_active = False
    model.eval()
    try:
        scores = []
        for inputs, _ in loader:
            logits = model(inputs.to(device=device, dtype=torch.float32, non_blocking=True))
            if isinstance(logits, tuple):
                logits = logits[-1]
            probabilities = torch.softmax(logits, dim=1)
            if probabilities.shape[1] < 2:
                batch_scores = torch.zeros(probabilities.shape[0], device=probabilities.device)
            else:
                top_two = torch.topk(probabilities, k=2, dim=1).values
                batch_scores = 1.0 - (top_two[:, 0] - top_two[:, 1])
            scores.append(batch_scores.detach().cpu().numpy())
        return np.concatenate(scores).astype(np.float32, copy=False)
    finally:
        if had_penultimate:
            model.penultimate_active = old_penultimate
        model.train(was_training)
        if original_device != device:
            model.to(original_device)
        if old_no_aug is not None:
            loader.dataset.no_aug = old_no_aug


class TrajectoryAdaptiveLIDCover:
    """Coverage-first active learning with a structural curriculum."""

    def __init__(
        self,
        cfg,
        lSet,
        uSet,
        budgetSize,
        delta0,
        alpha_max=1.0,
        coverage_target=0.9,
        coverage_epsilon=0.05,
        radius_min_factor=0.5,
        radius_max_factor=2.0,
        topology_weight=0.5,
        topology_quantiles=(0.25, 0.5, 0.75),
        min_component_size=5,
        use_topology=True,
        use_uncertainty=True,
        cache_root="./talc_cache",
        k_id=50,
        k_knn=50,
        l2_normalize_features=True,
        prefer_faiss=True,
        faiss_gpu=True,
        add_self_cover=True,
        clf_model=None,
        train_dataset=None,
        data_obj=None,
    ):
        self.cfg = cfg
        self.ds_name = self.cfg["DATASET"]["NAME"]
        self.seed = int(self.cfg["RNG_SEED"])
        self.lSet = np.asarray(lSet, dtype=np.int64)
        self.uSet = np.asarray(uSet, dtype=np.int64)
        self.budgetSize = int(budgetSize)
        self.delta0 = float(delta0)
        self.alpha_max = float(alpha_max)
        self.coverage_target = float(coverage_target)
        self.coverage_epsilon = float(coverage_epsilon)
        self.radius_min_factor = float(radius_min_factor)
        self.radius_max_factor = float(radius_max_factor)
        self.topology_weight = float(topology_weight)
        self.topology_quantiles = tuple(float(q) for q in topology_quantiles)
        self.min_component_size = int(min_component_size)
        self.use_topology = bool(use_topology)
        self.use_uncertainty = bool(use_uncertainty) and len(self.lSet) > 0
        self.cache_root = cache_root or "./talc_cache"
        self.k_id = int(k_id)
        self.k_knn = int(k_knn)
        self.add_self_cover = bool(add_self_cover)

        self._validate_parameters()

        self.all_features = ds_utils.load_features(self.ds_name, self.seed).astype(np.float32)
        if l2_normalize_features:
            self.all_features /= np.linalg.norm(self.all_features, axis=1, keepdims=True) + 1e-12

        cache_base = os.path.join(self.cache_root, self.ds_name, "seed{}".format(self.seed))
        ids_path = os.path.join(cache_base, "mle_local_k{}.npy".format(self.k_id))
        knn_path = os.path.join(cache_base, "knn_k{}.npz".format(self.k_knn))
        self.ids_all, id_meta = compute_or_load_ids_mle(self.all_features, ids_path, k_id=self.k_id)
        self.knn_idx_all, self.knn_dist_all, knn_meta = compute_or_load_knn(
            self.all_features,
            knn_path,
            k_knn=self.k_knn,
            prefer_faiss=bool(prefer_faiss),
            faiss_gpu=bool(faiss_gpu),
        )

        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(np.int64)
        self.Nr = int(len(self.relevant_indices))
        self.L = int(len(self.lSet))
        self.g2r = -np.ones(len(self.all_features), dtype=np.int32)
        self.g2r[self.relevant_indices] = np.arange(self.Nr, dtype=np.int32)
        self.rel_ids = self.ids_all[self.relevant_indices].astype(np.float32)
        self.id_preference = 1.0 - percentile_ranks(self.rel_ids)

        base_neighbors, _ = self._build_adjacency(np.full(self.Nr, self.delta0, dtype=np.float32))
        base_covered = self._covered_by_labelled(base_neighbors)
        self.base_coverage_before = float(base_covered.mean()) if self.Nr else 0.0
        self.progress = trajectory_progress(self.base_coverage_before, self.coverage_target)
        self.alpha_t = self.alpha_max * (1.0 - self.progress)
        self.delta_per_x = self._dynamic_radii()
        self.out_neighbors, self.in_sources = self._build_adjacency(self.delta_per_x)

        self.component_thresholds = np.zeros(0, dtype=np.float32)
        self.rel_component_labels = np.empty((self.Nr, 0), dtype=np.int32)
        self.component_sizes = []
        component_cache_hit = False
        component_cache = ""
        if self.use_topology and self.Nr:
            component_cache = _component_cache_path(
                self.cache_root, self.ds_name, self.seed, self.k_knn, self.topology_quantiles
            )
            thresholds, labels_all, component_cache_hit = compute_or_load_components(
                self.knn_idx_all,
                self.knn_dist_all,
                component_cache,
                self.topology_quantiles,
            )
            self.component_thresholds = thresholds
            self.rel_component_labels = labels_all[self.relevant_indices]
            self.component_sizes = [
                np.bincount(labels_all[:, scale]).astype(np.int64)
                for scale in range(labels_all.shape[1])
            ]

        self.rel_uncertainty = np.zeros(self.Nr, dtype=np.float32)
        if self.use_uncertainty:
            if clf_model is None or train_dataset is None or data_obj is None:
                raise ValueError("TALC uncertainty requires clf_model, train_dataset, and data_obj.")
            raw_uncertainty = margin_uncertainty_scores(
                data_obj, cfg, clf_model, train_dataset, self.uSet
            )
            self.rel_uncertainty[self.L:] = percentile_ranks(raw_uncertainty)

        self.selection_metadata = {
            "strategy": "trajectory_adaptive_lid_cover",
            "selection_mode": "talc",
            "effective_delta": float(self.delta0),
            "delta_phase": "trajectory_adaptive",
            "alpha_max": float(self.alpha_max),
            "alpha": float(self.alpha_t),
            "trajectory_progress": float(self.progress),
            "base_coverage_fraction_before": float(self.base_coverage_before),
            "coverage_target": float(self.coverage_target),
            "coverage_epsilon": float(self.coverage_epsilon),
            "topology_active": bool(self.use_topology),
            "uncertainty_active": bool(self.use_uncertainty),
            "uncertainty_mode": "margin" if self.use_uncertainty else "disabled",
            "topology_weight": float(self.topology_weight),
            "topology_quantiles": list(self.topology_quantiles),
            "topology_thresholds": self.component_thresholds.tolist(),
            "topology_component_cache_hit": bool(component_cache_hit),
            "topology_component_cache_path": component_cache,
            "k_id": int(self.k_id),
            "k_knn": int(self.k_knn),
            "cache_root": self.cache_root,
            "id_cache_hit": bool(id_meta["cache_hit"]),
            "knn_cache_hit": bool(knn_meta["cache_hit"]),
        }

    def _validate_parameters(self):
        if self.budgetSize < 0:
            raise ValueError("budgetSize must be non-negative.")
        if self.delta0 <= 0.0:
            raise ValueError("delta0 must be positive.")
        if self.alpha_max < 0.0:
            raise ValueError("alpha_max must be non-negative.")
        if not 0.0 <= self.coverage_epsilon < 1.0:
            raise ValueError("coverage_epsilon must lie in [0, 1).")
        if not 0.0 <= self.topology_weight <= 1.0:
            raise ValueError("topology_weight must lie in [0, 1].")
        if not 0.0 < self.radius_min_factor <= self.radius_max_factor:
            raise ValueError("radius factors must satisfy 0 < min <= max.")
        trajectory_progress(0.0, self.coverage_target)

    def _dynamic_radii(self):
        median_id = float(np.median(self.rel_ids)) if self.Nr else 1.0
        ratio = (self.rel_ids + 1e-12) / (median_id + 1e-12)
        radii = self.delta0 * np.power(ratio, -self.alpha_t)
        return np.clip(
            radii,
            self.delta0 * self.radius_min_factor,
            self.delta0 * self.radius_max_factor,
        ).astype(np.float32)

    def _build_adjacency(self, radii):
        out_neighbors = [None] * self.Nr
        in_sources = [[] for _ in range(self.Nr)]
        for rel_x, global_x in enumerate(self.relevant_indices):
            neigh_r = self.g2r[self.knn_idx_all[global_x]]
            neigh_d = self.knn_dist_all[global_x]
            mask = (neigh_r >= 0) & (neigh_d < float(radii[rel_x]))
            neighbors = neigh_r[mask].astype(np.int32)
            if self.add_self_cover and (neighbors.size == 0 or not np.any(neighbors == rel_x)):
                neighbors = np.concatenate([np.asarray([rel_x], dtype=np.int32), neighbors])
            neighbors = np.unique(neighbors).astype(np.int32)
            out_neighbors[rel_x] = neighbors
            for rel_y in neighbors:
                in_sources[int(rel_y)].append(rel_x)
        return out_neighbors, [np.asarray(values, dtype=np.int32) for values in in_sources]

    def _covered_by_labelled(self, neighbors):
        covered = np.zeros(self.Nr, dtype=bool)
        for rel_x in range(self.L):
            covered[neighbors[rel_x]] = True
        return covered

    def _initial_component_counts(self):
        counts = []
        for scale in range(self.rel_component_labels.shape[1]):
            num_components = len(self.component_sizes[scale])
            scale_counts = np.zeros(num_components, dtype=np.int32)
            if self.L:
                np.add.at(scale_counts, self.rel_component_labels[:self.L, scale], 1)
            counts.append(scale_counts)
        return counts

    def _topology_scores(self, candidates, component_counts):
        candidates = np.asarray(candidates, dtype=np.int32)
        scores = np.zeros(len(candidates), dtype=np.float64)
        if not self.use_topology or self.rel_component_labels.shape[1] == 0:
            return scores.astype(np.float32)
        for scale, counts in enumerate(component_counts):
            labels = self.rel_component_labels[candidates, scale]
            sizes = self.component_sizes[scale][labels]
            valid = sizes >= self.min_component_size
            scores[valid] += np.log1p(sizes[valid]) / (1.0 + counts[labels[valid]])
        if scores.size and scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores.fill(0.0)
        return scores.astype(np.float32)

    def _select_candidate(self, candidates, component_counts):
        candidates = np.asarray(candidates, dtype=np.int32)
        topo = self._topology_scores(candidates, component_counts)
        uncertainty = self.rel_uncertainty[candidates]
        late_score = self.topology_weight * topo + (1.0 - self.topology_weight) * uncertainty
        score = (1.0 - self.progress) * self.id_preference[candidates] + self.progress * late_score

        # Highest curriculum score, then lower ID, then original pool order.
        ordering = np.lexsort((candidates, self.rel_ids[candidates], -score))
        chosen_pos = int(ordering[0])
        return int(candidates[chosen_pos]), float(score[chosen_pos]), float(topo[chosen_pos])

    def select_samples(self):
        if self.budgetSize == 0 or len(self.uSet) == 0:
            self.selection_metadata.update({"selected_count": 0, "stopping_reason": "empty_budget_or_pool"})
            return np.empty(0, dtype=np.int64), self.uSet.copy()

        covered = self._covered_by_labelled(self.out_neighbors)
        coverage_before = covered.copy()
        current_gain = np.asarray(
            [np.count_nonzero(~covered[neighbors]) for neighbors in self.out_neighbors],
            dtype=np.int32,
        )
        selected_mask = np.zeros(self.Nr, dtype=bool)
        component_counts = self._initial_component_counts()
        selected_rel = []
        selected_gains = []
        max_gains = []
        coverage_ratios = []
        curriculum_scores = []
        topology_scores = []

        stopping_reason = "budget_exhausted"
        for _ in range(min(self.budgetSize, len(self.uSet))):
            candidates = np.where((np.arange(self.Nr) >= self.L) & (~selected_mask))[0].astype(np.int32)
            if candidates.size == 0:
                stopping_reason = "pool_exhausted"
                break
            shortlist, best_gain = coverage_shortlist(
                current_gain, candidates, self.coverage_epsilon
            )
            rel_x, curriculum_score, topology_score = self._select_candidate(
                shortlist, component_counts
            )
            selected_gain = int(current_gain[rel_x])
            selected_rel.append(rel_x)
            selected_gains.append(selected_gain)
            max_gains.append(best_gain)
            coverage_ratios.append(
                float(selected_gain) / float(best_gain) if best_gain > 0 else 1.0
            )
            curriculum_scores.append(curriculum_score)
            topology_scores.append(topology_score)
            selected_mask[rel_x] = True

            newly = self.out_neighbors[rel_x]
            newly = newly[~covered[newly]]
            if newly.size:
                covered[newly] = True
                for rel_y in newly:
                    sources = self.in_sources[int(rel_y)]
                    if sources.size:
                        current_gain[sources] -= 1
            current_gain[rel_x] = 0

            for scale, counts in enumerate(component_counts):
                counts[self.rel_component_labels[rel_x, scale]] += 1

        selected_rel = np.asarray(selected_rel, dtype=np.int32)
        active_set = self.relevant_indices[selected_rel]
        selected_lookup = set(int(index) for index in active_set)
        remain_set = np.asarray(
            [index for index in self.uSet if int(index) not in selected_lookup], dtype=np.int64
        )

        gain_stats = _summary_stats(selected_gains)
        ratio_stats = _summary_stats(coverage_ratios)
        selected_ids = self.rel_ids[selected_rel] if selected_rel.size else np.zeros(0)
        selected_radii = self.delta_per_x[selected_rel] if selected_rel.size else np.zeros(0)
        self.selection_metadata.update({
            "coverage_fraction_before": float(coverage_before.mean()) if self.Nr else 0.0,
            "coverage_fraction_after": float(covered.mean()) if self.Nr else 0.0,
            "selected_count": int(len(active_set)),
            "returned_fewer_than_budget": bool(len(active_set) < self.budgetSize),
            "stopping_reason": stopping_reason,
            "selected_id_mean": _summary_stats(selected_ids)["mean"],
            "selected_radius_mean": _summary_stats(selected_radii)["mean"],
            "selected_coverage_gain_mean": gain_stats["mean"],
            "selected_coverage_gain_min": gain_stats["min"],
            "selected_coverage_gain_max": gain_stats["max"],
            "coverage_safeguard_ratio_mean": ratio_stats["mean"],
            "coverage_safeguard_ratio_min": ratio_stats["min"],
            "selected_curriculum_score_mean": _summary_stats(curriculum_scores)["mean"],
            "selected_topology_score_mean": _summary_stats(topology_scores)["mean"],
            "selected_ids_sha256": hashlib.sha256(
                np.ascontiguousarray(active_set).view(np.uint8)
            ).hexdigest(),
        })
        print(
            "TALC selected {} samples. progress={:.3f} alpha_t={:.3f} "
            "coverage={:.4f}->{:.4f} safeguard_min={:.3f}".format(
                len(active_set),
                self.progress,
                self.alpha_t,
                self.selection_metadata["coverage_fraction_before"],
                self.selection_metadata["coverage_fraction_after"],
                self.selection_metadata["coverage_safeguard_ratio_min"],
            )
        )
        return active_set, remain_set


TALC = TrajectoryAdaptiveLIDCover
