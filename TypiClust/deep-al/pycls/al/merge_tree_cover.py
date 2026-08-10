"""Residual H0 merge-tree active learning.

Unlike the static checklist version, merges are rebuilt each greedy pick on the
still-uncovered residual of the pool (points outside δ-balls of L ∪ selected).
That keeps topological signal alive as labeling progresses, instead of covering
a fixed merge list once and collapsing to ProbCover-style point covering.

Defaults:
- point_lambda = 0 (pure merge-event gain)
- ProbCover fallback = False (point-only residual covering if merges vanish)
"""

from __future__ import annotations

import os

import numpy as np

import pycls.datasets.utils as ds_utils

from .IDProbCover import _safe_mkdir, compute_or_load_knn


def _summary_stats(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std()),
    }


def _default_cache_paths(cache_root, dataset, seed, k_knn):
    base = os.path.join(cache_root, dataset, "seed{}".format(seed))
    edge_path = os.path.join(base, "merge_tree_edges_k{}.npz".format(k_knn))
    return edge_path


def _compute_or_load_sorted_edges(knn_idx, knn_dist, edge_cache_path):
    _safe_mkdir(os.path.dirname(edge_cache_path))
    if os.path.exists(edge_cache_path):
        cached = np.load(edge_cache_path)
        return (
            cached["src"].astype(np.int32),
            cached["dst"].astype(np.int32),
            cached["dist"].astype(np.float32),
            {"cache_hit": True, "cache_path": edge_cache_path},
        )

    n, k = knn_idx.shape
    src = np.repeat(np.arange(n, dtype=np.int32), k)
    dst = knn_idx.reshape(-1).astype(np.int32)
    dist = knn_dist.reshape(-1).astype(np.float32)
    mask = src != dst
    src, dst, dist = src[mask], dst[mask], dist[mask]

    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    pair = (lo.astype(np.int64) << 32) | hi.astype(np.int64)
    order = np.lexsort((dist, pair))
    pair, lo, hi, dist = pair[order], lo[order], hi[order], dist[order]
    keep = np.ones(pair.shape[0], dtype=bool)
    if pair.shape[0] > 1:
        keep[1:] = pair[1:] != pair[:-1]
    src, dst, dist = lo[keep], hi[keep], dist[keep]
    np.savez_compressed(edge_cache_path, src=src, dst=dst, dist=dist)
    return src, dst, dist, {"cache_hit": False, "cache_path": edge_cache_path}


class _UnionFind(object):
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int32)

    def find(self, x):
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra, rb, ra, False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return ra, rb, ra, True


class MergeTreeCover(object):
    """Residual merge-tree covering on currently uncovered embedding mass."""

    def __init__(
        self,
        cfg,
        lSet,
        uSet,
        budgetSize,
        delta0,
        cache_root="./merge_tree_cover_cache",
        k_knn=50,
        l2_normalize_features=True,
        prefer_faiss=True,
        faiss_gpu=True,
        min_component_size=5,
        max_merge_events=2000,
        events_per_budget=25,
        rep_k=10,
        radius_multiplier=1.0,
        radius_min=0.05,
        radius_max=None,
        scale_exp=1.0,
        size_exp=0.5,
        balance_exp=1.0,
        point_lambda=0.0,
        use_probcover_fallback=False,
        rebuild_every_pick=True,
        eps=1e-8,
    ):
        self.cfg = cfg
        self.ds_name = self.cfg["DATASET"]["NAME"]
        self.seed = self.cfg["RNG_SEED"]
        self.lSet = np.asarray(lSet, dtype=np.int64)
        self.uSet = np.asarray(uSet, dtype=np.int64)
        self.budgetSize = int(budgetSize)
        self.delta0 = float(delta0)
        self.cache_root = str(cache_root)
        self.k_knn = int(k_knn)
        self.l2_normalize_features = bool(l2_normalize_features)
        self.prefer_faiss = bool(prefer_faiss)
        self.faiss_gpu = bool(faiss_gpu)
        self.min_component_size = int(min_component_size)
        self.max_merge_events = int(max_merge_events)
        self.events_per_budget = int(events_per_budget)
        self.rep_k = int(rep_k)
        self.radius_multiplier = float(radius_multiplier)
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max) if radius_max is not None else float(delta0)
        self.scale_exp = float(scale_exp)
        self.size_exp = float(size_exp)
        self.balance_exp = float(balance_exp)
        self.point_lambda = float(point_lambda)
        self.use_probcover_fallback = bool(use_probcover_fallback)
        self.rebuild_every_pick = bool(rebuild_every_pick)
        self.eps = float(eps)

        self.selection_metadata = {
            "strategy": "merge_tree_cover",
            "selection_mode": "merge_tree_cover_residual",
            "effective_delta": float(self.delta0),
            "cache_root": self.cache_root,
            "k_knn": self.k_knn,
            "min_component_size": self.min_component_size,
            "max_merge_events": self.max_merge_events,
            "events_per_budget": self.events_per_budget,
            "radius_multiplier": self.radius_multiplier,
            "radius_min": self.radius_min,
            "radius_max": self.radius_max,
            "scale_exp": self.scale_exp,
            "size_exp": self.size_exp,
            "balance_exp": self.balance_exp,
            "point_lambda": self.point_lambda,
            "use_probcover_fallback": self.use_probcover_fallback,
            "rebuild_every_pick": self.rebuild_every_pick,
        }

        self.all_features = ds_utils.load_features(self.ds_name, self.seed).astype(np.float32)
        if self.l2_normalize_features:
            self.all_features /= (
                np.linalg.norm(self.all_features, axis=1, keepdims=True) + self.eps
            )

        edge_cache_path = _default_cache_paths(
            self.cache_root, self.ds_name, self.seed, self.k_knn
        )
        knn_cache_path = os.path.join(
            self.cache_root,
            self.ds_name,
            "seed{}".format(self.seed),
            "knn_k{}.npz".format(self.k_knn),
        )
        self.knn_idx_all, self.knn_dist_all, knn_cache_meta = compute_or_load_knn(
            self.all_features,
            knn_cache_path,
            k_knn=self.k_knn,
            prefer_faiss=self.prefer_faiss,
            faiss_gpu=self.faiss_gpu,
        )
        (
            self.edge_src_all,
            self.edge_dst_all,
            self.edge_dist_all,
            edge_cache_meta,
        ) = _compute_or_load_sorted_edges(
            self.knn_idx_all, self.knn_dist_all, edge_cache_path
        )
        self.selection_metadata.update({
            "knn_cache_hit": bool(knn_cache_meta["cache_hit"]),
            "knn_cache_path": knn_cache_meta["cache_path"],
            "knn_backend": knn_cache_meta["backend"],
            "edge_cache_hit": bool(edge_cache_meta["cache_hit"]),
            "edge_cache_path": edge_cache_meta["cache_path"],
        })

        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(np.int64)
        self.Nr = len(self.relevant_indices)
        self.g2r = -np.ones(self.all_features.shape[0], dtype=np.int32)
        self.g2r[self.relevant_indices] = np.arange(self.Nr, dtype=np.int32)
        self.u_set_mask = np.zeros(self.all_features.shape[0], dtype=bool)
        self.u_set_mask[self.uSet] = True

        rep_k_eff = max(1, min(self.rep_k, self.knn_dist_all.shape[1]))
        self.point_centrality = (
            self.knn_dist_all[self.relevant_indices, :rep_k_eff]
            .mean(axis=1)
            .astype(np.float32)
        )

    def _merge_importance(self, scale, size_a, size_b):
        min_size = float(min(size_a, size_b))
        max_size = float(max(size_a, size_b))
        balance = min_size / max(max_size, 1.0)
        return (
            float(scale + self.eps) ** self.scale_exp
            * float(max(min_size, 1.0)) ** self.size_exp
            * float(max(balance, self.eps)) ** self.balance_exp
        )

    def _event_radius(self, scale):
        return float(
            np.clip(self.radius_multiplier * float(scale), self.radius_min, self.radius_max)
        )

    def _point_ball(self, global_idx):
        neigh_g = self.knn_idx_all[int(global_idx)]
        neigh_d = self.knn_dist_all[int(global_idx)]
        within = neigh_g[neigh_d <= self.delta0]
        ball = {int(global_idx)}
        for neigh in within.tolist():
            neigh = int(neigh)
            if self.g2r[neigh] >= 0:
                ball.add(neigh)
        return ball

    def _build_residual_events(self, residual_rel):
        """Kruskal merges restricted to currently uncovered residual points."""
        residual_rel = np.asarray(residual_rel, dtype=np.int32)
        if residual_rel.size < 2:
            return []

        # Local ids on residual subset.
        n_res = int(residual_rel.size)
        rel_to_local = -np.ones(self.Nr, dtype=np.int32)
        rel_to_local[residual_rel] = np.arange(n_res, dtype=np.int32)
        residual_global = self.relevant_indices[residual_rel]
        local_centrality = self.point_centrality[residual_rel]

        # Edges with both ends in residual, in global sorted order.
        rel_src = self.g2r[self.edge_src_all]
        rel_dst = self.g2r[self.edge_dst_all]
        mask = (rel_src >= 0) & (rel_dst >= 0)
        rel_src = rel_src[mask]
        rel_dst = rel_dst[mask]
        rel_dist = self.edge_dist_all[mask]
        local_src = rel_to_local[rel_src]
        local_dst = rel_to_local[rel_dst]
        keep = (local_src >= 0) & (local_dst >= 0)
        local_src = local_src[keep].astype(np.int32)
        local_dst = local_dst[keep].astype(np.int32)
        local_dist = rel_dist[keep].astype(np.float32)

        uf = _UnionFind(n_res)
        comp_rep = np.arange(n_res, dtype=np.int32)
        events = []
        for a, b, dist in zip(local_src, local_dst, local_dist):
            ra, rb = uf.find(int(a)), uf.find(int(b))
            if ra == rb:
                continue
            size_a, size_b = int(uf.size[ra]), int(uf.size[rb])
            rep_a, rep_b = int(comp_rep[ra]), int(comp_rep[rb])
            merged_rep = min(
                [rep_a, rep_b, int(a), int(b)],
                key=lambda idx: float(local_centrality[idx]),
            )
            events.append({
                "scale": float(dist),
                "size_a": size_a,
                "size_b": size_b,
                "min_size": int(min(size_a, size_b)),
                "importance": float(self._merge_importance(float(dist), size_a, size_b)),
                # Store global anchors for covering.
                "anchors_global": np.unique(
                    np.asarray(
                        [
                            residual_global[int(a)],
                            residual_global[int(b)],
                            residual_global[rep_a],
                            residual_global[rep_b],
                            residual_global[merged_rep],
                        ],
                        dtype=np.int64,
                    )
                ),
            })
            _n, _o, keep_root, merged = uf.union(ra, rb)
            if merged:
                comp_rep[keep_root] = merged_rep

        return events

    def _select_important_events(self, events):
        if not events:
            return []
        target = max(self.budgetSize * self.events_per_budget, self.budgetSize)
        target = min(target, self.max_merge_events)
        filtered = [e for e in events if e["min_size"] >= self.min_component_size]
        if not filtered:
            filtered = list(events)
        filtered.sort(key=lambda e: e["importance"], reverse=True)
        return filtered[:target]

    def _event_coverers(self, anchors_global, radius, eligible_mask):
        """eligible_mask: currently selectable unlabeled residual points."""
        cover_all = set()
        cover_eligible = set()
        for anchor in anchors_global:
            anchor = int(anchor)
            cover_all.add(anchor)
            if eligible_mask[anchor]:
                cover_eligible.add(anchor)
            neigh_g = self.knn_idx_all[anchor]
            neigh_d = self.knn_dist_all[anchor]
            for neigh in neigh_g[neigh_d <= radius].tolist():
                neigh = int(neigh)
                if self.g2r[neigh] < 0:
                    continue
                cover_all.add(neigh)
                if eligible_mask[neigh]:
                    cover_eligible.add(neigh)

        if not cover_eligible:
            best, best_d = None, None
            for anchor in anchors_global:
                neigh_g = self.knn_idx_all[int(anchor)]
                neigh_d = self.knn_dist_all[int(anchor)]
                for neigh, dist in zip(neigh_g.tolist(), neigh_d.tolist()):
                    neigh = int(neigh)
                    if self.g2r[neigh] < 0 or not eligible_mask[neigh]:
                        continue
                    if best_d is None or dist < best_d:
                        best_d = float(dist)
                        best = neigh
            if best is not None:
                cover_eligible.add(int(best))
                cover_all.add(int(best))
        return cover_all, cover_eligible

    def _point_only_pick(self, selected_set, point_covered):
        best, best_gain, best_cent = None, -1, None
        for cand in self.uSet:
            cand = int(cand)
            if cand in selected_set or point_covered[cand]:
                continue
            ball = self._point_ball(cand)
            gain = int(sum(1 for p in ball if not point_covered[p]))
            rel = int(self.g2r[cand])
            cent = float(self.point_centrality[rel]) if rel >= 0 else float("inf")
            if gain > best_gain or (
                gain == best_gain and (best_cent is None or cent < best_cent)
            ):
                best, best_gain, best_cent = cand, gain, cent
        return best, best_gain

    def _probcover_fill(self, selected, remaining):
        if remaining <= 0 or not self.use_probcover_fallback:
            return selected, 0
        from .prob_cover import ProbCover

        selected_set = set(map(int, selected))
        current_l = np.concatenate(
            [self.lSet, np.asarray(selected, dtype=np.int64)]
        ).astype(np.int64)
        current_u = np.asarray(
            [i for i in self.uSet if int(i) not in selected_set], dtype=np.int64
        )
        if current_u.size == 0:
            return selected, 0
        print("[MergeTreeCover] ProbCover fallback remaining={}".format(remaining))
        more, _ = ProbCover(
            self.cfg, current_l, current_u, budgetSize=remaining, delta=self.delta0
        ).select_samples()
        n_add = 0
        for idx in map(int, more):
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
                n_add += 1
        self.selection_metadata["fallback"] = "probcover"
        return selected, n_add

    def select_samples(self):
        point_covered = np.zeros(self.all_features.shape[0], dtype=bool)
        for lab in self.lSet:
            for p in self._point_ball(int(lab)):
                point_covered[p] = True
        residual_before = float((~point_covered[self.relevant_indices]).mean()) if self.Nr else 0.0

        selected = []
        selected_set = set()
        n_merge_tree_selected = 0
        n_point_only_selected = 0
        n_probcover_fallback = 0
        selected_event_gains = []
        selected_point_gains = []
        selected_scores = []
        selected_scales = []
        selected_n_events = []
        active_event_counts = []

        cached_events = None

        for it in range(self.budgetSize):
            # Residual = not yet δ-covered by L ∪ batch-so-far.
            residual_rel = np.asarray(
                [
                    rel
                    for rel, g in enumerate(self.relevant_indices)
                    if not point_covered[int(g)]
                ],
                dtype=np.int32,
            )
            eligible_mask = (~point_covered) & self.u_set_mask
            for s in selected_set:
                eligible_mask[int(s)] = False

            if residual_rel.size < max(2, self.min_component_size):
                # No residual topology left; fill without ProbCover by default.
                remaining = self.budgetSize - len(selected)
                if self.use_probcover_fallback:
                    selected, n_add = self._probcover_fill(selected, remaining)
                    n_probcover_fallback += n_add
                else:
                    for _ in range(remaining):
                        cand, gain = self._point_only_pick(selected_set, point_covered)
                        if cand is None or gain <= 0:
                            break
                        selected.append(cand)
                        selected_set.add(cand)
                        n_point_only_selected += 1
                        for p in self._point_ball(cand):
                            point_covered[p] = True
                break

            if self.rebuild_every_pick or cached_events is None:
                events = self._build_residual_events(residual_rel)
                important = self._select_important_events(events)
                cached_events = important
            else:
                important = cached_events

            active_event_counts.append(int(len(important)))
            if not important:
                cand, gain = self._point_only_pick(selected_set, point_covered)
                if cand is None or gain <= 0:
                    remaining = self.budgetSize - len(selected)
                    if self.use_probcover_fallback:
                        selected, n_add = self._probcover_fill(selected, remaining)
                        n_probcover_fallback += n_add
                    break
                selected.append(cand)
                selected_set.add(cand)
                n_point_only_selected += 1
                for p in self._point_ball(cand):
                    point_covered[p] = True
                selected_event_gains.append(0.0)
                selected_point_gains.append(float(gain))
                selected_scores.append(float(self.point_lambda * gain))
                selected_scales.append(0.0)
                selected_n_events.append(0)
                print(
                    "[MergeTreeCover] it={:03d} mode=point_only residual={} "
                    "pick={} point_gain={}".format(
                        it, int(residual_rel.size), cand, gain
                    )
                )
                continue

            # Build candidate -> events map for residual active events.
            candidate_to_events = {}
            event_weights = []
            event_scales = []
            for eid, event in enumerate(important):
                radius = self._event_radius(event["scale"])
                _cover_all, cover_eligible = self._event_coverers(
                    event["anchors_global"], radius, eligible_mask
                )
                event_weights.append(float(event["importance"]))
                event_scales.append(float(event["scale"]))
                for cand in cover_eligible:
                    candidate_to_events.setdefault(int(cand), []).append(eid)

            if not candidate_to_events:
                cand, gain = self._point_only_pick(selected_set, point_covered)
                if cand is None or gain <= 0:
                    break
                selected.append(cand)
                selected_set.add(cand)
                n_point_only_selected += 1
                for p in self._point_ball(cand):
                    point_covered[p] = True
                selected_event_gains.append(0.0)
                selected_point_gains.append(float(gain))
                selected_scores.append(float(self.point_lambda * gain))
                selected_scales.append(0.0)
                selected_n_events.append(0)
                continue

            weights = np.asarray(event_weights, dtype=np.float32)
            scales = np.asarray(event_scales, dtype=np.float32)
            best_cand = None
            best_score = -1.0
            best_event_gain = -1.0
            best_point_gain = -1
            best_cent = None
            best_eids = None

            for cand, eids in candidate_to_events.items():
                eids = np.asarray(eids, dtype=np.int32)
                event_gain = float(weights[eids].sum())
                ball = self._point_ball(cand)
                point_gain = int(sum(1 for p in ball if not point_covered[p]))
                score = event_gain + self.point_lambda * float(point_gain)
                rel = int(self.g2r[cand])
                cent = float(self.point_centrality[rel]) if rel >= 0 else float("inf")
                if score > best_score or (
                    score == best_score and (best_cent is None or cent < best_cent)
                ):
                    best_cand = int(cand)
                    best_score = score
                    best_event_gain = event_gain
                    best_point_gain = point_gain
                    best_cent = cent
                    best_eids = eids

            if best_cand is None or best_score <= 0:
                remaining = self.budgetSize - len(selected)
                if self.use_probcover_fallback:
                    selected, n_add = self._probcover_fill(selected, remaining)
                    n_probcover_fallback += n_add
                break

            selected.append(best_cand)
            selected_set.add(best_cand)
            if best_event_gain > 0:
                n_merge_tree_selected += 1
                mode = "merge_tree"
            else:
                n_point_only_selected += 1
                mode = "point_only"

            for p in self._point_ball(best_cand):
                point_covered[p] = True

            selected_event_gains.append(float(best_event_gain))
            selected_point_gains.append(float(best_point_gain))
            selected_scores.append(float(best_score))
            selected_scales.append(
                float(scales[best_eids].mean()) if best_eids is not None and best_eids.size else 0.0
            )
            selected_n_events.append(int(best_eids.size) if best_eids is not None else 0)
            # Force rebuild next pick so residual topology updates.
            cached_events = None
            print(
                "[MergeTreeCover] it={:03d} mode={} residual={} active_events={} "
                "pick={} score={:.4f} event_gain={:.4f} point_gain={} n_events={}".format(
                    it,
                    mode,
                    int(residual_rel.size),
                    int(len(important)),
                    best_cand,
                    best_score,
                    best_event_gain,
                    best_point_gain,
                    int(best_eids.size) if best_eids is not None else 0,
                )
            )

        active_set = np.asarray(selected, dtype=np.int64)
        remain_set = np.array(
            sorted(list(set(map(int, self.uSet)) - set(map(int, active_set)))),
            dtype=np.int64,
        )
        residual_after = float((~point_covered[self.relevant_indices]).mean()) if self.Nr else 0.0

        self.selection_metadata.update({
            "coverage_fraction_before": float(1.0 - residual_before),
            "coverage_fraction_after": float(1.0 - residual_after),
            "residual_fraction_before": float(residual_before),
            "residual_fraction_after": float(residual_after),
            "selected_count": int(len(active_set)),
            "n_merge_tree_selected": int(n_merge_tree_selected),
            "n_point_only_selected": int(n_point_only_selected),
            "n_probcover_fallback": int(n_probcover_fallback),
            "used_fallback": bool(n_probcover_fallback > 0),
            "used_point_only_tail": bool(n_point_only_selected > 0),
            "active_events_mean": _summary_stats(active_event_counts)["mean"],
            "selected_coverage_gain_mean": _summary_stats(selected_event_gains)["mean"],
            "selected_point_gain_mean": _summary_stats(selected_point_gains)["mean"],
            "selected_score_mean": _summary_stats(selected_scores)["mean"],
            "selected_merge_scale_mean": _summary_stats(selected_scales)["mean"],
            "selected_merge_scale_min": _summary_stats(selected_scales)["min"],
            "selected_merge_scale_max": _summary_stats(selected_scales)["max"],
            "selected_merge_scale_std": _summary_stats(selected_scales)["std"],
            "selected_events_per_pick_mean": _summary_stats(selected_n_events)["mean"],
        })
        print(
            "MergeTreeCover residual selected {} samples. "
            "residual {:.4f}->{:.4f} counts merge_tree={} point_only={} probcover={}".format(
                len(active_set),
                residual_before,
                residual_after,
                n_merge_tree_selected,
                n_point_only_selected,
                n_probcover_fallback,
            )
        )
        return active_set, remain_set
