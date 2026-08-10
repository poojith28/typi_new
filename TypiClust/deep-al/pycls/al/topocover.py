import os
from collections import deque

import numpy as np

import pycls.datasets.utils as ds_utils

from .adaptive_cover.common import compute_or_load_knn, safe_mkdir, summary_stats


def _cache_paths(cache_root, dataset, seed, k_knn):
    base = os.path.join(cache_root, dataset, "seed{}".format(seed))
    safe_mkdir(base)
    return os.path.join(base, "knn_k{}.npz".format(k_knn))


class TopoCover:
    """Topology-guided component coverage in a fixed representation graph.

    The method builds a sparse kNN approximation of the fixed-radius graph,
    marks vertices covered by the current labeled set, computes connected
    components in the uncovered subgraph, and greedily selects candidates whose
    neighborhoods touch the largest number of uncovered components.
    """

    def __init__(
        self,
        cfg,
        lSet,
        uSet,
        budgetSize,
        delta,
        cache_root="./topocover_cache",
        k_knn=50,
        eps=1e-8,
        l2_normalize_features=True,
        prefer_faiss=True,
        faiss_gpu=True,
        add_self_cover=True,
        recompute_components=True,
    ):
        self.cfg = cfg
        self.ds_name = self.cfg["DATASET"]["NAME"]
        self.seed = self.cfg["RNG_SEED"]
        self.lSet = np.asarray(lSet, dtype=np.int64)
        self.uSet = np.asarray(uSet, dtype=np.int64)
        self.budgetSize = int(budgetSize)
        self.delta = float(delta)
        self.cache_root = cache_root
        self.k_knn = int(k_knn)
        self.eps = float(eps)
        self.l2_normalize_features = bool(l2_normalize_features)
        self.prefer_faiss = bool(prefer_faiss)
        self.faiss_gpu = bool(faiss_gpu)
        self.add_self_cover = bool(add_self_cover)
        self.recompute_components = True

        self.selection_metadata = {
            "strategy": "topocover",
            "effective_delta": float(self.delta),
            "k_knn": int(self.k_knn),
            "cache_root": self.cache_root,
            "recompute_components": bool(self.recompute_components),
        }

        self.all_features = ds_utils.load_features(self.ds_name, self.seed).astype(np.float32)
        if self.l2_normalize_features:
            self.all_features /= (np.linalg.norm(self.all_features, axis=1, keepdims=True) + self.eps)

        knn_path = _cache_paths(self.cache_root, self.ds_name, self.seed, self.k_knn)
        self.knn_idx_all, self.knn_dist_all, knn_meta = compute_or_load_knn(
            self.all_features,
            knn_path,
            k_knn=self.k_knn,
            prefer_faiss=self.prefer_faiss,
            faiss_gpu=self.faiss_gpu,
        )
        self.selection_metadata.update({
            "knn_cache_hit": bool(knn_meta["cache_hit"]),
            "knn_cache_path": knn_meta["cache_path"],
            "knn_backend": knn_meta["backend"],
        })

        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(np.int64)
        self.Nr = int(len(self.relevant_indices))
        self.L = int(len(self.lSet))
        self.g2r = -np.ones(self.all_features.shape[0], dtype=np.int32)
        self.g2r[self.relevant_indices] = np.arange(self.Nr, dtype=np.int32)
        self.out_neighbors, self.undirected_neighbors = self._build_adjacency()

    def _build_adjacency(self):
        out_neighbors = [None] * self.Nr
        in_sources = [[] for _ in range(self.Nr)]

        for rel_x, global_x in enumerate(self.relevant_indices):
            neigh_g = self.knn_idx_all[global_x]
            neigh_d = self.knn_dist_all[global_x]
            neigh_r = self.g2r[neigh_g]
            # Algorithm 1 uses the closed radius neighborhood.
            mask = (neigh_r >= 0) & (neigh_d <= self.delta)
            neigh_r = neigh_r[mask].astype(np.int32)
            if neigh_r.size == 0 or not np.any(neigh_r == rel_x):
                neigh_r = np.concatenate([np.asarray([rel_x], dtype=np.int32), neigh_r], axis=0)
            neigh_r = np.unique(neigh_r).astype(np.int32)
            out_neighbors[rel_x] = neigh_r
            for rel_y in neigh_r:
                in_sources[int(rel_y)].append(rel_x)

        undirected = [None] * self.Nr
        for rel_x in range(self.Nr):
            if in_sources[rel_x]:
                merged = np.concatenate([out_neighbors[rel_x], np.asarray(in_sources[rel_x], dtype=np.int32)])
                undirected[rel_x] = np.unique(merged).astype(np.int32)
            else:
                undirected[rel_x] = out_neighbors[rel_x]
        # The radius relation is undirected; kNN only discovers sparse radius edges.
        # Coverage and connectivity must use the same symmetrized graph.
        return undirected, undirected

    def _component_ids(self, uncovered):
        comp_id = -np.ones(self.Nr, dtype=np.int32)
        comp_sizes = []
        cid = 0
        for start in np.where(uncovered)[0]:
            if comp_id[start] >= 0:
                continue
            q = deque([int(start)])
            comp_id[start] = cid
            size = 0
            while q:
                rel_x = q.popleft()
                size += 1
                for rel_y in self.undirected_neighbors[rel_x]:
                    rel_y = int(rel_y)
                    if uncovered[rel_y] and comp_id[rel_y] < 0:
                        comp_id[rel_y] = cid
                        q.append(rel_y)
            comp_sizes.append(size)
            cid += 1
        return comp_id, np.asarray(comp_sizes, dtype=np.int32)

    def _best_candidate(self, comp_id, covered, selected_mask):
        best_rel = -1
        best_gain = -1
        best_points = -1
        candidate_range = range(self.L, self.Nr)
        for rel_x in candidate_range:
            # Algorithm 1, lines 20--23: covered candidates are ineligible.
            if covered[rel_x] or selected_mask[rel_x]:
                continue
            comps = comp_id[self.out_neighbors[rel_x]]
            comps = comps[comps >= 0]
            if comps.size:
                gain = int(np.unique(comps).size)
                points = int(comps.size)
            else:
                gain = 0
                points = 0
            # Preserve pool order on ties, as in the paper's strict greater-than update.
            if gain > best_gain:
                best_rel = rel_x
                best_gain = gain
                best_points = points
        return best_rel, best_gain, best_points

    def select_samples(self):
        covered = np.zeros(self.Nr, dtype=bool)
        for rel_x in range(self.L):
            covered[self.out_neighbors[rel_x]] = True

        selected_rel = []
        selected_mask = np.zeros(self.Nr, dtype=bool)
        gains = []
        touched_points = []
        component_counts = []
        coverage_before = covered.copy()

        for _ in range(self.budgetSize):
            if np.all(covered):
                break
            # Algorithm 1 recomputes components after every greedy selection.
            comp_id, comp_sizes = self._component_ids(~covered)
            component_counts.append(int(len(comp_sizes)))

            rel_x, gain, points = self._best_candidate(comp_id, covered, selected_mask)
            # Algorithm 1, lines 29--30: do not pad with arbitrary samples.
            if rel_x < 0 or gain <= 0:
                break

            selected_rel.append(rel_x)
            selected_mask[rel_x] = True
            gains.append(gain)
            touched_points.append(points)
            covered[self.out_neighbors[rel_x]] = True

        selected_rel = np.asarray(selected_rel, dtype=np.int32)
        activeSet = self.relevant_indices[selected_rel]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))), dtype=np.int64)

        gain_stats = summary_stats(gains)
        point_stats = summary_stats(touched_points)
        comp_stats = summary_stats(component_counts)
        self.selection_metadata.update({
            "coverage_fraction_before": float(coverage_before.mean()) if len(coverage_before) else 0.0,
            "coverage_fraction_after": float(covered.mean()) if len(covered) else 0.0,
            "selected_count": int(len(activeSet)),
            "selected_component_gain_mean": gain_stats["mean"],
            "selected_component_gain_min": gain_stats["min"],
            "selected_component_gain_max": gain_stats["max"],
            "selected_component_gain_std": gain_stats["std"],
            "selected_uncovered_points_mean": point_stats["mean"],
            "uncovered_components_mean": comp_stats["mean"],
            "uncovered_components_min": comp_stats["min"],
            "uncovered_components_max": comp_stats["max"],
        })

        print("TopoCover selected {} samples. coverage {:.4f} -> {:.4f}".format(
            len(activeSet),
            self.selection_metadata["coverage_fraction_before"],
            self.selection_metadata["coverage_fraction_after"],
        ))
        return activeSet, remainSet
