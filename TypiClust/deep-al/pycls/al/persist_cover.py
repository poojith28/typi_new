"""Persistence-aware covering acquisitions.

1) PersistProbCover
   ProbCover-style δ-ball covering, but the greedy score only counts newly
   covered points that lie in large connected components of the δ-graph.
   Tiny components (topological dust) do not contribute to gain.

2) W1ProxyCover
   Approximate H0 structure via a kNN-Kruskal MST forest. Each MST edge is a
   merge event (death time). Greedy selection maximises a mix of:
     - ordinary δ-coverage of uncovered points (ProbCover term)
     - coverage of yet-uncovered long-persistence merge edges (barcode term)
   This is a practical proxy for reducing structural W1 to the pool H0 diagram.
"""

from __future__ import annotations

import numpy as np
import torch

import pycls.datasets.utils as ds_utils

from .adaptive_cover.common import compute_or_load_knn, safe_mkdir, summary_stats


def _cache_knn_path(cache_root, dataset, seed, k_knn):
    import os

    base = os.path.join(cache_root, dataset, "seed{}".format(seed))
    safe_mkdir(base)
    return os.path.join(base, "knn_k{}.npz".format(k_knn))


class _UnionFind(object):
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)
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
            return -1, 0, 0
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        # rb dies into ra
        killed_size = int(self.size[rb])
        kept_size = int(self.size[ra])
        self.parent[rb] = ra
        self.size[ra] = kept_size + killed_size
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return rb, killed_size, kept_size


class PersistProbCover(object):
    """ProbCover whose gain ignores points in micro δ-components."""

    def __init__(
        self,
        cfg,
        lSet,
        uSet,
        budgetSize,
        delta,
        min_component_size=8,
        graph_batch_size=500,
        device="cuda",
    ):
        self.cfg = cfg
        self.ds_name = cfg["DATASET"]["NAME"]
        self.seed = int(cfg["RNG_SEED"])
        self.lSet = np.asarray(lSet, dtype=np.int64)
        self.uSet = np.asarray(uSet, dtype=np.int64)
        self.budgetSize = int(budgetSize)
        self.delta = float(delta)
        self.min_component_size = int(min_component_size)
        self.graph_batch_size = int(graph_batch_size)

        requested = str(device)
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)

        self.all_features = np.asarray(
            ds_utils.load_features(self.ds_name, self.seed), dtype=np.float32
        )
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(np.int64)
        self.num_labelled = int(len(self.lSet))
        self.Nr = int(len(self.relevant_indices))
        self.rel_features = self.all_features[self.relevant_indices]

        self.selection_metadata = {
            "strategy": "persist_probcover",
            "effective_delta": float(self.delta),
            "min_component_size": int(self.min_component_size),
        }

        self.neighbors = self._build_radius_neighbors()
        self.signal_mask = self._signal_mask_from_components()
        self.selection_metadata["signal_fraction"] = float(self.signal_mask.mean()) if self.Nr else 0.0
        print(
            "PersistProbCover: delta={:.4f} min_cc={} signal_fraction={:.4f}".format(
                self.delta, self.min_component_size, self.selection_metadata["signal_fraction"]
            )
        )

    @torch.no_grad()
    def _build_radius_neighbors(self):
        n = self.Nr
        x = torch.as_tensor(self.rel_features, dtype=torch.float32, device=self.device)
        neighbors = [np.asarray([i], dtype=np.int32) for i in range(n)]
        for start in range(0, n, self.graph_batch_size):
            stop = min(start + self.graph_batch_size, n)
            dist = torch.cdist(x[start:stop], x, p=2)
            mask = dist <= self.delta
            for local, global_row in enumerate(range(start, stop)):
                idx = torch.nonzero(mask[local], as_tuple=False).flatten().cpu().numpy().astype(np.int32)
                if idx.size == 0:
                    idx = np.asarray([global_row], dtype=np.int32)
                neighbors[global_row] = idx
            del dist, mask
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return neighbors

    def _signal_mask_from_components(self):
        """Connected components of the undirected δ-graph; mark large ones as signal."""
        n = self.Nr
        uf = _UnionFind(n)
        for i, neigh in enumerate(self.neighbors):
            for j in neigh:
                j = int(j)
                if i < j:
                    uf.union(i, j)
        roots = np.array([uf.find(i) for i in range(n)], dtype=np.int32)
        sizes = np.bincount(roots, minlength=n)
        signal = sizes[roots] >= self.min_component_size
        return signal.astype(bool)

    def select_samples(self):
        if self.budgetSize <= 0 or len(self.uSet) == 0:
            return np.empty(0, dtype=np.int64), self.uSet.copy()

        n = self.Nr
        covered = np.zeros(n, dtype=bool)
        for rel in range(self.num_labelled):
            covered[self.neighbors[rel]] = True

        selected = []
        selected_mask = np.zeros(n, dtype=bool)
        gains = []
        budget = min(self.budgetSize, int((~selected_mask[self.num_labelled:]).sum()))

        for _ in range(budget):
            best_rel = -1
            best_gain = -1
            for rel in range(self.num_labelled, n):
                if selected_mask[rel] or covered[rel]:
                    continue
                # Count newly covered *signal* points.
                ball = self.neighbors[rel]
                gain = int(np.count_nonzero((~covered[ball]) & self.signal_mask[ball]))
                if gain > best_gain:
                    best_gain = gain
                    best_rel = rel

            if best_rel < 0 or best_gain <= 0:
                # Fallback: ordinary ProbCover gain if signal is exhausted.
                for rel in range(self.num_labelled, n):
                    if selected_mask[rel]:
                        continue
                    ball = self.neighbors[rel]
                    gain = int(np.count_nonzero(~covered[ball]))
                    if gain > best_gain:
                        best_gain = gain
                        best_rel = rel
                if best_rel < 0 or best_gain <= 0:
                    break

            selected.append(best_rel)
            selected_mask[best_rel] = True
            gains.append(best_gain)
            covered[self.neighbors[best_rel]] = True

        active = self.relevant_indices[np.asarray(selected, dtype=np.int64)]
        remain = np.asarray(
            [idx for idx in self.uSet if int(idx) not in set(map(int, active))],
            dtype=np.int64,
        )
        self.selection_metadata.update({
            "selected_count": int(len(active)),
            "coverage_fraction": float(covered.mean()) if n else 0.0,
            "selected_gain_mean": summary_stats(gains)["mean"],
        })
        print(
            "PersistProbCover selected {} samples. coverage={:.4f}".format(
                len(active), self.selection_metadata["coverage_fraction"]
            )
        )
        return active, remain


class W1ProxyCover(object):
    """Coverage + coverage of long H0 merge edges (MST barcode proxy)."""

    def __init__(
        self,
        cfg,
        lSet,
        uSet,
        budgetSize,
        delta,
        k_knn=50,
        persist_quantile=0.5,
        barcode_lambda=1.0,
        cache_root="./topocover_cache",
        l2_normalize_features=True,
        prefer_faiss=True,
        faiss_gpu=True,
        eps=1e-8,
        graph_batch_size=500,
        device="cuda",
    ):
        self.cfg = cfg
        self.ds_name = cfg["DATASET"]["NAME"]
        self.seed = int(cfg["RNG_SEED"])
        self.lSet = np.asarray(lSet, dtype=np.int64)
        self.uSet = np.asarray(uSet, dtype=np.int64)
        self.budgetSize = int(budgetSize)
        self.delta = float(delta)
        self.k_knn = int(k_knn)
        self.persist_quantile = float(persist_quantile)
        self.barcode_lambda = float(barcode_lambda)
        self.cache_root = cache_root or "./topocover_cache"
        self.eps = float(eps)
        self.graph_batch_size = int(graph_batch_size)

        requested = str(device)
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)

        feats = np.asarray(
            ds_utils.load_features(self.ds_name, self.seed), dtype=np.float32
        )
        if l2_normalize_features:
            feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + self.eps)
        self.all_features = feats

        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(np.int64)
        self.num_labelled = int(len(self.lSet))
        self.Nr = int(len(self.relevant_indices))
        self.rel_features = self.all_features[self.relevant_indices]

        self.selection_metadata = {
            "strategy": "w1_proxy_cover",
            "effective_delta": float(self.delta),
            "k_knn": int(self.k_knn),
            "persist_quantile": float(self.persist_quantile),
            "barcode_lambda": float(self.barcode_lambda),
        }

        knn_path = _cache_knn_path(self.cache_root, self.ds_name, self.seed, self.k_knn)
        self.knn_idx, self.knn_dist, knn_meta = compute_or_load_knn(
            self.all_features,
            knn_path,
            k_knn=self.k_knn,
            prefer_faiss=prefer_faiss,
            faiss_gpu=faiss_gpu,
        )
        self.selection_metadata["knn_backend"] = knn_meta.get("backend", "unknown")

        self.neighbors = self._build_radius_neighbors()
        self.merge_edges, self.merge_deaths = self._mst_merges()
        if self.merge_deaths.size:
            self.death_thresh = float(np.quantile(self.merge_deaths, self.persist_quantile))
        else:
            self.death_thresh = 0.0
        keep = self.merge_deaths >= self.death_thresh
        self.sig_edges = self.merge_edges[keep]
        self.sig_deaths = self.merge_deaths[keep]
        self.selection_metadata["num_mst_edges"] = int(self.merge_edges.shape[0])
        self.selection_metadata["num_significant_edges"] = int(self.sig_edges.shape[0])
        self.selection_metadata["death_thresh"] = float(self.death_thresh)
        print(
            "W1ProxyCover: delta={:.4f} mst_edges={} significant={} death_thresh={:.4f} lambda={:.2f}".format(
                self.delta,
                self.selection_metadata["num_mst_edges"],
                self.selection_metadata["num_significant_edges"],
                self.death_thresh,
                self.barcode_lambda,
            )
        )

    @torch.no_grad()
    def _build_radius_neighbors(self):
        n = self.Nr
        x = torch.as_tensor(self.rel_features, dtype=torch.float32, device=self.device)
        neighbors = [np.asarray([i], dtype=np.int32) for i in range(n)]
        for start in range(0, n, self.graph_batch_size):
            stop = min(start + self.graph_batch_size, n)
            dist = torch.cdist(x[start:stop], x, p=2)
            mask = dist <= self.delta
            for local, global_row in enumerate(range(start, stop)):
                idx = (
                    torch.nonzero(mask[local], as_tuple=False)
                    .flatten()
                    .cpu()
                    .numpy()
                    .astype(np.int32)
                )
                if idx.size == 0:
                    idx = np.asarray([global_row], dtype=np.int32)
                neighbors[global_row] = idx
            del dist, mask
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return neighbors

    def _mst_merges(self):
        """Kruskal on sparse kNN edges among relevant indices → MST forest merges."""
        n = self.Nr
        g2r = -np.ones(self.all_features.shape[0], dtype=np.int32)
        g2r[self.relevant_indices] = np.arange(n, dtype=np.int32)

        edge_map = {}
        for rel_x, global_x in enumerate(self.relevant_indices):
            neigh_g = self.knn_idx[global_x]
            neigh_d = self.knn_dist[global_x]
            for g, d in zip(neigh_g, neigh_d):
                rel_y = int(g2r[int(g)])
                if rel_y < 0 or rel_y == rel_x:
                    continue
                a, b = (rel_x, rel_y) if rel_x < rel_y else (rel_y, rel_x)
                prev = edge_map.get((a, b))
                if prev is None or d < prev:
                    edge_map[(a, b)] = float(d)

        edges = np.array([(a, b, d) for (a, b), d in edge_map.items()], dtype=np.float64)
        if edges.size == 0:
            return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.float32)

        order = np.argsort(edges[:, 2], kind="mergesort")
        edges = edges[order]
        uf = _UnionFind(n)
        kept = []
        deaths = []
        for a, b, d in edges:
            a = int(a)
            b = int(b)
            killed, killed_size, kept_size = uf.union(a, b)
            if killed < 0:
                continue
            # Elder rule persistence with birth=0 ≈ death distance.
            # Weight importance by killed component size later via thresholding deaths.
            kept.append((a, b))
            deaths.append(d)
            if len(kept) >= n - 1:
                break

        if not kept:
            return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.float32)
        return np.asarray(kept, dtype=np.int32), np.asarray(deaths, dtype=np.float32)

    def select_samples(self):
        if self.budgetSize <= 0 or len(self.uSet) == 0:
            return np.empty(0, dtype=np.int64), self.uSet.copy()

        n = self.Nr
        covered = np.zeros(n, dtype=bool)
        for rel in range(self.num_labelled):
            covered[self.neighbors[rel]] = True

        n_edges = int(self.sig_edges.shape[0])
        if n_edges:
            endpoint_count = (
                covered[self.sig_edges[:, 0]].astype(np.int8)
                + covered[self.sig_edges[:, 1]].astype(np.int8)
            )
            incident = [[] for _ in range(n)]
            for ei, (a, b) in enumerate(self.sig_edges):
                incident[int(a)].append(ei)
                incident[int(b)].append(ei)
        else:
            endpoint_count = np.zeros(0, dtype=np.int8)
            incident = [[] for _ in range(n)]

        selected = []
        selected_mask = np.zeros(n, dtype=bool)
        scores = []

        def barcode_gain_for_ball(ball):
            if n_edges == 0:
                return 0
            delta = {}
            for p in ball:
                p = int(p)
                if covered[p]:
                    continue
                for ei in incident[p]:
                    delta[ei] = delta.get(ei, 0) + 1
            gain = 0
            for ei, d in delta.items():
                before = int(endpoint_count[ei])
                if before < 2 and before + d >= 2:
                    gain += 1
            return gain

        budget = min(self.budgetSize, n - self.num_labelled)
        for _ in range(budget):
            best_rel = -1
            best_score = -1.0

            for rel in range(self.num_labelled, n):
                if selected_mask[rel]:
                    continue
                ball = self.neighbors[rel]
                point_gain = int(np.count_nonzero(~covered[ball]))
                if point_gain <= 0:
                    # Still allow pure structural completions.
                    b_gain = barcode_gain_for_ball(ball)
                    if b_gain <= 0:
                        continue
                    score = self.barcode_lambda * float(b_gain)
                else:
                    b_gain = barcode_gain_for_ball(ball)
                    score = float(point_gain) + self.barcode_lambda * float(b_gain)

                if score > best_score:
                    best_score = score
                    best_rel = rel

            if best_rel < 0 or best_score <= 0:
                break

            selected.append(best_rel)
            selected_mask[best_rel] = True
            scores.append(best_score)
            ball = self.neighbors[best_rel]
            # Update MST endpoint coverage counts before flipping covered bits.
            if n_edges:
                for p in ball:
                    p = int(p)
                    if covered[p]:
                        continue
                    for ei in incident[p]:
                        endpoint_count[ei] += 1
            covered[ball] = True

        active = self.relevant_indices[np.asarray(selected, dtype=np.int64)]
        remain = np.asarray(
            [idx for idx in self.uSet if int(idx) not in set(map(int, active))],
            dtype=np.int64,
        )
        edge_cov = float((endpoint_count >= 2).mean()) if n_edges else 0.0
        self.selection_metadata.update({
            "selected_count": int(len(active)),
            "coverage_fraction": float(covered.mean()) if n else 0.0,
            "significant_edge_coverage": edge_cov,
            "selected_score_mean": summary_stats(scores)["mean"],
        })
        print(
            "W1ProxyCover selected {} samples. point_cov={:.4f} edge_cov={:.4f}".format(
                len(active),
                self.selection_metadata["coverage_fraction"],
                self.selection_metadata["significant_edge_coverage"],
            )
        )
        return active, remain
