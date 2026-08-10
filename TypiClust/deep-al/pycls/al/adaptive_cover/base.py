import numpy as np

import pycls.datasets.utils as ds_utils

from .common import (
    compute_or_load_knn,
    compute_or_load_signal,
    default_cache_paths,
    summary_stats,
)


class AdaptiveRadiusCover:
    def __init__(
        self,
        cfg,
        lSet,
        uSet,
        budgetSize,
        delta0,
        signal_name,
        signal_direction,
        strategy_name,
        selection_mode,
        alpha=1.0,
        cache_root="./adaptive_cover_cache",
        k_signal=50,
        k_knn=50,
        eps=1e-8,
        l2_normalize_features=True,
        prefer_faiss=True,
        faiss_gpu=True,
        add_self_cover=True,
    ):
        self.cfg = cfg
        self.ds_name = self.cfg["DATASET"]["NAME"]
        self.seed = self.cfg["RNG_SEED"]
        self.lSet = np.asarray(lSet).astype(np.int64)
        self.uSet = np.asarray(uSet).astype(np.int64)
        self.budgetSize = int(budgetSize)
        self.delta0 = float(delta0)
        self.signal_name = signal_name
        self.signal_direction = int(signal_direction)
        self.strategy_name = strategy_name
        self.selection_mode = selection_mode
        self.alpha = float(alpha)
        self.cache_root = cache_root
        self.k_signal = int(k_signal)
        self.k_knn = int(k_knn)
        self.eps = float(eps)
        self.l2_normalize_features = bool(l2_normalize_features)
        self.prefer_faiss = bool(prefer_faiss)
        self.faiss_gpu = bool(faiss_gpu)
        self.add_self_cover = bool(add_self_cover)

        self.selection_metadata = {
            "strategy": self.strategy_name,
            "selection_mode": self.selection_mode,
            "effective_delta": float(self.delta0),
            "delta_phase": "adaptive_signal",
            "signal_name": self.signal_name,
            "k_signal": int(self.k_signal),
            "k_knn": int(self.k_knn),
            "alpha": float(self.alpha),
            "cache_root": self.cache_root,
        }

        self.all_features = ds_utils.load_features(self.ds_name, self.seed).astype(np.float32)
        if self.l2_normalize_features:
            self.all_features /= (np.linalg.norm(self.all_features, axis=1, keepdims=True) + self.eps)

        signal_path, knn_path = default_cache_paths(
            self.cache_root,
            self.ds_name,
            self.seed,
            self.k_signal,
            self.k_knn,
            self.signal_name,
        )
        self.knn_idx_all, self.knn_dist_all, knn_cache_meta = compute_or_load_knn(
            self.all_features,
            knn_path,
            k_knn=self.k_knn,
            prefer_faiss=self.prefer_faiss,
            faiss_gpu=self.faiss_gpu,
        )
        self.signal_all, signal_cache_meta = compute_or_load_signal(
            signal_path,
            self.all_features.shape[0],
            self._compute_signal_all,
        )

        self.selection_metadata.update({
            "signal_cache_hit": bool(signal_cache_meta["cache_hit"]),
            "signal_cache_path": signal_cache_meta["cache_path"],
            "knn_cache_hit": bool(knn_cache_meta["cache_hit"]),
            "knn_cache_path": knn_cache_meta["cache_path"],
            "knn_backend": knn_cache_meta["backend"],
        })

        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(np.int64)
        self.Nr = len(self.relevant_indices)
        self.L = len(self.lSet)
        self.g2r = -np.ones(self.all_features.shape[0], dtype=np.int32)
        self.g2r[self.relevant_indices] = np.arange(self.Nr, dtype=np.int32)
        self.rel_signal = self.signal_all[self.relevant_indices].astype(np.float32)
        self.delta_per_x = self._compute_delta_per_x(self.rel_signal)
        self.out_neighbors, self.in_sources = self._build_rel_adjacency()

    def _compute_signal_all(self):
        raise NotImplementedError

    def _compute_delta_per_x(self, signal):
        median_signal = float(np.median(signal) + self.eps)
        ratio = (signal + self.eps) / median_signal
        exponent = self.alpha * self.signal_direction
        scale = ratio ** exponent
        return (self.delta0 * scale).astype(np.float32)

    def _build_rel_adjacency(self):
        out_neighbors = [None] * self.Nr
        in_sources = [[] for _ in range(self.Nr)]

        for rel_x, global_x in enumerate(self.relevant_indices):
            neigh_g = self.knn_idx_all[global_x]
            neigh_d = self.knn_dist_all[global_x]

            neigh_r = self.g2r[neigh_g]
            mask_rel = neigh_r >= 0
            neigh_r = neigh_r[mask_rel]
            neigh_d = neigh_d[mask_rel]

            thr = self.delta_per_x[rel_x]
            neigh_r = neigh_r[neigh_d < thr].astype(np.int32)

            if self.add_self_cover and (neigh_r.size == 0 or not np.any(neigh_r == rel_x)):
                neigh_r = np.concatenate([np.array([rel_x], dtype=np.int32), neigh_r], axis=0)

            out_neighbors[rel_x] = neigh_r
            for rel_y in neigh_r:
                in_sources[int(rel_y)].append(rel_x)

        return out_neighbors, [np.asarray(srcs, dtype=np.int32) for srcs in in_sources]

    def select_samples(self):
        covered = np.zeros(self.Nr, dtype=bool)
        for rel_x in range(self.L):
            covered[self.out_neighbors[rel_x]] = True

        current_degree = np.zeros(self.Nr, dtype=np.int32)
        for rel_x in range(self.Nr):
            current_degree[rel_x] = int(np.sum(~covered[self.out_neighbors[rel_x]]))

        selected_rel = []
        selected_mask = np.zeros(self.Nr, dtype=bool)
        selected_gains = []
        coverage_before = covered.copy()

        for _ in range(self.budgetSize):
            cand_deg = current_degree.copy()
            cand_deg[:self.L] = -1
            cand_deg[selected_mask] = -1
            best = int(cand_deg.max())
            if best < 0:
                break

            cands = np.where(cand_deg == best)[0]
            rel_x = int(cands[np.argmin(self.delta_per_x[cands])])

            if best <= 0:
                pool = np.where((~selected_mask) & (np.arange(self.Nr) >= self.L))[0]
                if pool.size == 0:
                    break
                rel_x = int(pool[np.argmin(self.delta_per_x[pool])])

            selected_rel.append(rel_x)
            selected_mask[rel_x] = True
            selected_gains.append(best)

            newly = self.out_neighbors[rel_x]
            newly = newly[~covered[newly]]
            if newly.size == 0:
                current_degree[rel_x] = 0
                continue

            covered[newly] = True
            for rel_y in newly:
                srcs = self.in_sources[int(rel_y)]
                if srcs.size:
                    current_degree[srcs] -= 1
            current_degree[rel_x] = 0

        selected_rel = np.asarray(selected_rel, dtype=np.int32)
        activeSet = self.relevant_indices[selected_rel]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))), dtype=np.int64)

        selected_signal = self.rel_signal[selected_rel] if len(selected_rel) else np.zeros(0, dtype=np.float32)
        selected_radii = self.delta_per_x[selected_rel] if len(selected_rel) else np.zeros(0, dtype=np.float32)
        signal_stats = summary_stats(selected_signal)
        radius_stats = summary_stats(selected_radii)
        gain_stats = summary_stats(selected_gains)

        self.selection_metadata.update({
            "coverage_fraction_before": float(coverage_before.mean()) if len(coverage_before) else 0.0,
            "coverage_fraction_after": float(covered.mean()) if len(covered) else 0.0,
            "selected_count": int(len(activeSet)),
            "selected_signal_mean": signal_stats["mean"],
            "selected_signal_min": signal_stats["min"],
            "selected_signal_max": signal_stats["max"],
            "selected_signal_std": signal_stats["std"],
            "selected_radius_mean": radius_stats["mean"],
            "selected_radius_min": radius_stats["min"],
            "selected_radius_max": radius_stats["max"],
            "selected_radius_std": radius_stats["std"],
            "selected_coverage_gain_mean": gain_stats["mean"],
            "selected_coverage_gain_min": gain_stats["min"],
            "selected_coverage_gain_max": gain_stats["max"],
            "selected_coverage_gain_std": gain_stats["std"],
            "median_signal": float(np.median(self.rel_signal)) if len(self.rel_signal) else 0.0,
        })
        return activeSet, remainSet

