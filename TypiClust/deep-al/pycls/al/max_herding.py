import numpy as np

import pycls.datasets.utils as ds_utils


def _l2_normalize(features, eps=1e-12):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / (norms + eps)


def _nearest_neighbors(query, database, k):
    """Return squared L2 distances and indices for k nearest neighbors."""
    k = min(int(k), int(database.shape[0]))
    try:
        import faiss

        index = faiss.IndexFlatL2(database.shape[1])
        index.add(database.astype(np.float32, copy=False))
        dists, indices = index.search(query.astype(np.float32, copy=False), k)
        return dists.astype(np.float32, copy=False), indices.astype(np.int64, copy=False)
    except Exception:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
        nn.fit(database)
        dists, indices = nn.kneighbors(query, return_distance=True)
        return (dists ** 2).astype(np.float32, copy=False), indices.astype(np.int64, copy=False)


class MaxHerding:
    """Greedy generalized coverage with a Gaussian kernel.

    This implements the MaxHerding objective from Bae et al.,
    "Generalized Coverage for More Robust Low-Budget Active Learning":

        argmax_x sum_i max(k(x_i, x) - max_{l in L} k(x_i, l), 0)

    For scalability in the full AL loop, we use a truncated Gaussian kernel:
    each candidate contributes only to its k nearest pool points. Setting
    MAXHERDING_KNN larger gives a closer approximation at higher cost.
    """

    def __init__(self, cfg, lSet, uSet, budgetSize):
        self.cfg = cfg
        self.ds_name = self.cfg["DATASET"]["NAME"]
        self.seed = self.cfg["RNG_SEED"]
        self.lSet = np.asarray(lSet, dtype=np.int64)
        self.uSet = np.asarray(uSet, dtype=np.int64)
        self.budgetSize = int(budgetSize)
        self.sigma = float(getattr(self.cfg.ACTIVE_LEARNING, "MAXHERDING_SIGMA", 1.0))
        self.knn = int(getattr(self.cfg.ACTIVE_LEARNING, "MAXHERDING_KNN", 50))
        self.l2_normalize = bool(getattr(self.cfg.ACTIVE_LEARNING, "MAXHERDING_L2_NORMALIZE_FEATURES", True))
        self.selection_metadata = {}

        self.all_features = ds_utils.load_features(
            self.ds_name, self.seed, train=True, normalized=False
        ).astype(np.float32, copy=False)
        if self.l2_normalize:
            self.all_features = _l2_normalize(self.all_features).astype(np.float32, copy=False)

    def _kernel_from_sqdist(self, sqdist):
        denom = 2.0 * (self.sigma ** 2)
        return np.exp(-sqdist / denom).astype(np.float32, copy=False)

    def _initial_coverage(self, rel_features):
        if len(self.lSet) == 0:
            return np.zeros(rel_features.shape[0], dtype=np.float32)

        labeled_features = self.all_features[self.lSet]
        dists, _ = _nearest_neighbors(rel_features, labeled_features, k=1)
        return self._kernel_from_sqdist(dists[:, 0])

    def select_samples(self):
        assert self.budgetSize > 0, "Expected positive budgetSize"
        assert self.budgetSize <= len(self.uSet), "Budget cannot exceed unlabelled set length"
        assert self.sigma > 0, "MAXHERDING_SIGMA must be positive"

        relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(np.int64, copy=False)
        rel_features = self.all_features[relevant_indices]
        u_features = self.all_features[self.uSet]

        coverage = self._initial_coverage(rel_features)
        dists, neigh = _nearest_neighbors(u_features, rel_features, k=self.knn)
        kvals = self._kernel_from_sqdist(dists)

        available = np.ones(len(self.uSet), dtype=bool)
        selected_local = []
        selected_scores = []
        coverage_before = float(np.mean(coverage)) if coverage.size else 0.0

        for _ in range(self.budgetSize):
            gains = np.maximum(kvals - coverage[neigh], 0.0).sum(axis=1)
            gains[~available] = -np.inf
            chosen = int(np.argmax(gains))
            if not np.isfinite(gains[chosen]):
                break

            selected_local.append(chosen)
            selected_scores.append(float(gains[chosen]))
            available[chosen] = False
            coverage[neigh[chosen]] = np.maximum(coverage[neigh[chosen]], kvals[chosen])

        if len(selected_local) != self.budgetSize:
            remaining = np.flatnonzero(available)
            needed = self.budgetSize - len(selected_local)
            selected_local.extend(remaining[:needed].astype(int).tolist())
            available[remaining[:needed]] = False

        selected_local = np.asarray(selected_local, dtype=np.int64)
        activeSet = self.uSet[selected_local]
        remainSet = self.uSet[available]
        selected_kernel = kvals[selected_local]

        self.selection_metadata = {
            "strategy": "maxherding_policy",
            "selection_mode": "gaussian_generalized_coverage_greedy",
            "approximation": "knn_truncated_gaussian",
            "feature_source": "precomputed",
            "sigma": float(self.sigma),
            "knn": int(self.knn),
            "l2_normalize_features": bool(self.l2_normalize),
            "target_pool_size": int(len(relevant_indices)),
            "labeled_count_before_sampling": int(len(self.lSet)),
            "unlabeled_count_before_sampling": int(len(self.uSet)),
            "selected_count": int(len(activeSet)),
            "coverage_before": coverage_before,
            "coverage_after": float(np.mean(coverage)) if coverage.size else 0.0,
            "selected_gain_mean": float(np.mean(selected_scores)) if selected_scores else 0.0,
            "selected_gain_min": float(np.min(selected_scores)) if selected_scores else 0.0,
            "selected_gain_max": float(np.max(selected_scores)) if selected_scores else 0.0,
            "selected_kernel_mean": float(np.mean(selected_kernel)) if selected_kernel.size else 0.0,
        }

        return activeSet.astype(np.int64, copy=False), remainSet.astype(np.int64, copy=False)
