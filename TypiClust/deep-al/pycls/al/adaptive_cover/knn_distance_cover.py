import numpy as np

from .base import AdaptiveRadiusCover


class KnnDistanceCover(AdaptiveRadiusCover):
    def __init__(self, cfg, lSet, uSet, budgetSize, delta0, **kwargs):
        super().__init__(
            cfg=cfg,
            lSet=lSet,
            uSet=uSet,
            budgetSize=budgetSize,
            delta0=delta0,
            signal_name="knn_distance",
            signal_direction=1,
            strategy_name="adaptive_knn_distance_policy",
            selection_mode="adaptive_knn_distance_cover",
            **kwargs,
        )

    def _compute_signal_all(self):
        k_eff = min(self.k_signal, self.knn_dist_all.shape[1])
        return np.mean(self.knn_dist_all[:, :k_eff], axis=1).astype(np.float32)

