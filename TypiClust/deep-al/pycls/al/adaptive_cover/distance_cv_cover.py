import numpy as np

from .base import AdaptiveRadiusCover


class DistanceCVCover(AdaptiveRadiusCover):
    def __init__(self, cfg, lSet, uSet, budgetSize, delta0, **kwargs):
        super().__init__(
            cfg=cfg,
            lSet=lSet,
            uSet=uSet,
            budgetSize=budgetSize,
            delta0=delta0,
            signal_name="distance_cv",
            signal_direction=1,
            strategy_name="adaptive_distance_cv_policy",
            selection_mode="adaptive_distance_cv_cover",
            **kwargs,
        )

    def _compute_signal_all(self):
        k_eff = min(self.k_signal, self.knn_dist_all.shape[1])
        dist = self.knn_dist_all[:, :k_eff]
        mean = np.mean(dist, axis=1)
        std = np.std(dist, axis=1)
        return (std / (mean + self.eps)).astype(np.float32)
