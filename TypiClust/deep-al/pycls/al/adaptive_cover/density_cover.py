import numpy as np

from .base import AdaptiveRadiusCover


class DensityCover(AdaptiveRadiusCover):
    def __init__(self, cfg, lSet, uSet, budgetSize, delta0, **kwargs):
        super().__init__(
            cfg=cfg,
            lSet=lSet,
            uSet=uSet,
            budgetSize=budgetSize,
            delta0=delta0,
            signal_name="density",
            signal_direction=-1,
            strategy_name="adaptive_density_policy",
            selection_mode="adaptive_density_cover",
            **kwargs,
        )

    def _compute_signal_all(self):
        k_eff = min(self.k_signal, self.knn_dist_all.shape[1])
        dist = self.knn_dist_all[:, :k_eff]
        sigma = float(np.median(self.knn_dist_all[:, k_eff - 1]) + self.eps)
        weights = np.exp(-(dist ** 2) / (2.0 * sigma * sigma))
        return np.mean(weights, axis=1).astype(np.float32)

