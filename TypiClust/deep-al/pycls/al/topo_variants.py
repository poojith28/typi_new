"""TopoCover variants: size-weighted, frontier, and phased ProbCover→Topo hybrids."""

from __future__ import annotations

import numpy as np

from .topocover import TopoCover
from .adaptive_cover.common import summary_stats


class WeightedTopoCover(TopoCover):
    """TopoCover with gain = sum of sizes of touched uncovered components.

    Prefers massed regions over many tiny (outlier-like) components.
    """

    def __init__(self, *args, **kwargs):
        super(WeightedTopoCover, self).__init__(*args, **kwargs)
        self.selection_metadata["strategy"] = "weighted_topocover"
        self.selection_metadata["gain_mode"] = "sum_component_sizes"

    def _best_candidate(self, comp_id, covered, selected_mask):
        # Recompute size lookup from current component labels.
        max_cid = int(comp_id.max()) if comp_id.size else -1
        if max_cid < 0:
            return -1, -1, -1
        comp_sizes = np.bincount(comp_id[comp_id >= 0], minlength=max_cid + 1).astype(np.int64)

        best_rel = -1
        best_gain = -1.0
        best_points = -1
        for rel_x in range(self.L, self.Nr):
            if covered[rel_x] or selected_mask[rel_x]:
                continue
            comps = comp_id[self.out_neighbors[rel_x]]
            comps = comps[comps >= 0]
            if comps.size == 0:
                continue
            uniq = np.unique(comps)
            gain = float(comp_sizes[uniq].sum())
            points = int(comps.size)
            if gain > best_gain:
                best_rel = rel_x
                best_gain = gain
                best_points = points
        return best_rel, best_gain, best_points

    def select_samples(self):
        active_set, remain_set = super(WeightedTopoCover, self).select_samples()
        # Override print-facing strategy label already set in metadata.
        print(
            "WeightedTopoCover selected {} samples. coverage {:.4f} -> {:.4f}".format(
                self.selection_metadata.get("selected_count", len(active_set)),
                self.selection_metadata.get("coverage_fraction_before", 0.0),
                self.selection_metadata.get("coverage_fraction_after", 0.0),
            )
        )
        return active_set, remain_set


class FrontierTopoCover(TopoCover):
    """Only credit uncovered components that touch the current covered frontier.

    Discourages teleporting to isolated component islands early.
    Gain is the size-weighted mass of frontier components touched.
    """

    def __init__(self, *args, **kwargs):
        super(FrontierTopoCover, self).__init__(*args, **kwargs)
        self.selection_metadata["strategy"] = "frontier_topocover"
        self.selection_metadata["gain_mode"] = "sum_frontier_component_sizes"

    def _frontier_component_mask(self, comp_id, covered):
        max_cid = int(comp_id.max()) if comp_id.size else -1
        if max_cid < 0:
            return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64)
        is_frontier = np.zeros(max_cid + 1, dtype=bool)
        comp_sizes = np.bincount(comp_id[comp_id >= 0], minlength=max_cid + 1).astype(np.int64)
        uncovered = np.where(~covered)[0]
        for rel_x in uncovered:
            cid = int(comp_id[rel_x])
            if cid < 0 or is_frontier[cid]:
                continue
            for rel_y in self.undirected_neighbors[rel_x]:
                if covered[int(rel_y)]:
                    is_frontier[cid] = True
                    break
        # If nothing touches the covered set yet (empty L), treat all as frontier.
        if not np.any(is_frontier) and np.any(comp_id >= 0):
            is_frontier[:] = True
        return is_frontier, comp_sizes

    def _best_candidate(self, comp_id, covered, selected_mask):
        is_frontier, comp_sizes = self._frontier_component_mask(comp_id, covered)
        if is_frontier.size == 0:
            return -1, -1, -1

        best_rel = -1
        best_gain = -1.0
        best_points = -1
        for rel_x in range(self.L, self.Nr):
            if covered[rel_x] or selected_mask[rel_x]:
                continue
            comps = comp_id[self.out_neighbors[rel_x]]
            comps = comps[comps >= 0]
            if comps.size == 0:
                continue
            uniq = np.unique(comps)
            uniq = uniq[is_frontier[uniq]]
            if uniq.size == 0:
                continue
            gain = float(comp_sizes[uniq].sum())
            points = int(comps.size)
            if gain > best_gain:
                best_rel = rel_x
                best_gain = gain
                best_points = points
        return best_rel, best_gain, best_points

    def select_samples(self):
        active_set, remain_set = super(FrontierTopoCover, self).select_samples()
        print(
            "FrontierTopoCover selected {} samples. coverage {:.4f} -> {:.4f}".format(
                self.selection_metadata.get("selected_count", len(active_set)),
                self.selection_metadata.get("coverage_fraction_before", 0.0),
                self.selection_metadata.get("coverage_fraction_after", 0.0),
            )
        )
        return active_set, remain_set


class PhasedTopoCover(object):
    """Early ProbCover, late WeightedTopoCover once enough labels exist."""

    def __init__(
        self,
        cfg,
        lSet,
        uSet,
        budgetSize,
        delta,
        switch_labeled=1000,
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
        self.lSet = np.asarray(lSet, dtype=np.int64)
        self.uSet = np.asarray(uSet, dtype=np.int64)
        self.budgetSize = int(budgetSize)
        self.delta = float(delta)
        self.switch_labeled = int(switch_labeled)
        self.cache_root = cache_root
        self.k_knn = int(k_knn)
        self.eps = float(eps)
        self.l2_normalize_features = bool(l2_normalize_features)
        self.prefer_faiss = bool(prefer_faiss)
        self.faiss_gpu = bool(faiss_gpu)
        self.add_self_cover = bool(add_self_cover)
        self.recompute_components = bool(recompute_components)

        self.phase = "probcover" if len(self.lSet) < self.switch_labeled else "weighted_topocover"
        self.selection_metadata = {
            "strategy": "phased_topocover",
            "phase": self.phase,
            "switch_labeled": self.switch_labeled,
            "labeled_count": int(len(self.lSet)),
            "effective_delta": float(self.delta),
        }

    def select_samples(self):
        if self.phase == "probcover":
            from .prob_cover import ProbCover

            sampler = ProbCover(
                self.cfg,
                self.lSet,
                self.uSet,
                budgetSize=self.budgetSize,
                delta=self.delta,
            )
            active_set, remain_set = sampler.select_samples()
            self.selection_metadata["delegated_strategy"] = "probcover"
            print(
                "PhasedTopoCover phase=probcover (|L|={} < {}). selected {}".format(
                    len(self.lSet), self.switch_labeled, len(active_set)
                )
            )
            return active_set, remain_set

        sampler = WeightedTopoCover(
            cfg=self.cfg,
            lSet=self.lSet,
            uSet=self.uSet,
            budgetSize=self.budgetSize,
            delta=self.delta,
            cache_root=self.cache_root,
            k_knn=self.k_knn,
            eps=self.eps,
            l2_normalize_features=self.l2_normalize_features,
            prefer_faiss=self.prefer_faiss,
            faiss_gpu=self.faiss_gpu,
            add_self_cover=self.add_self_cover,
            recompute_components=self.recompute_components,
        )
        active_set, remain_set = sampler.select_samples()
        meta = getattr(sampler, "selection_metadata", {})
        self.selection_metadata.update(meta)
        self.selection_metadata["strategy"] = "phased_topocover"
        self.selection_metadata["phase"] = "weighted_topocover"
        self.selection_metadata["delegated_strategy"] = "weighted_topocover"
        print(
            "PhasedTopoCover phase=weighted_topocover (|L|={} >= {}). selected {}".format(
                len(self.lSet), self.switch_labeled, len(active_set)
            )
        )
        return active_set, remain_set
