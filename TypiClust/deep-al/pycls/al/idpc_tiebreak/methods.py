#!/usr/bin/env python3

import os

import numpy as np

from pycls.al.IDProbCover import IDProbCover
from pycls.al.IDProbCover import _summary_stats
from pycls.al.IDProbCover import _write_idpc_diagnostics


class _IDProbCoverTieBreakBase(IDProbCover):
    def __init__(self, *args, tie_break_mode="min_id", **kwargs):
        self.tie_break_mode = str(tie_break_mode)
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(int(self.seed))
        self.selection_metadata.update({
            "strategy": "id_probcover_tiebreak_policy",
            "selection_mode": "id_prob_cover_tiebreak",
            "tie_break_mode": self.tie_break_mode,
        })

    def _pick_from_tied_candidates(self, candidates):
        candidates = np.asarray(candidates, dtype=np.int32)
        if candidates.size == 0:
            raise ValueError("Expected at least one candidate for tie-break.")

        if self.tie_break_mode == "min_id":
            return int(candidates[np.argmin(self.rel_ids[candidates])])
        if self.tie_break_mode == "random":
            return int(self.rng.choice(candidates))
        if self.tie_break_mode == "first_max":
            return int(candidates[0])
        raise ValueError(f"Unknown tie_break_mode: {self.tie_break_mode}")

    def _pick_fallback_candidate(self, selected_mask):
        pool = np.where((~selected_mask) & (np.arange(self.Nr) >= self.L))[0]
        if pool.size == 0:
            return None
        return self._pick_from_tied_candidates(pool)

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

        for it in range(self.budgetSize):
            cand_deg = current_degree.copy()
            cand_deg[:self.L] = -1
            cand_deg[selected_mask] = -1
            best = int(cand_deg.max())
            if best < 0:
                break

            tied = np.where(cand_deg == best)[0]
            rel_x = self._pick_from_tied_candidates(tied)

            if best <= 0:
                rel_x = self._pick_fallback_candidate(selected_mask)
                if rel_x is None:
                    break

            selected_rel.append(rel_x)
            selected_mask[rel_x] = True
            selected_gains.append(best)

            newly = self.out_neighbors[rel_x]
            newly = newly[~covered[newly]]

            pick_id = float(self.rel_ids[rel_x])
            pick_global = int(self.relevant_indices[rel_x])
            if newly.size == 0:
                current_degree[rel_x] = 0
                cov_ratio = float(np.mean(covered))
                print(
                    f"[IDProbCover/{self.tie_break_mode}] it={it:03d} pick_rel={rel_x} ID={pick_id:.3f} "
                    f"pick_global={pick_global} best_gain={best} covered={cov_ratio:.3f} "
                    f"maxdeg={int(current_degree.max())}"
                )
                continue

            covered[newly] = True
            for rel_y in newly:
                srcs = self.in_sources[int(rel_y)]
                if srcs.size:
                    current_degree[srcs] -= 1
            current_degree[rel_x] = 0

            cov_ratio = float(np.mean(covered))
            print(
                f"[IDProbCover/{self.tie_break_mode}] it={it:03d} pick_rel={rel_x} ID={pick_id:.3f} "
                f"pick_global={pick_global} best_gain={best} covered={cov_ratio:.3f} "
                f"maxdeg={int(current_degree.max())}"
            )

        selected_rel = np.asarray(selected_rel, dtype=np.int32)
        activeSet = self.relevant_indices[selected_rel]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))), dtype=np.int64)

        selected_ids = self.rel_ids[selected_rel] if len(selected_rel) else np.zeros(0, dtype=np.float32)
        selected_radii = self.delta_per_x[selected_rel] if len(selected_rel) else np.zeros(0, dtype=np.float32)
        id_stats = _summary_stats(selected_ids)
        radius_stats = _summary_stats(selected_radii)
        gain_stats = _summary_stats(selected_gains)
        median_id = float(np.median(self.rel_ids)) if len(self.rel_ids) else 0.0

        self.selection_metadata.update({
            "coverage_fraction_before": float(coverage_before.mean()) if len(coverage_before) else 0.0,
            "coverage_fraction_after": float(covered.mean()) if len(covered) else 0.0,
            "selected_count": int(len(activeSet)),
            "selected_id_mean": id_stats["mean"],
            "selected_id_min": id_stats["min"],
            "selected_id_max": id_stats["max"],
            "selected_id_std": id_stats["std"],
            "selected_radius_mean": radius_stats["mean"],
            "selected_radius_min": radius_stats["min"],
            "selected_radius_max": radius_stats["max"],
            "selected_radius_std": radius_stats["std"],
            "selected_coverage_gain_mean": gain_stats["mean"],
            "selected_coverage_gain_min": gain_stats["min"],
            "selected_coverage_gain_max": gain_stats["max"],
            "selected_coverage_gain_std": gain_stats["std"],
            "median_local_id": median_id,
        })

        csv_path = str(getattr(self.cfg.ACTIVE_LEARNING, "IDPC_LOG_CSV", "") or "")
        if csv_path:
            if not os.path.isabs(csv_path):
                csv_path = os.path.join(getattr(self.cfg, "EXP_DIR", "."), csv_path)
            _write_idpc_diagnostics(csv_path, {
                "selected_size": int(len(activeSet)),
                "selected_id_mean": id_stats["mean"],
                "selected_id_min": id_stats["min"],
                "selected_id_max": id_stats["max"],
                "selected_id_std": id_stats["std"],
                "selected_radius_mean": radius_stats["mean"],
                "selected_radius_min": radius_stats["min"],
                "selected_radius_max": radius_stats["max"],
                "selected_radius_std": radius_stats["std"],
                "selected_gain_mean": gain_stats["mean"],
                "selected_gain_min": gain_stats["min"],
                "selected_gain_max": gain_stats["max"],
                "selected_gain_std": gain_stats["std"],
                "coverage_fraction_before": self.selection_metadata["coverage_fraction_before"],
                "coverage_fraction_after": self.selection_metadata["coverage_fraction_after"],
                "k_id": int(self.k_id),
                "k_knn": int(self.k_knn),
                "base_delta": float(self.delta0),
                "alpha": float(self.alpha),
                "median_id": median_id,
            })

        return activeSet, remainSet


class IDProbCoverMinIDTieBreak(_IDProbCoverTieBreakBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, tie_break_mode="min_id", **kwargs)


class IDProbCoverRandomTieBreak(_IDProbCoverTieBreakBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, tie_break_mode="random", **kwargs)


class IDProbCoverFirstMaxTieBreak(_IDProbCoverTieBreakBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, tie_break_mode="first_max", **kwargs)
