"""New samplers for the 2026 extensions; historical sampler files are untouched."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import pycls.datasets.utils as ds_utils
from pycls.al.IDProbCover import IDProbCover, compute_or_load_knn


def _append_csv(path: Path, row: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _summary(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if values.size else float("nan")


class LIDCoverTieFallbackFactorial(IDProbCover):
    """Frozen LIDCover geometry with only tie and fallback decisions varied."""

    ROUND_FIELDS = [
        "method", "seed", "selection_event", "labelled_budget", "acquisition_batch_size",
        "positive_gain_tie_count", "mean_positive_gain_tie_set_size",
        "max_positive_gain_tie_set_size", "fallback_count", "fallback_fraction",
        "first_fallback_query_position", "number_of_selections_after_first_fallback",
        "mean_selected_lid_before_fallback", "mean_selected_lid_after_fallback",
        "mean_selected_adaptive_radius", "mean_selected_uncovered_gain",
    ]

    def __init__(self, *args, positive_tie_rule: str, fallback_rule: str, method: str, **kwargs):
        self.positive_tie_rule = positive_tie_rule
        self.fallback_rule = fallback_rule
        self.method = method
        super().__init__(*args, **kwargs)

    def _choice(self, candidates: np.ndarray, query_position: int, stream: int) -> int:
        episode = int(len(self.lSet) // self.budgetSize)
        sequence = np.random.SeedSequence([int(self.seed), episode, int(query_position), int(stream)])
        rng = np.random.default_rng(sequence)
        return int(rng.choice(candidates))

    def select_samples(self):
        covered = np.zeros(self.Nr, dtype=bool)
        for rel_x in range(self.L):
            covered[self.out_neighbors[rel_x]] = True
        current_degree = np.asarray(
            [np.sum(~covered[neighbours]) for neighbours in self.out_neighbors], dtype=np.int32
        )
        selected_mask = np.zeros(self.Nr, dtype=bool)
        selected_rel: list[int] = []
        per_selection: list[dict] = []
        positive_tie_sizes: list[int] = []
        fallback_positions: list[int] = []

        selection_event = int(len(self.lSet) // self.budgetSize)
        for query_position in range(self.budgetSize):
            degrees = current_degree.copy()
            degrees[: self.L] = -1
            degrees[selected_mask] = -1
            best = int(degrees.max())
            if best < 0:
                break
            tied = np.where(degrees == best)[0]
            was_positive_tie = bool(best > 0 and tied.size > 1)
            if was_positive_tie:
                positive_tie_sizes.append(int(tied.size))

            fallback = best <= 0
            if fallback:
                pool = np.where((~selected_mask) & (np.arange(self.Nr) >= self.L))[0]
                if pool.size == 0:
                    break
                fallback_positions.append(query_position)
                if self.fallback_rule == "min_lid":
                    rel_x = int(pool[np.argmin(self.rel_ids[pool])])
                elif self.fallback_rule == "random":
                    rel_x = self._choice(pool, query_position, stream=2)
                else:
                    raise ValueError(f"unknown fallback rule: {self.fallback_rule}")
            elif self.positive_tie_rule == "min_lid":
                rel_x = int(tied[np.argmin(self.rel_ids[tied])])
            elif self.positive_tie_rule == "random":
                rel_x = self._choice(tied, query_position, stream=1)
            else:
                raise ValueError(f"unknown positive tie rule: {self.positive_tie_rule}")

            newly = self.out_neighbors[rel_x]
            newly = newly[~covered[newly]]
            actual_gain = int(newly.size)
            selected_rel.append(rel_x)
            selected_mask[rel_x] = True
            per_selection.append({
                "method": self.method,
                "seed": int(self.seed),
                "selection_event": selection_event,
                "labelled_budget_before": int(len(self.lSet)),
                "query_position_zero_based": query_position,
                "query_position_one_based": query_position + 1,
                "sample_index": int(self.relevant_indices[rel_x]),
                "pointwise_lid": float(self.rel_ids[rel_x]),
                "adaptive_radius": float(self.delta_per_x[rel_x]),
                "maximum_gain": best,
                "actual_uncovered_gain": actual_gain,
                "positive_gain_tied": was_positive_tie,
                "tie_set_size": int(tied.size),
                "zero_gain_fallback": fallback,
                "positive_tie_rule": self.positive_tie_rule,
                "fallback_rule": self.fallback_rule,
                "rng_key": [int(self.seed), selection_event, query_position],
            })
            if newly.size:
                covered[newly] = True
                for rel_y in newly:
                    sources = self.in_sources[int(rel_y)]
                    if sources.size:
                        current_degree[sources] -= 1
            current_degree[rel_x] = 0

        selected_array = np.asarray(selected_rel, dtype=np.int32)
        active_set = self.relevant_indices[selected_array]
        remain_set = np.asarray(sorted(set(self.uSet) - set(active_set)), dtype=np.int64)
        fallback_count = len(fallback_positions)
        first_fallback = fallback_positions[0] if fallback_positions else None
        lids = [row["pointwise_lid"] for row in per_selection]
        radii = [row["adaptive_radius"] for row in per_selection]
        gains = [row["actual_uncovered_gain"] for row in per_selection]
        before = lids[:first_fallback] if first_fallback is not None else lids
        after = lids[first_fallback:] if first_fallback is not None else []
        round_row = {
            "method": self.method,
            "seed": int(self.seed),
            "selection_event": selection_event,
            "labelled_budget": int(len(self.lSet)),
            "acquisition_batch_size": int(self.budgetSize),
            "positive_gain_tie_count": len(positive_tie_sizes),
            "mean_positive_gain_tie_set_size": _summary(positive_tie_sizes),
            "max_positive_gain_tie_set_size": max(positive_tie_sizes, default=0),
            "fallback_count": fallback_count,
            "fallback_fraction": fallback_count / len(per_selection) if per_selection else 0.0,
            "first_fallback_query_position": (first_fallback + 1) if first_fallback is not None else "",
            "number_of_selections_after_first_fallback": (
                len(per_selection) - first_fallback - 1 if first_fallback is not None else 0
            ),
            "mean_selected_lid_before_fallback": _summary(before),
            "mean_selected_lid_after_fallback": _summary(after),
            "mean_selected_adaptive_radius": _summary(radii),
            "mean_selected_uncovered_gain": _summary(gains),
        }
        exp_dir = Path(str(self.cfg.EXP_DIR))
        _append_csv(exp_dir / "tie_fallback_round_metrics.csv", round_row, self.ROUND_FIELDS)
        _append_jsonl(exp_dir / "tie_fallback_selection_metrics.jsonl", per_selection)
        self.selection_metadata.update(round_row)
        self.selection_metadata.update({
            "strategy": "lidcover_tie_fallback_factorial_v1",
            "selection_mode": self.method,
            "positive_tie_rule": self.positive_tie_rule,
            "fallback_rule": self.fallback_rule,
            "selected_count": int(len(active_set)),
        })
        return active_set, remain_set


class SparseFixedCover:
    """Fixed-radius sparse LIDCover-style graph with deterministic first-index decisions."""

    def __init__(
        self, cfg, lSet, uSet, budgetSize, delta0, cache_root, k_knn=50,
        l2_normalize_features=True, prefer_faiss=True, faiss_gpu=True, add_self_cover=True,
    ):
        self.cfg = cfg
        self.seed = int(cfg.RNG_SEED)
        self.ds_name = str(cfg.DATASET.NAME)
        self.lSet = np.asarray(lSet, dtype=np.int64)
        self.uSet = np.asarray(uSet, dtype=np.int64)
        self.budgetSize = int(budgetSize)
        self.delta0 = float(delta0)
        self.k_knn = int(k_knn)
        self.add_self_cover = bool(add_self_cover)
        features = ds_utils.load_features(self.ds_name, self.seed).astype(np.float32)
        if l2_normalize_features:
            features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
        digest = hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()
        cache_path = Path(cache_root) / self.ds_name / f"seed{self.seed}" / f"knn_k{self.k_knn}_{digest[:16]}.npz"
        self.knn_idx, self.knn_dist, cache_meta = compute_or_load_knn(
            features, str(cache_path), k_knn=self.k_knn,
            prefer_faiss=prefer_faiss, faiss_gpu=faiss_gpu,
        )
        self.relevant_indices = np.concatenate([self.lSet, self.uSet])
        self.nr = len(self.relevant_indices)
        self.labelled = len(self.lSet)
        global_to_relative = -np.ones(features.shape[0], dtype=np.int32)
        global_to_relative[self.relevant_indices] = np.arange(self.nr, dtype=np.int32)
        self.out_neighbors = []
        self.in_sources = [[] for _ in range(self.nr)]
        for relative_x, global_x in enumerate(self.relevant_indices):
            neighbour_relative = global_to_relative[self.knn_idx[global_x]]
            distance = self.knn_dist[global_x]
            keep = (neighbour_relative >= 0) & (distance < self.delta0)
            neighbours = neighbour_relative[keep].astype(np.int32)
            if self.add_self_cover and not np.any(neighbours == relative_x):
                neighbours = np.concatenate([np.asarray([relative_x], dtype=np.int32), neighbours])
            self.out_neighbors.append(neighbours)
            for relative_y in neighbours:
                self.in_sources[int(relative_y)].append(relative_x)
        self.in_sources = [np.asarray(values, dtype=np.int32) for values in self.in_sources]
        self.selection_metadata = {
            "strategy": "sparse_fixed_cover_v1",
            "selection_mode": "sparsefixedcover_v1",
            "effective_delta": self.delta0,
            "k_knn": self.k_knn,
            "feature_normalisation": "rowwise_l2_float32_eps_1e-12",
            "radius_adaptation": False,
            "lid_computed_or_used": False,
            "tie_rule": "first_relative_index",
            "fallback_rule": "first_remaining_relative_index",
            "strict_radius_test": True,
            "explicit_self_coverage": self.add_self_cover,
            "knn_cache_path": str(cache_path),
            "knn_backend": cache_meta.get("backend"),
        }

    def select_samples(self):
        covered = np.zeros(self.nr, dtype=bool)
        for relative_x in range(self.labelled):
            covered[self.out_neighbors[relative_x]] = True
        degree = np.asarray([np.sum(~covered[n]) for n in self.out_neighbors], dtype=np.int32)
        selected_mask = np.zeros(self.nr, dtype=bool)
        selected = []
        rows = []
        for position in range(self.budgetSize):
            candidate_degree = degree.copy()
            candidate_degree[: self.labelled] = -1
            candidate_degree[selected_mask] = -1
            best = int(candidate_degree.max())
            if best < 0:
                break
            tied = np.where(candidate_degree == best)[0]
            relative_x = int(tied[0])
            fallback = best <= 0
            if fallback:
                remaining = np.where((~selected_mask) & (np.arange(self.nr) >= self.labelled))[0]
                if remaining.size == 0:
                    break
                relative_x = int(remaining[0])
            newly = self.out_neighbors[relative_x]
            newly = newly[~covered[newly]]
            selected.append(relative_x)
            selected_mask[relative_x] = True
            rows.append({
                "method": "sparsefixedcover_v1", "seed": self.seed,
                "selection_event": int(len(self.lSet) // self.budgetSize),
                "labelled_budget_before": int(len(self.lSet)),
                "query_position_one_based": position + 1,
                "sample_index": int(self.relevant_indices[relative_x]),
                "fixed_radius": self.delta0, "maximum_gain": best,
                "actual_uncovered_gain": int(newly.size),
                "tie_set_size": int(tied.size), "zero_gain_fallback": fallback,
            })
            if newly.size:
                covered[newly] = True
                for relative_y in newly:
                    sources = self.in_sources[int(relative_y)]
                    if sources.size:
                        degree[sources] -= 1
            degree[relative_x] = 0
        active = self.relevant_indices[np.asarray(selected, dtype=np.int32)]
        remain = np.asarray(sorted(set(self.uSet) - set(active)), dtype=np.int64)
        _append_jsonl(Path(str(self.cfg.EXP_DIR)) / "sparse_fixed_selection_metrics.jsonl", rows)
        self.selection_metadata["selected_count"] = int(len(active))
        self.selection_metadata["mean_selected_uncovered_gain"] = _summary(
            [row["actual_uncovered_gain"] for row in rows]
        )
        return active, remain
