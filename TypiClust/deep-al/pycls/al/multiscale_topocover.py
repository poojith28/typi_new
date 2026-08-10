#!/usr/bin/env python3
"""
Original-style MultiScaleTopoCover acquisition strategy.

This module implements the early two-scale TopoCover variant used through the
following public interface:

    MultiScaleTopoCover(
        cfg,
        lSet,
        uSet,
        budgetSize,
        delta_coarse,
        delta_fine,
        alpha=0.7,
        graph_batch_size=500,
        device="cuda",
    )

The acquisition score is

    alpha * coarse_component_gain
    + (1 - alpha) * fine_component_gain.

No diversity regulariser, anchor term, score normalisation, or unique-component
constraint is included.
"""

from __future__ import annotations

from collections import deque
from typing import List, Sequence, Tuple

import numpy as np
import torch

import pycls.datasets.utils as ds_utils


class MultiScaleTopoCover:
    """Greedy component-aware coverage at coarse and fine graph scales."""

    def __init__(
        self,
        cfg,
        lSet,
        uSet,
        budgetSize,
        delta_coarse,
        delta_fine,
        alpha=0.7,
        graph_batch_size=500,
        device="cuda",
    ):
        self.cfg = cfg
        self.lSet = np.asarray(lSet, dtype=np.int64)
        self.uSet = np.asarray(uSet, dtype=np.int64)
        self.budgetSize = int(budgetSize)

        self.delta_coarse = float(delta_coarse)
        self.delta_fine = float(delta_fine)
        self.alpha = float(alpha)
        self.graph_batch_size = int(graph_batch_size)

        if self.budgetSize < 0:
            raise ValueError("budgetSize must be non-negative.")
        if self.delta_coarse < 0 or self.delta_fine < 0:
            raise ValueError("TopoCover radii must be non-negative.")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1].")
        if self.graph_batch_size <= 0:
            raise ValueError("graph_batch_size must be positive.")

        requested_device = str(device)
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        self.dataset_name = self._resolve_dataset_name(cfg)
        self.seed = int(getattr(cfg, "RNG_SEED", 0))

        self.selection_metadata = {
            "strategy": "multiscale_topocover",
            "delta_coarse": float(self.delta_coarse),
            "delta_fine": float(self.delta_fine),
            "alpha": float(self.alpha),
            "graph_batch_size": int(self.graph_batch_size),
            "device": str(self.device),
        }

        all_features = self._load_features()
        self.relevant_indices = np.concatenate([self.lSet, self.uSet])
        self.num_labelled = len(self.lSet)
        self.num_relevant = len(self.relevant_indices)

        if self.num_relevant == 0:
            self.features = np.empty((0, 0), dtype=np.float32)
            self.coarse_neighbours = []
            self.fine_neighbours = []
            return

        if np.any(self.relevant_indices < 0):
            raise ValueError("lSet and uSet must contain non-negative indices.")
        if int(self.relevant_indices.max()) >= len(all_features):
            raise IndexError(
                "An index in lSet/uSet exceeds the number of stored features: "
                "max index={}, feature rows={}.".format(
                    int(self.relevant_indices.max()),
                    len(all_features),
                )
            )

        self.features = np.asarray(
            all_features[self.relevant_indices], dtype=np.float32
        )

        self.coarse_neighbours, self.fine_neighbours = (
            self._build_multiscale_neighbourhoods()
        )

    @staticmethod
    def _resolve_dataset_name(cfg):
        if hasattr(cfg, "DATASET") and hasattr(cfg.DATASET, "NAME"):
            return str(cfg.DATASET.NAME)
        if hasattr(cfg, "TRAIN") and hasattr(cfg.TRAIN, "DATASET"):
            return str(cfg.TRAIN.DATASET)
        raise AttributeError(
            "Could not determine the dataset name from cfg.DATASET.NAME "
            "or cfg.TRAIN.DATASET."
        )

    def _load_features(self):
        """
        Load the frozen dataset features used by the other geometry-based
        acquisition strategies in the project.
        """
        try:
            features = ds_utils.load_features(
                self.dataset_name,
                self.seed,
                normalized=True,
            )
        except TypeError:
            # Compatibility with older load_features signatures.
            features = ds_utils.load_features(self.dataset_name, self.seed)

        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(
                "Expected a 2-D feature matrix, received shape {}.".format(
                    features.shape
                )
            )
        return features

    @torch.no_grad()
    def _build_multiscale_neighbourhoods(self):
        """
        Construct threshold-neighbour lists for both scales.

        Distances are evaluated in batches to avoid materialising the full
        pairwise distance matrix. The coarse and fine neighbourhoods include
        each point itself when the corresponding radius is non-negative.
        """
        n = self.num_relevant
        x_all = torch.as_tensor(
            self.features,
            dtype=torch.float32,
            device=self.device,
        )

        coarse = [np.empty(0, dtype=np.int32) for _ in range(n)]
        fine = [np.empty(0, dtype=np.int32) for _ in range(n)]

        for start in range(0, n, self.graph_batch_size):
            stop = min(start + self.graph_batch_size, n)
            distances = torch.cdist(x_all[start:stop], x_all, p=2)

            coarse_mask = distances <= self.delta_coarse
            fine_mask = distances <= self.delta_fine

            for local_row, global_row in enumerate(range(start, stop)):
                coarse[global_row] = (
                    torch.nonzero(coarse_mask[local_row], as_tuple=False)
                    .flatten()
                    .cpu()
                    .numpy()
                    .astype(np.int32, copy=False)
                )
                fine[global_row] = (
                    torch.nonzero(fine_mask[local_row], as_tuple=False)
                    .flatten()
                    .cpu()
                    .numpy()
                    .astype(np.int32, copy=False)
                )

            del distances, coarse_mask, fine_mask

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return coarse, fine

    @staticmethod
    def _mark_covered(covered, centres, neighbours):
        for centre in centres:
            covered[neighbours[int(centre)]] = True

    @staticmethod
    def _component_labels(uncovered, neighbours):
        """
        Label connected components in the graph induced by uncovered vertices.

        Vertices outside the uncovered set retain label -1.
        """
        n = len(uncovered)
        labels = np.full(n, -1, dtype=np.int32)
        component_id = 0

        for root in np.flatnonzero(uncovered):
            root = int(root)
            if labels[root] != -1:
                continue

            labels[root] = component_id
            queue = deque([root])

            while queue:
                current = queue.popleft()
                for neighbour in neighbours[current]:
                    neighbour = int(neighbour)
                    if uncovered[neighbour] and labels[neighbour] == -1:
                        labels[neighbour] = component_id
                        queue.append(neighbour)

            component_id += 1

        return labels, component_id

    @staticmethod
    def _components_touched(candidate, component_labels, neighbours):
        labels = component_labels[neighbours[candidate]]
        labels = labels[labels >= 0]
        if labels.size == 0:
            return 0
        return int(np.unique(labels).size)

    def _score_candidates(self, available_candidates, coarse_labels, fine_labels):
        scores = np.zeros(len(available_candidates), dtype=np.float32)

        for position, candidate in enumerate(available_candidates):
            candidate = int(candidate)

            coarse_gain = self._components_touched(
                candidate,
                coarse_labels,
                self.coarse_neighbours,
            )
            fine_gain = self._components_touched(
                candidate,
                fine_labels,
                self.fine_neighbours,
            )

            scores[position] = (
                self.alpha * coarse_gain
                + (1.0 - self.alpha) * fine_gain
            )

        return scores

    def select_samples(self):
        """
        Select a query batch and return:

            activeSet: selected global dataset indices
            remainSet: unselected global indices in the original uSet order
        """
        if self.budgetSize == 0 or len(self.uSet) == 0:
            return np.empty(0, dtype=np.int64), self.uSet.copy()

        budget = min(self.budgetSize, len(self.uSet))
        n = self.num_relevant

        covered_coarse = np.zeros(n, dtype=bool)
        covered_fine = np.zeros(n, dtype=bool)

        labelled_relative = np.arange(self.num_labelled, dtype=np.int32)
        self._mark_covered(
            covered_coarse,
            labelled_relative,
            self.coarse_neighbours,
        )
        self._mark_covered(
            covered_fine,
            labelled_relative,
            self.fine_neighbours,
        )

        candidate_relative = np.arange(
            self.num_labelled,
            n,
            dtype=np.int32,
        )
        selected_mask = np.zeros(n, dtype=bool)
        selected_relative = []
        selected_scores = []

        for _ in range(budget):
            available = candidate_relative[~selected_mask[candidate_relative]]
            if available.size == 0:
                break

            uncovered_coarse = ~covered_coarse
            uncovered_fine = ~covered_fine

            coarse_labels, _ = self._component_labels(
                uncovered_coarse,
                self.coarse_neighbours,
            )
            fine_labels, _ = self._component_labels(
                uncovered_fine,
                self.fine_neighbours,
            )

            scores = self._score_candidates(
                available,
                coarse_labels,
                fine_labels,
            )

            # Original-style deterministic first-maximum tie handling.
            best_position = int(np.argmax(scores))
            chosen = int(available[best_position])
            best_score = float(scores[best_position])

            selected_relative.append(chosen)
            selected_scores.append(best_score)
            selected_mask[chosen] = True

            # A selected sample acts as an additional coverage centre at both
            # graph scales for the remaining greedy selections.
            covered_coarse[self.coarse_neighbours[chosen]] = True
            covered_fine[self.fine_neighbours[chosen]] = True

        selected_relative_array = np.asarray(
            selected_relative,
            dtype=np.int64,
        )
        active_set = self.relevant_indices[selected_relative_array]

        selected_global = set(map(int, active_set))
        remain_set = np.asarray(
            [idx for idx in self.uSet if int(idx) not in selected_global],
            dtype=np.int64,
        )

        self.selection_metadata.update({
            "selected_count": int(len(active_set)),
            "coverage_fraction_coarse": float(covered_coarse.mean()) if n else 0.0,
            "coverage_fraction_fine": float(covered_fine.mean()) if n else 0.0,
            "selected_score_mean": float(np.mean(selected_scores)) if selected_scores else 0.0,
            "selected_score_max": float(np.max(selected_scores)) if selected_scores else 0.0,
        })

        print(
            "MultiScaleTopoCover selected {} samples. "
            "delta_c={:.4f} delta_f={:.4f} alpha={:.2f} "
            "coverage_c={:.4f} coverage_f={:.4f}".format(
                len(active_set),
                self.delta_coarse,
                self.delta_fine,
                self.alpha,
                self.selection_metadata["coverage_fraction_coarse"],
                self.selection_metadata["coverage_fraction_fine"],
            )
        )
        return active_set.astype(np.int64, copy=False), remain_set
