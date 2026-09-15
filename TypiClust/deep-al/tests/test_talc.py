from types import SimpleNamespace
from unittest import mock

import numpy as np

from pycls.al.talc import (
    TrajectoryAdaptiveLIDCover,
    coverage_shortlist,
    multiscale_component_partitions,
    percentile_ranks,
    trajectory_progress,
)


def test_percentile_ranks_average_ties():
    ranks = percentile_ranks([3.0, 1.0, 1.0, 5.0])
    assert np.allclose(ranks, [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0])


def test_trajectory_progress_is_coverage_controlled_and_clipped():
    assert trajectory_progress(0.0, 0.9) == 0.0
    assert np.isclose(trajectory_progress(0.45, 0.9), 0.5)
    assert trajectory_progress(0.95, 0.9) == 1.0


def test_coverage_shortlist_enforces_relative_gain_floor():
    gains = np.asarray([10, 9, 8, 0], dtype=np.int32)
    shortlist, best = coverage_shortlist(gains, np.arange(4), epsilon=0.1)
    assert best == 10
    assert shortlist.tolist() == [0, 1]


def test_multiscale_partitions_follow_mst_death_quantiles():
    # Sparse undirected chain with merge scales 0.1, 0.3, and 0.9.
    knn_idx = np.asarray([[1], [0], [1], [2]], dtype=np.int32)
    knn_dist = np.asarray([[0.1], [0.1], [0.3], [0.9]], dtype=np.float32)
    thresholds, labels = multiscale_component_partitions(
        knn_idx, knn_dist, quantiles=(0.25, 0.75)
    )

    assert np.allclose(thresholds, [0.2, 0.6])
    assert labels[0, 0] == labels[1, 0]
    assert len(np.unique(labels[:, 0])) == 3
    assert labels[0, 1] == labels[1, 1] == labels[2, 1]
    assert len(np.unique(labels[:, 1])) == 2


def test_curriculum_moves_from_low_id_to_underrepresented_component():
    sampler = TrajectoryAdaptiveLIDCover.__new__(TrajectoryAdaptiveLIDCover)
    sampler.rel_ids = np.asarray([1.0, 4.0], dtype=np.float32)
    sampler.id_preference = np.asarray([1.0, 0.0], dtype=np.float32)
    sampler.rel_uncertainty = np.zeros(2, dtype=np.float32)
    sampler.use_topology = True
    sampler.topology_weight = 1.0
    sampler.min_component_size = 1
    sampler.rel_component_labels = np.asarray([[0], [1]], dtype=np.int32)
    sampler.component_sizes = [np.asarray([10, 10], dtype=np.int64)]
    component_counts = [np.asarray([5, 0], dtype=np.int32)]

    sampler.progress = 0.0
    early, _, _ = sampler._select_candidate(np.asarray([0, 1]), component_counts)
    sampler.progress = 1.0
    late, _, _ = sampler._select_candidate(np.asarray([0, 1]), component_counts)

    assert early == 0
    assert late == 1


def test_greedy_selection_respects_configured_coverage_safeguard():
    sampler = TrajectoryAdaptiveLIDCover.__new__(TrajectoryAdaptiveLIDCover)
    sampler.cfg = SimpleNamespace()
    sampler.Nr = 4
    sampler.L = 0
    sampler.budgetSize = 2
    sampler.uSet = np.arange(4, dtype=np.int64)
    sampler.relevant_indices = np.arange(4, dtype=np.int64)
    sampler.rel_ids = np.asarray([4.0, 1.0, 2.0, 3.0], dtype=np.float32)
    sampler.id_preference = 1.0 - percentile_ranks(sampler.rel_ids)
    sampler.delta_per_x = np.ones(4, dtype=np.float32)
    sampler.progress = 0.0
    sampler.alpha_t = 1.0
    sampler.coverage_epsilon = 0.25
    sampler.topology_weight = 0.5
    sampler.use_topology = False
    sampler.rel_component_labels = np.empty((4, 0), dtype=np.int32)
    sampler.component_sizes = []
    sampler.rel_uncertainty = np.zeros(4, dtype=np.float32)
    sampler.out_neighbors = [
        np.asarray([0, 1, 2, 3], dtype=np.int32),
        np.asarray([0, 1, 2], dtype=np.int32),
        np.asarray([2], dtype=np.int32),
        np.asarray([3], dtype=np.int32),
    ]
    sampler.in_sources = [
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([0, 1, 2], dtype=np.int32),
        np.asarray([0, 3], dtype=np.int32),
    ]
    sampler.selection_metadata = {}

    active, remain = sampler.select_samples()

    # Candidate 1 has lower ID and exactly 75% of the maximum initial gain.
    assert active[0] == 1
    assert len(active) == 2
    assert len(remain) == 2
    assert sampler.selection_metadata["coverage_safeguard_ratio_min"] >= 0.75


def test_constructor_builds_dynamic_curriculum_without_optional_signals(tmp_path):
    features = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32
    )
    ids = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    knn_idx = np.asarray(
        [[1, 2], [0, 2], [3, 1], [2, 1]], dtype=np.int32
    )
    knn_dist = np.asarray(
        [[0.1, 0.8], [0.1, 0.7], [0.1, 0.7], [0.1, 0.8]], dtype=np.float32
    )
    cfg = {"DATASET": {"NAME": "TOY"}, "RNG_SEED": 1}

    with mock.patch("pycls.al.talc.ds_utils.load_features", return_value=features), mock.patch(
        "pycls.al.talc.compute_or_load_ids_mle",
        return_value=(ids, {"cache_hit": False}),
    ), mock.patch(
        "pycls.al.talc.compute_or_load_knn",
        return_value=(knn_idx, knn_dist, {"cache_hit": False}),
    ):
        sampler = TrajectoryAdaptiveLIDCover(
            cfg=cfg,
            lSet=np.asarray([0]),
            uSet=np.asarray([1, 2, 3]),
            budgetSize=1,
            delta0=0.5,
            cache_root=str(tmp_path),
            k_id=2,
            k_knn=2,
            use_topology=False,
            use_uncertainty=False,
        )

    assert sampler.alpha_t <= sampler.alpha_max
    assert sampler.delta_per_x.shape == (4,)
    active, remain = sampler.select_samples()
    assert len(active) == 1
    assert len(remain) == 2
