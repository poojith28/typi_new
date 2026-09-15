from types import SimpleNamespace

import numpy as np

from pycls.al.idpc_tiebreak.methods import IDProbCoverFallbackRandom
from pycls.al.idpc_tiebreak.methods import IDProbCoverMinIDTieBreak


def _handcrafted_sampler(cls):
    sampler = cls.__new__(cls)
    sampler.cfg = SimpleNamespace(
        ACTIVE_LEARNING=SimpleNamespace(IDPC_LOG_CSV=""),
        EXP_DIR=".",
    )
    sampler.Nr = 4
    sampler.L = 0
    sampler.budgetSize = 3
    sampler.seed = 7
    sampler.rng = np.random.default_rng(sampler.seed)
    sampler.tie_break_mode = "min_id"
    sampler.fallback_mode = "random" if cls is IDProbCoverFallbackRandom else "min_id"
    sampler.rel_ids = np.asarray([4.0, 1.0, 2.0, 3.0], dtype=np.float32)
    sampler.delta_per_x = np.ones(4, dtype=np.float32)
    sampler.relevant_indices = np.arange(4, dtype=np.int64)
    sampler.lSet = np.asarray([], dtype=np.int64)
    sampler.uSet = np.arange(4, dtype=np.int64)
    sampler.out_neighbors = [
        np.asarray([0, 1, 2, 3], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
        np.asarray([2], dtype=np.int32),
        np.asarray([3], dtype=np.int32),
    ]
    sampler.in_sources = [
        np.asarray([0], dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([0, 2], dtype=np.int32),
        np.asarray([0, 3], dtype=np.int32),
    ]
    sampler.delta0 = 0.25
    sampler.alpha = 1.0
    sampler.k_id = 50
    sampler.k_knn = 50
    sampler.selection_metadata = {}
    return sampler


def test_fallback_random_keeps_min_id_positive_gain_ties():
    sampler = _handcrafted_sampler(IDProbCoverFallbackRandom)
    candidates = np.asarray([0, 1, 2], dtype=np.int32)
    assert sampler._pick_from_tied_candidates(candidates) == 1


def test_fallback_diagnostics_count_only_zero_gain_batch_filling():
    sampler = _handcrafted_sampler(IDProbCoverFallbackRandom)
    active, remain = sampler.select_samples()

    assert active[0] == 0
    assert len(active) == 3
    assert len(remain) == 1
    assert sampler.selection_metadata["positive_gain_count"] == 1
    assert sampler.selection_metadata["zero_gain_fallback_count"] == 2
    assert sampler.selection_metadata["zero_gain_fallback_fraction"] == 2.0 / 3.0
    assert sampler.selection_metadata["first_zero_gain_selection_index"] == 1
    assert sampler.selection_metadata["first_zero_gain_selection_number"] == 2


def test_min_id_diagnostic_control_retains_historical_fallback():
    sampler = _handcrafted_sampler(IDProbCoverMinIDTieBreak)
    active, _ = sampler.select_samples()

    assert active.tolist() == [0, 1, 2]
    assert sampler.fallback_mode == "min_id"
    assert sampler.selection_metadata["zero_gain_fallback_count"] == 2
