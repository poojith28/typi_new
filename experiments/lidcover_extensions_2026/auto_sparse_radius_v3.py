"""Label-free sparse adaptive-graph calibration for v3 coverage runs.

The rule deliberately calibrates a graph property, not a numeric radius.  On
the acquisition-eligible, L2-normalised representation it estimates local
intrinsic dimension from exact 50-neighbour distances and selects the smallest
base radius whose *post-cold-start* adaptive graph has a mean non-self out
degree of at least two.  Labels and validation/test measurements are never
read.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.lidcover_extensions_2026.auto_radius_v2 import (
    _exact_knn,
    _feature_path,
    _git_revision,
    _sha256_array,
    _sha256_file,
)


SCHEMA = "lidcover_auto_sparse_radius_v3"
RULE = "candidate_lid_adaptive_target_mean_degree_v3"
K = 50
TARGET_MEAN_NONSELF_DEGREE = 2.0
POST_COLD_START_SCALE = 0.85
LID_EPSILON = 1e-12


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(str(temporary), str(path))


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w+b", suffix=".npz", dir=str(path.parent), delete=False) as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(str(temporary), str(path))


@contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_or_compute_knn(features, cache_path, metadata_path, lock_path, expected):
    with _exclusive_lock(lock_path):
        if cache_path.is_file() and metadata_path.is_file():
            try:
                cached_metadata = json.loads(metadata_path.read_text())
                if all(cached_metadata.get(key) == value for key, value in expected.items()):
                    with np.load(str(cache_path)) as cached:
                        indices = np.asarray(cached["indices"], dtype=np.int32)
                        distances = np.asarray(cached["distances"], dtype=np.float32)
                    if indices.shape == distances.shape == (features.shape[0], K):
                        return indices, distances, str(cached_metadata["knn_backend"]), True
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass

        indices, distances, backend = _exact_knn(features, K)
        if indices.shape != distances.shape or distances.shape != (features.shape[0], K):
            raise RuntimeError("exact kNN returned an unexpected shape")
        _atomic_npz(cache_path, indices=indices, distances=distances)
        _atomic_json(metadata_path, {**expected, "knn_backend": backend})
        return indices, distances, backend, False


def _local_lid_hill(distances):
    """Explicit Hill/MLE local-ID estimate using the kth distance as radius."""
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[1] != K:
        raise ValueError("v3 calibration requires an N by 50 distance matrix")
    if not np.isfinite(distances).all() or np.any(distances < 0.0):
        raise ValueError("nearest-neighbour distances must be finite and non-negative")
    radius = distances[:, -1:]
    log_ratios = np.log((distances[:, :-1] + LID_EPSILON) / (radius + LID_EPSILON))
    denominator = np.sum(log_ratios, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lids = -(K - 1.0) / denominator
    valid = np.isfinite(lids) & (lids > 0.0)
    if not np.any(valid):
        raise RuntimeError("local-ID calibration produced no positive finite estimates")
    replacement = float(np.median(lids[valid]))
    lids[~valid] = replacement
    return lids


def _calibrate(distances):
    lids = _local_lid_hill(distances)
    median_lid = float(np.median(lids))
    relative_lid = lids / (median_lid + LID_EPSILON)

    # A neighbour at distance d is present after cold start precisely when
    # d < (POST_COLD_START_SCALE * delta_base) / relative_lid.  Therefore
    # d * relative_lid / scale is its required base-radius threshold.
    required_base = distances.astype(np.float64) * relative_lid[:, None] / POST_COLD_START_SCALE
    flat = required_base.reshape(-1)
    target_edges = int(round(TARGET_MEAN_NONSELF_DEGREE * distances.shape[0]))
    if target_edges <= 0 or target_edges > flat.size:
        raise RuntimeError("target degree is incompatible with the calibration graph")
    boundary = float(np.partition(flat, target_edges - 1)[target_edges - 1])
    delta_auto = float(np.nextafter(boundary, np.inf))
    degrees = np.sum(required_base < delta_auto, axis=1).astype(np.int32)
    if not np.isfinite(delta_auto) or delta_auto <= 0.0:
        raise RuntimeError("automatic sparse radius is not positive and finite")
    if int(degrees.sum()) < target_edges:
        raise RuntimeError("strict-threshold calibration failed to reach the target edge count")
    return delta_auto, lids, relative_lid, degrees, boundary


def resolve_auto_delta(cfg, candidate_indices):
    """Compatibility hook used by the frozen active-learning driver."""
    dataset = str(cfg.DATASET.NAME)
    backbone = str(cfg.MODEL.TYPE).lower()
    seed = int(cfg.RNG_SEED)
    k = int(getattr(cfg.ACTIVE_LEARNING, "AUTO_DELTA_K", K))
    if k != K:
        raise ValueError("auto-sparse-radius-v3 is frozen to exact k=50")

    candidate_indices = np.ascontiguousarray(candidate_indices, dtype=np.int64)
    if candidate_indices.ndim != 1 or candidate_indices.size <= K:
        raise ValueError("v3 calibration needs more than 50 one-dimensional candidate indices")
    if np.unique(candidate_indices).size != candidate_indices.size:
        raise ValueError("candidate indices are not unique")

    representation_path = _feature_path(dataset, backbone, seed)
    representation = np.load(str(representation_path), mmap_mode="r")
    if candidate_indices.min() < 0 or candidate_indices.max() >= representation.shape[0]:
        raise ValueError("candidate indices are outside the representation matrix")
    features = np.asarray(representation[candidate_indices], dtype=np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
    features = np.ascontiguousarray(features)
    if not np.isfinite(features).all():
        raise ValueError("normalised candidate representation contains non-finite values")

    representation_sha = _sha256_file(representation_path)
    feature_sha = _sha256_array(features)
    candidate_sha = _sha256_array(candidate_indices)
    cache_root = Path(
        str(getattr(cfg.ACTIVE_LEARNING, "AUTO_DELTA_CACHE_ROOT", ""))
        or "/vast/s219110279/experiments/lidcover_extensions_2026/cache/auto_sparse_radius_v3"
    )
    key = f"{dataset}_{backbone}_seed{seed}_k50_{representation_sha[:12]}_{candidate_sha[:12]}"
    cache_path = cache_root / f"{key}.npz"
    metadata_path = cache_root / f"{key}.json"
    lock_path = cache_root / f"{key}.lock"
    expected = {
        "schema": SCHEMA,
        "representation_sha256": representation_sha,
        "normalised_candidate_feature_sha256": feature_sha,
        "candidate_index_sha256": candidate_sha,
        "k": K,
    }
    _, distances, backend, cache_hit = _load_or_compute_knn(
        features, cache_path, metadata_path, lock_path, expected
    )
    delta_auto, lids, relative_lid, degrees, boundary = _calibrate(distances)

    repo = Path(__file__).resolve().parents[2]
    payload = {
        "schema": SCHEMA,
        "rule": RULE,
        "dataset": dataset,
        "backbone": backbone,
        "requested_seed": seed,
        "representation_path": str(representation_path),
        "representation_sha256": representation_sha,
        "feature_sha256": feature_sha,
        "normalised_candidate_feature_sha256": feature_sha,
        "candidate_indices_sha256": candidate_sha,
        "candidate_index_sha256": candidate_sha,
        "candidate_count": int(candidate_indices.size),
        "feature_dimension": int(features.shape[1]),
        "k": K,
        # Retained because the frozen driver logs this compatibility field.
        "quantile": None,
        "delta_auto": delta_auto,
        "strict_order_statistic_boundary": boundary,
        "initial_effective_radius": 1.35 * delta_auto,
        "post_cold_start_effective_radius": POST_COLD_START_SCALE * delta_auto,
        "calibration_phase": "post_label_shrink",
        "calibration_driver_scale": POST_COLD_START_SCALE,
        "target_mean_nonself_out_degree": TARGET_MEAN_NONSELF_DEGREE,
        "target_nonself_edge_count": int(round(TARGET_MEAN_NONSELF_DEGREE * len(degrees))),
        "realised_mean_nonself_out_degree": float(np.mean(degrees)),
        "realised_nonself_edge_count": int(np.sum(degrees)),
        "nonzero_degree_fraction": float(np.mean(degrees > 0)),
        "degree_median": float(np.median(degrees)),
        "degree_p95": float(np.percentile(degrees, 95)),
        "local_lid_estimator": "hill_mle_explicit_k50",
        "local_lid_epsilon": LID_EPSILON,
        "median_local_lid": float(np.median(lids)),
        "relative_lid_median": float(np.median(relative_lid)),
        "metric": "euclidean",
        "feature_normalisation": "rowwise_l2_float32_eps_1e-12",
        "candidate_scope": "acquisition_eligible_train_indices_only",
        "held_out_validation_or_test_included": False,
        "labels_used": False,
        "validation_or_test_accuracy_used": False,
        "knn_exact": True,
        "knn_backend": backend,
        "knn_cache_path": str(cache_path),
        "knn_cache_metadata_path": str(metadata_path),
        "auto_delta_cache_hit": cache_hit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_revision(repo),
        "extension_source_path": str(Path(__file__).resolve()),
        "extension_source_sha256": _sha256_file(Path(__file__).resolve()),
    }
    return delta_auto, payload


def write_run_provenance(exp_dir, metadata):
    path = Path(exp_dir) / "auto_delta_provenance.json"
    _atomic_json(path, metadata)
    return str(path)
