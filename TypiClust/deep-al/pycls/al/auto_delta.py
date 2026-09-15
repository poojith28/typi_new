"""Deterministic, label-free base-radius selection for coverage methods."""

import contextlib
import fcntl
import hashlib
import json
import os
import tempfile

import numpy as np

import pycls.datasets.utils as ds_utils
from .IDProbCover import compute_or_load_knn


AUTO_DELTA_SCHEMA = "lidcover_auto_delta_v1"
AUTO_DELTA_RULE = "median_50th_neighbour_distance"


def _sha256_array(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _atomic_write_json(path, payload):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=directory,
        prefix=".auto_delta_",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = handle.name
    os.replace(temporary_path, path)


@contextlib.contextmanager
def _exclusive_lock(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def select_auto_delta_from_knn_distances(knn_distances, quantile=0.5):
    """Return the requested quantile of each point's k-th-neighbour distance."""
    distances = np.asarray(knn_distances, dtype=np.float32)
    if distances.ndim != 2 or distances.shape[1] == 0:
        raise ValueError("kNN distances must be a non-empty two-dimensional array")
    quantile = float(quantile)
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"AUTO_DELTA_QUANTILE must be in [0, 1], got {quantile}")
    kth_distances = distances[:, -1]
    if not np.all(np.isfinite(kth_distances)) or np.any(kth_distances <= 0.0):
        raise ValueError("k-th-neighbour distances must be positive and finite")
    selected = float(np.quantile(kth_distances.astype(np.float64), quantile))
    if not np.isfinite(selected) or selected <= 0.0:
        raise ValueError(f"automatic base radius is invalid: {selected}")
    return selected


def resolve_auto_delta(cfg, candidate_indices):
    """Resolve and cache an unsupervised base radius for one AL candidate pool.

    The rule is the median distance to the 50th nearest neighbour in the
    L2-normalized pretrained representation. Labels, validation accuracy and
    test accuracy are never read. The candidate indices exclude the held-out
    validation split and are content-hashed with the normalized features.
    """
    dataset = str(cfg.DATASET.NAME)
    backbone = str(cfg.MODEL.TYPE).lower()
    seed = int(cfg.RNG_SEED)
    k = int(getattr(cfg.ACTIVE_LEARNING, "AUTO_DELTA_K", 50))
    quantile = float(getattr(cfg.ACTIVE_LEARNING, "AUTO_DELTA_QUANTILE", 0.5))
    cache_root = str(
        getattr(cfg.ACTIVE_LEARNING, "AUTO_DELTA_CACHE_ROOT", "")
        or getattr(cfg.ACTIVE_LEARNING, "IDPC_CACHE_ROOT", "")
        or "./lidcover_auto_delta_cache"
    )
    prefer_faiss = bool(getattr(cfg.ACTIVE_LEARNING, "AUTO_DELTA_PREFER_FAISS", True))
    faiss_gpu = bool(getattr(cfg.ACTIVE_LEARNING, "AUTO_DELTA_FAISS_GPU", True))
    if k <= 0:
        raise ValueError(f"AUTO_DELTA_K must be positive, got {k}")

    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if candidate_indices.ndim != 1 or candidate_indices.size <= k:
        raise ValueError(
            f"automatic delta needs more than k={k} candidate points; "
            f"received {candidate_indices.size}"
        )
    if np.unique(candidate_indices).size != candidate_indices.size:
        raise ValueError("automatic-delta candidate indices contain duplicates")

    all_features = ds_utils.load_features(
        dataset,
        seed,
        train=True,
        normalized=False,
        backbone=backbone,
    ).astype(np.float32, copy=False)
    if candidate_indices.min() < 0 or candidate_indices.max() >= all_features.shape[0]:
        raise ValueError(
            f"candidate index range [{candidate_indices.min()}, {candidate_indices.max()}] "
            f"does not fit feature rows={all_features.shape[0]}"
        )

    candidate_features = np.ascontiguousarray(all_features[candidate_indices], dtype=np.float32)
    norms = np.linalg.norm(candidate_features, axis=1, keepdims=True)
    candidate_features /= norms + 1e-12
    if not np.all(np.isfinite(candidate_features)):
        raise ValueError("normalized candidate features contain non-finite values")

    feature_sha256 = _sha256_array(candidate_features)
    candidate_indices_sha256 = _sha256_array(candidate_indices)
    cache_dir = os.path.join(cache_root, "auto_delta", dataset, f"seed{seed}")
    key = f"k{k}_q{quantile:.6f}_{feature_sha256[:16]}_{candidate_indices_sha256[:16]}"
    provenance_path = os.path.join(cache_dir, f"{key}.json")
    knn_cache_path = os.path.join(cache_dir, f"{key}.npz")
    lock_path = os.path.join(cache_dir, f"{key}.lock")

    expected = {
        "schema": AUTO_DELTA_SCHEMA,
        "rule": AUTO_DELTA_RULE,
        "dataset": dataset,
        "backbone": backbone,
        "requested_seed": seed,
        "candidate_count": int(candidate_indices.size),
        "feature_dimension": int(candidate_features.shape[1]),
        "feature_sha256": feature_sha256,
        "candidate_indices_sha256": candidate_indices_sha256,
        "k": k,
        "quantile": quantile,
        "metric": "euclidean_on_l2_normalized_features",
        "labels_used": False,
        "validation_or_test_accuracy_used": False,
    }

    with _exclusive_lock(lock_path):
        if os.path.exists(provenance_path):
            with open(provenance_path, "r", encoding="utf-8") as handle:
                cached = json.load(handle)
            if all(cached.get(field) == value for field, value in expected.items()):
                selected = float(cached.get("delta_auto", 0.0))
                if np.isfinite(selected) and selected > 0.0:
                    cached["auto_delta_cache_hit"] = True
                    return selected, cached

        _, knn_distances, knn_metadata = compute_or_load_knn(
            candidate_features,
            knn_cache_path,
            k_knn=k,
            prefer_faiss=prefer_faiss,
            faiss_gpu=faiss_gpu,
        )
        selected = select_auto_delta_from_knn_distances(knn_distances, quantile=quantile)
        payload = {
            **expected,
            "delta_auto": selected,
            "knn_cache_path": knn_cache_path,
            "knn_backend": str(knn_metadata.get("backend", "unknown")),
            "auto_delta_cache_hit": False,
        }
        _atomic_write_json(provenance_path, payload)
        return selected, payload


def write_run_provenance(exp_dir, metadata):
    path = os.path.join(exp_dir, "auto_delta_provenance.json")
    _atomic_write_json(path, metadata)
    return path
