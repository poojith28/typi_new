"""Label-free automatic base-radius calibration with content-addressed evidence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCHEMA = "lidcover_auto_radius_v2"
RULE = "median_exact_50th_neighbour_euclidean_l2_normalized_v2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(str(temporary), str(path))


def _feature_path(dataset: str, backbone: str, seed: int) -> Path:
    dataset_slug = {
        "CIFAR10": "cifar-10",
        "CIFAR100": "cifar-100",
        "TINYIMAGENET": "tiny-imagenet",
    }[dataset]
    root = Path(os.environ.get("TYPI_FEATURES_ROOT", "/vast/s219110279/results/results"))
    requested = root / backbone / dataset_slug / "pretext" / f"features_seed{seed}.npy"
    fallback = root / backbone / dataset_slug / "pretext" / "features_seed1.npy"
    if requested.is_file():
        return requested.resolve()
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(f"missing representation; tried {requested} and {fallback}")


def _exact_knn(features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Return exact non-self neighbours using FAISS Flat or sklearn brute force."""
    try:
        import faiss  # type: ignore

        index = faiss.IndexFlatL2(features.shape[1])
        backend = "faiss_indexflatl2_cpu_exact"
        if bool(int(os.environ.get("LIDCOVER_AUTO_RADIUS_FAISS_GPU", "1"))):
            try:
                resources = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(resources, 0, index)
                backend = "faiss_indexflatl2_gpu_exact"
            except Exception:
                pass
        index.add(features)
        squared, indices = index.search(features, k + 1)
        return (
            indices[:, 1:].astype(np.int32),
            np.sqrt(np.maximum(squared[:, 1:], 0.0)).astype(np.float32),
            backend,
        )
    except ImportError:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="brute")
        nn.fit(features)
        distances, indices = nn.kneighbors(features, return_distance=True)
        return indices[:, 1:].astype(np.int32), distances[:, 1:].astype(np.float32), "sklearn_brute_exact"


def _git_revision(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def resolve_auto_delta(cfg, candidate_indices):
    """Compatibility hook used by the frozen active-learning driver."""
    dataset = str(cfg.DATASET.NAME)
    backbone = str(cfg.MODEL.TYPE).lower()
    seed = int(cfg.RNG_SEED)
    k = int(getattr(cfg.ACTIVE_LEARNING, "AUTO_DELTA_K", 50))
    quantile = float(getattr(cfg.ACTIVE_LEARNING, "AUTO_DELTA_QUANTILE", 0.5))
    if k != 50 or quantile != 0.5:
        raise ValueError("auto-radius-v2 is frozen to k=50 and the median (quantile=0.5)")

    candidate_indices = np.ascontiguousarray(candidate_indices, dtype=np.int64)
    if candidate_indices.ndim != 1 or candidate_indices.size <= k:
        raise ValueError(f"need more than {k} one-dimensional candidate indices")
    if np.unique(candidate_indices).size != candidate_indices.size:
        raise ValueError("candidate indices are not unique")

    representation_path = _feature_path(dataset, backbone, seed)
    representation = np.load(representation_path, mmap_mode="r")
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
        or "/vast/s219110279/experiments/lidcover_extensions_2026/cache/auto_radius_v2"
    )
    key = f"{dataset}_{backbone}_seed{seed}_k50_{representation_sha[:12]}_{candidate_sha[:12]}"
    cache_path = cache_root / f"{key}.npz"
    cache_metadata_path = cache_root / f"{key}.json"
    cache_root.mkdir(parents=True, exist_ok=True)

    distances = None
    backend = None
    cache_hit = False
    if cache_path.is_file() and cache_metadata_path.is_file():
        cached_meta = json.loads(cache_metadata_path.read_text())
        expected = {
            "schema": SCHEMA,
            "representation_sha256": representation_sha,
            "normalised_candidate_feature_sha256": feature_sha,
            "candidate_index_sha256": candidate_sha,
            "k": 50,
        }
        if all(cached_meta.get(key_) == value for key_, value in expected.items()):
            cached = np.load(cache_path)
            distances = np.asarray(cached["distances"], dtype=np.float32)
            if distances.shape != (candidate_indices.size, 50):
                distances = None
            else:
                backend = str(cached_meta.get("knn_backend", "verified_cache"))
                cache_hit = True

    if distances is None:
        indices, distances, backend = _exact_knn(features, 50)
        if indices.shape != distances.shape or distances.shape != (candidate_indices.size, 50):
            raise RuntimeError("exact kNN returned an unexpected shape")
        np.savez_compressed(cache_path, indices=indices, distances=distances)
        _atomic_json(cache_metadata_path, {
            "schema": SCHEMA,
            "representation_sha256": representation_sha,
            "normalised_candidate_feature_sha256": feature_sha,
            "candidate_index_sha256": candidate_sha,
            "k": 50,
            "knn_backend": backend,
        })

    kth = distances[:, 49].astype(np.float64)
    if not np.isfinite(kth).all() or np.any(kth <= 0):
        raise RuntimeError("invalid 50th-neighbour distances")
    delta_auto = float(np.median(kth))
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
        "k": 50,
        "quantile": 0.5,
        "delta_auto": delta_auto,
        "initial_effective_radius": 1.35 * delta_auto,
        "post_cold_start_effective_radius": 0.85 * delta_auto,
        "metric": "euclidean",
        "feature_normalisation": "rowwise_l2_float32_eps_1e-12",
        "candidate_scope": "acquisition_eligible_train_indices_only",
        "held_out_validation_or_test_included": False,
        "labels_used": False,
        "validation_or_test_accuracy_used": False,
        "knn_exact": True,
        "knn_backend": backend,
        "knn_cache_path": str(cache_path),
        "knn_cache_metadata_path": str(cache_metadata_path),
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
