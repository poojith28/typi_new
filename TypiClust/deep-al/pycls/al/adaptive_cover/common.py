import os
import zipfile

import numpy as np


def summary_stats(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std()),
    }


def safe_mkdir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def default_cache_paths(cache_root, dataset, seed, k_signal, k_knn, signal_name):
    base = os.path.join(cache_root, dataset, f"seed{seed}")
    knn_path = os.path.join(base, f"knn_k{k_knn}.npz")
    signal_path = os.path.join(base, f"{signal_name}_k{k_signal}.npy")
    return signal_path, knn_path


def knn_faiss(x, k, use_gpu=True):
    import faiss  # type: ignore

    x = np.asarray(x, dtype=np.float32)
    _, dim = x.shape
    index = faiss.IndexFlatL2(dim)
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            backend = "faiss_gpu"
        except Exception:
            backend = "faiss_cpu"
    else:
        backend = "faiss_cpu"
    index.add(x)
    d2, idx = index.search(x, k + 1)
    idx = idx[:, 1:].astype(np.int32)
    dist = np.sqrt(np.maximum(d2[:, 1:], 0.0)).astype(np.float32)
    return idx, dist, backend


def knn_sklearn(x, k):
    from sklearn.neighbors import NearestNeighbors

    x = np.asarray(x, dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="auto")
    nn.fit(x)
    dist, idx = nn.kneighbors(x, return_distance=True)
    idx = idx[:, 1:].astype(np.int32)
    dist = dist[:, 1:].astype(np.float32)
    return idx, dist, "sklearn"


def compute_or_load_knn(x, knn_cache_path, k_knn=50, prefer_faiss=True, faiss_gpu=True):
    safe_mkdir(os.path.dirname(knn_cache_path))
    if os.path.exists(knn_cache_path):
        try:
            cached = np.load(knn_cache_path)
            idx = cached["idx"].astype(np.int32)
            dist = cached["dist"].astype(np.float32)
        except (OSError, ValueError, zipfile.BadZipFile, EOFError) as exc:
            print(f"[adaptive kNN cache] invalid cache at {knn_cache_path}: {exc}. Recomputing.")
            try:
                os.remove(knn_cache_path)
            except OSError:
                pass
        else:
            if idx.shape[0] != x.shape[0]:
                raise RuntimeError(
                    f"[adaptive kNN cache] N mismatch: cached idx {idx.shape} vs X {x.shape} at {knn_cache_path}"
                )
            if idx.shape[1] != k_knn:
                raise RuntimeError(
                    f"[adaptive kNN cache] k mismatch: cached k={idx.shape[1]} but requested k_knn={k_knn}. "
                    "Delete cache or request the cached k."
                )
            return idx, dist, {
                "cache_hit": True,
                "cache_path": knn_cache_path,
                "backend": "cache",
            }

    idx = dist = None
    backend = None
    if prefer_faiss:
        try:
            idx, dist, backend = knn_faiss(x, k=int(k_knn), use_gpu=faiss_gpu)
        except Exception:
            idx = dist = None
    if idx is None or dist is None:
        idx, dist, backend = knn_sklearn(x, k=int(k_knn))

    np.savez_compressed(knn_cache_path, idx=idx, dist=dist)
    return idx, dist, {
        "cache_hit": False,
        "cache_path": knn_cache_path,
        "backend": backend,
    }


def compute_or_load_signal(signal_cache_path, x_shape0, compute_fn):
    safe_mkdir(os.path.dirname(signal_cache_path))
    if os.path.exists(signal_cache_path):
        signal = np.load(signal_cache_path).astype(np.float32)
        if signal.shape[0] != x_shape0:
            raise RuntimeError(
                f"[adaptive signal cache] shape mismatch: cached {signal.shape} vs expected ({x_shape0},) at {signal_cache_path}"
            )
        return signal, {
            "cache_hit": True,
            "cache_path": signal_cache_path,
        }

    signal = np.asarray(compute_fn(), dtype=np.float32)
    signal = np.nan_to_num(signal, nan=float(np.nanmedian(signal)), posinf=float(np.nanmedian(signal)), neginf=float(np.nanmedian(signal))).astype(np.float32)
    np.save(signal_cache_path, signal)
    return signal, {
        "cache_hit": False,
        "cache_path": signal_cache_path,
    }

