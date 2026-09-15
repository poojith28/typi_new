#!/usr/bin/env python3
"""Exact Levina--Bickel local-ID reference for full-train or pool-only rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from posthoc_graph_diagnostics.id_diagnostics import _levina_bickel_from_sorted_distances
from posthoc_graph_diagnostics.neighbors import knn_query


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--eligible-indices", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    started = time.time()
    faiss.omp_set_num_threads(args.threads)
    all_features = np.load(args.features, mmap_mode="r")
    if args.eligible_indices:
        eligible = np.load(args.eligible_indices, allow_pickle=True)
        if eligible.dtype == object:
            eligible = np.asarray(eligible.tolist())
        eligible = np.sort(np.asarray(eligible, dtype=np.int64).reshape(-1))
        reference_type = "POOL_ONLY_REFERENCE"
    else:
        eligible = np.arange(len(all_features), dtype=np.int64)
        reference_type = "FULL_TRAIN_REFERENCE"
    features = np.asarray(all_features[eligible], dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = np.ascontiguousarray(features / np.maximum(norms, 1e-12), dtype=np.float32)
    print(f"[load] {reference_type} {args.dataset}/{args.backbone}/seed={args.seed} {features.shape}", flush=True)
    distances, _ = knn_query(
        features, k=args.k, metric="euclidean", use_faiss=True,
        self_query=True, approx=False,
    )
    local_id = _levina_bickel_from_sorted_distances(distances, args.k)
    median = float(np.nanmedian(local_id))
    local_id = np.nan_to_num(local_id, nan=median if np.isfinite(median) else 0.0).astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, local_id)
    code_path = Path(__file__).resolve()
    meta = {
        "dataset": args.dataset, "backbone": args.backbone, "seed": args.seed,
        "reference_type": reference_type, "estimator": "Levina-Bickel MLE",
        "k": args.k, "metric": "euclidean", "l2_normalize": True,
        "exact_knn": True, "approximate": False, "n_reference_vertices": len(eligible),
        "feature_path": str(args.features.resolve()), "feature_sha256": file_sha256(args.features),
        "eligible_index_path": str(args.eligible_indices.resolve()) if args.eligible_indices else None,
        "eligible_index_file_sha256": file_sha256(args.eligible_indices) if args.eligible_indices else None,
        "code_path": str(code_path), "code_sha256": file_sha256(code_path),
        "faiss_version": faiss.__version__, "numpy_version": np.__version__,
        "runtime_seconds": time.time() - started,
        "local_id_min": float(local_id.min()), "local_id_median": float(np.median(local_id)),
        "local_id_mean": float(local_id.mean()), "local_id_max": float(local_id.max()),
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"[done] runtime_min={(time.time()-started)/60:.2f}", flush=True)


if __name__ == "__main__":
    main()
