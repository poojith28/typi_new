#!/usr/bin/env python3
"""Exact pool-only H0 reference with original embedding-row vertex labels.

This preserves the historical convention: L2-normalised fixed features,
Euclidean complete-graph MST via the repository Prim implementation, followed
by GUDHI SimplexTree H0 persistence.  The only change is restricting vertices
to the supplied acquisition-eligible index array.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import gudhi as gd
import numpy as np
import pandas as pd

from compute_h0_reference_mst import l2_normalize, persistence_summary, prim_mst


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def simplex_text(simplex) -> str:
    return "[" + ", ".join(str(int(value)) for value in simplex) + "]"


def h0_rows(u: np.ndarray, v: np.ndarray, w: np.ndarray, original_ids: np.ndarray) -> list[dict]:
    tree = gd.SimplexTree()
    for vertex in original_ids:
        tree.insert([int(vertex)], filtration=0.0)
    for local_u, local_v, weight in zip(u, v, w):
        tree.insert(
            [int(original_ids[int(local_u)]), int(original_ids[int(local_v)])],
            filtration=float(weight),
        )
    tree.make_filtration_non_decreasing()
    tree.compute_persistence()
    rows: list[dict] = []
    for birth_simplex, death_simplex in tree.persistence_pairs():
        if len(birth_simplex) != 1:
            continue
        birth = float(tree.filtration(birth_simplex))
        death = float("inf") if not death_simplex else float(tree.filtration(death_simplex))
        rows.append({
            "dim": 0,
            "birth": birth,
            "death": death,
            "persistence": float("inf") if not np.isfinite(death) else death - birth,
            "birth_simplex": simplex_text(birth_simplex),
            "death_simplex": simplex_text(death_simplex),
            "scope": "pool_only_reference",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--eligible-indices", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output prefix without extension")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    started = time.time()
    all_features = np.load(args.features, mmap_mode="r")
    eligible = np.load(args.eligible_indices, allow_pickle=True)
    if eligible.dtype == object:
        eligible = np.asarray(eligible.tolist())
    eligible = np.sort(np.asarray(eligible, dtype=np.int64).reshape(-1))
    if len(np.unique(eligible)) != len(eligible):
        raise ValueError("eligible indices contain duplicates")
    if eligible.min(initial=0) < 0 or eligible.max(initial=-1) >= len(all_features):
        raise ValueError("eligible indices are outside feature rows")
    features = l2_normalize(np.asarray(all_features[eligible], dtype=np.float64))
    print(f"[load] {args.dataset}/{args.backbone}/seed{args.seed}: {features.shape}", flush=True)

    u, v, weights = prim_mst(features, assume_unit=True)
    rows = h0_rows(u, v, weights, eligible)
    frame = pd.DataFrame(rows)
    finite = frame[np.isfinite(frame["death"])].copy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    finite.to_csv(Path(str(args.out) + ".csv"), index=False)
    frame.to_csv(Path(str(args.out) + "_with_infinite.csv"), index=False)
    np.save(Path(str(args.out) + ".npy"), finite[["dim", "birth", "death"]].to_numpy(np.float64))
    np.savez_compressed(
        Path(str(args.out) + "_mst_edges.npz"),
        u_original=eligible[u], v_original=eligible[v], weight=weights,
    )
    code_path = Path(__file__).resolve()
    meta = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "seed": args.seed,
        "reference_type": "POOL_ONLY_REFERENCE",
        "method": "exact Euclidean MST via historical Prim + GUDHI SimplexTree H0",
        "metric": "euclidean",
        "l2_normalize": True,
        "feature_path": str(args.features.resolve()),
        "feature_sha256": file_sha256(args.features),
        "eligible_index_path": str(args.eligible_indices.resolve()),
        "eligible_index_file_sha256": file_sha256(args.eligible_indices),
        "eligible_indices_canonical_sha256": ids_sha256(eligible),
        "n_feature_rows": int(len(all_features)),
        "n_reference_vertices": int(len(eligible)),
        "n_finite_h0_pairs": int(len(finite)),
        "code_path": str(code_path),
        "code_sha256": file_sha256(code_path),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "gudhi_version": gd.__version__,
        "runtime_seconds": time.time() - started,
        "summary": persistence_summary(rows),
    }
    Path(str(args.out) + "_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] finite={len(finite)} runtime_min={(time.time()-started)/60:.2f}", flush=True)


if __name__ == "__main__":
    main()
