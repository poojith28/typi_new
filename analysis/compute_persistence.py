"""
Compute Vietoris-Rips persistence (H0 and optionally H1+) of a feature matrix.

This is a generalised, thesis-friendly version of the user's original
generate_h0_persistence_csv.py:

  * supports H0, H1, ... up to --max-dim (default H0+H1)
  * supports class-stratified subsampling so it runs on 50k-point pools
  * supports sparse Rips (epsilon parameter) for memory control
  * exposes per-class persistence (`--per-class`)
  * always writes a tidy CSV + an .npy with (dim, birth, death) triples,
    plus a summary _meta.json with TDA features that are useful as covariates
    in tables and plots (total persistence, persistence entropy, max
    persistence, ...).

Typical thesis use:

    python compute_persistence.py \\
        --features /vast/.../features_seed1.npy \\
        --labels   /vast/.../labels_train.npy \\
        --out      /vast/.../analysis/persistence/cifar10_simclr \\
        --max-dim  1 --subsample 5000 --normalize

CSV columns:
    dim, birth, death, persistence, birth_simplex, death_simplex,
    [class_id, class_name] if --per-class
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from _common import (
    CIFAR10_CLASSES,
    RunMeta,
    Timer,
    collect_library_versions,
    load_features,
    load_index_array,
    maybe_load_labels,
    out_prefix_paths,
    save_meta,
    subsample_indices,
)


# ---------------------------------------------------------------------------

def _simplex_to_str(simplex) -> str:
    if simplex is None:
        return "[]"
    return "[" + ", ".join(str(int(x)) for x in simplex) + "]"


def _remap_simplex(simplex, original_indices) -> List[int]:
    if simplex is None:
        return []
    return [int(original_indices[int(v)]) for v in simplex]


def _persistence_summary(rows: List[dict], dim: int) -> dict:
    """TDA features useful for thesis tables (per dimension)."""
    in_dim = [r for r in rows if r["dim"] == dim]
    empty = {
        "count": 0,
        "count_finite": 0,
        "total_persistence": 0.0,
        "max_persistence": 0.0,
        "mean_persistence": 0.0,
        "persistence_entropy": 0.0,
    }
    if not in_dim:
        return empty
    df = pd.DataFrame(in_dim)
    finite = df[np.isfinite(df["death"])].copy()
    if finite.empty:
        empty["count"] = int(df.shape[0])
        return empty
    persists = finite["persistence"].to_numpy(dtype=np.float64)
    total = float(persists.sum())
    p_norm = persists / total if total > 0 else persists
    entropy = float(-np.sum(p_norm * np.log(np.maximum(p_norm, 1e-12)))) if total > 0 else 0.0
    return {
        "count": int(df.shape[0]),
        "count_finite": int(finite.shape[0]),
        "total_persistence": total,
        "max_persistence": float(persists.max()),
        "mean_persistence": float(persists.mean()),
        "persistence_entropy": entropy,
    }


def _compute_one(
    X: np.ndarray,
    original_indices: np.ndarray,
    max_dim: int,
    max_edge_length: float,
    sparse: Optional[float],
    label_value: Optional[int] = None,
    label_name: Optional[str] = None,
) -> List[dict]:
    """Compute persistence of a single feature subset; return tidy rows."""
    import gudhi as gd

    rips_kwargs = dict(points=X, max_edge_length=max_edge_length)
    if sparse is not None and sparse > 0.0:
        rips_kwargs["sparse"] = float(sparse)
    rips = gd.RipsComplex(**rips_kwargs)
    simplex_tree = rips.create_simplex_tree(max_dimension=max(1, int(max_dim)))

    simplex_tree.compute_persistence()
    pairs = simplex_tree.persistence_pairs()

    rows: List[dict] = []
    for birth_simplex, death_simplex in pairs:
        birth_dim = len(birth_simplex) - 1
        if birth_dim > max_dim:
            continue
        birth = float(simplex_tree.filtration(birth_simplex))
        if len(death_simplex) == 0:
            death = math.inf
            persistence = math.inf
            death_mapped: List[int] = []
        else:
            death = float(simplex_tree.filtration(death_simplex))
            persistence = death - birth
            death_mapped = _remap_simplex(death_simplex, original_indices)
        birth_mapped = _remap_simplex(birth_simplex, original_indices)
        row = {
            "dim": int(birth_dim),
            "birth": birth,
            "death": death,
            "persistence": persistence,
            "birth_simplex": _simplex_to_str(birth_mapped),
            "death_simplex": _simplex_to_str(death_mapped),
        }
        if label_value is not None:
            row["class_id"] = int(label_value)
            row["class_name"] = label_name
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--indices", default=None,
                        help="Optional .npy mapping feature rows back to original dataset indices")
    parser.add_argument("--label-names", default=None)
    parser.add_argument("--dataset", choices=["CIFAR10", "CIFAR100", "TINYIMAGENET", "OTHER"], default="OTHER")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-dim", type=int, default=1, help="Largest homology dimension to record")
    parser.add_argument("--max-edge-length", type=float, default=float("inf"),
                        help="Maximum Rips edge length")
    parser.add_argument("--sparse", type=float, default=None,
                        help="Sparse Rips epsilon (e.g. 0.3). Reduces memory at the cost of approximation.")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize feature rows first")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Class-stratified subsample size if --labels given, otherwise random.")
    parser.add_argument("--per-class", action="store_true",
                        help="Compute persistence separately per class as well")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    paths = out_prefix_paths(args.out)

    X = load_features(args.features, normalize=args.normalize)
    labels = maybe_load_labels(args.labels)
    if args.indices is not None:
        ds_indices = load_index_array(args.indices)
    else:
        ds_indices = np.arange(X.shape[0], dtype=np.int64)

    sub_idx = subsample_indices(X.shape[0], args.subsample, seed=args.seed, stratify_labels=labels)
    if sub_idx.shape[0] != X.shape[0]:
        X = X[sub_idx]
        if labels is not None:
            labels = labels[sub_idx]
        ds_indices = ds_indices[sub_idx]
        print(f"Sub-sampled to {X.shape[0]} points")

    name_lookup: Optional[List[str]] = None
    if args.label_names:
        name_lookup = [s.strip() for s in args.label_names.split(",")]
    elif args.dataset == "CIFAR10":
        name_lookup = list(CIFAR10_CLASSES)

    rows: List[dict] = []
    per_class_summaries = []

    with Timer() as timer:
        print(f"Computing persistence on n={X.shape[0]}, d={X.shape[1]}, max_dim={args.max_dim} ...")
        pooled = _compute_one(
            X,
            ds_indices,
            max_dim=args.max_dim,
            max_edge_length=args.max_edge_length,
            sparse=args.sparse,
            label_value=None,
        )
        for r in pooled:
            r["scope"] = "all"
        rows.extend(pooled)

        if args.per_class:
            if labels is None:
                raise SystemExit("--per-class requires --labels")
            for cls in np.unique(labels):
                mask = labels == cls
                if mask.sum() < 3:
                    continue
                cname = name_lookup[int(cls)] if name_lookup and int(cls) < len(name_lookup) else str(int(cls))
                print(f"  class {int(cls)} ({cname}): n={int(mask.sum())}")
                per_cls = _compute_one(
                    X[mask],
                    ds_indices[mask],
                    max_dim=args.max_dim,
                    max_edge_length=args.max_edge_length,
                    sparse=args.sparse,
                    label_value=int(cls),
                    label_name=cname,
                )
                for r in per_cls:
                    r["scope"] = f"class_{int(cls)}"
                rows.extend(per_cls)
                per_class_summaries.append({
                    "class_id": int(cls),
                    "class_name": cname,
                    **{f"h{d}": _persistence_summary(per_cls, d) for d in range(args.max_dim + 1)},
                })

    df = pd.DataFrame(rows)
    finite_df = df[np.isfinite(df["death"])].copy().sort_values(["scope", "dim", "death"]).reset_index(drop=True)
    finite_df.to_csv(paths["csv"], index=False)

    if (~np.isfinite(df["death"])).any():
        inf_csv = paths["prefix"].with_name(paths["prefix"].name + "_with_infinite.csv")
        df.sort_values(["scope", "dim", "birth"]).to_csv(inf_csv, index=False)
        print(f"Also saved full CSV (including infinite class) to: {inf_csv}")

    triples = finite_df[["dim", "birth", "death"]].to_numpy(dtype=np.float64)
    np.save(paths["npy"], triples)

    summary = {f"h{d}": _persistence_summary([r for r in rows if r.get("scope") == "all"], d)
               for d in range(args.max_dim + 1)}

    meta = RunMeta(
        script=Path(__file__).name,
        params=vars(args),
        inputs={
            "features": str(args.features),
            "labels": str(args.labels) if args.labels else None,
            "indices": str(args.indices) if args.indices else None,
        },
        outputs=[str(paths["npy"]), str(paths["csv"])],
        runtime_sec=timer.elapsed,
        library_versions=collect_library_versions(["numpy", "pandas", "gudhi"]),
        notes={
            "n": int(X.shape[0]),
            "d": int(X.shape[1]),
            "rows_total": int(df.shape[0]),
            "rows_finite": int(finite_df.shape[0]),
            "summary": summary,
            "per_class_summaries": per_class_summaries,
        },
    )
    save_meta(meta, paths["meta"])

    print(f"Saved persistence triples: {paths['npy']}  rows={triples.shape[0]}")
    print(f"Saved CSV:                 {paths['csv']}  rows={len(finite_df)}")
    print(f"Saved meta:                {paths['meta']}")
    for d, s in summary.items():
        print(f"  {d}: count={s['count']}, finite={s['count_finite']}, total={s['total_persistence']:.4f}, "
              f"max={s['max_persistence']:.4f}, entropy={s['persistence_entropy']:.3f}")
    print(f"Elapsed: {timer.elapsed:.1f}s")


if __name__ == "__main__":
    main()
