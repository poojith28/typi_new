"""
Per-round persistence summary for a single AL experiment.

Given an experiment directory (e.g.
    output/CIFAR10/resnet18/LIDCOVER_1_50b/
)
this script:

  1. Loads the SimCLR / SCAN feature matrix once.
  2. Walks every episode_<r> subfolder in order.
  3. Computes persistence (H0 + optionally H1) of the labeled set lSet.npy
     at that round (and optionally of the unlabeled set or of the new
     activeSet selection).
  4. Writes one row per (round, scope, dim) to a CSV plus a JSON sidecar
     containing per-round summary statistics (entropy of persistence,
     total persistence, max persistence, count) that can be plotted
     against round / test accuracy / coverage radius.

Why bother?
  These TDA-style summaries (especially `total_persistence` and
  `persistence_entropy` of H0/H1 of the labeled set) act as geometric
  diagnostics of how well an AL strategy fills the feature space, and
  correlate nicely with downstream accuracy gains. They make a strong
  empirical chapter for the thesis.

Example:
    python compute_persistence_per_round.py \\
        --experiment-dir /vast/.../output/CIFAR10/resnet18/LIDCOVER_1_50b \\
        --features       /vast/.../scan/results/cifar-10/pretext/features_seed1.npy \\
        --out            /vast/.../analysis/persistence_rounds/cifar10_lidcover_s1 \\
        --scope labeled --max-dim 1 --max-edge-length 1.0 --normalize
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from _common import (
    RunMeta,
    Timer,
    collect_library_versions,
    find_episode_dirs,
    load_features,
    load_index_array,
    out_prefix_paths,
    save_meta,
)


def _persistence_features(X: np.ndarray, max_dim: int, max_edge_length: float, sparse: float | None):
    import gudhi as gd

    kwargs = dict(points=X, max_edge_length=max_edge_length)
    if sparse is not None and sparse > 0.0:
        kwargs["sparse"] = float(sparse)
    rips = gd.RipsComplex(**kwargs)
    st = rips.create_simplex_tree(max_dimension=max(1, int(max_dim)))
    diag = st.persistence()  # [(dim, (birth, death)), ...]

    per_dim: dict = {d: [] for d in range(max_dim + 1)}
    for d, (b, dth) in diag:
        if d <= max_dim:
            per_dim[d].append((float(b), float(dth)))

    summary: dict = {}
    for d, pairs in per_dim.items():
        if not pairs:
            summary[f"h{d}"] = dict(count=0, count_finite=0, total_persistence=0.0,
                                    max_persistence=0.0, mean_persistence=0.0,
                                    persistence_entropy=0.0)
            continue
        arr = np.asarray(pairs, dtype=np.float64)  # [K, 2]
        finite_mask = np.isfinite(arr[:, 1])
        finite = arr[finite_mask]
        persists = finite[:, 1] - finite[:, 0]
        if persists.size > 0 and persists.sum() > 0:
            p_norm = persists / persists.sum()
            ent = float(-np.sum(p_norm * np.log(np.maximum(p_norm, 1e-12))))
        else:
            ent = 0.0
        summary[f"h{d}"] = dict(
            count=int(arr.shape[0]),
            count_finite=int(finite.shape[0]),
            total_persistence=float(persists.sum()) if persists.size else 0.0,
            max_persistence=float(persists.max()) if persists.size else 0.0,
            mean_persistence=float(persists.mean()) if persists.size else 0.0,
            persistence_entropy=ent,
        )
    return summary, per_dim


def _load_episode_indices(ep_dir: Path, scope: str) -> np.ndarray:
    if scope == "labeled":
        return load_index_array(ep_dir / "lSet.npy")
    if scope == "unlabeled":
        return load_index_array(ep_dir / "uSet.npy")
    if scope == "active":
        return load_index_array(ep_dir / "activeSet.npy")
    raise ValueError(f"Unknown scope {scope!r}")


def _load_episode_summary(ep_dir: Path) -> dict:
    p = ep_dir / "episode_summary.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-dir", required=True,
                        help="A single experiment folder (contains episode_0/ ...)")
    parser.add_argument("--features", required=True,
                        help="Feature matrix .npy for the dataset")
    parser.add_argument("--out", required=True, help="Output prefix")
    parser.add_argument("--scope", choices=["labeled", "unlabeled", "active"], default="labeled",
                        help="Which selection to score at each round")
    parser.add_argument("--max-dim", type=int, default=1)
    parser.add_argument("--max-edge-length", type=float, default=float("inf"))
    parser.add_argument("--sparse", type=float, default=None,
                        help="Sparse Rips epsilon (e.g. 0.3) for memory control")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--max-points", type=int, default=2000,
                        help="Cap per-round point count (random subsample, deterministic per round)")
    parser.add_argument("--rounds", type=int, nargs="*", default=None,
                        help="Optional explicit list of round indices to compute")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir).expanduser().resolve()
    if not exp_dir.is_dir():
        raise SystemExit(f"experiment-dir not found: {exp_dir}")

    paths = out_prefix_paths(args.out)
    features = load_features(args.features, normalize=args.normalize)

    eps = find_episode_dirs(exp_dir)
    if args.rounds is not None:
        keep = set(args.rounds)
        eps = [p for p in eps if int(p.name.split("_", 1)[1]) in keep]

    rng = np.random.default_rng(args.seed)
    rows: List[dict] = []
    per_round_summary: List[dict] = []

    with Timer() as timer:
        for ep_dir in eps:
            r = int(ep_dir.name.split("_", 1)[1])
            try:
                idx = _load_episode_indices(ep_dir, args.scope)
            except FileNotFoundError:
                print(f"  round {r}: missing {args.scope} file, skipping")
                continue
            if idx.size < 3:
                print(f"  round {r}: only {idx.size} points, skipping")
                continue
            sub = idx
            if args.max_points and sub.size > args.max_points:
                sub = np.sort(rng.choice(sub, size=int(args.max_points), replace=False))
            X = features[sub]
            print(f"  round {r}: n={X.shape[0]}")
            summary, per_dim = _persistence_features(
                X, max_dim=args.max_dim, max_edge_length=args.max_edge_length, sparse=args.sparse,
            )
            ep_summary = _load_episode_summary(ep_dir)
            row_base = {
                "round": r,
                "n_points": int(X.shape[0]),
                "n_total_in_scope": int(idx.size),
                "scope": args.scope,
                "test_accuracy": ep_summary.get("test_accuracy"),
                "best_val_accuracy": ep_summary.get("best_val_accuracy"),
                "round_time_sec": ep_summary.get("round_time_sec"),
                "labeled_count_before_sampling": ep_summary.get("labeled_count_before_sampling"),
                "labeled_count_after_sampling": ep_summary.get("labeled_count_after_sampling"),
            }
            for d, pairs in per_dim.items():
                for b, dth in pairs:
                    rows.append({
                        **row_base,
                        "dim": int(d),
                        "birth": float(b),
                        "death": float(dth),
                        "persistence": float(dth - b) if math.isfinite(dth) else math.inf,
                    })
            per_round_summary.append({**row_base, **summary})

    df = pd.DataFrame(rows)
    df.to_csv(paths["csv"], index=False)

    summary_df = pd.json_normalize(per_round_summary, sep="_")
    summary_csv = paths["prefix"].with_name(paths["prefix"].name + "_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    np.save(paths["npy"], df[["round", "dim", "birth", "death"]].to_numpy(dtype=np.float64) if not df.empty else np.zeros((0, 4)))

    meta = RunMeta(
        script=Path(__file__).name,
        params=vars(args),
        inputs={
            "experiment_dir": str(exp_dir),
            "features": str(args.features),
        },
        outputs=[str(paths["csv"]), str(summary_csv), str(paths["npy"])],
        runtime_sec=timer.elapsed,
        library_versions=collect_library_versions(["numpy", "pandas", "gudhi"]),
        notes={"rounds": [r["round"] for r in per_round_summary]},
    )
    save_meta(meta, paths["meta"])

    print(f"Saved long-form persistence CSV: {paths['csv']}  rows={len(df)}")
    print(f"Saved per-round summary CSV:    {summary_csv}  rows={len(summary_df)}")
    print(f"Saved persistence triples npy:  {paths['npy']}")
    print(f"Elapsed: {timer.elapsed:.1f}s")


if __name__ == "__main__":
    main()
