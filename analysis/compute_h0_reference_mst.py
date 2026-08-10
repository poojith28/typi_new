#!/usr/bin/env python3
"""
Exact H0 reference persistence diagram for a fixed embedding pool.

For H0, Vietoris–Rips persistence is equivalent to the Euclidean MST:
finite death values = MST edge weights. Building the full Rips complex at
N≈100k is intractable, so we:

  1. Compute the exact Euclidean MST via Prim (O(N^2) time, O(N) memory)
  2. Insert vertices + MST edges into a GUDHI SimplexTree
  3. Run GUDHI persistence to obtain birth/death simplices

This matches dense GUDHI VR H0 on CIFAR-scale pools (validated) and scales
to TinyImageNet (N=100k) with ~500G RAM unnecessary — only O(N) memory.

Outputs (same layout as compute_persistence.py):
  <out>.csv / <out>_with_infinite.csv / <out>.npy / <out>_meta.json

CSV columns:
  dim, birth, death, persistence, birth_simplex, death_simplex, scope
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _simplex_to_str(simplex) -> str:
    if simplex is None:
        return "[]"
    return "[" + ", ".join(str(int(x)) for x in simplex) + "]"


def l2_normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    return X / nrm


def prim_mst(X: np.ndarray, assume_unit: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact Euclidean MST via Prim. Returns (u, v, weight) length N-1.

    For L2-normalised rows, ||x-y||^2 = 2 - 2 x·y, so each Prim update is one
    matrix–vector product (BLAS) instead of an explicit broadcast subtract.
    """
    n, d = X.shape
    if n < 2:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )

    X = np.ascontiguousarray(X, dtype=np.float64)
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    parent = np.zeros(n, dtype=np.int64)

    if assume_unit:
        dots = X @ X[0]
        min_d = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * dots))
    else:
        min_d = np.linalg.norm(X - X[0], axis=1)
    min_d[0] = np.inf

    u_out = np.empty(n - 1, dtype=np.int64)
    v_out = np.empty(n - 1, dtype=np.int64)
    w_out = np.empty(n - 1, dtype=np.float64)

    step = max(1, (n - 1) // 20)
    t0 = time.time()
    for k in range(n - 1):
        v = int(np.argmin(min_d))
        u_out[k] = int(parent[v])
        v_out[k] = v
        w_out[k] = float(min_d[v])
        in_tree[v] = True
        min_d[v] = np.inf

        if assume_unit:
            dots = X @ X[v]
            dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * dots))
        else:
            diff = X - X[v]
            dist = np.sqrt(np.einsum("ij,ij->i", diff, diff, optimize=True))
        upd = (~in_tree) & (dist < min_d)
        min_d[upd] = dist[upd]
        parent[upd] = v

        if (k + 1) % step == 0 or k == n - 2:
            elapsed = time.time() - t0
            print(
                f"[MST Prim] {k+1}/{n-1} edges  "
                f"({100.0*(k+1)/(n-1):.1f}%)  elapsed={elapsed/60:.1f} min",
                flush=True,
            )
    return u_out, v_out, w_out


def gudhi_h0_from_mst(
    u: np.ndarray, v: np.ndarray, w: np.ndarray, n_points: int
) -> List[dict]:
    """Insert MST into GUDHI SimplexTree and extract H0 persistence pairs."""
    import gudhi as gd

    st = gd.SimplexTree()
    for i in range(n_points):
        st.insert([int(i)], filtration=0.0)
    for a, b, weight in zip(u, v, w):
        st.insert([int(a), int(b)], filtration=float(weight))
    st.make_filtration_non_decreasing()
    st.compute_persistence()

    rows: List[dict] = []
    for birth_s, death_s in st.persistence_pairs():
        if len(birth_s) != 1:
            continue  # H0 only
        birth_val = float(st.filtration(birth_s))
        if len(death_s) == 0:
            death_val = float("inf")
            death_simplex: Optional[List[int]] = None
        else:
            death_val = float(st.filtration(death_s))
            death_simplex = [int(x) for x in death_s]
        pers = (
            float("inf")
            if not np.isfinite(death_val)
            else float(death_val - birth_val)
        )
        rows.append(
            {
                "dim": 0,
                "birth": birth_val,
                "death": death_val,
                "persistence": pers,
                "birth_simplex": _simplex_to_str([int(x) for x in birth_s]),
                "death_simplex": _simplex_to_str(death_simplex),
                "scope": "all",
            }
        )
    return rows


def persistence_summary(rows: List[dict]) -> dict:
    finite = [r for r in rows if np.isfinite(r["death"])]
    if not finite:
        return {
            "count": len(rows),
            "count_finite": 0,
            "total_persistence": 0.0,
            "max_persistence": 0.0,
            "mean_persistence": 0.0,
            "persistence_entropy": 0.0,
        }
    persists = np.asarray([r["persistence"] for r in finite], dtype=np.float64)
    total = float(persists.sum())
    p_norm = persists / total if total > 0 else persists
    entropy = (
        float(-np.sum(p_norm * np.log(np.maximum(p_norm, 1e-12)))) if total > 0 else 0.0
    )
    return {
        "count": len(rows),
        "count_finite": len(finite),
        "total_persistence": total,
        "max_persistence": float(persists.max()),
        "mean_persistence": float(persists.mean()),
        "persistence_entropy": entropy,
    }


def write_outputs(rows: List[dict], out_prefix: Path, meta: dict) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    # Finite-only (thesis default) and with-infinite
    finite = df[np.isfinite(df["death"])].copy()
    finite_path = Path(str(out_prefix) + ".csv")
    all_path = Path(str(out_prefix) + "_with_infinite.csv")
    npy_path = Path(str(out_prefix) + ".npy")
    meta_path = Path(str(out_prefix) + "_meta.json")

    finite.to_csv(finite_path, index=False)
    df.to_csv(all_path, index=False)
    arr = finite[["dim", "birth", "death"]].to_numpy(dtype=np.float64)
    np.save(npy_path, arr)

    meta = dict(meta)
    meta["outputs"] = [str(npy_path), str(finite_path), str(all_path)]
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"[write] {finite_path}  ({len(finite)} finite pairs)")
    print(f"[write] {all_path}")
    print(f"[write] {npy_path}")
    print(f"[write] {meta_path}")


def validate_against_existing(rows: List[dict], existing_csv: Path) -> None:
    """Compare death multisets to an existing dense-GUDHI CSV (scope=all, dim=0)."""
    if not existing_csv.exists():
        print(f"[validate] skip — missing {existing_csv}")
        return
    ref = pd.read_csv(existing_csv)
    ref = ref[(ref["scope"] == "all") & (ref["dim"] == 0)]
    ref = ref[np.isfinite(ref["death"])]
    new = pd.DataFrame([r for r in rows if np.isfinite(r["death"])])
    d_ref = np.sort(ref["death"].to_numpy(dtype=np.float64))
    d_new = np.sort(new["death"].to_numpy(dtype=np.float64))
    if len(d_ref) != len(d_new):
        print(f"[validate] FAIL count {len(d_new)} vs {len(d_ref)}")
        return
    mx = float(np.max(np.abs(d_ref - d_new)))
    ok = bool(np.allclose(d_ref, d_new, rtol=0, atol=1e-6))
    print(f"[validate] death multiset vs {existing_csv.name}: max|Δ|={mx:.3g} allclose={ok}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="Output prefix (no extension)")
    ap.add_argument("--dataset", type=str, default="")
    ap.add_argument("--backbone", type=str, default="")
    ap.add_argument("--normalize", action="store_true", default=True)
    ap.add_argument("--no-normalize", action="store_false", dest="normalize")
    ap.add_argument(
        "--validate-against",
        type=Path,
        default=None,
        help="Optional existing GUDHI full CSV to check death multiset equality",
    )
    ap.add_argument("--max-points", type=int, default=None, help="Debug: truncate pool")
    args = ap.parse_args()

    t_all = time.time()
    X = np.load(args.features)
    if args.max_points is not None:
        X = X[: int(args.max_points)]
    print(f"[load] {args.features}  shape={X.shape}", flush=True)
    if args.normalize:
        X = l2_normalize(X)
        print("[norm] L2 row-normalised", flush=True)

    t0 = time.time()
    u, v, w = prim_mst(X, assume_unit=bool(args.normalize))
    print(f"[MST] done in {(time.time()-t0)/60:.2f} min  edges={len(w)}", flush=True)

    t0 = time.time()
    rows = gudhi_h0_from_mst(u, v, w, n_points=X.shape[0])
    print(f"[GUDHI] persistence done in {time.time()-t0:.1f}s  pairs={len(rows)}", flush=True)

    summary = persistence_summary(rows)
    meta = {
        "script": "compute_h0_reference_mst.py",
        "method": "exact_euclidean_mst + gudhi.SimplexTree H0",
        "equivalence": (
            "H0 Vietoris–Rips persistence deaths equal Euclidean MST edge weights; "
            "birth/death simplices from GUDHI on the MST filtration."
        ),
        "params": {
            "features": str(args.features),
            "dataset": args.dataset,
            "backbone": args.backbone,
            "normalize": bool(args.normalize),
            "filtration": "MST edges (exact VR H0 equivalent)",
            "metric": "euclidean",
            "library": "gudhi (SimplexTree) + Prim MST",
            "sparse": None,
            "max_edge_length": None,
            "knn_restriction": False,
        },
        "inputs": {"features": str(args.features)},
        "runtime_sec": time.time() - t_all,
        "library_versions": {},
        "notes": {
            "n": int(X.shape[0]),
            "d": int(X.shape[1]),
            "summary": {"h0": summary},
        },
    }
    try:
        import gudhi as gd

        meta["library_versions"] = {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "gudhi": gd.__version__,
        }
    except Exception:
        pass

    write_outputs(rows, args.out, meta)

    if args.validate_against is not None:
        validate_against_existing(rows, args.validate_against)

    print(f"[done] total {(time.time()-t_all)/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
