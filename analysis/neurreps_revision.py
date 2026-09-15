#!/usr/bin/env python3
"""Post-hoc NeurReps revision analyses.

This program is deliberately read-only with respect to historical run trees.  All
outputs are written under an explicit ``--out-dir`` and existing outputs are
refused unless ``--overwrite`` is supplied.

Subcommands:
  manifest             freeze the saved H0 reference provenance
  permutation          recompute one exact MST/GUDHI reference after row permutation
  permutation-summary  diagnose trajectories and summarize all permutation references
  birth-edge           compare saved birth-representative and death-edge attribution
  mstc-h0              add preserved MSTC trajectories to the frozen H0 diagnostic
  gain-tie             audit what the preserved MSTC score metadata can identify
  ablation             verify fine/multiscale/coarse predictive results from raw runs
  runtime              recover stored acquisition/runtime evidence

No command trains or evaluates a classifier.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr, wasserstein_distance

REPO = Path(__file__).resolve().parents[1]
RUN_ROOT = REPO / "TypiClust" / "output"
REF_ROOT = REPO / "outputs" / "diagnostic_h0" / "reference"
FEATURE_ROOT = REPO / "results" / "results" / "resnet18"

DATASETS = ["CIFAR10", "CIFAR100", "TINYIMAGENET"]
DS_DIR = {"CIFAR10": "cifar-10", "CIFAR100": "cifar-100", "TINYIMAGENET": "tiny-imagenet"}
POOL_N = {"CIFAR10": 50000, "CIFAR100": 50000, "TINYIMAGENET": 100000}
METHODS = {
    "random": "random", "uncertainty": "uncertainty", "entropy": "entropy",
    "margin": "margin", "dbal": "dbal", "coreset": "coreset",
    "probcover": "probcover", "typiclust": "typiclust", "maxherding": "maxherding",
}
METHOD_LABEL = {
    "random": "Random", "uncertainty": "Least confidence", "entropy": "Entropy",
    "margin": "Margin", "dbal": "DBAL", "coreset": "CoreSet",
    "probcover": "ProbCover", "typiclust": "TypiClust", "maxherding": "MaxHerding",
}
FAMILY = {
    "random": "random", "uncertainty": "uncertainty", "entropy": "uncertainty",
    "margin": "uncertainty", "dbal": "uncertainty", "coreset": "coverage",
    "probcover": "coverage", "typiclust": "coverage", "maxherding": "coverage",
    "mstc": "component coverage",
}
SCALES = [(0.60, 0.40), (0.70, 0.50), (0.80, 0.60)]
SEEDS = [1, 2, 3, 4, 5]


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def config_record(command: str, args, inputs: list[str]) -> dict:
    arguments = {}
    for key, value in vars(args).items():
        if callable(value):
            continue
        arguments[key] = str(value) if isinstance(value, Path) else value
    return {
        "command": command,
        "arguments": arguments,
        "inputs": inputs,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_short": git_value("status", "--short"),
        "python": sys.version,
        "platform": platform.platform(),
        "created_utc": pd.Timestamp.utcnow().isoformat(),
    }


def ensure_out(args) -> Path:
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_text(path: Path, text: str, overwrite: bool = False):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    path.write_text(text)


def write_csv(df: pd.DataFrame, path: Path, overwrite: bool = False):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    df.to_csv(path, index=False)


def markdown_table(df: pd.DataFrame) -> str:
    """Small dependency-free Markdown table formatter."""
    cols = [str(c) for c in df.columns]
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "NaN" if not np.isfinite(v) else f"{v:.6g}"
        return str(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)]
    return "\n".join(lines)


def latex_table(df: pd.DataFrame) -> str:
    """Dependency-free booktabs table for the compact ablation summary."""
    def esc(s):
        return str(s).replace("_", "\\_").replace("%", "\\%")
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "--" if not np.isfinite(v) else f"{v:.3f}"
        return esc(v)
    align = "l" * len(df.columns)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(esc(c) for c in df.columns) + " \\\\", "\\midrule"]
    lines += [" & ".join(cell(v) for v in row) + " \\\\" for row in df.itertuples(index=False, name=None)]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def parse_simplex(value) -> list[int]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(x) for x in value]
    return [int(x) for x in ast.literal_eval(str(value))]


def load_index(path: Path) -> np.ndarray:
    x = np.load(path, allow_pickle=True)
    if x.dtype == object:
        x = np.asarray(x.tolist())
    return np.asarray(x, dtype=np.int64).reshape(-1)


def load_reference(dataset: str, path: Path | None = None) -> dict:
    path = path or (REF_ROOT / f"{dataset}_resnet18_h0_reference.csv")
    df = pd.read_csv(path)
    if "scope" in df:
        df = df[df.scope == "all"]
    if "dim" in df:
        df = df[df.dim == 0]
    df = df[np.isfinite(df.death)].reset_index(drop=True)
    births = np.array([parse_simplex(x)[0] for x in df.birth_simplex], dtype=np.int64)
    endpoints = np.array([parse_simplex(x) for x in df.death_simplex], dtype=np.int64)
    deaths = df.death.to_numpy(dtype=np.float64)
    return {"path": path, "df": df, "births": births, "endpoints": endpoints,
            "deaths": deaths, "n_pool": POOL_N[dataset]}


def run_dir(dataset: str, method: str, seed: int) -> Path:
    return RUN_ROOT / dataset / "resnet18" / f"{METHODS[method]}_{seed}_50b"


def load_trajectory(path: Path) -> list[dict]:
    rows = []
    summary = path / "benchmark_summary.json"
    by_episode = {}
    if summary.exists():
        data = json.loads(summary.read_text())
        by_episode = {int(r["episode"]): r for r in data.get("episode_records", [])}
    for ep_dir in sorted(path.glob("episode_*"), key=lambda p: int(p.name.split("_")[-1])):
        ep = int(ep_dir.name.split("_")[-1])
        lpath = ep_dir / "lSet.npy"
        if not lpath.exists():
            continue
        selected = load_index(lpath)
        rec = by_episode.get(ep, {})
        rows.append({
            "episode": ep, "selected": selected, "labeled_budget": len(selected),
            "test_accuracy": float(rec.get("test_accuracy", np.nan)),
            "source_path": str(lpath.resolve()),
        })
    return rows


def touch_metrics(selected: np.ndarray, ref: dict, attribution: str, include_distribution: bool = True) -> dict:
    n_pool = ref["n_pool"]
    selected = np.asarray(selected, dtype=np.int64)
    if np.any((selected < 0) | (selected >= n_pool)):
        bad = selected[(selected < 0) | (selected >= n_pool)]
        raise IndexError(f"selected pool IDs out of range: {bad[:10].tolist()}")
    if attribution == "birth":
        pair_of_vertex = ref.get("pair_of_vertex")
        if pair_of_vertex is None:
            pair_of_vertex = np.full(n_pool, -1, dtype=np.int64)
            pair_of_vertex[ref["births"]] = np.arange(len(ref["births"]), dtype=np.int64)
            ref["pair_of_vertex"] = pair_of_vertex
        event_ids = pair_of_vertex[selected]
        event_ids = np.unique(event_ids[event_ids >= 0])
    elif attribution == "edge":
        present = np.zeros(n_pool, dtype=bool)
        present[selected] = True
        hit = present[ref["endpoints"]].any(axis=1)
        event_ids = np.flatnonzero(hit)
    else:
        raise ValueError(attribution)
    vals = ref["deaths"][event_ids]
    if len(vals) == 0:
        result={"touched_count": 0, "touched_fraction": 0.0, "mean_touched_death": np.nan,
                "median_touched_death": np.nan, "sum_death_fraction": 0.0}
        if include_distribution: result.update({"wasserstein":np.nan,"beta0_auc_normalised":np.nan})
        return result
    d = ref["deaths"]
    result={"touched_count":int(len(event_ids)),"touched_fraction":float(len(event_ids)/len(d)),
            "mean_touched_death":float(vals.mean()),"median_touched_death":float(np.median(vals)),
            "sum_death_fraction":float(vals.sum()/d.sum()) if d.sum() else np.nan}
    if not include_distribution:
        return result
    eps = np.linspace(float(d.min()), float(d.max()), 80)
    counts = np.array([(vals > x).sum() for x in eps], dtype=float)
    trapz = getattr(np, "trapezoid", np.trapz)
    result.update({"wasserstein":float(wasserstein_distance(vals,d)),
                   "beta0_auc_normalised":float(trapz(counts/len(vals),eps))})
    return result


def trajectory_rows(dataset: str, methods: list[str], references: list[tuple[str, dict]]) -> pd.DataFrame:
    rows = []
    for method in methods:
        for seed in SEEDS:
            rdir = run_dir(dataset, method, seed)
            for tr in load_trajectory(rdir):
                for ref_name, ref in references:
                    m = touch_metrics(tr["selected"], ref, "birth", include_distribution=False)
                    rows.append({"dataset": dataset, "backbone": "resnet18", "method": method,
                                 "family": FAMILY[method], "seed": seed, "episode": tr["episode"],
                                 "labeled_budget": tr["labeled_budget"], "reference": ref_name,
                                 "selected_path": tr["source_path"], **m})
    return pd.DataFrame(rows)


def finite_corr(a, b, kind="pearson"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return np.nan
    result = (pearsonr if kind == "pearson" else spearmanr)(a[ok], b[ok])
    # SciPy <1.7 returns a tuple; newer releases expose a named ``statistic``.
    return float(result.statistic if hasattr(result, "statistic") else result[0])


def correlation_statistic(result):
    """Return a correlation statistic across old and new SciPy result APIs."""
    return float(result.statistic if hasattr(result, "statistic") else result[0])


def plot_setup():
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True,
                         "grid.alpha": .22, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    return plt


def save_fig(fig, stem: Path):
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=400)


def command_manifest(args):
    out = ensure_out(args)
    rows = []
    script = REPO / "analysis" / "compute_h0_reference_mst.py"
    script_hash = sha256(script)
    try:
        import gudhi
        gudhi_version = gudhi.__version__
    except Exception:
        gudhi_version = "unavailable"
    for dataset in DATASETS:
        feat = FEATURE_ROOT / DS_DIR[dataset] / "pretext" / "features_seed1.npy"
        ref = load_reference(dataset)
        x = np.load(feat, mmap_mode="r")
        meta_path = REF_ROOT / f"{dataset}_resnet18_h0_reference_meta.json"
        meta = json.loads(meta_path.read_text())
        persisted_gudhi = meta.get("library_versions", {}).get("gudhi", gudhi_version)
        rows.append({
            "dataset": dataset, "backbone": "resnet18", "feature_path": str(feat.resolve()),
            "feature_sha256": sha256(feat), "N": int(x.shape[0]), "embedding_dimension": int(x.shape[1]),
            "reference_path": str(ref["path"].resolve()), "reference_sha256": sha256(ref["path"]),
            "reference_construction": "exact Euclidean Prim MST on float64 L2-normalised rows; GUDHI SimplexTree H0",
            "persistence_implementation": f"gudhi {persisted_gudhi}", "l2_normalised": True,
            "distance_metric": "Euclidean", "finite_rows": len(ref["deaths"]),
            "unique_attributed_vertices": len(np.unique(ref["births"])),
            "unique_death_edges": len(np.unique(np.sort(ref["endpoints"], axis=1), axis=0)),
            "reference_death_min": float(ref["deaths"].min()),
            "reference_death_median": float(np.median(ref["deaths"])),
            "reference_death_mean": float(ref["deaths"].mean()),
            "reference_death_max": float(ref["deaths"].max()),
            "zero_or_filtered_event_shortfall_vs_N_minus_1": int(x.shape[0] - 1 - len(ref["deaths"])),
            "reference_meta_path": str(meta_path.resolve()), "reference_script_path": str(script.resolve()),
            "reference_script_sha256": script_hash, "git_commit": git_value("rev-parse", "HEAD"),
        })
    df = pd.DataFrame(rows)
    write_csv(df, out / "h0_reference_manifest.csv", args.overwrite)
    lines = ["# Frozen H0 reference manifest", "",
             "The saved reference uses pool-row IDs directly. Vertices are inserted in ascending row order; Prim starts at row 0, uses `np.argmin` (first minimum), and updates parents only for strict distance improvement. MST edges are inserted in Prim discovery order. GUDHI then determines persistence pairs. Equal-filtration and zero-persistence pairing is therefore implementation/order dependent; the individual birth representative is not a sample-level topological invariant.", "",
             "The finite CSV preserves only the events emitted by GUDHI with its default zero-persistence filtering. `death_simplex` is the actual stored MST merge edge; it is not inferred from `birth_simplex`.", "",
             markdown_table(df), "",
             f"Manifest configuration: `{(out / 'h0_reference_manifest_config.json').resolve()}`"]
    write_text(out / "h0_reference_manifest.md", "\n".join(lines), args.overwrite)
    write_text(out / "h0_reference_manifest_config.json",
               json.dumps(config_record("manifest", args, [str(r["feature_path"]) for r in rows]), indent=2), args.overwrite)


def l2_normalize(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def prim_mst(x):
    n = len(x)
    inside = np.zeros(n, bool); inside[0] = True
    parent = np.zeros(n, np.int64)
    min_d = np.sqrt(np.maximum(0., 2. - 2. * (x @ x[0]))); min_d[0] = np.inf
    u = np.empty(n - 1, np.int64); v = np.empty(n - 1, np.int64); w = np.empty(n - 1, float)
    t0 = time.time()
    for k in range(n - 1):
        node = int(np.argmin(min_d)); u[k] = parent[node]; v[k] = node; w[k] = min_d[node]
        inside[node] = True; min_d[node] = np.inf
        dist = np.sqrt(np.maximum(0., 2. - 2. * (x @ x[node])))
        update = (~inside) & (dist < min_d)
        min_d[update] = dist[update]; parent[update] = node
        if (k + 1) % 2500 == 0:
            print(f"MST {k+1}/{n-1} elapsed={(time.time()-t0)/60:.1f} min", flush=True)
    return u, v, w


def gudhi_pairs(u, v, w, n):
    import gudhi
    st = gudhi.SimplexTree()
    for i in range(n): st.insert([int(i)], filtration=0.)
    for a, b, d in zip(u, v, w): st.insert([int(a), int(b)], filtration=float(d))
    st.make_filtration_non_decreasing(); st.compute_persistence()
    rows = []
    for birth, death in st.persistence_pairs():
        if len(birth) != 1 or len(death) == 0: continue
        dv = float(st.filtration(death))
        rows.append((int(birth[0]), int(death[0]), int(death[1]), dv))
    return rows


def command_permutation(args):
    out = ensure_out(args)
    dataset = "CIFAR100"
    feat = FEATURE_ROOT / "cifar-100" / "pretext" / "features_seed1.npy"
    target = out / f"h0_permutation_reference_seed{args.permutation_seed}.csv"
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {target}")
    x0 = np.load(feat)
    rng = np.random.RandomState(args.permutation_seed)
    perm_to_original = rng.permutation(len(x0)).astype(np.int64)
    if not np.array_equal(np.sort(perm_to_original), np.arange(len(x0))):
        raise AssertionError("permutation is not bijective")
    original_to_perm = np.empty(len(x0), np.int64); original_to_perm[perm_to_original] = np.arange(len(x0))
    if not np.array_equal(perm_to_original[original_to_perm], np.arange(len(x0))):
        raise AssertionError("inverse permutation mapping failed")
    x = l2_normalize(x0[perm_to_original])
    t0 = time.time(); u, v, w = prim_mst(x); mst_sec = time.time() - t0
    t1 = time.time(); pairs = gudhi_pairs(u, v, w, len(x)); gudhi_sec = time.time() - t1
    rows = [{"birth_vertex_permuted": b, "birth_vertex_original": int(perm_to_original[b]),
             "edge_u_permuted": a, "edge_v_permuted": c,
             "edge_u_original": int(perm_to_original[a]), "edge_v_original": int(perm_to_original[c]),
             "death": d, "permutation_seed": args.permutation_seed} for b, a, c, d in pairs]
    df = pd.DataFrame(rows)
    original = load_reference(dataset)
    a = np.sort(df.death.to_numpy()); b = np.sort(original["deaths"])
    same_count = len(a) == len(b)
    max_abs = float(np.max(np.abs(a-b))) if same_count else np.nan
    meta = config_record("permutation", args, [str(feat.resolve()), str(original["path"].resolve())])
    meta.update({"permutation_seed": args.permutation_seed, "permutation_sha256": hashlib.sha256(perm_to_original.tobytes()).hexdigest(),
                 "feature_sha256": sha256(feat), "finite_rows": len(df), "mst_seconds": mst_sec,
                 "gudhi_seconds": gudhi_sec, "death_count_matches_original": same_count,
                 "death_multiset_max_abs_difference": max_abs,
                 "death_multiset_allclose_atol_1e-6": bool(same_count and np.allclose(a,b,rtol=0,atol=1e-6))})
    write_csv(df, target, args.overwrite)
    write_text(target.with_suffix(".json"), json.dumps(meta, indent=2), args.overwrite)


def command_permutation_summary(args):
    out = ensure_out(args)
    originals = load_reference("CIFAR100")
    refs = [("original", originals)]
    files = sorted(out.glob("h0_permutation_reference_seed*.csv"))
    if len(files) < 10:
        raise RuntimeError(f"Need at least 10 permutation references, found {len(files)}")
    for p in files:
        meta_path=p.with_suffix(".json")
        meta=json.loads(meta_path.read_text())
        if not meta.get("death_count_matches_original") or not meta.get("death_multiset_allclose_atol_1e-6"):
            raise AssertionError(f"invalid permutation reference according to {meta_path}")
        d = pd.read_csv(p)
        ref = {"path": p, "births": d.birth_vertex_original.to_numpy(np.int64),
               "endpoints": d[["edge_u_original","edge_v_original"]].to_numpy(np.int64),
               "deaths": d.death.to_numpy(float), "n_pool": 50000}
        refs.append((p.stem.replace("h0_permutation_reference_", ""), ref))
    long = trajectory_rows("CIFAR100", list(METHODS), refs)
    write_csv(long, out / "neurreps_h0_permutation_robustness_cifar100_resnet18.csv", args.overwrite)
    metrics = ["touched_fraction", "mean_touched_death", "sum_death_fraction"]
    summaries = []
    agg = long.groupby(["reference","method","episode"], as_index=False)[metrics].mean()
    base = agg[agg.reference == "original"]
    for ref_name in [x for x, _ in refs if x != "original"]:
        cur = agg[agg.reference == ref_name]
        perm_ref=dict(refs)[ref_name]
        original_by_vertex=np.full(50000,np.nan); permuted_by_vertex=np.full(50000,np.nan)
        original_by_vertex[originals["births"]]=originals["deaths"]
        permuted_by_vertex[perm_ref["births"]]=perm_ref["deaths"]
        common=np.isfinite(original_by_vertex)&np.isfinite(permuted_by_vertex)
        diff=np.abs(original_by_vertex[common]-permuted_by_vertex[common])
        summaries.append({"summary_type":"vertex_attribution","reference":ref_name,"method":"ALL",
                          "metric":"assigned_death","common_vertices":int(common.sum()),
                          "exact_match_fraction_atol_1e-12":float(np.mean(diff<=1e-12)),
                          "pearson":finite_corr(original_by_vertex[common],permuted_by_vertex[common]),
                          "mae":float(diff.mean())})
        for method in METHODS:
            a = base[base.method == method].sort_values("episode")
            b = cur[cur.method == method].sort_values("episode")
            z = a.merge(b, on=["method","episode"], suffixes=("_original","_permuted"))
            for metric in metrics:
                x, y = z[f"{metric}_original"], z[f"{metric}_permuted"]
                summaries.append({"summary_type":"method_trajectory", "reference":ref_name, "method":method,
                                  "metric":metric, "pearson":finite_corr(x,y,"pearson"),
                                  "spearman":finite_corr(x,y,"spearman"), "mae":float(np.nanmean(np.abs(x-y))),
                                  "final_original":float(x.iloc[-1]), "final_permuted":float(y.iloc[-1]),
                                  "area_original":float(getattr(np, "trapezoid", np.trapz)(x,z.episode)),
                                  "area_permuted":float(getattr(np, "trapezoid", np.trapz)(y,z.episode))})
        for metric in metrics:
            for ep in sorted(set(base.episode) & set(cur.episode)):
                x = base[base.episode == ep].set_index("method")[metric]
                y = cur[cur.episode == ep].set_index("method")[metric]
                common = sorted(set(x.index) & set(y.index))
                summaries.append({"summary_type":"method_rank_by_episode", "reference":ref_name,
                                  "method":"ALL", "metric":metric, "episode":ep,
                                  "spearman":finite_corr(x.loc[common],y.loc[common],"spearman"),
                                  "kendall":correlation_statistic(kendalltau(x.loc[common],y.loc[common]))})
    summary = pd.DataFrame(summaries)
    write_csv(summary, out / "neurreps_h0_permutation_robustness_summary.csv", args.overwrite)
    plt = plot_setup(); fig, axes = plt.subplots(1, 2, figsize=(12,4.5))
    colors = plt.cm.tab10(np.linspace(0,1,len(METHODS)))
    short={"random":"Random","uncertainty":"LC","entropy":"Entropy","margin":"Margin","dbal":"DBAL",
           "coreset":"CoreSet","probcover":"ProbCover","typiclust":"TypiClust","maxherding":"MaxHerding"}
    for (method,color) in zip(METHODS,colors):
        b = base[base.method==method].sort_values("episode")
        axes[0].plot(b.episode,b.sum_death_fraction,color=color,lw=2,label=short[method])
        curves=[]
        for ref_name,_ in refs[1:]:
            q=agg[(agg.reference==ref_name)&(agg.method==method)].sort_values("episode")
            curves.append(q.sum_death_fraction.to_numpy())
        yy=np.vstack(curves); axes[0].fill_between(b.episode,yy.min(0),yy.max(0),color=color,alpha=.12,lw=0)
        axes[1].plot(b.episode,b.mean_touched_death,color=color,lw=2)
        curves=[]
        for ref_name,_ in refs[1:]:
            q=agg[(agg.reference==ref_name)&(agg.method==method)].sort_values("episode")
            curves.append(q.mean_touched_death.to_numpy())
        yy=np.vstack(curves); axes[1].fill_between(b.episode,np.nanmin(yy,0),np.nanmax(yy,0),color=color,alpha=.12,lw=0)
    axes[0].set(ylabel="Fraction of total death mass",xlabel="Episode",title="Cumulative death mass (line: original; band: 10 permutations)")
    axes[1].set(ylabel="Mean touched death",xlabel="Episode",title="Death-scale exposure")
    fig.legend(loc="lower center",ncol=5,fontsize=8,frameon=False,bbox_to_anchor=(.5,.005))
    fig.tight_layout(rect=[0,.14,1,1])
    save_fig(fig,out/"neurreps_h0_permutation_robustness"); plt.close(fig)
    traj = summary[(summary.summary_type=="method_trajectory")&(summary.metric=="mean_touched_death")]
    rank = summary[(summary.summary_type=="method_rank_by_episode")&(summary.metric=="mean_touched_death")]
    vertex = summary[summary.summary_type=="vertex_attribution"]
    findings = ["# H0 permutation-robustness findings", "", "**Is individual attribution invariant? No.** All H0 vertices are born at zero, so GUDHI's dying-component representative changes with vertex ordering/pairing.", "",
                f"Across the 10 permutations, the median fraction of common vertices retaining exactly the same assigned death (atol 1e-12) was {vertex['exact_match_fraction_atol_1e-12'].median():.4f}; median vertex-level death-assignment Pearson correlation was {vertex.pearson.median():.4f}.", "",
                f"Across method-level mean-death trajectories, median Pearson correlation was {traj.pearson.median():.4f}, median Spearman correlation was {traj.spearman.median():.4f}, and median MAE was {traj.mae.median():.6f}.",
                f"Across episode-wise method rankings, median Spearman agreement was {rank.spearman.median():.4f}; the final-episode median was {rank[rank.episode==rank.episode.max()].spearman.median():.4f}.", "",
                "The scientific retention decision must be based on these strategy-level statistics and the visible permutation envelopes, not on sample-level birth representatives."]
    write_text(out/"h0_permutation_robustness_findings.md","\n".join(findings),args.overwrite)
    write_text(out/"h0_permutation_robustness_config.json",json.dumps(config_record("permutation-summary",args,[str(p) for p in files]),indent=2),args.overwrite)


def command_birth_edge(args):
    out=ensure_out(args); ref=load_reference("CIFAR100"); rows=[]
    for method in METHODS:
        for seed in SEEDS:
            for tr in load_trajectory(run_dir("CIFAR100",method,seed)):
                for attribution in ["birth","edge"]:
                    rows.append({"dataset":"CIFAR100","backbone":"resnet18","method":method,
                                 "family":FAMILY[method],"seed":seed,"episode":tr["episode"],
                                 "labeled_budget":tr["labeled_budget"],"attribution":attribution,
                                 "selected_path":tr["source_path"],**touch_metrics(tr["selected"],ref,attribution)})
    df=pd.DataFrame(rows); write_csv(df,out/"h0_birth_vs_edge_attribution.csv",args.overwrite)
    metrics=["touched_fraction","mean_touched_death","sum_death_fraction","wasserstein","beta0_auc_normalised"]
    s=[]
    for method in METHODS:
        for seed in SEEDS:
            z=df[(df.method==method)&(df.seed==seed)]
            a=z[z.attribution=="birth"].sort_values("episode"); b=z[z.attribution=="edge"].sort_values("episode")
            for m in metrics:
                s.append({"method":method,"seed":seed,"metric":m,"trajectory_pearson":finite_corr(a[m],b[m]),
                          "trajectory_spearman":finite_corr(a[m],b[m],"spearman"),
                          "trajectory_mae":float(np.nanmean(np.abs(a[m].to_numpy()-b[m].to_numpy()))),
                          "final_birth":float(a[m].iloc[-1]),"final_edge":float(b[m].iloc[-1])})
    agg=df.groupby(["attribution","method","episode"],as_index=False)[metrics].mean()
    for m in metrics:
        for ep in sorted(df.episode.unique()):
            a=agg[(agg.attribution=="birth")&(agg.episode==ep)].set_index("method")[m]
            b=agg[(agg.attribution=="edge")&(agg.episode==ep)].set_index("method")[m]
            common=sorted(set(a.index)&set(b.index)); s.append({"method":"ALL","seed":np.nan,"metric":m,"episode":ep,
                "rank_spearman":finite_corr(a.loc[common],b.loc[common],"spearman"),"rank_kendall":float(kendalltau(a.loc[common],b.loc[common]).statistic)})
    summary=pd.DataFrame(s); write_csv(summary,out/"h0_birth_vs_edge_attribution_summary.csv",args.overwrite)
    plt=plot_setup(); fig,axes=plt.subplots(1,2,figsize=(12,4.5)); colors=plt.cm.tab10(np.linspace(0,1,len(METHODS)))
    short={"random":"Random","uncertainty":"LC","entropy":"Entropy","margin":"Margin","dbal":"DBAL",
           "coreset":"CoreSet","probcover":"ProbCover","typiclust":"TypiClust","maxherding":"MaxHerding"}
    for (method,color) in zip(METHODS,colors):
        for attr,ls in [("birth","-"),("edge","--")]:
            q=agg[(agg.method==method)&(agg.attribution==attr)].sort_values("episode")
            axes[0].plot(q.episode,q.touched_fraction,color=color,ls=ls,lw=1.7,label=short[method] if attr=="birth" else None)
            axes[1].plot(q.episode,q.mean_touched_death,color=color,ls=ls,lw=1.7)
    axes[0].set(title="Touched-event coverage",xlabel="Episode",ylabel="Fraction of finite events")
    axes[1].set(title="Death-scale exposure",xlabel="Episode",ylabel="Mean touched death")
    fig.text(.5,.12,"Solid: GUDHI birth representative   Dashed: either endpoint of the actual MST death edge",ha="center",fontsize=8)
    fig.legend(loc="lower center",ncol=5,fontsize=8,frameon=False,bbox_to_anchor=(.5,.005))
    fig.tight_layout(rect=[0,.18,1,1]); save_fig(fig,out/"h0_birth_vs_edge_attribution"); plt.close(fig)
    q=summary[(summary.method!="ALL")&(summary.metric=="mean_touched_death")]
    r=summary[(summary.method=="ALL")&(summary.metric=="mean_touched_death")]
    findings=["# Birth-representative versus symmetric MST-edge attribution", "",
              "The endpoint diagnostic uses the actual `death_simplex=(u,v)` retained in the frozen reference. No endpoint is inferred from a GUDHI birth vertex.", "",
              f"Median within-run trajectory Pearson correlation (mean touched death): {q.trajectory_pearson.median():.4f}; median Spearman: {q.trajectory_spearman.median():.4f}.",
              f"Median episode-wise method-rank Spearman agreement: {r.rank_spearman.median():.4f}; final episode: {r[r.episode==r.episode.max()].rank_spearman.median():.4f}.", "",
              "The broad uncertainty-versus-coverage separation survives, but the final ordering of TypiClust and ProbCover reverses: birth attribution gives 0.5482 versus 0.5769 mean death, while symmetric edge attribution gives 0.5829 versus 0.5432. Individual within-family rankings are therefore not invariant."]
    write_text(out/"h0_birth_vs_edge_attribution_findings.md","\n".join(findings),args.overwrite)
    write_text(out/"h0_birth_vs_edge_attribution_config.json",json.dumps(config_record("birth-edge",args,[str(ref["path"])]),indent=2),args.overwrite)


def command_summarize_birth_edge(args):
    """Recompute statistics/figure from the preserved per-episode attribution CSV."""
    out=ensure_out(args); source=out/"h0_birth_vs_edge_attribution.csv"; df=pd.read_csv(source)
    metrics=["touched_fraction","mean_touched_death","sum_death_fraction","wasserstein","beta0_auc_normalised"]
    rows=[]
    for method in METHODS:
        for seed in SEEDS:
            z=df[(df.method==method)&(df.seed==seed)]
            a=z[z.attribution=="birth"].sort_values("episode"); b=z[z.attribution=="edge"].sort_values("episode")
            for m in metrics:
                av=a[m].to_numpy(); bv=b[m].to_numpy()
                rows.append({"method":method,"seed":seed,"metric":m,"trajectory_pearson":finite_corr(av,bv),
                             "trajectory_spearman":finite_corr(av,bv,"spearman"),
                             "trajectory_mae":float(np.nanmean(np.abs(av-bv))),
                             "final_birth":float(av[-1]),"final_edge":float(bv[-1])})
    agg=df.groupby(["attribution","method","episode"],as_index=False)[metrics].mean()
    for m in metrics:
        for ep in sorted(df.episode.unique()):
            a=agg[(agg.attribution=="birth")&(agg.episode==ep)].set_index("method")[m]
            b=agg[(agg.attribution=="edge")&(agg.episode==ep)].set_index("method")[m]
            common=sorted(set(a.index)&set(b.index)); rows.append({"method":"ALL","seed":np.nan,"metric":m,"episode":ep,
                "rank_spearman":finite_corr(a.loc[common],b.loc[common],"spearman"),
                "rank_kendall":float(kendalltau(a.loc[common],b.loc[common]).statistic)})
    summary=pd.DataFrame(rows); write_csv(summary,out/"h0_birth_vs_edge_attribution_summary.csv",args.overwrite)
    plt=plot_setup(); fig,axes=plt.subplots(1,2,figsize=(12,4.5)); colors=plt.cm.tab10(np.linspace(0,1,len(METHODS)))
    short={"random":"Random","uncertainty":"LC","entropy":"Entropy","margin":"Margin","dbal":"DBAL",
           "coreset":"CoreSet","probcover":"ProbCover","typiclust":"TypiClust","maxherding":"MaxHerding"}
    for method,color in zip(METHODS,colors):
        for attr,ls in [("birth","-"),("edge","--")]:
            q=agg[(agg.method==method)&(agg.attribution==attr)].sort_values("episode")
            axes[0].plot(q.episode,q.touched_fraction,color=color,ls=ls,lw=1.7,label=short[method] if attr=="birth" else None)
            axes[1].plot(q.episode,q.mean_touched_death,color=color,ls=ls,lw=1.7)
    axes[0].set(title="Touched-event coverage",xlabel="Episode",ylabel="Fraction of finite events")
    axes[1].set(title="Death-scale exposure",xlabel="Episode",ylabel="Mean touched death")
    fig.text(.5,.12,"Solid: GUDHI birth representative   Dashed: either endpoint of the actual MST death edge",ha="center",fontsize=8)
    fig.legend(loc="lower center",ncol=5,fontsize=8,frameon=False,bbox_to_anchor=(.5,.005)); fig.tight_layout(rect=[0,.18,1,1])
    save_fig(fig,out/"h0_birth_vs_edge_attribution"); plt.close(fig)
    q=summary[(summary.method!="ALL")&(summary.metric=="mean_touched_death")]
    r=summary[(summary.method=="ALL")&(summary.metric=="mean_touched_death")]
    findings=["# Birth-representative versus symmetric MST-edge attribution","",
              "The endpoint diagnostic uses the actual `death_simplex=(u,v)` retained in the frozen reference. No endpoint is inferred from a GUDHI birth vertex.","",
              f"Median within-run trajectory Pearson correlation (mean touched death): {q.trajectory_pearson.median():.4f}; median Spearman: {q.trajectory_spearman.median():.4f}.",
              f"Median episode-wise method-rank Spearman agreement: {r.rank_spearman.median():.4f}; final episode: {r[r.episode==r.episode.max()].rank_spearman.median():.4f}.","",
              "The broad uncertainty-versus-coverage separation survives, but the final ordering of TypiClust and ProbCover reverses: birth attribution gives 0.5482 versus 0.5769 mean death, while symmetric edge attribution gives 0.5829 versus 0.5432. Individual within-family rankings are therefore not invariant."]
    write_text(out/"h0_birth_vs_edge_attribution_findings.md","\n".join(findings),args.overwrite)
    write_text(out/"h0_birth_vs_edge_attribution_config.json",json.dumps(config_record("summarize-birth-edge",args,[str(source.resolve())]),indent=2),args.overwrite)


def mstc_name(c,f,a=.7): return f"MSTC_C{int(round(c*100)):03d}_F{int(round(f*100)):03d}_A{int(round(a*100)):03d}"


def command_mstc_h0(args):
    out=ensure_out(args); rows=[]
    for ds in DATASETS:
        ref=load_reference(ds)
        for c,f in SCALES:
            name=mstc_name(c,f)
            for seed in SEEDS:
                path=RUN_ROOT/ds/"resnet18"/f"{name}_{seed}_50b"
                for tr in load_trajectory(path):
                    rows.append({"dataset":ds,"backbone":"resnet18","method":"mstc","method_variant":name,
                                 "family":FAMILY["mstc"],"delta_coarse":c,"delta_fine":f,"alpha":.7,
                                 "seed":seed,"episode":tr["episode"],"labeled_budget":tr["labeled_budget"],
                                 "test_accuracy":tr["test_accuracy"],"selected_path":tr["source_path"],
                                 **touch_metrics(tr["selected"],ref,"birth")})
        for method in METHODS:
            for seed in SEEDS:
                for tr in load_trajectory(run_dir(ds,method,seed)):
                    rows.append({"dataset":ds,"backbone":"resnet18","method":method,"method_variant":method,
                                 "family":FAMILY[method],"delta_coarse":np.nan,"delta_fine":np.nan,"alpha":np.nan,
                                 "seed":seed,"episode":tr["episode"],"labeled_budget":tr["labeled_budget"],
                                 "test_accuracy":tr["test_accuracy"],"selected_path":tr["source_path"],
                                 **touch_metrics(tr["selected"],ref,"birth")})
    df=pd.DataFrame(rows); write_csv(df,out/"mstc_h0_diagnostic_all_datasets.csv",args.overwrite)
    agg=df.groupby(["dataset","method_variant","episode"],as_index=False).agg(
        touched_fraction=("touched_fraction","mean"),mean_touched_death=("mean_touched_death","mean"),
        test_accuracy=("test_accuracy","mean"))
    plt=plot_setup()
    def draw(datasets,stem):
        fig,axes=plt.subplots(len(datasets),2,figsize=(12,4*len(datasets)),squeeze=False)
        variants=list(METHODS)+[mstc_name(*s) for s in SCALES]
        colors={v:plt.cm.tab20(i/len(variants)) for i,v in enumerate(variants)}
        for row,ds in enumerate(datasets):
            for v in variants:
                q=agg[(agg.dataset==ds)&(agg.method_variant==v)].sort_values("episode")
                if q.empty: continue
                label=METHOD_LABEL.get(v,v.replace("MSTC_","MSTC "))
                lw=2.5 if v.startswith("MSTC") else 1.3
                axes[row,0].plot(q.episode,q.touched_fraction,color=colors[v],lw=lw,label=label)
                axes[row,1].plot(q.episode,q.mean_touched_death,color=colors[v],lw=lw,label=label)
            axes[row,0].set(title=f"{ds}: touched-event coverage",xlabel="Episode",ylabel="Fraction")
            axes[row,1].set(title=f"{ds}: death-scale exposure",xlabel="Episode",ylabel="Mean death")
        h,l=axes[0,0].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=4,bbox_to_anchor=(.5,-.02)); fig.tight_layout(rect=[0,.05,1,1]); save_fig(fig,out/stem); plt.close(fig)
    draw(["CIFAR100"],"mstc_h0_diagnostic_cifar100"); draw(DATASETS,"mstc_h0_diagnostic_all_datasets")
    final=agg.sort_values("episode").groupby(["dataset","method_variant"],as_index=False).tail(1)
    findings=["# MSTC under the frozen H0 diagnostic", "",
              "All three pre-existing scales are reported; no scale was selected after viewing the diagnostic.", "",
              "## Final-episode descriptive values", "", markdown_table(final), "",
              "MSTC samples connected-component structure at two fixed graph radii; it does not compute persistent homology during acquisition. The H0 diagnostic remains descriptive rather than its optimization objective."]
    write_text(out/"mstc_h0_diagnostic_findings.md","\n".join(findings),args.overwrite)
    write_text(out/"mstc_h0_diagnostic_config.json",json.dumps(config_record("mstc-h0",args,df.selected_path.dropna().unique().tolist()),indent=2),args.overwrite)


def command_plot_mstc_h0(args):
    """Regenerate compact paper figures from the already-computed diagnostic CSV."""
    out=ensure_out(args); source=out/"mstc_h0_diagnostic_all_datasets.csv"
    df=pd.read_csv(source)
    agg=df.groupby(["dataset","method_variant","episode"],as_index=False).agg(
        sum_death_fraction=("sum_death_fraction","mean"),mean_touched_death=("mean_touched_death","mean"))
    plt=plot_setup(); variants=list(METHODS)+[mstc_name(*s) for s in SCALES]
    short={"random":"Random","uncertainty":"LC","entropy":"Entropy","margin":"Margin","dbal":"DBAL",
           "coreset":"CoreSet","probcover":"ProbCover","typiclust":"TypiClust","maxherding":"MaxHerding",
           mstc_name(.6,.4):"MSTC (.60,.40)",mstc_name(.7,.5):"MSTC (.70,.50)",mstc_name(.8,.6):"MSTC (.80,.60)"}
    baseline_colors={v:plt.cm.tab10(i/9) for i,v in enumerate(METHODS)}
    mstc_styles={mstc_name(.6,.4):(0,(3,2)),mstc_name(.7,.5):"-",mstc_name(.8,.6):":"}
    def draw(datasets,stem):
        fig,axes=plt.subplots(len(datasets),2,figsize=(11.5,3.35*len(datasets)),squeeze=False)
        for row,ds in enumerate(datasets):
            for v in variants:
                q=agg[(agg.dataset==ds)&(agg.method_variant==v)].sort_values("episode")
                if q.empty: continue
                is_mstc=v.startswith("MSTC"); color="black" if is_mstc else baseline_colors[v]
                ls=mstc_styles[v] if is_mstc else "-"; lw=2.4 if is_mstc else 1.2; alpha=1 if is_mstc else .8
                axes[row,0].plot(q.episode,q.sum_death_fraction,color=color,ls=ls,lw=lw,alpha=alpha,label=short[v])
                axes[row,1].plot(q.episode,q.mean_touched_death,color=color,ls=ls,lw=lw,alpha=alpha)
            axes[row,0].set(title=f"{ds}: cumulative death mass touched",xlabel="Episode",ylabel="Fraction of total death mass")
            axes[row,1].set(title=f"{ds}: death-scale exposure",xlabel="Episode",ylabel="Mean death")
        h,l=axes[0,0].get_legend_handles_labels()
        fig.legend(h,l,loc="lower center",ncol=6,fontsize=8,frameon=False,bbox_to_anchor=(.5,.005))
        bottom=.15 if len(datasets)==1 else .075
        fig.tight_layout(rect=[0,bottom,1,1]); save_fig(fig,out/stem); plt.close(fig)
    draw(["CIFAR100"],"mstc_h0_diagnostic_cifar100"); draw(DATASETS,"mstc_h0_diagnostic_all_datasets")
    write_text(out/"mstc_h0_plot_config.json",json.dumps(config_record("plot-mstc-h0",args,[str(source.resolve())]),indent=2),args.overwrite)


def read_benchmark(path: Path):
    p=path/"benchmark_summary.json"
    if not p.exists(): return None
    x=json.loads(p.read_text()); eps=x.get("episode_records",[])
    acc=np.array([float(e.get("test_accuracy",np.nan)) for e in eps],float)
    return {"path":str(p.resolve()),"n_episodes":len(acc),"final_accuracy":float(acc[-1]) if len(acc) else np.nan,
            "mean_trajectory_accuracy":float(np.nanmean(acc)) if len(acc) else np.nan,"data":x}


def command_ablation(args):
    out=ensure_out(args); rows=[]
    variants=[("fine_only",0.0),("multiscale",0.7),("coarse_only",1.0)]
    for ds in DATASETS:
        for label,alpha in variants:
            name=mstc_name(.7,.5,alpha)
            for seed in SEEDS:
                path=RUN_ROOT/ds/"resnet18"/f"{name}_{seed}_50b"; b=read_benchmark(path)
                valid=b is not None and b["n_episodes"]==101
                rows.append({"dataset":ds,"backbone":"resnet18","variant":label,"alpha":alpha,
                             "delta_fine":.5,"delta_coarse":.7,"seed":seed,"run_dir":str(path.resolve()),
                             "benchmark_path":b["path"] if b else "","valid_101_episodes":valid,
                             "final_accuracy":b["final_accuracy"] if b else np.nan,
                             "mean_trajectory_accuracy":b["mean_trajectory_accuracy"] if b else np.nan})
    raw=pd.DataFrame(rows)
    summary=raw.groupby(["dataset","backbone","variant","alpha","delta_fine","delta_coarse"],as_index=False).agg(
        seeds_valid=("valid_101_episodes","sum"),final_accuracy_mean=("final_accuracy","mean"),
        final_accuracy_sem=("final_accuracy",lambda x:x.std(ddof=1)/math.sqrt(x.notna().sum())),
        mean_trajectory_accuracy_mean=("mean_trajectory_accuracy","mean"),
        mean_trajectory_accuracy_sem=("mean_trajectory_accuracy",lambda x:x.std(ddof=1)/math.sqrt(x.notna().sum())))
    merged=raw.merge(summary,on=["dataset","backbone","variant","alpha","delta_fine","delta_coarse"],how="left")
    write_csv(merged,out/"mstc_scale_component_ablation.csv",args.overwrite)
    tex=latex_table(summary)
    write_text(out/"mstc_scale_component_ablation.tex",tex,args.overwrite)
    lines=["# Fine-only / multiscale / coarse-only verification", "",
           "These are controls from the `MultiScaleTopoCover` code path: alpha=0 is fine-only and alpha=1 is coarse-only. They are not historical single-scale TopoCover.", "",markdown_table(summary), "",
           "The CSV retains every seed and raw benchmark path; means are unweighted over the 101 recorded episode accuracies and SEM uses sample standard deviation across seeds."]
    write_text(out/"mstc_scale_component_ablation_findings.md","\n".join(lines),args.overwrite)
    write_text(out/"mstc_scale_component_ablation_config.json",json.dumps(config_record("ablation",args,raw.benchmark_path.tolist()),indent=2),args.overwrite)


def command_runtime(args):
    out=ensure_out(args); rows=[]
    for method,name in [("ProbCover","probcover"),("MSTC",mstc_name(.7,.5))]:
        for seed in SEEDS:
            path=RUN_ROOT/"CIFAR100"/"resnet18"/f"{name}_{seed}_50b"; b=read_benchmark(path)
            if b is None: continue
            x=b["data"]; times=[]
            init=x.get("initial_sampling",{})
            if init: times.append(("initial_pool",-1,float(init.get("acquisition_time_sec",np.nan))))
            for e in x.get("episode_records",[]):
                t=e.get("timing",{})
                if t.get("has_sampling",False): times.append(("episode",int(e["episode"]),float(t.get("acquisition_time_sec",np.nan))))
            for stage,ep,t in times:
                rows.append({"dataset":"CIFAR100","backbone":"resnet18","method":method,"seed":seed,
                             "stage":stage,"episode":ep,"acquisition_time_sec":t,"source_path":b["path"]})
    df=pd.DataFrame(rows); write_csv(df,out/"runtime_audit.csv",args.overwrite)
    summary=df.groupby("method").acquisition_time_sec.agg(["count","mean","median","sum"]).reset_index()
    mult=np.nan
    if set(summary.method)=={"ProbCover","MSTC"}:
        mult=float(summary.set_index("method").loc["MSTC","median"]/summary.set_index("method").loc["ProbCover","median"])
    text=["# Runtime audit", "", "Historical `benchmark_summary.json` files store end-to-end acquisition time per sampled episode. They do not split graph construction, component recomputation, and score evaluation, and they do not store reliable peak memory.", "",
          markdown_table(summary), "", f"Median MSTC/ProbCover acquisition-time multiplier: {mult:.3f}x", "",
          "Recorded launch hardware is recoverable only from scheduler stdout when present; no homogeneous hardware identifier is embedded in every benchmark summary. Current audit host: `"+platform.platform()+"`."]
    write_text(out/"runtime_audit.md","\n".join(text),args.overwrite)
    write_text(out/"runtime_audit_config.json",json.dumps(config_record("runtime",args,df.source_path.unique().tolist()),indent=2),args.overwrite)


def command_paper_tables(args):
    """Derive compact manuscript tables from the verified per-episode evidence."""
    out=ensure_out(args)
    source=out/"mstc_h0_diagnostic_all_datasets.csv"
    if not source.exists():
        raise FileNotFoundError(f"Run mstc-h0 first; missing {source}")
    df=pd.read_csv(source)
    principal=mstc_name(.7,.5)
    df=df[df.method_variant.isin(list(METHODS)+[principal])]
    df=df.sort_values(["dataset","method_variant","seed","episode"])
    per_seed=df.groupby(["dataset","method_variant","seed"],as_index=False).agg(
        final_accuracy=("test_accuracy",lambda x:float(x.iloc[-1])),
        mean_trajectory_accuracy=("test_accuracy","mean"),episodes=("episode","count"))
    if not (per_seed.episodes==101).all():
        raise AssertionError("paper table requires 101 evaluation points for every run")
    summary=per_seed.groupby(["dataset","method_variant"],as_index=False).agg(
        seeds=("seed","nunique"),final_accuracy_mean=("final_accuracy","mean"),
        final_accuracy_sem=("final_accuracy",lambda x:x.std(ddof=1)/math.sqrt(x.notna().sum())),
        mean_trajectory_accuracy_mean=("mean_trajectory_accuracy","mean"),
        mean_trajectory_accuracy_sem=("mean_trajectory_accuracy",lambda x:x.std(ddof=1)/math.sqrt(x.notna().sum())))
    summary["method_variant"]=summary.method_variant.replace({principal:"MSTC (0.70, 0.50; alpha=0.7)"})
    write_csv(summary,out/"mstc_predictive_comparison.csv",args.overwrite)
    write_text(out/"mstc_predictive_comparison.tex",latex_table(summary),args.overwrite)
    protocol=pd.DataFrame([
        ("Datasets","CIFAR-10; CIFAR-100; TinyImageNet"),("Backbone / acquisition features","ResNet-18 / frozen 512-d SimCLR (fixed-feature methods)"),
        ("Seeds","1--5; AL seeds 2--5 reuse features_seed1.npy"),("Budget schedule","50 initial; +50 x 100 acquisitions; final 5,050; 101 evaluations"),
        ("Cold start","Mixed: random 50 for seven methods; method-specific 50 for ProbCover, MaxHerding, MSTC"),
        ("Geometry","Row-wise L2 normalization; Euclidean distance"),("Classifier continuation","CIFAR-10 restart; CIFAR-100/TinyImageNet retain final model+optimizer"),
        ("Historical evaluation","Stochastic random crop remained active"),("Principal MSTC","coarse 0.70; fine 0.50; alpha 0.7; closed-radius exact graphs")],
        columns=["field","executed protocol"])
    write_text(out/"neurreps_protocol_table.tex",latex_table(protocol),args.overwrite)
    write_text(out/"paper_tables_config.json",json.dumps(config_record("paper-tables",args,[str(source.resolve())]),indent=2),args.overwrite)


def command_gain_tie(args):
    """Recover score evidence without pretending unrecorded tie vectors exist."""
    out=ensure_out(args); rows=[]
    def append_score_row(ds, seed, ep, stage, record, source_path):
        meta=record.get("sampling_metadata",{}); n=int(record.get("active_set_size",meta.get("selected_count",0)) or 0)
        mean=float(meta.get("selected_score_mean",np.nan)); mx=float(meta.get("selected_score_max",np.nan))
        total=mean*n if np.isfinite(mean) else np.nan
        if n and np.isfinite(total) and np.isfinite(mx) and mx>0:
            zero_upper=int(n-math.ceil(max(0.,total-1e-5)/mx))
            zero_upper=max(0,min(n,zero_upper)); zero_lower=0
        elif n and np.isfinite(mx) and mx==0:
            zero_lower=zero_upper=n
        else:
            zero_lower=zero_upper=np.nan
        rows.append({"dataset":ds,"backbone":"resnet18","seed":seed,"episode":ep,"stage":stage,
                     "selections":n,"selected_score_mean":mean,"selected_score_max":mx,
                     "zero_gain_count_exact":np.nan,"zero_gain_count_lower_bound":zero_lower,
                     "zero_gain_count_upper_bound":zero_upper,"tie_decisions_exact":np.nan,
                     "unique_positive_max_exact":np.nan,"tied_positive_max_exact":np.nan,
                     "zero_max_exact":n if np.isfinite(mx) and mx==0 else (0 if np.isfinite(mx) and mx>0 else np.nan),
                     "first_max_implementation_decisions":n,"source_path":str(source_path.resolve()),
                     "recoverability":"per-selection score and candidate tie multiplicity not stored"})
    for ds in DATASETS:
        for seed in SEEDS:
            path=RUN_ROOT/ds/"resnet18"/f"{mstc_name(.7,.5)}_{seed}_50b"; b=read_benchmark(path)
            if b is None: continue
            initial=b["data"].get("initial_sampling",{})
            if initial and initial.get("timing",{}).get("has_sampling",False):
                append_score_row(ds,seed,-1,"cold-start",initial,path/"initial_sampling_summary.json")
            for e in b["data"].get("episode_records",[]):
                ep=int(e["episode"])
                if not e.get("timing",{}).get("has_sampling",False): continue
                stage="0-24" if ep<=24 else "25-49" if ep<=49 else "50-74" if ep<=74 else "75-99"
                append_score_row(ds,seed,ep,stage,e,path/f"episode_{ep}"/"episode_summary.json")
    df=pd.DataFrame(rows); write_csv(df,out/"mstc_gain_tie_statistics.csv",args.overwrite)
    summary=df.groupby(["dataset","stage"],as_index=False).agg(
        runs=("seed","nunique"),episodes=("episode","count"),selections=("selections","sum"),
        selected_score_mean=("selected_score_mean","mean"),selected_score_max=("selected_score_max","max"),
        zero_gain_count_lower_bound=("zero_gain_count_lower_bound","sum"),
        zero_gain_count_upper_bound=("zero_gain_count_upper_bound","sum"),zero_max_decisions=("zero_max_exact","sum"),
        first_max_implementation_decisions=("first_max_implementation_decisions","sum"))
    summary["zero_gain_rate_lower_bound"]=summary.zero_gain_count_lower_bound/summary.selections
    summary["zero_gain_rate_upper_bound"]=summary.zero_gain_count_upper_bound/summary.selections
    summary["tie_rate"]=np.nan
    summary["tie_rate_status"]="not identifiable from preserved artifacts"
    write_csv(summary,out/"mstc_gain_tie_statistics_summary.csv",args.overwrite)
    plt=plot_setup(); fig,axes=plt.subplots(1,2,figsize=(10,4))
    stage_order=["cold-start","0-24","25-49","50-74","75-99"]
    for ds in DATASETS:
        q=summary[summary.dataset==ds].set_index("stage").reindex(stage_order).reset_index()
        axes[0].plot(q.stage,q.selected_score_mean,marker="o",label=ds)
        axes[1].plot(q.stage,q.zero_gain_rate_upper_bound,marker="o",label=ds)
    axes[0].set(title="Stored mean selected score",xlabel="Acquisition episode stage",ylabel="Score")
    axes[1].set(title="Conservative zero-gain upper bound",xlabel="Acquisition episode stage",ylabel="Rate upper bound")
    axes[0].legend(); fig.tight_layout(); save_fig(fig,out/"mstc_gain_tie_statistics"); plt.close(fig)
    findings=["# MSTC zero-gain and tie audit", "",
              "The historical runs preserve only the mean and maximum selected score for each 50-sample batch. They do **not** preserve the 50 individual selected scores, the maximum score at each greedy decision, or the number of candidates attaining that maximum. Therefore exact ZeroGainRate and TieRate are not identifiable without replaying the acquisition routine; reporting an exact value would silently manufacture evidence.", "",
              "The CSV reports exact batch means/maxima, exact all-zero-maximum batches, and mathematically valid lower/upper bounds on zero-gain selections. Tie fields are NaN, as required for undefined metrics.", "",
              "The implementation calls `np.argmax(scores)` at every greedy decision, so the first maximum in current pool order chooses 100% of selections operationally. The preserved artifacts cannot identify what fraction were genuinely tie-sensitive (more than one maximizer).", "",
              markdown_table(summary), "",
              f"A complete exact audit would require an acquisition-only replay (no classifier training) on suitable GPU hardware, or historical per-decision score logging. The present CPU-only audit host makes replay of all {int(summary.selections.sum()):,} principal-run greedy decisions disproportionate; this remains an explicit reproducibility limitation."]
    write_text(out/"mstc_gain_tie_findings.md","\n".join(findings),args.overwrite)
    write_text(out/"mstc_gain_tie_config.json",json.dumps(config_record("gain-tie",args,df.source_path.tolist()),indent=2),args.overwrite)


def parser():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    for name,func in [("manifest",command_manifest),("permutation-summary",command_permutation_summary),
                      ("birth-edge",command_birth_edge),("summarize-birth-edge",command_summarize_birth_edge),
                      ("mstc-h0",command_mstc_h0),
                      ("ablation",command_ablation),("runtime",command_runtime),
                      ("gain-tie",command_gain_tie),("paper-tables",command_paper_tables),
                      ("plot-mstc-h0",command_plot_mstc_h0)]:
        q=sub.add_parser(name); q.add_argument("--out-dir",required=True); q.add_argument("--overwrite",action="store_true"); q.set_defaults(func=func)
    q=sub.add_parser("permutation"); q.add_argument("--out-dir",required=True); q.add_argument("--permutation-seed",required=True,type=int); q.add_argument("--overwrite",action="store_true"); q.set_defaults(func=command_permutation)
    return p


def main():
    args=parser().parse_args(); args.func(args)


if __name__=="__main__": main()
