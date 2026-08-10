#!/usr/bin/env python3
"""
H0 persistent-homology diagnostics for the thesis chapter
"Diagnostic Analysis of Active Learning".

Research question:
  Do active learning strategies expose the connectivity structure of the fixed
  embedding space at different rates and in different ways?

Method (important):
  Do NOT recompute a persistence diagram per labelled subset.
  Use ONE reference H0 PD per dataset/backbone (full embedding pool).
  A reference pair p is touched by sample set S iff BirthSimplex(p) intersects S.

Usage:
  python analysis/generate_diagnostic_h0_figures.py --backbones resnet18
  python analysis/generate_diagnostic_h0_figures.py --backbones resnet18 resnet50 alexnet
  python analysis/generate_diagnostic_h0_figures.py --backbones resnet18 --skip-figures
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr, spearmanr, wasserstein_distance

REPO_ROOT = Path("/vast/s219110279")
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_ROOT = REPO_ROOT / "TypiClust" / "output"
FIGURES_DIR = REPO_ROOT / "figures" / "diagnostic_h0"
OUTPUTS_DIR = REPO_ROOT / "outputs" / "diagnostic_h0"
REF_DIR = OUTPUTS_DIR / "reference"
CACHE_DIR = OUTPUTS_DIR / "h0_cache"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["CIFAR10", "CIFAR100", "TINYIMAGENET"]
DATASET_LABELS = {
    "CIFAR10": "CIFAR-10",
    "CIFAR100": "CIFAR-100",
    "TINYIMAGENET": "TinyImageNet",
}
BACKBONES = ["resnet18", "resnet50", "alexnet"]
BACKBONE_LABELS = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "alexnet": "AlexNet",
}
SEEDS = [1, 2, 3, 4, 5]
BATCH_SIZE = 50

# Cumulative / table key rounds (L_t exists at E100)
KEY_ROUNDS_CUM = [10, 30, 60, 100]
KEY_ROUNDS_TABLE = [10, 25, 50, 100]
KEY_ROUNDS_ACC = [10, 25, 50, 100]
# Exclusive batch diagnostics: no ΔL at E100 → use E90
KEY_ROUNDS_EXCL = [10, 30, 60, 90]
N_EPS = 80

METHOD_DIR_PREFIX = {
    "random": "random",
    "uncertainty": "uncertainty",
    "entropy": "entropy",
    "margin": "margin",
    "dbal": "dbal",
    "coreset": "coreset",
    "probcover": "probcover",
    "typiclust": "typiclust",
    "maxherding": "maxherding",
}
METHOD_ORDER = list(METHOD_DIR_PREFIX.keys())
METHOD_DISPLAY = {
    "random": "Random",
    "uncertainty": "Uncertainty (LC)",
    "entropy": "Entropy",
    "margin": "Margin",
    "dbal": "DBAL (BALD)",
    "coreset": "CoreSet",
    "probcover": "ProbCover",
    "typiclust": "TypiClust",
    "maxherding": "MaxHerding",
}
METHOD_STYLE = {
    "random":      {"color": "#000000", "linestyle": "-",  "linewidth": 2.0},
    "uncertainty": {"color": "#E69F00", "linestyle": "-",  "linewidth": 2.0},
    "entropy":     {"color": "#56B4E9", "linestyle": "--", "linewidth": 2.0},
    "margin":      {"color": "#009E73", "linestyle": "-.", "linewidth": 2.0},
    "dbal":        {"color": "#F0E442", "linestyle": ":",  "linewidth": 2.4},
    "coreset":     {"color": "#0072B2", "linestyle": "-",  "linewidth": 2.0},
    "probcover":   {"color": "#D55E00", "linestyle": "--", "linewidth": 2.2},
    "typiclust":   {"color": "#CC79A7", "linestyle": "-.", "linewidth": 2.2},
    "maxherding":  {"color": "#999999", "linestyle": ":",  "linewidth": 2.4},
}
ACCURACY_ZONE = {
    "CIFAR10": (0.0, 96.0),
    "CIFAR100": (0.0, 80.0),
    "TINYIMAGENET": (0.0, 70.0),
}
EPISODE_MARKERS = {10: "o", 25: "s", 30: "^", 50: "D", 60: "v", 90: "P", 100: "X"}


# ---------------------------------------------------------------------------
# Matplotlib
# ---------------------------------------------------------------------------
def configure_matplotlib():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _budget_to_episode(budget):
    return np.asarray(budget, dtype=float) / BATCH_SIZE - 1.0


def _episode_to_budget(episode):
    return BATCH_SIZE * (np.asarray(episode, dtype=float) + 1.0)


def _add_episode_axis(ax):
    ax.spines["top"].set_visible(True)
    sec = ax.secondary_xaxis("top", functions=(_budget_to_episode, _episode_to_budget))
    sec.set_xlabel("Episode", labelpad=4)
    return sec


def save_fig(fig, stem):
    pdf = FIGURES_DIR / f"{stem}.pdf"
    png = FIGURES_DIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=600)
    fig.savefig(png, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return pdf, png


# ---------------------------------------------------------------------------
# Run loaders (same conventions as ID diagnostics)
# ---------------------------------------------------------------------------
def _plausible_acc(acc, dataset):
    try:
        acc = float(acc)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(acc) or acc < 0 or acc > 100 + 1e-6:
        return False
    lo, hi = ACCURACY_ZONE.get(dataset, (0.0, 100.0))
    return lo <= acc <= hi


def _load_index_npy(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if arr.dtype == object:
        arr = np.asarray(arr.tolist())
    return np.asarray(arr, dtype=np.int64).reshape(-1)


def load_run_rounds(dataset, backbone, method, seed):
    prefix = METHOD_DIR_PREFIX[method]
    run_dir = OUTPUT_ROOT / dataset / backbone / f"{prefix}_{seed}_50b"
    summary_path = run_dir / "benchmark_summary.json"
    if not summary_path.exists():
        return None, f"missing:{summary_path}"

    with open(summary_path) as f:
        data = json.load(f)
    episodes = data.get("episode_records") or []
    if not episodes:
        return None, f"empty:{summary_path}"

    rows = []
    for ep in episodes:
        r = int(ep["episode"])
        acc = ep.get("test_accuracy")
        if not _plausible_acc(acc, dataset):
            continue

        labeled_before = ep.get("labeled_count_before_sampling")
        labeled_after = ep.get("labeled_count_after_sampling")
        if labeled_before is None:
            labeled_before = BATCH_SIZE + r * BATCH_SIZE

        batch = ep.get("active_set_ids")
        cum = None
        lset_p = run_dir / f"episode_{r}" / "lSet.npy"
        if lset_p.exists():
            cum = np.sort(_load_index_npy(lset_p))
            if len(cum) != int(labeled_before):
                print(
                    f"[WARN] size mismatch {dataset}/{backbone}/{method}/seed{seed}/E{r}: "
                    f"len(lSet)={len(cum)} vs labeled_before={labeled_before}"
                )

        if batch is not None:
            batch_arr = np.sort(np.asarray(batch, dtype=np.int64).reshape(-1))
        else:
            if labeled_after is not None and int(labeled_after) == int(labeled_before):
                batch_arr = np.array([], dtype=np.int64)
            elif cum is None:
                continue
            elif r == 0:
                batch_arr = cum.copy()
            else:
                prev_p = run_dir / f"episode_{r-1}" / "lSet.npy"
                if prev_p.exists():
                    prev = np.sort(_load_index_npy(prev_p))
                    batch_arr = np.setdiff1d(cum, prev, assume_unique=False)
                else:
                    batch_arr = np.array([], dtype=np.int64)

        rows.append({
            "round": r,
            "batch": batch_arr.astype(np.int64),
            "cumulative": None if cum is None else cum.astype(np.int64),
            "test_accuracy": float(acc),
            "labeled_budget": int(labeled_before),
        })

    if rows and any(r["cumulative"] is None for r in rows):
        running = np.array([], dtype=np.int64)
        for row in rows:
            if row["cumulative"] is None:
                running = np.union1d(running, row["batch"])
                row["cumulative"] = running.copy()
            else:
                running = row["cumulative"]

    if not rows:
        return None, f"no_valid_rounds:{summary_path}"
    return rows, None


# ---------------------------------------------------------------------------
# Reference H0 PD
# ---------------------------------------------------------------------------
def reference_csv_path(dataset: str, backbone: str) -> Path:
    return REF_DIR / f"{dataset}_{backbone}_h0_reference.csv"


def reference_meta_path(dataset: str, backbone: str) -> Path:
    return REF_DIR / f"{dataset}_{backbone}_h0_reference_meta.json"


def _parse_birth_vertex(s) -> int:
    if isinstance(s, (list, tuple)):
        return int(s[0])
    s = str(s).strip()
    vals = ast.literal_eval(s)
    return int(vals[0])


class ReferenceH0:
    """Full-pool H0 reference diagram + birth-simplex touch index."""

    def __init__(self, dataset: str, backbone: str):
        self.dataset = dataset
        self.backbone = backbone
        self.path = reference_csv_path(dataset, backbone)
        if not self.path.exists():
            raise FileNotFoundError(f"Missing reference PD: {self.path}")

        df = pd.read_csv(self.path)
        if "scope" in df.columns:
            df = df[df["scope"] == "all"]
        if "dim" in df.columns:
            df = df[df["dim"] == 0]
        df = df[np.isfinite(df["death"])].reset_index(drop=True)
        if df.empty:
            raise RuntimeError(f"No finite H0 pairs in {self.path}")

        self.birth = df["birth"].to_numpy(dtype=np.float64)
        self.death = df["death"].to_numpy(dtype=np.float64)
        self.persistence = df["persistence"].to_numpy(dtype=np.float64)
        # H0: persistence == death when birth==0
        self.birth_vertex = np.asarray(
            [_parse_birth_vertex(s) for s in df["birth_simplex"]], dtype=np.int64
        )
        self.n_pairs = int(len(df))
        self.n_pool = int(self.birth_vertex.max()) + 1

        # pair_of_vertex[v] = pair index, or -1 if v is not a finite birth simplex
        self.pair_of_vertex = -np.ones(self.n_pool, dtype=np.int32)
        # unique births expected; last write wins if duplicate (should not happen)
        self.pair_of_vertex[self.birth_vertex] = np.arange(self.n_pairs, dtype=np.int32)

        self.ref_pers_sorted = np.sort(self.persistence)
        self.ref_death_sorted = np.sort(self.death)

        # Common epsilon grid from full reference death range
        dmin = float(self.death.min())
        dmax = float(self.death.max())
        self.epsilon_grid = np.linspace(dmin, dmax, N_EPS)

        meta_p = reference_meta_path(dataset, backbone)
        self.meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}

    def describe(self) -> dict:
        return {
            "path": str(self.path),
            "n_finite_pairs": self.n_pairs,
            "n_pool_index_span": self.n_pool,
            "n_unique_birth_vertices": int(len(np.unique(self.birth_vertex))),
            "death_min": float(self.death.min()),
            "death_median": float(np.median(self.death)),
            "death_mean": float(self.death.mean()),
            "death_max": float(self.death.max()),
            "pers_min": float(self.persistence.min()),
            "pers_median": float(np.median(self.persistence)),
            "pers_mean": float(self.persistence.mean()),
            "pers_max": float(self.persistence.max()),
            "birth_simplex_encoding": "string list with one vertex index, e.g. '[40653]'",
            "columns": ["dim", "birth", "death", "persistence", "birth_simplex",
                        "death_simplex", "scope"],
            "meta": self.meta,
        }

    def touched_pair_ids(self, indices: np.ndarray) -> np.ndarray:
        """Unique reference pair ids whose birth vertex is in `indices`."""
        if indices is None or len(indices) == 0:
            return np.array([], dtype=np.int32)
        idx = np.asarray(indices, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < self.n_pool)]
        if len(idx) == 0:
            return np.array([], dtype=np.int32)
        pids = self.pair_of_vertex[idx]
        pids = pids[pids >= 0]
        if len(pids) == 0:
            return np.array([], dtype=np.int32)
        return np.unique(pids)

    def touched_persistence(self, indices: np.ndarray) -> np.ndarray:
        pids = self.touched_pair_ids(indices)
        return self.persistence[pids] if len(pids) else np.array([], dtype=np.float64)

    def touched_deaths(self, indices: np.ndarray) -> np.ndarray:
        pids = self.touched_pair_ids(indices)
        return self.death[pids] if len(pids) else np.array([], dtype=np.float64)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def ecdf_l1_gap(a: np.ndarray, b: np.ndarray) -> float:
    """L1 area between two ECDFs (same construction as posthoc utils)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    xs = np.sort(np.unique(np.concatenate([a, b])))
    # ECDF value just to the right of each x: fraction <= x
    Fa = np.searchsorted(np.sort(a), xs, side="right") / len(a)
    Fb = np.searchsorted(np.sort(b), xs, side="right") / len(b)
    if len(xs) == 1:
        return float(abs(Fa[0] - Fb[0]))
    dx = np.diff(xs)
    mid = 0.5 * (np.abs(Fa[:-1] - Fb[:-1]) + np.abs(Fa[1:] - Fb[1:]))
    return float(np.sum(mid * dx))


def dist_metrics(touched: np.ndarray, ref_values: np.ndarray) -> dict:
    touched = np.asarray(touched, dtype=np.float64)
    touched = touched[np.isfinite(touched)]
    if len(touched) == 0:
        return {
            "wasserstein": float("nan"),
            "ks": float("nan"),
            "l1_ecdf": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
        }
    return {
        "wasserstein": float(wasserstein_distance(touched, ref_values)),
        "ks": float(ks_2samp(touched, ref_values, alternative="two-sided", mode="asymp").statistic),
        "l1_ecdf": ecdf_l1_gap(touched, ref_values),
        "mean": float(np.mean(touched)),
        "median": float(np.median(touched)),
    }


def beta0_diagnostics(deaths: np.ndarray, eps_grid: np.ndarray) -> dict:
    """beta0(eps)=#{touched deaths > eps}; normalised by beta0(0)."""
    deaths = np.asarray(deaths, dtype=np.float64)
    deaths = deaths[np.isfinite(deaths)]
    if len(deaths) == 0:
        return {
            "beta0_raw": np.zeros_like(eps_grid),
            "beta0_norm": np.full_like(eps_grid, np.nan),
            "beta0_auc_raw": float("nan"),
            "beta0_auc_normalised": float("nan"),
            "epsilon_50": float("nan"),
            "epsilon_10": float("nan"),
        }
    # deaths > eps  <=>  count after searchsorted of sorted ascending
    d_sorted = np.sort(deaths)
    # number strictly greater than eps
    gt = len(d_sorted) - np.searchsorted(d_sorted, eps_grid, side="right")
    beta0_raw = gt.astype(np.float64)
    # beta0(0): deaths > 0 (almost all positive for our PDs)
    b0 = float(np.sum(d_sorted > 0.0))
    if b0 <= 0:
        b0 = float(len(d_sorted))
    beta0_norm = beta0_raw / b0

    _trapz = getattr(np, "trapezoid", None) or np.trapz
    auc_raw = float(_trapz(beta0_raw, eps_grid))
    auc_norm = float(_trapz(beta0_norm, eps_grid))

    def _eps_at(level: float) -> float:
        # smallest eps where normalised beta0 <= level
        ok = np.where(beta0_norm <= level)[0]
        if len(ok) == 0:
            return float(eps_grid[-1])
        return float(eps_grid[ok[0]])

    return {
        "beta0_raw": beta0_raw,
        "beta0_norm": beta0_norm,
        "beta0_auc_raw": auc_raw,
        "beta0_auc_normalised": auc_norm,
        "epsilon_50": _eps_at(0.5),
        "epsilon_10": _eps_at(0.1),
    }


def metrics_for_set(ref: ReferenceH0, indices: np.ndarray, prefix: str) -> dict:
    pids = ref.touched_pair_ids(indices)
    pers = ref.persistence[pids] if len(pids) else np.array([], dtype=np.float64)
    deaths = ref.death[pids] if len(pids) else np.array([], dtype=np.float64)
    dm = dist_metrics(pers, ref.ref_pers_sorted)
    b0 = beta0_diagnostics(deaths, ref.epsilon_grid)
    out = {
        f"{prefix}_touched_count": int(len(pids)),
        f"{prefix}_mean_persistence": dm["mean"],
        f"{prefix}_median_persistence": dm["median"],
        f"{prefix}_wasserstein": dm["wasserstein"],
        f"{prefix}_ks": dm["ks"],
        f"{prefix}_l1_ecdf": dm["l1_ecdf"],
    }
    if prefix == "cumulative":
        out["cumulative_touched_fraction"] = (
            float(len(pids) / ref.n_pairs) if ref.n_pairs else float("nan")
        )
        out["beta0_auc_raw"] = b0["beta0_auc_raw"]
        out["beta0_auc_normalised"] = b0["beta0_auc_normalised"]
        out["epsilon_50"] = b0["epsilon_50"]
        out["epsilon_10"] = b0["epsilon_10"]
    return out, pers, deaths, b0


# ---------------------------------------------------------------------------
# Per-run computation + cache
# ---------------------------------------------------------------------------
def run_cache_path(dataset, backbone, method, seed) -> Path:
    d = CACHE_DIR / backbone / dataset / method
    d.mkdir(parents=True, exist_ok=True)
    return d / f"seed{seed}_h0_metrics.npz"


def compute_run_h0(ref: ReferenceH0, rounds, dataset, backbone, method, seed,
                   refresh_cache: bool = False):
    cache_p = run_cache_path(dataset, backbone, method, seed)
    if refresh_cache and cache_p.exists():
        cache_p.unlink()

    if cache_p.exists():
        z = np.load(cache_p, allow_pickle=True)
        cached_rounds = set(z["rounds"].astype(int).tolist())
        needed = {r["round"] for r in rounds}
        if needed <= cached_rounds:
            by = {int(r): i for i, r in enumerate(z["rounds"].astype(int))}
            skip = {
                "rounds", "key_round", "key_cum_pers", "key_cum_death",
                "key_excl_pers", "eps_grid", "labeled_budget", "test_accuracy",
            }
            out = []
            for row in rounds:
                i = by[row["round"]]
                rec = {
                    "dataset": dataset,
                    "backbone": backbone,
                    "method": method,
                    "method_display": METHOD_DISPLAY[method],
                    "seed": seed,
                    "round": row["round"],
                    "labeled_budget": row["labeled_budget"],
                    "test_accuracy": row["test_accuracy"],
                }
                for k in z.files:
                    if k in skip:
                        continue
                    val = z[k][i]
                    rec[k] = val.item() if getattr(val, "ndim", 0) == 0 else val
                out.append(rec)
            print(f"[H0] cache hit {cache_p}")
            return out, {
                "key_round": z["key_round"],
                "key_cum_pers": list(z["key_cum_pers"]),
                "key_cum_death": list(z["key_cum_death"]),
                "key_excl_pers": list(z["key_excl_pers"]),
                "eps_grid": z["eps_grid"],
            }

    print(f"[H0] {dataset}/{backbone}/{method}/seed{seed}: {len(rounds)} rounds")
    records = []
    store = {
        "rounds": [],
        "exclusive_touched_count": [],
        "cumulative_touched_count": [],
        "cumulative_touched_fraction": [],
        "exclusive_mean_persistence": [],
        "cumulative_mean_persistence": [],
        "exclusive_median_persistence": [],
        "cumulative_median_persistence": [],
        "exclusive_wasserstein": [],
        "cumulative_wasserstein": [],
        "exclusive_ks": [],
        "cumulative_ks": [],
        "exclusive_l1_ecdf": [],
        "cumulative_l1_ecdf": [],
        "beta0_auc_raw": [],
        "beta0_auc_normalised": [],
        "epsilon_50": [],
        "epsilon_10": [],
        "test_accuracy": [],
        "labeled_budget": [],
    }
    key_rounds_needed = sorted(set(KEY_ROUNDS_CUM) | set(KEY_ROUNDS_EXCL) | set(KEY_ROUNDS_ACC) | set(KEY_ROUNDS_TABLE))
    key_pack = {
        "key_round": [],
        "key_cum_pers": [],
        "key_cum_death": [],
        "key_excl_pers": [],
        "eps_grid": ref.epsilon_grid.copy(),
    }

    for row in rounds:
        excl_m, excl_pers, excl_death, _ = metrics_for_set(ref, row["batch"], "exclusive")
        cum_m, cum_pers, cum_death, _ = metrics_for_set(ref, row["cumulative"], "cumulative")
        rec = {
            "dataset": dataset,
            "backbone": backbone,
            "method": method,
            "method_display": METHOD_DISPLAY[method],
            "seed": seed,
            "round": row["round"],
            "labeled_budget": row["labeled_budget"],
            "test_accuracy": row["test_accuracy"],
            **excl_m,
            **cum_m,
        }
        records.append(rec)
        store["rounds"].append(row["round"])
        store["test_accuracy"].append(row["test_accuracy"])
        store["labeled_budget"].append(row["labeled_budget"])
        for k in store:
            if k in ("rounds", "test_accuracy", "labeled_budget"):
                continue
            store[k].append(rec.get(k, np.nan))

        if row["round"] in key_rounds_needed:
            key_pack["key_round"].append(row["round"])
            key_pack["key_cum_pers"].append(cum_pers.astype(np.float32))
            key_pack["key_cum_death"].append(cum_death.astype(np.float32))
            key_pack["key_excl_pers"].append(excl_pers.astype(np.float32))

    # save numeric cache
    save_kw = {k: np.asarray(v) for k, v in store.items()}
    save_kw["key_round"] = np.asarray(key_pack["key_round"], dtype=np.int32)
    save_kw["key_cum_pers"] = np.asarray(key_pack["key_cum_pers"], dtype=object)
    save_kw["key_cum_death"] = np.asarray(key_pack["key_cum_death"], dtype=object)
    save_kw["key_excl_pers"] = np.asarray(key_pack["key_excl_pers"], dtype=object)
    save_kw["eps_grid"] = key_pack["eps_grid"]
    np.savez_compressed(cache_p, **save_kw)
    print(f"[H0] saved {cache_p}")
    return records, key_pack


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
METRIC_COLS = [
    "exclusive_touched_count",
    "cumulative_touched_count",
    "cumulative_touched_fraction",
    "exclusive_mean_persistence",
    "cumulative_mean_persistence",
    "exclusive_median_persistence",
    "cumulative_median_persistence",
    "exclusive_wasserstein",
    "cumulative_wasserstein",
    "exclusive_ks",
    "cumulative_ks",
    "exclusive_l1_ecdf",
    "cumulative_l1_ecdf",
    "beta0_auc_raw",
    "beta0_auc_normalised",
    "epsilon_50",
    "epsilon_10",
    "test_accuracy",
]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(["dataset", "backbone", "method", "method_display", "round"]):
        rec = {
            "dataset": keys[0],
            "backbone": keys[1],
            "method": keys[2],
            "method_display": keys[3],
            "round": int(keys[4]),
            "labeled_budget": int(g["labeled_budget"].median()),
            "n_seeds": int(g["seed"].nunique()),
            "n_missing_seeds": int(len(SEEDS) - g["seed"].nunique()),
        }
        for m in METRIC_COLS:
            if m not in g.columns:
                rec[f"{m}_mean"] = np.nan
                rec[f"{m}_sem"] = np.nan
                continue
            vals = g[m].astype(float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                rec[f"{m}_mean"] = np.nan
                rec[f"{m}_sem"] = np.nan
            else:
                rec[f"{m}_mean"] = float(vals.mean())
                rec[f"{m}_sem"] = (
                    float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
                )
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(
        ["dataset", "backbone", "method", "round"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_metric_panel(ax, summary, dataset, backbone, mean_col, sem_col, ylabel, title):
    panel = summary[(summary.dataset == dataset) & (summary.backbone == backbone)]
    present = []
    for method in METHOD_ORDER:
        sub = panel[panel.method == method].sort_values("labeled_budget")
        if sub.empty:
            continue
        style = METHOD_STYLE[method]
        x = sub["labeled_budget"].to_numpy()
        y = sub[mean_col].to_numpy(dtype=float)
        sem = sub[sem_col].to_numpy(dtype=float)
        mask = np.isfinite(y)
        if not mask.any():
            continue
        ax.plot(
            x[mask], y[mask], color=style["color"], linestyle=style["linestyle"],
            linewidth=style["linewidth"], label=METHOD_DISPLAY[method], zorder=3,
        )
        if np.any(np.isfinite(sem[mask]) & (sem[mask] > 0)):
            ax.fill_between(
                x[mask], y[mask] - sem[mask], y[mask] + sem[mask],
                color=style["color"], alpha=0.18, linewidth=0, zorder=2,
            )
        present.append(method)
    ax.set_title(title, pad=28)
    ax.set_xlabel("Labelled budget")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    ax.spines["right"].set_visible(False)
    _add_episode_axis(ax)
    return present


def _shared_legend(fig, methods, y=0.0):
    handles, labels = [], []
    for m in METHOD_ORDER:
        if m not in methods:
            continue
        s = METHOD_STYLE[m]
        handles.append(mpl.lines.Line2D(
            [0], [0], color=s["color"], linestyle=s["linestyle"], linewidth=s["linewidth"],
        ))
        labels.append(METHOD_DISPLAY[m])
    if handles:
        fig.legend(
            handles, labels, loc="lower center", ncol=min(5, len(handles)),
            frameon=True, fancybox=False, edgecolor="#cccccc",
            bbox_to_anchor=(0.5, y), handlelength=2.8,
        )


def make_combined_metric_figure(summary, backbone, mean_col, sem_col, ylabel, stem, suptitle):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), sharey=False)
    all_m = set()
    for ax, ds, letter in zip(axes, DATASETS, ["(a)", "(b)", "(c)"]):
        present = _plot_metric_panel(
            ax, summary, ds, backbone, mean_col, sem_col,
            ylabel if ax is axes[0] else "",
            f"{letter} {DATASET_LABELS[ds]}",
        )
        all_m.update(present)
        if not present:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle(suptitle, fontsize=15, y=1.04, fontweight="bold")
    _shared_legend(fig, all_m, y=0.0)
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    return save_fig(fig, stem)


def _empirical_ecdf(values):
    v = np.sort(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.array([0.0]), np.array([0.0])
    y = np.arange(1, len(v) + 1) / len(v)
    return v, y


def make_ecdf_figure(ref: ReferenceH0, key_store, backbone, dataset, stem):
    """Cumulative touched persistence ECDFs at KEY_ROUNDS_CUM vs reference."""
    rounds = KEY_ROUNDS_CUM
    fig, axes = plt.subplots(1, len(rounds), figsize=(3.4 * len(rounds), 4.8), sharey=True)
    if len(rounds) == 1:
        axes = [axes]
    xref, yref = _empirical_ecdf(ref.ref_pers_sorted)
    methods_present = set()
    for ax, r in zip(axes, rounds):
        ax.plot(xref, yref, color="#333333", linewidth=2.8, label="Reference (full pool)", zorder=5)
        for method in METHOD_ORDER:
            # average ECDF across seeds via pooled touched values
            pooled = []
            for seed in SEEDS:
                pack = key_store.get((dataset, backbone, method, seed))
                if pack is None:
                    continue
                kr = np.asarray(pack["key_round"], dtype=int)
                if r not in kr:
                    continue
                i = int(np.where(kr == r)[0][0])
                vals = np.asarray(pack["key_cum_pers"][i], dtype=float)
                if len(vals):
                    pooled.append(vals)
            if not pooled:
                continue
            vals = np.concatenate(pooled)
            x, y = _empirical_ecdf(vals)
            s = METHOD_STYLE[method]
            ax.plot(x, y, color=s["color"], linestyle=s["linestyle"],
                    linewidth=s["linewidth"], label=METHOD_DISPLAY[method], alpha=0.9)
            methods_present.add(method)
        ax.set_title(f"E{r} / B{_episode_to_budget(r):.0f}")
        ax.set_xlabel("Persistence")
        if ax is axes[0]:
            ax.set_ylabel("ECDF")
        ax.set_ylim(0, 1.02)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    handles = [
        mpl.lines.Line2D([0], [0], color="#333333", linewidth=2.8, label="Reference (full pool)")
    ]
    labels = ["Reference (full pool)"]
    for m in METHOD_ORDER:
        if m not in methods_present:
            continue
        s = METHOD_STYLE[m]
        handles.append(mpl.lines.Line2D(
            [0], [0], color=s["color"], linestyle=s["linestyle"], linewidth=s["linewidth"],
        ))
        labels.append(METHOD_DISPLAY[m])
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True,
               bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(
        f"Cumulative touched-persistence ECDFs — {BACKBONE_LABELS[backbone]} / {DATASET_LABELS[dataset]}",
        y=1.14, fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    return save_fig(fig, stem)


def make_excl_ecdf_figure(ref: ReferenceH0, key_store, backbone, dataset, stem):
    rounds = KEY_ROUNDS_EXCL
    fig, axes = plt.subplots(1, len(rounds), figsize=(3.4 * len(rounds), 4.8), sharey=True)
    if len(rounds) == 1:
        axes = [axes]
    xref, yref = _empirical_ecdf(ref.ref_pers_sorted)
    methods_present = set()
    for ax, r in zip(axes, rounds):
        ax.plot(xref, yref, color="#333333", linewidth=2.8, label="Reference", zorder=5)
        for method in METHOD_ORDER:
            pooled = []
            for seed in SEEDS:
                pack = key_store.get((dataset, backbone, method, seed))
                if pack is None:
                    continue
                kr = np.asarray(pack["key_round"], dtype=int)
                if r not in kr:
                    continue
                i = int(np.where(kr == r)[0][0])
                vals = np.asarray(pack["key_excl_pers"][i], dtype=float)
                if len(vals):
                    pooled.append(vals)
            if not pooled:
                continue
            x, y = _empirical_ecdf(np.concatenate(pooled))
            s = METHOD_STYLE[method]
            ax.plot(x, y, color=s["color"], linestyle=s["linestyle"],
                    linewidth=s["linewidth"], label=METHOD_DISPLAY[method], alpha=0.9)
            methods_present.add(method)
        ax.set_title(f"E{r}")
        ax.set_xlabel("Persistence")
        if ax is axes[0]:
            ax.set_ylabel("ECDF")
        ax.set_ylim(0, 1.02)
    fig.suptitle(
        f"Exclusive (batch) touched-persistence ECDFs — {BACKBONE_LABELS[backbone]} / {DATASET_LABELS[dataset]}",
        y=1.06, fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    return save_fig(fig, stem)


def make_beta0_norm_figure(ref: ReferenceH0, key_store, backbone, dataset, stem):
    rounds = KEY_ROUNDS_CUM
    fig, axes = plt.subplots(1, len(rounds), figsize=(3.4 * len(rounds), 4.8), sharey=True)
    if len(rounds) == 1:
        axes = [axes]
    eps = ref.epsilon_grid
    for ax, r in zip(axes, rounds):
        for method in METHOD_ORDER:
            curves = []
            for seed in SEEDS:
                pack = key_store.get((dataset, backbone, method, seed))
                if pack is None:
                    continue
                kr = np.asarray(pack["key_round"], dtype=int)
                if r not in kr:
                    continue
                i = int(np.where(kr == r)[0][0])
                deaths = np.asarray(pack["key_cum_death"][i], dtype=float)
                b0 = beta0_diagnostics(deaths, eps)
                if np.all(np.isfinite(b0["beta0_norm"])):
                    curves.append(b0["beta0_norm"])
            if not curves:
                continue
            arr = np.vstack(curves)
            mean = arr.mean(axis=0)
            sem = arr.std(axis=0, ddof=1) / math.sqrt(len(arr)) if len(arr) > 1 else np.zeros_like(mean)
            s = METHOD_STYLE[method]
            ax.plot(eps, mean, color=s["color"], linestyle=s["linestyle"],
                    linewidth=s["linewidth"], label=METHOD_DISPLAY[method])
            ax.fill_between(eps, mean - sem, mean + sem, color=s["color"], alpha=0.15, linewidth=0)
        ax.set_title(f"E{r} / B{_episode_to_budget(r):.0f}")
        ax.set_xlabel(r"$\varepsilon$")
        if ax is axes[0]:
            ax.set_ylabel(r"Normalised $\beta_0(\varepsilon)$")
        ax.set_ylim(0, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True,
                   bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(
        f"Normalised Betti-0 curves — {BACKBONE_LABELS[backbone]} / {DATASET_LABELS[dataset]}",
        y=1.14, fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    return save_fig(fig, stem)


def make_beta0_raw_figure(ref: ReferenceH0, key_store, backbone, dataset, stem):
    rounds = KEY_ROUNDS_CUM
    fig, axes = plt.subplots(1, len(rounds), figsize=(3.4 * len(rounds), 4.8), sharey=False)
    if len(rounds) == 1:
        axes = [axes]
    eps = ref.epsilon_grid
    for ax, r in zip(axes, rounds):
        for method in METHOD_ORDER:
            curves = []
            for seed in SEEDS:
                pack = key_store.get((dataset, backbone, method, seed))
                if pack is None:
                    continue
                kr = np.asarray(pack["key_round"], dtype=int)
                if r not in kr:
                    continue
                i = int(np.where(kr == r)[0][0])
                deaths = np.asarray(pack["key_cum_death"][i], dtype=float)
                b0 = beta0_diagnostics(deaths, eps)
                curves.append(b0["beta0_raw"])
            if not curves:
                continue
            arr = np.vstack(curves)
            mean = arr.mean(axis=0)
            s = METHOD_STYLE[method]
            ax.plot(eps, mean, color=s["color"], linestyle=s["linestyle"],
                    linewidth=s["linewidth"], label=METHOD_DISPLAY[method])
        ax.set_title(f"E{r}")
        ax.set_xlabel(r"$\varepsilon$")
        if ax is axes[0]:
            ax.set_ylabel(r"Raw $\beta_0(\varepsilon)$")
    fig.suptitle(
        f"Raw Betti-0 curves — {BACKBONE_LABELS[backbone]} / {DATASET_LABELS[dataset]}",
        y=1.02, fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    return save_fig(fig, stem)


def make_wasserstein_vs_accuracy(summary, backbone, stem):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2))
    all_m = set()
    for ax, ds, letter in zip(axes, DATASETS, ["(a)", "(b)", "(c)"]):
        panel = summary[(summary.backbone == backbone) & (summary.dataset == ds)
                        & (summary["round"].isin(KEY_ROUNDS_ACC))]
        for method in METHOD_ORDER:
            sub = panel[panel.method == method].sort_values("round")
            if sub.empty:
                continue
            s = METHOD_STYLE[method]
            x = sub["cumulative_wasserstein_mean"].to_numpy(dtype=float)
            y = sub["test_accuracy_mean"].to_numpy(dtype=float)
            rounds = sub["round"].to_numpy(dtype=int)
            mask = np.isfinite(x) & np.isfinite(y)
            if not mask.any():
                continue
            ax.plot(x[mask], y[mask], color=s["color"], linestyle=s["linestyle"],
                    linewidth=s["linewidth"], alpha=0.85, zorder=2)
            for xi, yi, r in zip(x[mask], y[mask], rounds[mask]):
                ax.scatter([xi], [yi], color=s["color"], marker=EPISODE_MARKERS.get(int(r), "o"),
                           s=36, zorder=3, edgecolors="white", linewidths=0.4)
            all_m.add(method)
        ax.set_title(f"{letter} {DATASET_LABELS[ds]}")
        ax.set_xlabel("Cumulative Wasserstein-1 (touched vs reference)")
        if ax is axes[0]:
            ax.set_ylabel("Test accuracy (%)")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    # episode marker legend notes
    fig.suptitle(
        f"H0 Wasserstein vs accuracy — {BACKBONE_LABELS[backbone]} (descriptive)",
        fontsize=15, y=1.04, fontweight="bold",
    )
    _shared_legend(fig, all_m, y=0.0)
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    return save_fig(fig, stem)


# ---------------------------------------------------------------------------
# Reports / tables
# ---------------------------------------------------------------------------
def write_reference_report(refs: dict, path: Path):
    lines = [
        "# H0 reference persistence-diagram report",
        "",
        "Thesis chapter: *Diagnostic Analysis of Active Learning*.",
        "",
        "Policy: **one reference H0 PD per dataset/backbone** on the complete fixed",
        "training embedding pool. Downstream diagnostics map selected indices to",
        "reference pairs via `BirthSimplex(p) ∩ S ≠ ∅`. No per-subset PD recomputation.",
        "",
        "## Column naming",
        "",
        "| Thesis brief | File column |",
        "|---|---|",
        "| Birth | `birth` |",
        "| Death | `death` |",
        "| Persistence | `persistence` |",
        "| Birth Simplex | `birth_simplex` |",
        "",
        "H0 births are `0.0`; for finite pairs `persistence == death`.",
        "`birth_simplex` is a single pool index string, e.g. `[40653]`.",
        "Indices match AL `lSet.npy` / `active_set_ids` (training-pool row indices).",
        "",
    ]
    for (ds, bb), ref in sorted(refs.items()):
        info = ref.describe()
        meta = info["meta"] or {}
        params = meta.get("params") or {}
        libs = meta.get("library_versions") or {}
        lines += [
            f"## {ds} / {bb}",
            "",
            f"- Path: `{info['path']}`",
            f"- Finite H0 pairs: **{info['n_finite_pairs']}**",
            f"- Unique birth vertices: {info['n_unique_birth_vertices']}",
            f"- Death: min={info['death_min']:.6g}, median={info['death_median']:.6g}, "
            f"mean={info['death_mean']:.6g}, max={info['death_max']:.6g}",
            f"- Persistence: min={info['pers_min']:.6g}, median={info['pers_median']:.6g}, "
            f"mean={info['pers_mean']:.6g}, max={info['pers_max']:.6g}",
            f"- Birth Simplex encoding: {info['birth_simplex_encoding']}",
            "",
            "### Construction",
            "",
            f"- Script / method: `{meta.get('script') or meta.get('method') or 'unresolved'}`",
            f"- Library: GUDHI {libs.get('gudhi', 'unresolved')}",
            f"- Filtration: {params.get('filtration') or meta.get('equivalence') or 'unresolved'}",
            f"- Metric: {params.get('metric', 'euclidean')}",
            f"- L2 normalisation: {params.get('normalize', 'unresolved')}",
            f"- Graph / kNN restriction: {params.get('knn_restriction', False)}",
            f"- Max edge length: {params.get('max_edge_length', 'n/a (MST filtration)')}",
            f"- Features: `{(meta.get('inputs') or {}).get('features') or params.get('features') or 'unresolved'}`",
            "",
        ]
        if meta.get("equivalence"):
            lines.append(f"- Equivalence note: {meta['equivalence']}")
            lines.append("")
    path.write_text("\n".join(lines) + "\n")


def write_key_rounds(summary: pd.DataFrame, csv_path: Path, tex_path: Path):
    key = summary[summary["round"].isin(KEY_ROUNDS_TABLE)].copy()
    key.to_csv(csv_path, index=False)
    cols = [
        "dataset", "backbone", "method_display", "round", "n_seeds",
        "cumulative_touched_fraction_mean", "cumulative_wasserstein_mean",
        "beta0_auc_normalised_mean", "epsilon_50_mean", "test_accuracy_mean",
    ]
    present = [c for c in cols if c in key.columns]
    lines = [
        "% Auto-generated H0 key-round summary",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Dataset & Method & Ep & TouchFrac & W1 \\\\",
        "\\midrule",
    ]
    for _, r in key.sort_values(["dataset", "backbone", "method", "round"]).iterrows():
        lines.append(
            f"{DATASET_LABELS.get(r.dataset, r.dataset)} & {r.method_display} & "
            f"E{int(r['round'])} & "
            f"{r.cumulative_touched_fraction_mean:.3f} & "
            f"{r.cumulative_wasserstein_mean:.4f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    tex_path.write_text("\n".join(lines))
    return key


def write_accuracy_relationship(df_long: pd.DataFrame, path: Path):
    rows = []
    pairs = [
        ("cumulative_wasserstein", "test_accuracy"),
        ("cumulative_touched_fraction", "test_accuracy"),
        ("beta0_auc_normalised", "test_accuracy"),
        ("epsilon_50", "test_accuracy"),
    ]
    for (ds, bb, method), g in df_long.groupby(["dataset", "backbone", "method"]):
        # average across seeds per episode first
        agg = g.groupby("round")[[p[0] for p in pairs] + ["test_accuracy"]].mean()
        rec = {
            "dataset": ds, "backbone": bb, "method": method,
            "method_display": METHOD_DISPLAY[method],
            "n_episodes": int(len(agg)),
        }
        for xcol, ycol in pairs:
            x = agg[xcol].to_numpy(dtype=float)
            y = agg[ycol].to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 3:
                rec[f"pearson_{xcol}"] = np.nan
                rec[f"spearman_{xcol}"] = np.nan
                rec[f"n_{xcol}"] = int(m.sum())
                continue
            rec[f"pearson_{xcol}"] = float(pearsonr(x[m], y[m])[0])
            rec[f"spearman_{xcol}"] = float(spearmanr(x[m], y[m])[0])
            rec[f"n_{xcol}"] = int(m.sum())
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def write_trend_summary(summary, corr, missing, path: Path):
    lines = [
        "# H0 diagnostic trend summary",
        "",
        "Descriptive only — **not causal**.",
        "",
        "## Research question",
        "",
        "Do active learning strategies expose the connectivity structure of the fixed",
        "embedding space at different rates and in different ways?",
        "",
        "## How to read the metrics",
        "",
        "- **Touched fraction**: how quickly birth-simplices of reference H0 pairs enter L_t.",
        "  (Nearly tracks labelled fraction when births are unique vertices — interpret",
        "  together with distributional metrics.)",
        "- **Wasserstein-1 / KS / L1-ECDF**: how the persistence *values* of touched pairs",
        "  compare to the full-pool reference distribution.",
        "- **Normalised β₀ AUC / ε₅₀ / ε₁₀**: shape of the survival curve of touched deaths.",
        "",
        "## Patterns to inspect in the figures",
        "",
        "- Which strategies increase cumulative touch fraction fastest?",
        "- Which keep cumulative Wasserstein lowest (closest to reference persistence law)?",
        "- Do coverage methods (ProbCover, TypiClust, CoreSet, MaxHerding) differ from",
        "  uncertainty methods (Entropy, Margin, LC, DBAL)?",
        "- Where does Random sit between families?",
        "- Do CIFAR-10 / CIFAR-100 / TinyImageNet tell the same story?",
        "- Does lower Wasserstein coincide with higher accuracy? (associative only)",
        "",
        "## Strongest |Spearman| (cumulative W1 vs accuracy)",
        "",
    ]
    if corr is not None and len(corr):
        tmp = corr.dropna(subset=["spearman_cumulative_wasserstein"]).copy()
        tmp["abs"] = tmp["spearman_cumulative_wasserstein"].abs()
        for _, r in tmp.nlargest(12, "abs").iterrows():
            lines.append(
                f"- {DATASET_LABELS.get(r.dataset, r.dataset)} / {BACKBONE_LABELS.get(r.backbone, r.backbone)} / "
                f"{r.method_display}: Spearman={r.spearman_cumulative_wasserstein:.3f} "
                f"(n={r.n_cumulative_wasserstein})"
            )
    lines += [
        "",
        "## Caveats",
        "",
        "- Reference PD is fixed; AL seeds vary selection only.",
        "- Touch uses birth-simplex membership (one vertex per finite H0 pair).",
        "- Cumulative metrics use L_t = `lSet.npy` aligned with `labeled_count_before_sampling`.",
        "- Exclusive metrics use `active_set_ids` (ΔL_t); E100 has no batch → use E90 for batch plots.",
        "- No causal claims from H0–accuracy correlations.",
        "",
        "## Missing runs",
        "",
    ]
    if missing:
        for m in missing[:80]:
            lines.append(f"- {m}")
        if len(missing) > 80:
            lines.append(f"- ... and {len(missing) - 80} more")
    else:
        lines.append("- None reported.")
    path.write_text("\n".join(lines) + "\n")


def write_figure_guide(backbone: str, path: Path):
    bb = backbone
    lines = [
        f"# What each H0 figure means ({BACKBONE_LABELS[bb]})",
        "",
        "All figures: `figures/diagnostic_h0/`.  ",
        "Code: `analysis/generate_diagnostic_h0_figures.py`.  ",
        "Reference PDs: `outputs/diagnostic_h0/reference/`.",
        "",
        "## Data used (every figure)",
        "",
        "- **Embeddings**: fixed SimCLR `features_seed1.npy` (shared across AL seeds).",
        "- **Reference H0 PD**: one full-pool diagram per dataset/backbone",
        "  (exact Euclidean MST + GUDHI SimplexTree; VR-H0 equivalent).",
        "- **Selections**: TypiClust AL logs — `active_set_ids` = newly acquired batch ΔL_t;",
        "  `lSet.npy` = cumulative labelled set L_t (aligned with accuracy at that episode).",
        "- **Touch rule**: pair p is touched by S iff its birth vertex ∈ S.",
        "- Curves: mean across seeds; shading = SEM. Colours match accuracy/ID figures.",
        "",
        "---",
        "",
        "## Main figures",
        "",
        "### 1. Cumulative reference-pair exposure",
        f"**File:** `diagnostic_h0_{bb}_touched_fraction_combined.pdf`",
        "",
        "- **Y-axis:** fraction of reference H0 pairs whose birth vertex is in L_t.",
        "- **X-axis:** labelled budget (episode on top).",
        "- **Shows:** how quickly each method “covers” reference connectivity events.",
        "- **Read as:** exposure *rate* of the fixed H0 skeleton — not a new PD each round.",
        "",
        "### 2. Cumulative Wasserstein-1 distance",
        f"**File:** `diagnostic_h0_{bb}_wasserstein_combined.pdf`",
        "",
        "- **Y-axis:** W₁ between persistence values of pairs touched by L_t and the",
        "  full reference persistence distribution.",
        "- **Shows:** whether the *kinds* of H0 events exposed (short vs long bars)",
        "  match the pool, not only how many pairs are touched.",
        "- Lower W₁ ⇒ touched persistence law closer to the full-pool reference.",
        "",
        "### 3. Cumulative persistence ECDFs at key episodes",
        f"**Files:** `diagnostic_h0_{bb}_ecdf_{{cifar10,cifar100,tinyimagenet}}.pdf`",
        "",
        "- Episodes **E10, E30, E60, E100**.",
        "- Thick dark line = full reference ECDF; coloured = method ECDFs (seeds pooled).",
        "- **Shows:** distributional bias toward short- or long-persistence events at",
        "  selected budgets.",
        "",
        "### 4. Normalised Betti-0 curves",
        f"**Files:** `diagnostic_h0_{bb}_beta0_normalised_{{cifar10,cifar100,tinyimagenet}}.pdf`",
        "",
        "- β₀(ε) = #{touched deaths > ε}, divided by β₀(0), on a shared ε-grid per dataset.",
        "- **Shows:** shape of the survival curve of exposed merge scales, comparable",
        "  across methods with different touch counts.",
        "",
        "### 5. Normalised Betti-0 AUC trajectory",
        f"**File:** `diagnostic_h0_{bb}_beta0_auc_combined.pdf`",
        "",
        "- **Y-axis:** AUC of the normalised β₀(ε) curve over the reference death range.",
        "- **Shows:** a scalar summary of how “spread out” exposed death scales remain",
        "  as labelling grows.",
        "",
        "### 6. Wasserstein vs accuracy (descriptive)",
        f"**File:** `diagnostic_h0_{bb}_wasserstein_vs_accuracy.pdf`",
        "",
        "- Episodes **E10, E25, E50, E100**; markers encode episode.",
        "- **Shows:** association between distributional H0 exposure and test accuracy.",
        "- **Not causal.**",
        "",
        "---",
        "",
        "## Appendix figures",
        "",
        f"- KS trajectory: `diagnostic_h0_{bb}_ks_combined.pdf`",
        f"- L1 ECDF-gap: `diagnostic_h0_{bb}_l1_ecdf_combined.pdf`",
        f"- ε₅₀ / ε₁₀: `diagnostic_h0_{bb}_epsilon50_combined.pdf`, `..._epsilon10_combined.pdf`",
        f"- Raw β₀ AUC: `diagnostic_h0_{bb}_beta0_auc_raw_combined.pdf`",
        f"- Raw β₀ curves: `diagnostic_h0_{bb}_beta0_raw_{{dataset}}.pdf`",
        f"- Exclusive ECDFs (E10/30/60/90): `diagnostic_h0_{bb}_ecdf_exclusive_{{dataset}}.pdf`",
        f"- ΔW₁ round-to-round: `diagnostic_h0_{bb}_wasserstein_change_combined.pdf`",
        "",
        "Exclusive batch plots **omit E100** (no new acquisition); use E90 instead.",
        "",
        "---",
        "",
        "## Suggested thesis order",
        "",
        "1. Accuracy curves (context)",
        "2. Touched fraction (exposure rate) → Fig 1",
        "3. Wasserstein / ECDFs (what kind of connectivity) → Figs 2–3",
        "4. Normalised β₀ → Figs 4–5",
        "5. H0 vs accuracy (descriptive) → Fig 6",
        "6. Appendix extras as needed",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_latex_snippets(backbone, out_dir: Path):
    snippets = {
        f"diagnostic_h0_{backbone}_touched_fraction_latex_figure.txt": (
            f"diagnostic_h0_{backbone}_touched_fraction_combined.pdf",
            "Cumulative fraction of reference H0 persistence pairs touched by the labelled "
            f"set using {BACKBONE_LABELS[backbone]} fixed embeddings. A pair is touched when "
            "its birth simplex intersects $L_t$. Curves are seed means with SEM.",
            f"fig:diagnostic_h0_{backbone}_touch_frac",
        ),
        f"diagnostic_h0_{backbone}_wasserstein_latex_figure.txt": (
            f"diagnostic_h0_{backbone}_wasserstein_combined.pdf",
            "Wasserstein-1 distance between the persistence distribution of H0 pairs touched "
            f"by $L_t$ and the full-pool reference distribution ({BACKBONE_LABELS[backbone]}). "
            "Lower values indicate closer agreement with the reference connectivity scales.",
            f"fig:diagnostic_h0_{backbone}_w1",
        ),
        f"diagnostic_h0_{backbone}_beta0_auc_latex_figure.txt": (
            f"diagnostic_h0_{backbone}_beta0_auc_combined.pdf",
            "Normalised Betti-0 AUC of touched reference death values over acquisition. "
            f"Computed from the fixed {BACKBONE_LABELS[backbone]} reference H0 diagram.",
            f"fig:diagnostic_h0_{backbone}_beta0_auc",
        ),
        f"diagnostic_h0_{backbone}_wasserstein_vs_accuracy_latex_figure.txt": (
            f"diagnostic_h0_{backbone}_wasserstein_vs_accuracy.pdf",
            "Descriptive relationship between cumulative H0 Wasserstein-1 distance and test "
            "accuracy at selected episodes. No causal interpretation is claimed.",
            f"fig:diagnostic_h0_{backbone}_w1_vs_acc",
        ),
    }
    paths = []
    for fname, (pdf, caption, label) in snippets.items():
        text = (
            "% Auto-generated\n"
            "\\begin{figure}[t]\n"
            "  \\centering\n"
            f"  \\includegraphics[width=\\textwidth]{{figures/diagnostic_h0/{pdf}}}\n"
            f"  \\caption{{{caption}}}\n"
            f"  \\label{{{label}}}\n"
            "\\end{figure}\n"
        )
        p = out_dir / fname
        p.write_text(text)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_for_backbones(backbones, skip_figures=False, refresh_cache=False):
    configure_matplotlib()
    warnings.filterwarnings("ignore", category=UserWarning)

    all_long = []
    missing = []
    generated = []
    refs = {}
    key_store = {}  # (ds, bb, method, seed) -> pack

    for backbone in backbones:
        for dataset in DATASETS:
            print(f"\n===== {dataset} / {backbone} =====")
            try:
                ref = ReferenceH0(dataset, backbone)
            except FileNotFoundError as e:
                missing.append(str(e))
                print("SKIP:", e)
                continue
            refs[(dataset, backbone)] = ref
            info = ref.describe()
            print(f"[ref] {info['n_finite_pairs']} finite pairs  "
                  f"death med={info['death_median']:.4f}")

            for method in METHOD_ORDER:
                for seed in SEEDS:
                    rounds, err = load_run_rounds(dataset, backbone, method, seed)
                    if err:
                        missing.append(f"{dataset}/{backbone}/{method}/s{seed}: {err}")
                        continue
                    # Index alignment check: birth vertices must lie in pool
                    max_idx = max(
                        (int(r["cumulative"].max()) if len(r["cumulative"]) else -1)
                        for r in rounds
                    )
                    if max_idx >= ref.n_pool:
                        # expand pair_of_vertex if needed (shouldn't for consistent indices)
                        print(f"[WARN] index {max_idx} >= ref.n_pool {ref.n_pool}")
                    rows, pack = compute_run_h0(
                        ref, rounds, dataset, backbone, method, seed,
                        refresh_cache=refresh_cache,
                    )
                    all_long.extend(rows)
                    key_store[(dataset, backbone, method, seed)] = pack

    if not all_long:
        raise SystemExit("No H0 rows computed — aborting.")

    df = pd.DataFrame(all_long)
    long_path = OUTPUTS_DIR / "diagnostic_h0_all_backbones_long.csv"
    # Also write flat name requested in brief
    long_path_flat = REPO_ROOT / "outputs" / "diagnostic_h0_all_backbones_long.csv"
    df.to_csv(long_path, index=False)
    df.to_csv(long_path_flat, index=False)
    generated += [str(long_path), str(long_path_flat)]
    print(f"Wrote {long_path} ({len(df)} rows)")

    summary = summarise(df)
    sum_path = OUTPUTS_DIR / "diagnostic_h0_all_backbones_summary.csv"
    sum_flat = REPO_ROOT / "outputs" / "diagnostic_h0_all_backbones_summary.csv"
    summary.to_csv(sum_path, index=False)
    summary.to_csv(sum_flat, index=False)
    generated += [str(sum_path), str(sum_flat)]

    # Reference report
    ref_report = OUTPUTS_DIR / "diagnostic_h0_reference_report.md"
    write_reference_report(refs, ref_report)
    generated.append(str(ref_report))

    # Key rounds
    key = write_key_rounds(
        summary,
        OUTPUTS_DIR / "diagnostic_h0_key_rounds.csv",
        OUTPUTS_DIR / "diagnostic_h0_key_rounds.tex",
    )
    generated += [
        str(OUTPUTS_DIR / "diagnostic_h0_key_rounds.csv"),
        str(OUTPUTS_DIR / "diagnostic_h0_key_rounds.tex"),
    ]

    corr = write_accuracy_relationship(
        df, OUTPUTS_DIR / "diagnostic_h0_accuracy_relationship.csv"
    )
    generated.append(str(OUTPUTS_DIR / "diagnostic_h0_accuracy_relationship.csv"))

    write_trend_summary(
        summary, corr, missing, OUTPUTS_DIR / "diagnostic_h0_trend_summary.md"
    )
    generated.append(str(OUTPUTS_DIR / "diagnostic_h0_trend_summary.md"))

    if not skip_figures:
        for backbone in backbones:
            # Fig 1
            pdf, png = make_combined_metric_figure(
                summary, backbone,
                "cumulative_touched_fraction_mean", "cumulative_touched_fraction_sem",
                "Cumulative touched-pair fraction",
                f"diagnostic_h0_{backbone}_touched_fraction_combined",
                f"Cumulative reference-pair exposure — {BACKBONE_LABELS[backbone]}",
            )
            generated += [str(pdf), str(png)]

            # Fig 2
            pdf, png = make_combined_metric_figure(
                summary, backbone,
                "cumulative_wasserstein_mean", "cumulative_wasserstein_sem",
                "Wasserstein-1 (touched vs reference)",
                f"diagnostic_h0_{backbone}_wasserstein_combined",
                f"Cumulative H0 Wasserstein distance — {BACKBONE_LABELS[backbone]}",
            )
            generated += [str(pdf), str(png)]

            # Fig 5
            pdf, png = make_combined_metric_figure(
                summary, backbone,
                "beta0_auc_normalised_mean", "beta0_auc_normalised_sem",
                r"Normalised $\beta_0$ AUC",
                f"diagnostic_h0_{backbone}_beta0_auc_combined",
                f"Normalised Betti-0 AUC — {BACKBONE_LABELS[backbone]}",
            )
            generated += [str(pdf), str(png)]

            # Fig 6
            pdf, png = make_wasserstein_vs_accuracy(
                summary, backbone, f"diagnostic_h0_{backbone}_wasserstein_vs_accuracy"
            )
            generated += [str(pdf), str(png)]

            # Appendix trajectories
            for mean_col, sem_col, ylabel, stem, title in [
                ("cumulative_ks_mean", "cumulative_ks_sem", "KS statistic",
                 f"diagnostic_h0_{backbone}_ks_combined",
                 f"Cumulative KS (touched vs reference) — {BACKBONE_LABELS[backbone]}"),
                ("cumulative_l1_ecdf_mean", "cumulative_l1_ecdf_sem", "L1 ECDF gap",
                 f"diagnostic_h0_{backbone}_l1_ecdf_combined",
                 f"Cumulative L1 ECDF gap — {BACKBONE_LABELS[backbone]}"),
                ("epsilon_50_mean", "epsilon_50_sem", r"$\varepsilon_{50}$",
                 f"diagnostic_h0_{backbone}_epsilon50_combined",
                 f"Touched-death $\\varepsilon_{{50}}$ — {BACKBONE_LABELS[backbone]}"),
                ("epsilon_10_mean", "epsilon_10_sem", r"$\varepsilon_{10}$",
                 f"diagnostic_h0_{backbone}_epsilon10_combined",
                 f"Touched-death $\\varepsilon_{{10}}$ — {BACKBONE_LABELS[backbone]}"),
                ("beta0_auc_raw_mean", "beta0_auc_raw_sem", r"Raw $\beta_0$ AUC",
                 f"diagnostic_h0_{backbone}_beta0_auc_raw_combined",
                 f"Raw Betti-0 AUC — {BACKBONE_LABELS[backbone]}"),
            ]:
                pdf, png = make_combined_metric_figure(
                    summary, backbone, mean_col, sem_col, ylabel, stem, title,
                )
                generated += [str(pdf), str(png)]

            # Round-to-round W1 change
            ch_rows = []
            for keys, g in summary[summary.backbone == backbone].groupby(
                ["dataset", "method", "method_display"]
            ):
                g = g.sort_values("round")
                w = g["cumulative_wasserstein_mean"].to_numpy(dtype=float)
                dw = np.diff(w, prepend=np.nan)
                tmp = g.copy()
                tmp["wasserstein_change_mean"] = dw
                tmp["wasserstein_change_sem"] = 0.0
                ch_rows.append(tmp)
            if ch_rows:
                ch = pd.concat(ch_rows, ignore_index=True)
                pdf, png = make_combined_metric_figure(
                    ch, backbone,
                    "wasserstein_change_mean", "wasserstein_change_sem",
                    r"$\Delta$ Wasserstein-1",
                    f"diagnostic_h0_{backbone}_wasserstein_change_combined",
                    f"Round-to-round Wasserstein change — {BACKBONE_LABELS[backbone]}",
                )
                generated += [str(pdf), str(png)]

            # Per-dataset ECDF / beta0
            for dataset in DATASETS:
                if (dataset, backbone) not in refs:
                    continue
                ref = refs[(dataset, backbone)]
                tag = dataset.lower().replace("tinyimagenet", "tinyimagenet")
                if dataset == "CIFAR10":
                    tag = "cifar10"
                elif dataset == "CIFAR100":
                    tag = "cifar100"
                else:
                    tag = "tinyimagenet"

                pdf, png = make_ecdf_figure(
                    ref, key_store, backbone, dataset,
                    f"diagnostic_h0_{backbone}_ecdf_{tag}",
                )
                generated += [str(pdf), str(png)]

                pdf, png = make_excl_ecdf_figure(
                    ref, key_store, backbone, dataset,
                    f"diagnostic_h0_{backbone}_ecdf_exclusive_{tag}",
                )
                generated += [str(pdf), str(png)]

                pdf, png = make_beta0_norm_figure(
                    ref, key_store, backbone, dataset,
                    f"diagnostic_h0_{backbone}_beta0_normalised_{tag}",
                )
                generated += [str(pdf), str(png)]

                pdf, png = make_beta0_raw_figure(
                    ref, key_store, backbone, dataset,
                    f"diagnostic_h0_{backbone}_beta0_raw_{tag}",
                )
                generated += [str(pdf), str(png)]

            # Figure guide + latex
            guide = OUTPUTS_DIR / "FIGURE_GUIDE.md"
            write_figure_guide(backbone, guide)
            generated.append(str(guide))
            generated += [str(p) for p in write_latex_snippets(backbone, OUTPUTS_DIR)]

    # Final report
    report = OUTPUTS_DIR / "diagnostic_h0_generation_report.md"
    report.write_text(
        "# Diagnostic H0 generation — final report\n\n"
        "## Policy\n\n"
        "- One reference H0 PD per dataset/backbone (full pool).\n"
        "- Touch via birth-simplex membership; pairs deduplicated per set S.\n"
        "- `active_set_ids` = ΔL_t; `lSet.npy` = L_t aligned with accuracy.\n"
        "- Epsilon grid = linspace over reference death range "
        f"({N_EPS} points), shared within dataset/backbone.\n"
        "- Missing runs listed; values never invented.\n\n"
        f"## Rows\n\n- long: {len(df)}\n- summary: {len(summary)}\n"
        f"- missing entries: {len(missing)}\n"
        f"- reference diagrams loaded: {len(refs)}\n\n"
        "## Reference PD paths\n\n"
        + "\n".join(f"- `{refs[k].path}` ({refs[k].n_pairs} finite pairs)" for k in sorted(refs))
        + "\n\n## Generated files\n\n"
        + "\n".join(f"- `{g}`" for g in generated)
        + "\n\n## Missing / skipped\n\n"
        + ("\n".join(f"- {m}" for m in missing) if missing else "- None")
        + "\n"
    )
    generated.append(str(report))
    print("\n========== FINAL REPORT ==========")
    print(report.read_text())
    return generated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["resnet18"], choices=BACKBONES)
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--refresh-cache", action="store_true")
    args = ap.parse_args()
    run_for_backbones(
        args.backbones,
        skip_figures=args.skip_figures,
        refresh_cache=args.refresh_cache,
    )


if __name__ == "__main__":
    main()
