#!/usr/bin/env python3
"""Audit whether birth-vertex H0 diagnostics survive arbitrary pairing choices.

The existing diagnostic touches an H0 pair when its GUDHI birth vertex is in a
selected set. H0 vertices all enter at filtration zero, so that representative
is not intrinsically distinguished. This audit compares:

* the saved GUDHI birth representative;
* random elder-rule representatives induced by vertex permutations and
  randomized ordering within equal-weight edge groups;
* symmetric either-endpoint and both-endpoint death-edge definitions;
* size-matched and exact-MST-degree-matched random subsets.

It is read-only with respect to experimental outputs.
"""
from __future__ import print_function

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


ROOT = Path("/vast/s219110279")
RUN_ROOT = ROOT / "TypiClust" / "output"
REF_ROOT = ROOT / "outputs" / "diagnostic_h0" / "reference"
OUT_ROOT = ROOT / "outputs" / "diagnostic_h0_robustness"
REFERENCE_CACHE = {}
PAIRING_CACHE = {}

METHOD_PREFIX = {
    "random": "random", "uncertainty": "uncertainty", "entropy": "entropy",
    "margin": "margin", "dbal": "dbal", "coreset": "coreset",
    "probcover": "probcover", "typiclust": "typiclust",
    "maxherding": "maxherding",
}


def parse_simplex(value):
    parsed = ast.literal_eval(str(value))
    return [int(x) for x in parsed]


def load_reference(dataset, backbone):
    cache_key = (dataset, backbone)
    if cache_key in REFERENCE_CACHE:
        return REFERENCE_CACHE[cache_key]
    path = REF_ROOT / ("{}_{}_h0_reference.csv".format(dataset, backbone))
    df = pd.read_csv(str(path))
    births = np.asarray([parse_simplex(x)[0] for x in df.birth_simplex], dtype=np.int64)
    endpoints = np.asarray([parse_simplex(x) for x in df.death_simplex], dtype=np.int64)
    deaths = df.death.to_numpy(dtype=np.float64)
    n_pool = 100000 if dataset == "TINYIMAGENET" else 50000
    result = (path, births, endpoints, deaths, n_pool)
    REFERENCE_CACHE[cache_key] = result
    return result


def cached_pairings(dataset, backbone, endpoints, deaths, n_pool, count, master_seed):
    key = (dataset, backbone, count, master_seed)
    if key not in PAIRING_CACHE:
        # Common pairing perturbations are shared by all methods/seeds/episodes,
        # making their differences directly comparable.
        stable = sum(ord(c) for c in dataset + backbone)
        rng = np.random.RandomState(master_seed + stable * 7919)
        PAIRING_CACHE[key] = [
            paired_births(endpoints, deaths, n_pool, rng) for _ in range(count)
        ]
    return PAIRING_CACHE[key]


def load_selected(dataset, backbone, method, seed, episode):
    run = RUN_ROOT / dataset / backbone / "{}_{}_50b".format(METHOD_PREFIX[method], seed)
    path = run / "episode_{}".format(episode) / "lSet.npy"
    if not path.exists():
        return None, str(path)
    return np.asarray(np.load(str(path), allow_pickle=True), dtype=np.int64).reshape(-1), str(path)


def edge_degrees(endpoints, n_pool):
    return np.bincount(endpoints.reshape(-1), minlength=n_pool).astype(np.int32)


def paired_births(endpoints, deaths, n_pool, rng):
    """Return finite dying-component representatives under random birth ranks."""
    parent = np.arange(n_pool, dtype=np.int64)
    representative = np.arange(n_pool, dtype=np.int64)
    birth_rank = rng.permutation(n_pool)

    # Stable death ordering, randomized only within exactly equal death groups.
    order = np.argsort(deaths, kind="mergesort")
    sorted_deaths = deaths[order]
    starts = np.r_[0, 1 + np.flatnonzero(sorted_deaths[1:] != sorted_deaths[:-1])]
    stops = np.r_[starts[1:], len(order)]
    for start, stop in zip(starts, stops):
        rng.shuffle(order[start:stop])

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != x:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    out = np.full(len(endpoints), -1, dtype=np.int64)
    for edge_id in order:
        a, b = endpoints[edge_id]
        ra, rb = find(int(a)), find(int(b))
        if ra == rb:
            continue
        va, vb = representative[ra], representative[rb]
        if birth_rank[va] <= birth_rank[vb]:
            elder, dying = ra, rb
        else:
            elder, dying = rb, ra
        out[edge_id] = representative[dying]
        parent[dying] = elder
        # The elder component retains its oldest representative.
        representative[elder] = representative[elder]
    return out


def touched_values(selected, definition, births, endpoints, deaths, n_pool):
    selected = selected[(selected >= 0) & (selected < n_pool)]
    if definition == "birth":
        # H0 finite birth representatives are unique. Indexing by selected
        # vertex avoids scanning every reference edge for every null replicate.
        death_of_birth = np.full(n_pool, np.nan, dtype=np.float64)
        death_of_birth[births] = deaths
        return death_of_birth[selected][np.isfinite(death_of_birth[selected])]
    present = np.zeros(n_pool, dtype=bool)
    present[selected] = True
    hits = present[endpoints]
    if definition == "either_endpoint":
        return deaths[hits.any(axis=1)]
    if definition == "both_endpoints":
        return deaths[hits.all(axis=1)]
    raise ValueError(definition)


def metrics(values, full_deaths):
    if not len(values):
        return dict(touched_count=0, touched_fraction=0.0, mean_death=np.nan,
                    wasserstein=np.nan, beta0_auc_normalised=np.nan)
    dmin, dmax = float(full_deaths.min()), float(full_deaths.max())
    eps = np.linspace(dmin, dmax, 80)
    counts = np.asarray([(values > x).sum() for x in eps], dtype=np.float64)
    denom = float((values > 0).sum()) or float(len(values))
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return dict(
        touched_count=int(len(values)),
        touched_fraction=float(len(values) / len(full_deaths)),
        mean_death=float(values.mean()),
        wasserstein=float(wasserstein_distance(values, full_deaths)),
        beta0_auc_normalised=float(trapz(counts / denom, eps)),
    )


def degree_matched_sample(selected, degree, rng):
    pieces = []
    selected_degree = degree[selected]
    for value in np.unique(selected_degree):
        count = int((selected_degree == value).sum())
        population = np.flatnonzero(degree == value)
        pieces.append(rng.choice(population, size=count, replace=False))
    return np.concatenate(pieces).astype(np.int64) if pieces else np.empty(0, dtype=np.int64)


def add_record(rows, base, definition, replicate, selected, births, endpoints, deaths,
               n_pool, touch_definition=None):
    values = touched_values(selected, touch_definition or definition, births,
                            endpoints, deaths, n_pool)
    rows.append(dict(base, definition=definition, replicate=replicate,
                     selected_count=int(len(selected)), **metrics(values, deaths)))


def run_one(dataset, backbone, method, seed, episode, n_pairings, n_nulls, master_seed):
    ref_path, saved_births, endpoints, deaths, n_pool = load_reference(dataset, backbone)
    selected, selected_path = load_selected(dataset, backbone, method, seed, episode)
    if selected is None:
        return [], {"missing": selected_path}
    rng = np.random.RandomState(master_seed + seed * 1009 + episode * 9173)
    base = dict(dataset=dataset, backbone=backbone, method=method, seed=seed,
                episode=episode, reference_path=str(ref_path), selected_path=selected_path)
    rows = []
    add_record(rows, base, "birth", 0, selected, saved_births, endpoints, deaths, n_pool)
    add_record(rows, base, "either_endpoint", 0, selected, saved_births, endpoints, deaths, n_pool)
    add_record(rows, base, "both_endpoints", 0, selected, saved_births, endpoints, deaths, n_pool)

    pairings = cached_pairings(dataset, backbone, endpoints, deaths, n_pool,
                               n_pairings, master_seed)
    for rep, births in enumerate(pairings):
        valid = births >= 0
        # Forest edges omitted by GUDHI zero-persistence filtering stay invalid.
        values = touched_values(selected, "birth", births[valid], endpoints[valid],
                                deaths[valid], n_pool)
        rows.append(dict(base, definition="permuted_pairing", replicate=rep,
                         selected_count=int(len(selected)), **metrics(values, deaths[valid])))

    degree = edge_degrees(endpoints, n_pool)
    for rep in range(n_nulls):
        size_null = rng.choice(n_pool, size=len(selected), replace=False).astype(np.int64)
        degree_null = degree_matched_sample(selected, degree, rng)
        add_record(rows, base, "size_matched_null", rep, size_null,
                   saved_births, endpoints, deaths, n_pool, touch_definition="birth")
        add_record(rows, base, "degree_matched_null", rep, degree_null,
                   saved_births, endpoints, deaths, n_pool, touch_definition="birth")
    return rows, None


def summarize(long_df):
    keys = ["dataset", "backbone", "method", "episode", "definition"]
    metrics_ = ["touched_fraction", "mean_death", "wasserstein", "beta0_auc_normalised"]
    agg = long_df.groupby(keys, dropna=False)[metrics_].agg(["mean", "std", "min", "max"]).reset_index()
    agg.columns = ["_".join([str(x) for x in col if x]) for col in agg.columns]
    return agg


def write_report(long_df, summary, missing, args):
    lines = [
        "# H0 birth-pairing robustness audit", "",
        "This audit tests whether conclusions based on a saved GUDHI birth vertex survive",
        "random elder-rule pairings, symmetric death-edge endpoints, and matched nulls.", "",
        "- Pairing replicates: {}".format(args.pairings),
        "- Null replicates: {}".format(args.nulls),
        "- Episodes: {}".format(args.episodes),
        "- Empty/omitted zero-persistence pairs follow the saved finite reference CSV.", "",
        "## Interpretation rule", "",
        "A structural claim is pairing-robust only if its direction and practical magnitude",
        "remain stable across permuted pairings and symmetric endpoint definitions, and if",
        "the observed value separates from both size- and exact-MST-degree-matched nulls.", "",
        "## Coverage", "",
        "- Long-form rows: {}".format(len(long_df)),
        "- Missing requested runs: {}".format(len(missing)), "",
    ]
    if missing:
        lines += ["## Missing inputs", "", "```json", json.dumps(missing, indent=2), "```", ""]
    lines += ["## Outputs", "", "- `h0_pairing_robustness_long.csv`",
              "- `h0_pairing_robustness_summary.csv`", ""]
    (OUT_ROOT / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["CIFAR10", "CIFAR100", "TINYIMAGENET"])
    parser.add_argument("--backbones", nargs="+", default=["resnet18", "resnet50", "alexnet"])
    parser.add_argument("--methods", nargs="+", default=list(METHOD_PREFIX))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--episodes", nargs="+", type=int, default=[10, 25, 50, 100])
    parser.add_argument("--pairings", type=int, default=25)
    parser.add_argument("--nulls", type=int, default=100)
    parser.add_argument("--master-seed", type=int, default=20260819)
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for ds in args.datasets:
        for bb in args.backbones:
            for method in args.methods:
                for seed in args.seeds:
                    for episode in args.episodes:
                        recs, error = run_one(ds, bb, method, seed, episode,
                                              args.pairings, args.nulls, args.master_seed)
                        rows.extend(recs)
                        if error:
                            missing.append(dict(dataset=ds, backbone=bb, method=method,
                                                seed=seed, episode=episode, **error))
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise RuntimeError("No robustness records were generated")
    summary = summarize(long_df)
    long_df.to_csv(str(OUT_ROOT / "h0_pairing_robustness_long.csv"), index=False)
    summary.to_csv(str(OUT_ROOT / "h0_pairing_robustness_summary.csv"), index=False)
    write_report(long_df, summary, missing, args)
    print("wrote {} rows to {}".format(len(long_df), OUT_ROOT))


if __name__ == "__main__":
    main()
