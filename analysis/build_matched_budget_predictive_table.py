#!/usr/bin/env python3
"""Build label-budget-matched predictive tables with explicit dispersion.

Historical results used stochastic evaluation transforms; this script does not
silently relabel them as corrected reruns. It provides the strongest valid
descriptive comparison available before deterministic-protocol reruns finish.
"""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/vast/s219110279")
OUT = ROOT / "outputs" / "matched_budget_predictive"
COMMON_MIN = 50
COMMON_MAX = 5050


def load_curves():
    base = pd.read_csv(str(ROOT / "outputs/diagnostic_accuracy/diagnostic_accuracy_all_backbones_long.csv"))
    topo = pd.read_csv(str(ROOT / "outputs/topocover/topocover_all_backbones_long.csv"))
    topo = topo[(topo.delta == 0.70) & (topo.budget_per_round == 50)].copy()
    # The preserved directories record sampling_fn=dbal. DBAL and BALD are
    # distinct implementations in Sampling.py; do not retain the stale alias.
    base.loc[base.method == "dbal", "method_display"] = "DBAL"
    keep = ["dataset", "backbone", "method", "method_display", "seed", "round",
            "labeled_budget", "test_accuracy"]
    return pd.concat([base[keep], topo[keep]], ignore_index=True)


def per_seed(curves):
    rows = []
    keys = ["dataset", "backbone", "method", "method_display", "seed"]
    for key, group in curves.groupby(keys):
        g = group[(group.labeled_budget >= COMMON_MIN) &
                  (group.labeled_budget <= COMMON_MAX)].sort_values("labeled_budget")
        g = g.drop_duplicates("labeled_budget", keep="last")
        if g.empty or int(g.labeled_budget.iloc[0]) != COMMON_MIN or int(g.labeled_budget.iloc[-1]) != COMMON_MAX:
            continue
        x = g.labeled_budget.to_numpy(dtype=float)
        y = g.test_accuracy.to_numpy(dtype=float)
        trapz = getattr(np, "trapezoid", None) or np.trapz
        normalized_aulc = float(trapz(y, x) / (COMMON_MAX - COMMON_MIN))
        rows.append(dict(zip(keys, key), final_accuracy=float(y[-1]),
                         matched_budget_aulc=normalized_aulc,
                         min_budget=COMMON_MIN, max_budget=COMMON_MAX,
                         n_budget_points=int(len(g))))
    return pd.DataFrame(rows)


def aggregate(seed):
    keys = ["dataset", "backbone", "method", "method_display"]
    rows = []
    for key, g in seed.groupby(keys):
        rec = dict(zip(keys, key), n_seeds=int(g.seed.nunique()))
        for metric in ["final_accuracy", "matched_budget_aulc"]:
            values = g[metric].to_numpy(dtype=float)
            rec[metric + "_mean"] = float(values.mean())
            rec[metric + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            rec[metric + "_sem"] = rec[metric + "_sd"] / np.sqrt(len(values)) if len(values) > 1 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["dataset", "backbone", "final_accuracy_mean"],
                                          ascending=[True, True, False])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    curves = load_curves()
    seed = per_seed(curves)
    summary = aggregate(seed)
    seed.to_csv(str(OUT / "matched_budget_per_seed.csv"), index=False)
    summary.to_csv(str(OUT / "matched_budget_summary.csv"), index=False)
    lines = [
        "# Matched-budget predictive results", "",
        "All methods are evaluated over the common labelled interval 50--5,050.",
        "AULC is trapezoidal integration over labelled-sample count divided by the",
        "interval width. Mean, sample SD, SEM, and completed seed count are retained.", "",
        "**Protocol warning:** these are historical runs evaluated with stochastic",
        "validation/test crops. They are descriptive evidence, not replacements for",
        "the required deterministic-transform reruns.", "",
        "- `matched_budget_per_seed.csv`", "- `matched_budget_summary.csv`", "",
    ]
    (OUT / "README.md").write_text("\n".join(lines))
    print("wrote {} seed rows and {} summary rows".format(len(seed), len(summary)))


if __name__ == "__main__":
    main()
