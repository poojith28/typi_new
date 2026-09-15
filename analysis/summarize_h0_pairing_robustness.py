#!/usr/bin/env python3
"""Create paper-ready summaries and figures from the H0 robustness audit."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


ROOT = Path("/vast/s219110279")
IN = ROOT / "outputs" / "diagnostic_h0_robustness" / "h0_pairing_robustness_long.csv"
OUT = IN.parent
METRICS = ["touched_fraction", "mean_death", "wasserstein", "beta0_auc_normalised"]


def observed(df, definition):
    return df[(df.definition == definition) & (df.replicate == 0)].copy()


def rank_stability(df):
    keys = ["dataset", "backbone", "episode"]
    idcols = keys + ["method", "seed"]
    birth = observed(df, "birth")[idcols + METRICS]
    rows = []
    for definition in ["either_endpoint", "both_endpoints"]:
        other = observed(df, definition)[idcols + METRICS]
        merged = birth.merge(other, on=idcols, suffixes=("_birth", "_other"))
        for key, g in merged.groupby(keys):
            for metric in METRICS:
                x, y = g[metric + "_birth"], g[metric + "_other"]
                valid = x.notna() & y.notna()
                rho = spearmanr(x[valid], y[valid])[0] if valid.sum() > 2 else np.nan
                rows.append(dict(zip(keys, key), definition=definition, metric=metric,
                                 spearman_rho=rho, n=int(valid.sum())))
    return pd.DataFrame(rows)


def pairing_variability(df):
    pair = df[df.definition == "permuted_pairing"]
    keys = ["dataset", "backbone", "method", "seed", "episode"]
    return pair.groupby(keys)[METRICS].std().reset_index().rename(
        columns={m: m + "_pairing_sd" for m in METRICS})


def null_effects(df):
    keys = ["dataset", "backbone", "method", "seed", "episode"]
    birth = observed(df, "birth")[keys + METRICS]
    rows = []
    for null_name in ["size_matched_null", "degree_matched_null"]:
        null = df[df.definition == null_name]
        for key, g in null.groupby(keys):
            target = birth
            for col, val in zip(keys, key):
                target = target[target[col] == val]
            if target.empty:
                continue
            for metric in METRICS:
                obs = float(target.iloc[0][metric])
                vals = g[metric].dropna().to_numpy(float)
                if not len(vals) or not np.isfinite(obs):
                    continue
                sd = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
                # Two-sided Monte Carlo p with the standard plus-one correction.
                centre = float(vals.mean())
                p = (1 + np.sum(np.abs(vals - centre) >= abs(obs - centre))) / (len(vals) + 1)
                rows.append(dict(zip(keys, key), null=null_name, metric=metric,
                                 observed=obs, null_mean=centre, difference=obs-centre,
                                 standardized_difference=(obs-centre)/sd if sd > 0 else np.nan,
                                 empirical_p_two_sided=float(p), null_replicates=len(vals)))
    return pd.DataFrame(rows)


def make_figure(rank, effects):
    pretty = {"touched_fraction": "Touched fraction", "mean_death": "Mean death",
              "wasserstein": "Wasserstein", "beta0_auc_normalised": "Normalised beta0 AUC"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    r = rank[rank.episode == rank.episode.max()]
    positions = np.arange(len(METRICS))
    for i, definition in enumerate(["either_endpoint", "both_endpoints"]):
        vals = [r[(r.definition == definition) & (r.metric == m)].spearman_rho.median()
                for m in METRICS]
        axes[0].bar(positions + (i-.5)*.34, vals, .34, label=definition.replace("_", " "))
    axes[0].axhline(0, color="black", lw=.7)
    axes[0].set_ylim(-1, 1)
    axes[0].set_ylabel("Median Spearman rank correlation")
    axes[0].set_title("Birth representative vs symmetric definitions")
    axes[0].set_xticks(positions, [pretty[m] for m in METRICS], rotation=25, ha="right")
    axes[0].legend(frameon=False)

    e = effects[(effects.episode == effects.episode.max()) &
                (effects.metric.isin(["mean_death", "wasserstein", "beta0_auc_normalised"]))]
    labels, data = [], []
    for null in ["size_matched_null", "degree_matched_null"]:
        for metric in ["mean_death", "wasserstein", "beta0_auc_normalised"]:
            labels.append(("Size" if null.startswith("size") else "Degree") + "\n" + pretty[metric])
            data.append(e[(e.null == null) & (e.metric == metric)].standardized_difference.dropna())
    axes[1].boxplot(data, labels=labels, showfliers=False)
    axes[1].axhline(0, color="black", lw=.7)
    axes[1].set_ylabel("Observed minus null mean (null SD units)")
    axes[1].set_title("Separation from matched null controls")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(OUT / "h0_robustness_summary.pdf", bbox_inches="tight")
    fig.savefig(OUT / "h0_robustness_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def report(df, rank, pairing, effects):
    max_ep = int(df.episode.max())
    r = rank[rank.episode == max_ep]
    birth = observed(df, "birth")
    # Under a birth representative, every finite H0 class has one unique vertex.
    ratio = birth.touched_count / birth.selected_count
    pair_cols = [c for c in pairing if c.endswith("_pairing_sd")]
    e = effects[effects.episode == max_ep]
    lines = [
        "# Decision memo: H0 diagnostic robustness", "",
        "## Decision", "",
        "The original **birth-vertex touched-event coverage claim does not survive**",
        "the invariance audit and must not remain a central claim. In degree-0 persistence,",
        "all vertices are born at filtration zero; choosing one birth representative is an",
        "algorithmic pairing convention, not an intrinsic topological attribution.", "",
        "The defensible replacement is narrower: selected sets can be compared by the",
        "**distribution of merge scales incident to either endpoint of an MST death edge**.",
        "Mean-death and normalised-beta0 summaries are usually rank-stable under symmetric",
        "endpoint definitions, while Wasserstein is less uniform and touched fraction is",
        "not stable enough to support the former interpretation.", "",
        "## Quantitative audit", "",
        "- Episode analysed for the headline summary: {}.".format(max_ep),
        "- Birth touched-count / selected-count median: {:.3f} (range {:.3f}--{:.3f}).".format(
            ratio.median(), ratio.min(), ratio.max()),
    ]
    for metric in METRICS:
        x = r[(r.definition == "either_endpoint") & (r.metric == metric)].spearman_rho
        lines.append("- Birth vs either-endpoint {} rank correlation: median {:.3f}, range {:.3f}--{:.3f}.".format(
            metric, x.median(), x.min(), x.max()))
    lines += [
        "- Pairing-replicate SD medians: " + ", ".join(
            "{}={:.6g}".format(c.replace("_pairing_sd", ""), pairing[c].median()) for c in pair_cols) + ".",
        "- Null separation is reported per run in `null_effects_per_run.csv`; positive values",
        "  mean the observed birth-based statistic exceeded its matched-null mean.", "",
        "## Required manuscript change", "",
        "1. Remove language that treats a GUDHI birth vertex as the unique creator or owner",
        "   of an H0 death event.",
        "2. Make either-death-edge-endpoint incidence the primary diagnostic; retain both-endpoint",
        "   incidence as a stricter sensitivity analysis.",
        "3. Report size- and exact-MST-degree-matched nulls beside every post-hoc statistic.",
        "4. Label birth-representative results as implementation sensitivity only.",
        "5. Do not use touched-event count/fraction as evidence of superior topology coverage;",
        "   under unique birth representatives it is largely determined by selected-set size.", "",
        "## Files", "",
        "- `rank_stability.csv`", "- `pairing_variability_per_run.csv`",
        "- `null_effects_per_run.csv`", "- `h0_robustness_summary.pdf`",
        "- `h0_robustness_summary.png`", "",
    ]
    (OUT / "robustness_decision.md").write_text("\n".join(lines))


def main():
    df = pd.read_csv(IN)
    rank = rank_stability(df)
    pairing = pairing_variability(df)
    effects = null_effects(df)
    rank.to_csv(OUT / "rank_stability.csv", index=False)
    pairing.to_csv(OUT / "pairing_variability_per_run.csv", index=False)
    effects.to_csv(OUT / "null_effects_per_run.csv", index=False)
    make_figure(rank, effects)
    report(df, rank, pairing, effects)
    print("wrote robustness decision, tables, and figure")


if __name__ == "__main__":
    main()
