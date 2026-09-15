#!/usr/bin/env python3
"""Aggregate five strictly replayed seeds into the mandated mechanism package."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
STAGING = ROOT / "analysis/lidcover_extensions_2026/mechanism_staging"
OUT = ROOT / "analysis/lidcover_extensions_2026/mechanism_results"
MATRIX = ROOT / "thesis_appendix_audit/_regenerated_lidcover/lidcover_completion_matrix.csv"


def sem(values):
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")


def seed_correlations(frame):
    rows = []
    for seed, group in frame.groupby("seed"):
        rows.append({
            "seed": seed,
            "spearman_lid_adaptive_gain": spearmanr(group.pointwise_lid, group.actual_uncovered_gain_immediately_before_selection).statistic,
            "spearman_lid_radius": spearmanr(group.pointwise_lid, group.adaptive_radius).statistic,
            "spearman_purity_adaptive_gain_posthoc": spearmanr(group.local_class_purity_k50_posthoc, group.actual_uncovered_gain_immediately_before_selection).statistic,
            "fallback_rate": group.selected_through_zero_gain_fallback.mean(),
            "tie_rate": group.maximum_gain_tied.mean(),
            "adaptive_minus_fixed_gain_mean": group.adaptive_minus_fixed_counterfactual_gain.mean(),
        })
    seeds = pd.DataFrame(rows)
    summary = []
    for column in seeds.columns.drop("seed"):
        summary.append({"metric": column, "mean_across_seeds": seeds[column].mean(), "sem_across_seeds": sem(seeds[column]), "n_seeds": len(seeds)})
    return seeds, pd.DataFrame(summary)


def quartile_summary(frame):
    per_seed = frame.groupby(["seed", "lid_quartile"]).agg(
        selected_count=("sample_index", "size"), radius_ratio=("radius_ratio", "mean"),
        adaptive_gain=("actual_uncovered_gain_immediately_before_selection", "mean"),
        fixed_counterfactual_gain=("fixed_radius_counterfactual_uncovered_gain", "mean"),
        adaptive_minus_fixed_gain=("adaptive_minus_fixed_counterfactual_gain", "mean"),
        purity_posthoc=("local_class_purity_k50_posthoc", "mean"),
    ).reset_index()
    per_seed["selected_fraction"] = per_seed.selected_count / per_seed.groupby("seed").selected_count.transform("sum")
    rows = []
    for quartile, group in per_seed.groupby("lid_quartile"):
        row = {"lid_quartile": quartile, "n_seeds": group.seed.nunique()}
        for column in ("selected_fraction", "radius_ratio", "adaptive_gain", "fixed_counterfactual_gain", "adaptive_minus_fixed_gain", "purity_posthoc"):
            row[f"{column}_mean"] = group[column].mean(); row[f"{column}_sem"] = sem(group[column])
        rows.append(row)
    return pd.DataFrame(rows)


def tie_fallback_summary(frame):
    per_seed = frame.groupby(["seed", "labelled_budget"]).agg(
        fallback_rate=("selected_through_zero_gain_fallback", "mean"), tie_rate=("maximum_gain_tied", "mean")
    ).reset_index()
    return per_seed.groupby("labelled_budget").agg(
        fallback_rate_mean=("fallback_rate", "mean"), fallback_rate_sem=("fallback_rate", sem),
        tie_rate_mean=("tie_rate", "mean"), tie_rate_sem=("tie_rate", sem), n_seeds=("seed", "nunique"),
    ).reset_index()


def matched_performance():
    selected = []
    with MATRIX.open(newline="") as handle:
        for row in csv.DictReader(handle):
            radius = row["delta0"] if row["method"] == "LIDCover" else row["delta"]
            if (
                row["status"] == "COMPLETE" and "candidate" not in row["experiment_id"]
                and row["dataset"] == "CIFAR100" and row["backbone"] == "resnet18"
                and row["method"] in {"LIDCover", "ProbCover"} and radius == "0.25"
            ):
                summary = json.loads((Path(row["source_path"]) / "benchmark_summary.json").read_text())
                records = summary["episode_records"]
                x = np.asarray([record["labeled_count_before_sampling"] for record in records], dtype=float)
                y = np.asarray([record["test_accuracy"] for record in records], dtype=float)
                selected.append({"method": row["method"], "seed": int(row["seed"]), "final_accuracy": y[-1], "normalized_aulc": np.trapz(y, x) / (x[-1] - x[0])})
    frame = pd.DataFrame(selected)
    if len(frame) != 10: raise RuntimeError(f"expected ten matched canonical performance runs, found {len(frame)}")
    return frame


def figures(frame, quartiles, tie_fallback):
    figure, axis = plt.subplots(figsize=(6.2, 4.5)); sample = frame.sample(min(len(frame), 12000), random_state=0)
    axis.scatter(sample.pointwise_lid, sample.radius_ratio, s=4, alpha=.18); axis.set(xlabel="Pointwise LID", ylabel="Adaptive/base radius ratio")
    axis.grid(alpha=.2); figure.tight_layout(); figure.savefig(OUT / "mechanism_lid_vs_radius.pdf", bbox_inches="tight"); plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.2, 4.5)); axis.errorbar(quartiles.lid_quartile, quartiles.adaptive_gain_mean, yerr=quartiles.adaptive_gain_sem, marker="o", label="Adaptive")
    axis.errorbar(quartiles.lid_quartile, quartiles.fixed_counterfactual_gain_mean, yerr=quartiles.fixed_counterfactual_gain_sem, marker="s", label="Fixed-radius counterfactual")
    axis.set(xlabel="LID quartile", ylabel="Uncovered gain"); axis.grid(alpha=.2); axis.legend(); figure.tight_layout(); figure.savefig(OUT / "mechanism_lid_quartile_gain.pdf", bbox_inches="tight"); plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.5, 5)); axis.scatter(frame.fixed_radius_counterfactual_uncovered_gain, frame.actual_uncovered_gain_immediately_before_selection, s=4, alpha=.15)
    limit = max(frame.fixed_radius_counterfactual_uncovered_gain.max(), frame.actual_uncovered_gain_immediately_before_selection.max()); axis.plot([0, limit], [0, limit], "k--", lw=1)
    axis.set(xlabel="Fixed-radius counterfactual gain", ylabel="Actual adaptive gain"); axis.grid(alpha=.2); figure.tight_layout(); figure.savefig(OUT / "mechanism_adaptive_vs_counterfactual_gain.pdf", bbox_inches="tight"); plt.close(figure)

    for stem, mean_col, sem_col, ylabel in (
        ("mechanism_fallback_over_budget.pdf", "fallback_rate_mean", "fallback_rate_sem", "Fallback rate"),
        ("mechanism_tie_over_budget.pdf", "tie_rate_mean", "tie_rate_sem", "Positive-gain tie rate"),
    ):
        figure, axis = plt.subplots(figsize=(6.4, 4.3)); axis.plot(tie_fallback.labelled_budget, tie_fallback[mean_col]); axis.fill_between(tie_fallback.labelled_budget, tie_fallback[mean_col] - tie_fallback[sem_col], tie_fallback[mean_col] + tie_fallback[sem_col], alpha=.2)
        axis.set(xlabel="Labelled budget", ylabel=ylabel); axis.grid(alpha=.2); figure.tight_layout(); figure.savefig(OUT / stem, bbox_inches="tight"); plt.close(figure)


def main():
    frames, provenance = [], []
    for seed in range(1, 6):
        path = STAGING / f"mechanism_seed{seed}.csv"; prov = STAGING / f"mechanism_seed{seed}_provenance.json"
        if not path.is_file() or not prov.is_file(): raise FileNotFoundError(f"missing strict replay output for seed {seed}")
        metadata = json.loads(prov.read_text())
        if metadata.get("status") != "STRICT_REPLAY_PASS" or metadata.get("historical_geometry_cache_used") is not False:
            raise RuntimeError(f"seed {seed} did not pass strict replay")
        frames.append(pd.read_csv(path)); provenance.append(metadata)
    frame = pd.concat(frames, ignore_index=True); OUT.mkdir(parents=True, exist_ok=True)
    try: frame.to_parquet(OUT / "lidcover_mechanism_per_selection.parquet", index=False)
    except ImportError as exc: raise RuntimeError("pyarrow or fastparquet is required for the mandated Parquet output") from exc
    seed_stats, summary = seed_correlations(frame); quartiles = quartile_summary(frame); tie_fallback = tie_fallback_summary(frame)
    performance = matched_performance()
    seed_stats.to_csv(OUT / "mechanism_seed_summary.csv", index=False); summary.to_csv(OUT / "mechanism_summary.csv", index=False)
    quartiles.to_csv(OUT / "mechanism_quartile_summary.csv", index=False); tie_fallback.to_csv(OUT / "mechanism_tie_fallback_summary.csv", index=False)
    performance.to_csv(OUT / "mechanism_matched_performance_context.csv", index=False)
    (OUT / "table_mechanism_summary.tex").write_text(summary.to_latex(index=False, float_format=lambda value: f"{value:.4f}", caption="LIDCover mechanism associations aggregated at seed level."))
    figures(frame, quartiles, tie_fallback)
    report = f"""# LIDCover mechanism analysis

## Canonical scope and validation

CIFAR100 / ResNet18 / configured delta0 0.25 / alpha 1 / k_id=k_knn=50 / empty cold start / five AL seeds. All {len(frame):,} selections were replayed sequentially from the exact saved labelled, unlabelled, and ordered selected-ID artifacts. The fixed representation and exact k=50 geometry were recomputed; no historical geometry cache was trusted. Every selected ID matched the frozen minimum-LID tie/fallback rule at its actual query position.

## Interpretation boundaries

- **Mathematical consequence:** with alpha=1, radius ratio is the reciprocal of relative LID. This is definitional, not empirical evidence.
- **Empirical selection effects:** quartile gain, adaptive-minus-fixed counterfactual gain, tie rate, and fallback rate describe realised selections in these runs.
- **Post-hoc label diagnostics:** purity, same-class-neighbour fraction, and class frequency use labels only after replay. Labels never enter acquisition reconstruction.
- **Correlations:** Spearman associations are descriptive and are aggregated with the AL seed as the replicate where possible.
- **Causal claims:** these observational and same-candidate counterfactual diagnostics do not identify causal effects.

Matched-radius ProbCover performance is included only as descriptive predictive context in `mechanism_matched_performance_context.csv`; it is not substituted for the sparse fixed-radius counterfactual gain of the same selected LIDCover candidates.
"""
    (OUT / "MECHANISM_ANALYSIS_REPORT.md").write_text(report)
    (OUT / "mechanism_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {len(frame)} strict per-selection rows")


if __name__ == "__main__": main()
