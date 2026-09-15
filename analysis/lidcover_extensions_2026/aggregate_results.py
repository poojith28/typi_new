#!/usr/bin/env python3
"""Strict aggregation for completed extension runs; no accuracy interpolation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence/lidcover_extensions_2026"
OUT = ROOT / "analysis/lidcover_extensions_2026/results"
COMMON_BATCH_BUDGETS = np.arange(500, 5001, 500, dtype=int)


def load_manifest(name):
    return list(csv.DictReader((EVIDENCE / name).open(newline="")))


def load_run(row):
    run = Path(row["output_dir"])
    summary_path = run / "benchmark_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing completed summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    records = summary.get("episode_records", [])
    expected = int(row["max_iter"]) + 1
    if len(records) != expected or int(summary.get("num_rounds_completed", -1)) != expected:
        raise RuntimeError(f"non-canonical episode sequence: {run}")
    if str(summary.get("sampling_fn", "")).lower() != row["method"]:
        raise RuntimeError(f"method mismatch: {run}")
    points = []
    for record in records:
        budget = int(record["labeled_count_before_sampling"])
        accuracy = float(record["test_accuracy"])
        if not math.isfinite(accuracy):
            raise RuntimeError(f"non-finite accuracy: {run}")
        points.append((budget, accuracy, int(record["episode"])))
    budgets = [point[0] for point in points]
    expected_budgets = list(range(int(row["acquisition_batch_size"]), int(row["final_evaluated_budget"]) + 1, int(row["acquisition_batch_size"])))
    if budgets != expected_budgets:
        raise RuntimeError(f"unexpected exact checkpoint sequence: {run}")
    return summary, points


def normalized_aulc(points):
    x = np.asarray([point[0] for point in points], dtype=float)
    y = np.asarray([point[1] for point in points], dtype=float)
    if len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("AULC requires at least two increasing exact checkpoints")
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def sem(values):
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")


def write_tex(path, frame, caption):
    path.write_text(frame.to_latex(index=False, float_format=lambda value: f"{value:.4f}", caption=caption))


def plot_curves(frame, group_columns, path, title):
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for key, group in frame.groupby(group_columns):
        key = key if isinstance(key, tuple) else (key,)
        stats = group.groupby("labelled_budget").test_accuracy.agg(["mean", "sem"]).reset_index()
        label = ", ".join(map(str, key))
        axis.plot(stats.labelled_budget, stats["mean"], label=label)
        axis.fill_between(stats.labelled_budget, stats["mean"] - stats["sem"], stats["mean"] + stats["sem"], alpha=0.16)
    axis.set(xlabel="Labelled budget", ylabel="Test accuracy", title=title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def aggregate_auto():
    rows, run_stats, scales, provenance_rows = [], [], [], []
    for manifest_row in load_manifest("auto_radius_manifest.csv"):
        summary, points = load_run(manifest_row)
        for budget, accuracy, episode in points:
            rows.append({**{key: manifest_row[key] for key in ("dataset", "method", "seed")}, "episode": episode, "labelled_budget": budget, "test_accuracy": accuracy})
        run_stats.append({
            "dataset": manifest_row["dataset"], "method": manifest_row["method"], "seed": int(manifest_row["seed"]),
            "final_accuracy": points[-1][1], "normalized_trapezoidal_aulc": normalized_aulc(points),
        })
        provenance = json.loads((Path(manifest_row["output_dir"]) / "auto_delta_provenance.json").read_text())
        provenance_rows.append(provenance)
        scales.append({
            "dataset": manifest_row["dataset"], "method": manifest_row["method"], "seed": int(manifest_row["seed"]),
            "delta_auto": float(provenance["delta_auto"]), "representation_sha256": provenance["representation_sha256"],
            "candidate_index_sha256": provenance["candidate_index_sha256"],
        })
    curves, runs, scale_frame = pd.DataFrame(rows), pd.DataFrame(run_stats), pd.DataFrame(scales)
    for (_, seed), pair in scale_frame.groupby(["dataset", "seed"]):
        if pair.delta_auto.nunique() != 1 or pair.representation_sha256.nunique() != 1 or pair.candidate_index_sha256.nunique() != 1:
            raise RuntimeError("matched auto-radius pair does not share identical calibrated geometry")
    summary = runs.groupby(["dataset", "method"]).agg(
        final_accuracy_mean=("final_accuracy", "mean"), final_accuracy_sem=("final_accuracy", sem),
        normalized_aulc_mean=("normalized_trapezoidal_aulc", "mean"), normalized_aulc_sem=("normalized_trapezoidal_aulc", sem),
    ).reset_index()
    curves.to_csv(OUT / "auto_radius_learning_curves.csv", index=False)
    scale_frame.to_csv(OUT / "auto_radius_selected_scales.csv", index=False)
    pd.DataFrame(provenance_rows).to_csv(OUT / "auto_radius_provenance.csv", index=False)
    write_tex(OUT / "table_auto_radius.tex", summary, "Automatic-radius results (mean and SEM over five AL seeds).")
    plot_curves(curves, ["dataset", "method"], OUT / "figure_auto_radius_learning_curves.pdf", "Automatic base radius")
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    unique = scale_frame.drop_duplicates(["dataset", "seed"])
    for dataset, group in unique.groupby("dataset"):
        axis.plot(group.seed, group.delta_auto, marker="o", label=dataset)
    axis.set(xlabel="AL seed", ylabel=r"Selected $\delta_{auto}$")
    axis.grid(alpha=0.25); axis.legend(); figure.tight_layout()
    figure.savefig(OUT / "figure_auto_radius_selected_scales.pdf", bbox_inches="tight"); plt.close(figure)


def aggregate_batch():
    common_rows, run_rows = [], []
    for manifest_row in load_manifest("batch_size_manifest.csv"):
        _, points = load_run(manifest_row)
        lookup = {budget: (accuracy, episode) for budget, accuracy, episode in points}
        missing = [int(budget) for budget in COMMON_BATCH_BUDGETS if int(budget) not in lookup]
        if missing:
            raise RuntimeError(f"no exact checkpoint(s) {missing}: {manifest_row['output_dir']}")
        common = [(int(budget), lookup[int(budget)][0], lookup[int(budget)][1]) for budget in COMMON_BATCH_BUDGETS]
        for budget, accuracy, episode in common:
            common_rows.append({
                "method": manifest_row["method"], "batch_size": int(manifest_row["acquisition_batch_size"]),
                "seed": int(manifest_row["seed"]), "labelled_budget": budget, "episode": episode,
                "test_accuracy": accuracy, "checkpoint_status": "EXACT_SAVED",
            })
        run_rows.append({
            "method": manifest_row["method"], "batch_size": int(manifest_row["acquisition_batch_size"]),
            "seed": int(manifest_row["seed"]), "final_accuracy_5000": common[-1][1],
            "normalized_aulc_common_500_5000": normalized_aulc(common),
        })
    common, runs = pd.DataFrame(common_rows), pd.DataFrame(run_rows)
    common.to_csv(OUT / "batch_size_common_budget.csv", index=False)
    final = runs.groupby(["method", "batch_size"]).agg(mean=("final_accuracy_5000", "mean"), sem=("final_accuracy_5000", sem)).reset_index()
    aulc = runs.groupby(["method", "batch_size"]).agg(mean=("normalized_aulc_common_500_5000", "mean"), sem=("normalized_aulc_common_500_5000", sem)).reset_index()
    write_tex(OUT / "table_batch_size_5000.tex", final, "Accuracy at exactly 5,000 labels.")
    write_tex(OUT / "table_batch_size_aulc_common_range.tex", aulc, "Normalised trapezoidal AULC on exact common checkpoints from 500 to 5,000 labels.")
    plot_curves(common, ["method", "batch_size"], OUT / "figure_batch_size_learning_curves.pdf", "Matched batch-size study")


def aggregate_tie_fallback():
    curve_rows, run_rows, round_frames, selection_rows = [], [], [], []
    for manifest_row in load_manifest("tie_fallback_manifest.csv"):
        _, points = load_run(manifest_row)
        for budget, accuracy, episode in points:
            curve_rows.append({"method": manifest_row["method"], "seed": int(manifest_row["seed"]), "episode": episode, "labelled_budget": budget, "test_accuracy": accuracy})
        run_rows.append({"method": manifest_row["method"], "seed": int(manifest_row["seed"]), "final_accuracy": points[-1][1], "normalized_aulc": normalized_aulc(points)})
        run = Path(manifest_row["output_dir"])
        metrics = pd.read_csv(run / "tie_fallback_round_metrics.csv")
        round_frames.append(metrics)
        with (run / "tie_fallback_selection_metrics.jsonl").open() as handle:
            selection_rows.extend(json.loads(line) for line in handle if line.strip())
    curves, runs = pd.DataFrame(curve_rows), pd.DataFrame(run_rows)
    rounds, selections = pd.concat(round_frames, ignore_index=True), pd.DataFrame(selection_rows)
    rounds.to_csv(OUT / "tie_fallback_round_metrics.csv", index=False)
    try:
        selections.to_parquet(OUT / "tie_fallback_selection_metrics.parquet", index=False)
    except ImportError as exc:
        raise RuntimeError("pyarrow or fastparquet is required for the mandated Parquet output") from exc
    reference = runs[runs.method == "lidcover_tf_minlid_minlid"].set_index("seed")
    paired_rows = []
    for method, group in runs.groupby("method"):
        aligned = group.set_index("seed").join(
            reference[["final_accuracy", "normalized_aulc"]], rsuffix="_reference", how="inner"
        )
        if len(aligned) != 5:
            raise RuntimeError(f"paired seed alignment failed for {method}")
        final_diff = aligned.final_accuracy - aligned.final_accuracy_reference
        aulc_diff = aligned.normalized_aulc - aligned.normalized_aulc_reference
        paired_rows.append({
            "method": method,
            "final_mean": group.final_accuracy.mean(), "final_sem": sem(group.final_accuracy),
            "aulc_mean": group.normalized_aulc.mean(), "aulc_sem": sem(group.normalized_aulc),
            "paired_final_difference_vs_minlid_minlid_mean": final_diff.mean(),
            "paired_final_difference_vs_minlid_minlid_sem": sem(final_diff),
            "paired_aulc_difference_vs_minlid_minlid_mean": aulc_diff.mean(),
            "paired_aulc_difference_vs_minlid_minlid_sem": sem(aulc_diff),
        })
    performance = pd.DataFrame(paired_rows)
    frequency = rounds.groupby("method").agg(positive_tie_count_mean=("positive_gain_tie_count", "mean"), fallback_fraction_mean=("fallback_fraction", "mean")).reset_index()
    write_tex(OUT / "table_tie_fallback_performance.tex", performance, "Tie/fallback factorial performance over paired seeds.")
    write_tex(OUT / "table_tie_fallback_frequency.tex", frequency, "Tie and fallback frequency summaries.")
    plot_curves(curves, ["method"], OUT / "figure_tie_fallback_learning_curves.pdf", "Tie/fallback factorial")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method, group in rounds.groupby("method"):
        axes[0].plot(group.labelled_budget, group.fallback_fraction, alpha=0.4, label=method)
        axes[1].plot(group.labelled_budget, group.positive_gain_tie_count, alpha=0.4, label=method)
    axes[0].set(xlabel="Labelled budget", ylabel="Fallback fraction"); axes[1].set(xlabel="Labelled budget", ylabel="Positive-gain tie count")
    axes[0].grid(alpha=.2); axes[1].grid(alpha=.2); axes[1].legend(fontsize=6)
    figure.tight_layout(); figure.savefig(OUT / "figure_tie_fallback_frequency.pdf", bbox_inches="tight"); plt.close(figure)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    aggregate_auto()
    aggregate_batch()
    aggregate_tie_fallback()
    print(f"wrote strict aggregate outputs to {OUT}")


if __name__ == "__main__":
    main()
