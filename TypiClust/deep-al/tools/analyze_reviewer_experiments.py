#!/usr/bin/env python3

import argparse
import json
import math
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


METHOD_ORDER = [
    "random",
    "uncertainty",
    "entropy",
    "margin",
    "coreset",
    "dbal",
    "probcover",
    "knn_distance_cover",
    "density_cover",
    "distance_variance_cover",
    "distance_cv_cover",
    "idprobcover",
    "idprobcover_tiebreak_min_id",
    "idprobcover_tiebreak_random",
    "idprobcover_tiebreak_first_max",
]


def _safe_mean(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def _safe_std(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0


def _safe_se(values):
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.nanstd(arr, ddof=1) / math.sqrt(arr.size))


def _bootstrap_ci(values, rng, num_bootstrap=10000, alpha=0.05):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    samples = rng.choice(arr, size=(num_bootstrap, arr.size), replace=True)
    means = np.mean(samples, axis=1)
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return low, high


def _parse_seed(exp_name, summary_seed):
    if summary_seed is not None:
        return int(summary_seed)
    match = re.search(r"_s(\d+)$", str(exp_name))
    if match:
        return int(match.group(1))
    match = re.search(r"seed[_-]?(\d+)", str(exp_name))
    if match:
        return int(match.group(1))
    return None


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _load_run_summary(run_dir):
    bench_path = run_dir / "benchmark_summary.json"
    if bench_path.exists():
        summary = _load_json(bench_path)
        summary["_source"] = str(bench_path)
        return summary

    episode_paths = sorted(run_dir.glob("episode_*/episode_summary.json"))
    if not episode_paths:
        return None
    records = [_load_json(path) for path in episode_paths]
    first = records[0]
    return {
        "sampling_fn": first.get("sampling_fn"),
        "dataset": run_dir.parts[-3] if len(run_dir.parts) >= 3 else "unknown",
        "model": run_dir.parts[-2] if len(run_dir.parts) >= 2 else "unknown",
        "seed": first.get("seed"),
        "exp_name": first.get("exp_name", run_dir.name),
        "exp_dir": str(run_dir),
        "episode_records": records,
        "_source": "episode_summary_only",
    }


def discover_runs(output_root, dataset, model, methods):
    run_root = Path(output_root) / dataset / model
    if not run_root.exists():
        return []

    discovered = []
    for run_dir in sorted([p for p in run_root.iterdir() if p.is_dir()]):
        summary = _load_run_summary(run_dir)
        if not summary:
            continue
        method = str(summary.get("sampling_fn", "")).lower()
        if methods and method not in methods:
            continue
        exp_name = summary.get("exp_name", run_dir.name)
        seed = _parse_seed(exp_name, summary.get("seed"))
        discovered.append({
            "dataset": summary.get("dataset", dataset),
            "model": summary.get("model", model),
            "method": method,
            "seed": seed,
            "exp_name": exp_name,
            "exp_dir": str(run_dir),
            "episode_records": summary.get("episode_records", []),
            "timing": summary.get("timing", {}),
            "initial_sampling": summary.get("initial_sampling", {}),
            "source": summary.get("_source"),
        })
    return discovered


def build_episode_frame(runs):
    rows = []
    for run in runs:
        initial_sampling = run.get("initial_sampling", {}) or {}
        if initial_sampling:
            initial_timing = initial_sampling.get("timing", {})
            rows.append({
                "dataset": run["dataset"],
                "model": run["model"],
                "method": run["method"],
                "seed": run["seed"],
                "exp_name": run["exp_name"],
                "episode": -1,
                "test_accuracy": float("nan"),
                "best_val_accuracy": float("nan"),
                "train_time_sec": float("nan"),
                "test_time_sec": float("nan"),
                "acquisition_time_sec": float(initial_sampling.get("acquisition_time_sec", initial_timing.get("acquisition_time_sec", float("nan")))),
                "round_time_sec": float(initial_sampling.get("acquisition_time_sec", initial_timing.get("acquisition_time_sec", float("nan")))),
                "has_sampling": True,
                "is_initial_sampling": True,
            })
        for record in run["episode_records"]:
            timing = record.get("timing", {})
            rows.append({
                "dataset": run["dataset"],
                "model": run["model"],
                "method": run["method"],
                "seed": run["seed"],
                "exp_name": run["exp_name"],
                "episode": int(record.get("episode", -1)),
                "test_accuracy": float(record.get("test_accuracy", float("nan"))),
                "best_val_accuracy": float(record.get("best_val_accuracy", float("nan"))),
                "train_time_sec": float(record.get("train_time_sec", timing.get("train_time_sec", float("nan")))),
                "test_time_sec": float(record.get("test_time_sec", timing.get("test_time_sec", float("nan")))),
                "acquisition_time_sec": float(record.get("acquisition_time_sec", timing.get("acquisition_time_sec", float("nan")))),
                "round_time_sec": float(record.get("round_time_sec", timing.get("round_time_sec", float("nan")))),
                "has_sampling": bool(timing.get("has_sampling", record.get("active_set_size", 0) > 0)),
                "is_initial_sampling": False,
            })
    return pd.DataFrame(rows)


def checkpoint_table(df, checkpoints):
    rows = []
    for method in sorted(df["method"].dropna().unique(), key=lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else x):
        method_df = df[df["method"] == method]
        for checkpoint in checkpoints:
            cp_df = method_df[method_df["episode"] == checkpoint]
            values = cp_df["test_accuracy"].to_numpy(dtype=float)
            rows.append({
                "method": method,
                "checkpoint": checkpoint,
                "n": int(np.sum(~np.isnan(values))),
                "mean": _safe_mean(values),
                "std": _safe_std(values),
                "se": _safe_se(values),
            })
    return pd.DataFrame(rows)


def per_seed_checkpoint_table(df, checkpoints):
    rows = []
    for checkpoint in checkpoints:
        cp_df = df[df["episode"] == checkpoint]
        for _, row in cp_df.iterrows():
            rows.append({
                "method": row["method"],
                "seed": row["seed"],
                "checkpoint": checkpoint,
                "test_accuracy": row["test_accuracy"],
            })
    return pd.DataFrame(rows)


def timing_table(df):
    sampled = df[df["has_sampling"]]
    rows = []
    for method in sorted(df["method"].dropna().unique(), key=lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else x):
        method_df = sampled[sampled["method"] == method]
        if method_df.empty:
            continue
        grouped = method_df.groupby(["exp_name", "seed"], dropna=False)
        cumulative_acq = grouped["acquisition_time_sec"].sum().to_numpy(dtype=float)
        cumulative_round = grouped["round_time_sec"].sum().to_numpy(dtype=float)
        initial_df = method_df[method_df.get("is_initial_sampling", False)]
        initial_times = initial_df["acquisition_time_sec"].to_numpy(dtype=float) if not initial_df.empty else np.asarray([], dtype=float)
        rows.append({
            "method": method,
            "rounds": int(len(method_df)),
            "runs": int(len(grouped)),
            "initial_sampling_events": int(len(initial_df)),
            "initial_sampling_acquisition_time_mean_sec": _safe_mean(initial_times),
            "initial_sampling_acquisition_time_median_sec": float(np.nanmedian(initial_times)) if initial_times.size else float("nan"),
            "acquisition_time_mean_sec": _safe_mean(method_df["acquisition_time_sec"]),
            "acquisition_time_median_sec": float(np.nanmedian(method_df["acquisition_time_sec"])),
            "train_time_mean_sec": _safe_mean(method_df["train_time_sec"]),
            "train_time_median_sec": float(np.nanmedian(method_df["train_time_sec"])),
            "test_time_mean_sec": _safe_mean(method_df["test_time_sec"]),
            "test_time_median_sec": float(np.nanmedian(method_df["test_time_sec"])),
            "round_time_mean_sec": _safe_mean(method_df["round_time_sec"]),
            "round_time_median_sec": float(np.nanmedian(method_df["round_time_sec"])),
            "cumulative_acquisition_time_mean_sec": _safe_mean(cumulative_acq),
            "cumulative_acquisition_time_median_sec": float(np.nanmedian(cumulative_acq)),
            "cumulative_round_time_mean_sec": _safe_mean(cumulative_round),
            "cumulative_round_time_median_sec": float(np.nanmedian(cumulative_round)),
        })
    return pd.DataFrame(rows)


def paired_significance(df, reference_method, target_method, checkpoints, seed_filter=None):
    rng = np.random.default_rng(0)
    rows = []
    for checkpoint in checkpoints:
        ref = df[(df["method"] == reference_method) & (df["checkpoint"] == checkpoint)]
        tgt = df[(df["method"] == target_method) & (df["checkpoint"] == checkpoint)]
        merged = ref.merge(tgt, on="seed", suffixes=("_ref", "_tgt"))
        if seed_filter is not None:
            merged = merged[merged["seed"].isin(seed_filter)]
        diffs = merged["test_accuracy_tgt"].to_numpy(dtype=float) - merged["test_accuracy_ref"].to_numpy(dtype=float)
        ttest_p = float("nan")
        wilcoxon_p = float("nan")
        if len(diffs) >= 2 and stats is not None:
            try:
                ttest_p = float(stats.ttest_rel(merged["test_accuracy_tgt"], merged["test_accuracy_ref"], nan_policy="omit").pvalue)
            except Exception:
                pass
            try:
                wilcoxon_p = float(stats.wilcoxon(diffs).pvalue)
            except Exception:
                pass
        ci_low, ci_high = _bootstrap_ci(diffs, rng)
        rows.append({
            "checkpoint": checkpoint,
            "reference_method": reference_method,
            "target_method": target_method,
            "paired_n": int(len(diffs)),
            "mean_difference": _safe_mean(diffs),
            "t_test_pvalue": ttest_p,
            "wilcoxon_pvalue": wilcoxon_p,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
        })
    return pd.DataFrame(rows)


def significance_vs_all(df, target_method, checkpoints, methods=None, seed_filter=None):
    methods = methods or sorted(df["method"].dropna().unique())
    frames = []
    for reference_method in methods:
        if reference_method == target_method:
            continue
        frame = paired_significance(
            df,
            reference_method=reference_method,
            target_method=target_method,
            checkpoints=checkpoints,
            seed_filter=seed_filter,
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[
            "checkpoint",
            "reference_method",
            "target_method",
            "paired_n",
            "mean_difference",
            "t_test_pvalue",
            "wilcoxon_pvalue",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
        ])
    return pd.concat(frames, ignore_index=True)


def reviewer_table(checkpoint_summary_df, significance_df, timing_df, reference_method="probcover", target_method="idprobcover"):
    ref_rows = checkpoint_summary_df[checkpoint_summary_df["method"] == reference_method].set_index("checkpoint")
    tgt_rows = checkpoint_summary_df[checkpoint_summary_df["method"] == target_method].set_index("checkpoint")
    timing_rows = timing_df.set_index("method")
    sig_rows = significance_df.set_index("checkpoint")
    rows = []
    if reference_method not in timing_rows.index or target_method not in timing_rows.index:
        acquisition_ratio = float("nan")
        end_to_end_ratio = float("nan")
    else:
        acquisition_ratio = (
            float(timing_rows.loc[target_method, "acquisition_time_mean_sec"]) /
            max(float(timing_rows.loc[reference_method, "acquisition_time_mean_sec"]), 1e-12)
        )
        end_to_end_ratio = (
            float(timing_rows.loc[target_method, "round_time_mean_sec"]) /
            max(float(timing_rows.loc[reference_method, "round_time_mean_sec"]), 1e-12)
        )
    for checkpoint in sorted(sig_rows.index):
        if checkpoint not in ref_rows.index or checkpoint not in tgt_rows.index:
            continue
        rows.append({
            "checkpoint": checkpoint,
            "probcover_mean_pm_se": f"{ref_rows.loc[checkpoint, 'mean']:.3f} ± {ref_rows.loc[checkpoint, 'se']:.3f}",
            "idprobcover_mean_pm_se": f"{tgt_rows.loc[checkpoint, 'mean']:.3f} ± {tgt_rows.loc[checkpoint, 'se']:.3f}",
            "mean_difference": float(sig_rows.loc[checkpoint, "mean_difference"]),
            "t_test_pvalue": float(sig_rows.loc[checkpoint, "t_test_pvalue"]),
            "wilcoxon_pvalue": float(sig_rows.loc[checkpoint, "wilcoxon_pvalue"]),
            "bootstrap_ci_low": float(sig_rows.loc[checkpoint, "bootstrap_ci_low"]),
            "bootstrap_ci_high": float(sig_rows.loc[checkpoint, "bootstrap_ci_high"]),
            "acquisition_overhead_ratio": acquisition_ratio,
            "end_to_end_overhead_ratio": end_to_end_ratio,
        })
    return pd.DataFrame(rows)


def plot_accuracy_curves(df, output_path, title):
    plt.figure(figsize=(8, 5))
    for method in sorted(df["method"].dropna().unique(), key=lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else x):
        method_df = df[df["method"] == method]
        grouped = method_df.groupby("episode")["test_accuracy"]
        mean = grouped.mean()
        se = grouped.std(ddof=1).fillna(0.0) / np.sqrt(grouped.count().clip(lower=1))
        x = mean.index.to_numpy(dtype=int)
        y = mean.to_numpy(dtype=float)
        yerr = se.to_numpy(dtype=float)
        plt.plot(x, y, label=method)
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)
    plt.xlabel("AL round")
    plt.ylabel("Test accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_timing_curves(df, output_path, title, column, ylabel, cumulative=False):
    plt.figure(figsize=(8, 5))
    sampled = df[df["has_sampling"]]
    for method in sorted(sampled["method"].dropna().unique(), key=lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else x):
        method_df = sampled[sampled["method"] == method]
        grouped = method_df.groupby(["seed", "episode"])[column].mean().reset_index()
        if cumulative:
            grouped[column] = grouped.groupby("seed")[column].cumsum()
        mean = grouped.groupby("episode")[column].mean()
        x = mean.index.to_numpy(dtype=int)
        y = mean.to_numpy(dtype=float)
        plt.plot(x, y, label=method)
    plt.xlabel("AL round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Aggregate reviewer-facing AL experiment results.")
    parser.add_argument("--output_root", type=str, default="/scratch/s219110279/TypiClust/output")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--methods", nargs="*", default=METHOD_ORDER)
    parser.add_argument("--checkpoints", nargs="*", type=int, default=[25, 50, 75, 100])
    parser.add_argument("--report_dir", type=str, required=True)
    parser.add_argument("--reference_method", type=str, default="probcover")
    parser.add_argument("--target_method", type=str, default="idprobcover")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.output_root, args.dataset, args.model, set(m.lower() for m in args.methods))
    if not runs:
        raise SystemExit(f"No runs found for dataset={args.dataset} model={args.model}")

    episode_df = build_episode_frame(runs)
    episode_df.to_csv(report_dir / f"{args.dataset.lower()}_{args.model.lower()}_episode_records.csv", index=False)

    checkpoint_seed_df = per_seed_checkpoint_table(episode_df, args.checkpoints)
    checkpoint_seed_df.to_csv(report_dir / f"{args.dataset.lower()}_{args.model.lower()}_per_seed_checkpoints.csv", index=False)

    checkpoint_summary_df = checkpoint_table(episode_df, args.checkpoints)
    checkpoint_summary_df.to_csv(report_dir / f"{args.dataset.lower()}_{args.model.lower()}_checkpoint_summary.csv", index=False)

    timing_df = timing_table(episode_df)
    timing_df.to_csv(report_dir / f"{args.dataset.lower()}_{args.model.lower()}_timing_summary.csv", index=False)

    significance_df = paired_significance(
        checkpoint_seed_df,
        reference_method=args.reference_method,
        target_method=args.target_method,
        checkpoints=args.checkpoints,
    )
    significance_df.to_csv(report_dir / f"{args.dataset.lower()}_{args.model.lower()}_{args.target_method}_vs_{args.reference_method}_significance.csv", index=False)

    all_significance_df = significance_vs_all(
        checkpoint_seed_df,
        target_method=args.target_method,
        checkpoints=args.checkpoints,
        methods=sorted(set(m.lower() for m in args.methods), key=lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else x),
    )
    all_significance_df.to_csv(
        report_dir / f"{args.dataset.lower()}_{args.model.lower()}_{args.target_method}_vs_all_significance.csv",
        index=False,
    )

    reviewer_df = reviewer_table(
        checkpoint_summary_df,
        significance_df,
        timing_df,
        reference_method=args.reference_method,
        target_method=args.target_method,
    )
    reviewer_df.to_csv(report_dir / f"{args.dataset.lower()}_{args.model.lower()}_reviewer_table.csv", index=False)

    plot_accuracy_curves(
        episode_df,
        report_dir / f"{args.dataset.lower()}_{args.model.lower()}_accuracy_curves.png",
        f"{args.dataset} {args.model} accuracy curves",
    )
    plot_timing_curves(
        episode_df,
        report_dir / f"{args.dataset.lower()}_{args.model.lower()}_acquisition_time_curves.png",
        f"{args.dataset} {args.model} acquisition time",
        column="acquisition_time_sec",
        ylabel="Acquisition time (sec)",
        cumulative=False,
    )
    plot_timing_curves(
        episode_df,
        report_dir / f"{args.dataset.lower()}_{args.model.lower()}_cumulative_acquisition_time_curves.png",
        f"{args.dataset} {args.model} cumulative acquisition time",
        column="acquisition_time_sec",
        ylabel="Cumulative acquisition time (sec)",
        cumulative=True,
    )
    plot_timing_curves(
        episode_df,
        report_dir / f"{args.dataset.lower()}_{args.model.lower()}_round_time_curves.png",
        f"{args.dataset} {args.model} end-to-end round time",
        column="round_time_sec",
        ylabel="Round time (sec)",
        cumulative=False,
    )

    print(f"Wrote reports to {report_dir}")


if __name__ == "__main__":
    main()
