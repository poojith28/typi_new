#!/usr/bin/env python3
"""Export resnet18 / 50b AL curves to per-dataset CSVs (all methods)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_ROOT = Path("/vast/s219110279/TypiClust/output")
DEFAULT_OUT_DIR = Path("/vast/s219110279/analysis/al_csv_resnet18_50b")
DATASETS = ("CIFAR10", "CIFAR100", "TINYIMAGENET")
BACKBONE = "resnet18"
BUDGET = 50
SEEDS = (1, 2, 3, 4, 5)
METHODS = (
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
    "LIDCOVER",
    "maxherding",
)


def load_run_summary(run_dir: Path, method: str) -> dict | None:
    bench = run_dir / "benchmark_summary.json"
    if bench.exists():
        summary = json.loads(bench.read_text())
        summary["_status"] = "complete"
        summary["_source"] = str(bench)
        return summary

    episode_paths = sorted(run_dir.glob("episode_*/episode_summary.json"))
    if not episode_paths:
        return None

    records = [json.loads(p.read_text()) for p in episode_paths]
    first = records[0]
    return {
        "sampling_fn": first.get("sampling_fn", method),
        "dataset": run_dir.parts[-3],
        "model": run_dir.parts[-2],
        "seed": first.get("seed"),
        "exp_name": first.get("exp_name", run_dir.name),
        "exp_dir": str(run_dir),
        "episode_records": records,
        "initial_sampling": {},
        "timing": {},
        "final_val_accuracy": records[-1].get("best_val_accuracy"),
        "final_test_accuracy": records[-1].get("test_accuracy"),
        "num_rounds_completed": len(records),
        "_status": "in_progress",
        "_source": "episode_summary_only",
    }


def episode_rows(summary: dict, method: str) -> list[dict]:
    rows = []
    base = {
        "dataset": summary.get("dataset"),
        "model": summary.get("model"),
        "method": str(summary.get("sampling_fn", method)).lower(),
        "seed": summary.get("seed"),
        "exp_name": summary.get("exp_name"),
        "run_status": summary.get("_status", "unknown"),
    }
    initial = summary.get("initial_sampling") or {}
    if initial:
        rows.append(
            {
                **base,
                "episode": -1,
                "labeled_count": int(initial.get("labeled_count_after_sampling", 50)),
                "best_val_accuracy": np.nan,
                "best_val_epoch": np.nan,
                "test_accuracy": np.nan,
                "train_time_sec": np.nan,
                "test_time_sec": np.nan,
                "acquisition_time_sec": float(initial.get("acquisition_time_sec", np.nan)),
                "round_time_sec": float(initial.get("acquisition_time_sec", np.nan)),
                "is_initial_sampling": True,
            }
        )

    for record in summary.get("episode_records", []):
        rows.append(
            {
                **base,
                "episode": int(record.get("episode", -1)),
                "labeled_count": int(record.get("labeled_count_after_sampling", np.nan)),
                "best_val_accuracy": float(record.get("best_val_accuracy", np.nan)),
                "best_val_epoch": int(record.get("best_val_epoch", -1)),
                "test_accuracy": float(record.get("test_accuracy", np.nan)),
                "train_time_sec": float(record.get("train_time_sec", np.nan)),
                "test_time_sec": float(record.get("test_time_sec", np.nan)),
                "acquisition_time_sec": float(record.get("acquisition_time_sec", np.nan)),
                "round_time_sec": float(record.get("round_time_sec", np.nan)),
                "is_initial_sampling": False,
            }
        )
    return rows


def summary_rows(summaries: list[dict], method: str) -> list[dict]:
    rows = []
    for summary in summaries:
        episodes = summary.get("episode_records", [])
        val_curve = [float(r.get("best_val_accuracy", np.nan)) for r in episodes]
        test_curve = [float(r.get("test_accuracy", np.nan)) for r in episodes]
        timing = summary.get("timing") or {}
        rows.append(
            {
                "dataset": summary.get("dataset"),
                "model": summary.get("model"),
                "method": str(summary.get("sampling_fn", method)).lower(),
                "seed": summary.get("seed"),
                "exp_name": summary.get("exp_name"),
                "run_status": summary.get("_status"),
                "num_rounds": int(summary.get("num_rounds_completed", len(episodes))),
                "final_val_accuracy": float(
                    summary.get("final_val_accuracy", val_curve[-1] if val_curve else np.nan)
                ),
                "final_test_accuracy": float(
                    summary.get("final_test_accuracy", test_curve[-1] if test_curve else np.nan)
                ),
                "val_auc": float(np.nanmean(val_curve)) if val_curve else np.nan,
                "test_auc": float(np.nanmean(test_curve)) if test_curve else np.nan,
                "cumulative_train_time_sec": float(
                    timing.get("train_time_sec", {}).get(
                        "cumulative",
                        np.nansum([r.get("train_time_sec") for r in episodes]),
                    )
                ),
                "cumulative_acquisition_time_sec": float(
                    timing.get("acquisition_time_sec", {}).get(
                        "cumulative",
                        np.nansum([r.get("acquisition_time_sec") for r in episodes]),
                    )
                ),
                "cumulative_round_time_sec": float(
                    timing.get("round_time_sec", {}).get(
                        "cumulative",
                        np.nansum([r.get("round_time_sec") for r in episodes]),
                    )
                ),
                "exp_dir": summary.get("exp_dir"),
            }
        )

    if not rows:
        return rows

    df = pd.DataFrame(rows)
    agg = {
        "dataset": rows[0]["dataset"],
        "model": rows[0]["model"],
        "method": rows[0]["method"],
        "seed": "mean_std",
        "exp_name": f"aggregate_n={len(rows)}",
        "run_status": "mixed" if df["run_status"].nunique() > 1 else rows[0]["run_status"],
        "num_rounds": int(df["num_rounds"].mean()),
    }
    for col in ("final_val_accuracy", "final_test_accuracy", "val_auc", "test_auc"):
        vals = df[col].astype(float)
        agg[col] = (
            f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}" if len(vals) > 1 else f"{vals.iloc[0]:.4f}"
        )
    for col in ("cumulative_train_time_sec", "cumulative_acquisition_time_sec", "cumulative_round_time_sec"):
        vals = df[col].astype(float)
        agg[col] = float(vals.mean())
    agg["exp_dir"] = ""
    rows.append(agg)
    return rows


def export_dataset(dataset: str, methods: tuple[str, ...], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = dataset.lower()

    all_round_rows: list[dict] = []
    all_summary_rows: list[dict] = []

    for method in methods:
        summaries = []
        for seed in SEEDS:
            run_dir = OUTPUT_ROOT / dataset / BACKBONE / f"{method}_{seed}_{BUDGET}b"
            summary = load_run_summary(run_dir, method)
            if summary is None:
                print(f"WARN missing: {run_dir}")
                continue
            summaries.append(summary)

        if not summaries:
            continue

        method_slug = method.lower()
        rounds_path = out_dir / f"{slug}_{method_slug}_all_rounds.csv"
        summary_path = out_dir / f"{slug}_{method_slug}_summary.csv"

        round_rows: list[dict] = []
        for summary in summaries:
            round_rows.extend(episode_rows(summary, method))
        rounds_df = pd.DataFrame(round_rows).sort_values(["seed", "episode"])
        rounds_df.to_csv(rounds_path, index=False)

        method_summary_rows = summary_rows(summaries, method)
        summary_df = pd.DataFrame(method_summary_rows)
        summary_df.to_csv(summary_path, index=False)

        all_round_rows.extend(round_rows)
        all_summary_rows.extend([r for r in method_summary_rows if r.get("seed") != "mean_std"])

        print(
            f"{dataset}/{method}: {len(summaries)} runs -> "
            f"{rounds_path.name} ({len(rounds_df)} rows), {summary_path.name}"
        )

    combined_rounds = out_dir / f"{slug}_all_methods_all_rounds.csv"
    combined_summary = out_dir / f"{slug}_all_methods_summary.csv"
    pd.DataFrame(all_round_rows).sort_values(["method", "seed", "episode"]).to_csv(
        combined_rounds, index=False
    )

    summary_df = pd.DataFrame(all_summary_rows)
    if not summary_df.empty:
        agg_rows = []
        for method, group in summary_df.groupby("method"):
            vals = group["final_test_accuracy"].astype(float)
            agg_rows.append(
                {
                    "dataset": dataset,
                    "model": BACKBONE,
                    "method": method,
                    "n_seeds": len(group),
                    "final_test_accuracy_mean": float(vals.mean()),
                    "final_test_accuracy_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "final_val_accuracy_mean": float(group["final_val_accuracy"].astype(float).mean()),
                    "final_val_accuracy_std": float(
                        group["final_val_accuracy"].astype(float).std(ddof=1)
                    )
                    if len(group) > 1
                    else 0.0,
                }
            )
        pd.DataFrame(agg_rows).sort_values("final_test_accuracy_mean", ascending=False).to_csv(
            combined_summary, index=False
        )

    print(
        f"{dataset}: combined -> {combined_rounds.name} ({len(all_round_rows)} rows), "
        f"{combined_summary.name}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=list(METHODS),
        help="Methods to export (default: all core baselines + maxherding)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASETS),
        choices=list(DATASETS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = tuple(args.methods)
    for dataset in args.datasets:
        export_dataset(dataset, methods, args.out_dir)


if __name__ == "__main__":
    main()
