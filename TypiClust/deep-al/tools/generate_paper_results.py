#!/usr/bin/env python3

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEEP_AL_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))

import sys

if _DEEP_AL_ROOT not in sys.path:
    sys.path.insert(0, _DEEP_AL_ROOT)

from analyze_id_stability import _sample_indices
from analyze_id_stability import _safe_spearman
from analyze_id_stability import discover_runs
from analyze_id_stability import load_pool_indices
from analyze_id_stability import local_mle_id
from analyze_id_stability import local_twonn_ratio_id
from analyze_id_stability import normalize_ratio
import pycls.datasets.utils as ds_utils


def _safe_mean(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def _safe_se(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size <= 1:
        return 0.0 if arr.size == 1 else float("nan")
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def _safe_pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(stats.pearsonr(a, b).statistic)


def _mean_pm_se(mean_val, se_val, digits=3):
    if np.isnan(mean_val):
        return "NA"
    if np.isnan(se_val):
        return f"{mean_val:.{digits}f}"
    return f"{mean_val:.{digits}f} ± {se_val:.{digits}f}"


def _write_tex_table(path, columns, rows, caption=None, label=None):
    with open(path, "w") as handle:
        handle.write("\\begin{table}[t]\n\\centering\n")
        if caption:
            handle.write(f"\\caption{{{caption}}}\n")
        if label:
            handle.write(f"\\label{{{label}}}\n")
        handle.write("\\begin{tabular}{" + "l" * len(columns) + "}\n")
        handle.write("\\toprule\n")
        handle.write(" & ".join(columns) + " \\\\\n")
        handle.write("\\midrule\n")
        for row in rows:
            handle.write(" & ".join(str(row.get(col, "")) for col in columns) + " \\\\\n")
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")
        handle.write("\\end{table}\n")


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _resolve_existing_path(*candidates):
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return str(candidates[0])


def _hardware_context():
    ctx = {}
    try:
        gpu_name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()
        ctx["gpu_names"] = gpu_name
    except Exception as exc:
        ctx["gpu_names"] = [f"unavailable: {exc}"]
    try:
        ctx["cpu_cores_logical"] = int(os.cpu_count() or 0)
    except Exception:
        ctx["cpu_cores_logical"] = None
    try:
        with open("/proc/meminfo", "r") as handle:
            mem_kb = None
            for line in handle:
                if line.startswith("MemTotal:"):
                    mem_kb = int(line.split()[1])
                    break
        ctx["memory_gb"] = round(mem_kb / (1024 ** 2), 2) if mem_kb else None
    except Exception:
        ctx["memory_gb"] = None
    return ctx


def compute_id_stability(output_root, report_dir, max_points=1500):
    report_dir = Path(report_dir)
    features = ds_utils.load_features("CIFAR100", seed=1, train=True, normalized=True).astype(np.float64)
    runs = discover_runs(output_root, "CIFAR100", "resnet18", "idprobcover")
    episodes = [0, 25, 50, 75, 100]

    details = []
    snapshot_cache = {}

    for run in runs:
        for episode in episodes:
            pool_indices = load_pool_indices(run["run_dir"], episode)
            sampled_pool = _sample_indices(pool_indices, max_points, seed=run["seed"] * 1000 + int(episode))
            x = features[sampled_pool]
            ids20, _ = local_mle_id(x, k=20)
            ids50, _ = local_mle_id(x, k=50)
            ids75, _ = local_mle_id(x, k=75)
            twonn, _ = local_twonn_ratio_id(x)
            rho20 = normalize_ratio(ids20)
            rho50 = normalize_ratio(ids50)
            rho75 = normalize_ratio(ids75)
            rho2 = normalize_ratio(twonn)

            def add_detail(metric, comparison, a, b):
                details.append({
                    "metric": metric,
                    "comparison": comparison,
                    "seed": int(run["seed"]),
                    "episode": int(episode),
                    "spearman": _safe_spearman(a, b),
                    "pearson": _safe_pearson(a, b),
                })

            add_detail("k-sensitivity", "k=20 vs k=50", ids20, ids50)
            add_detail("k-sensitivity", "k=75 vs k=50", ids75, ids50)
            add_detail("normalized rho", "k=20 vs k=50", rho20, rho50)
            add_detail("normalized rho", "k=75 vs k=50", rho75, rho50)
            add_detail("optional estimator", "local 2-NN vs MLE k=50", twonn, ids50)

            snapshot_cache[(run["exp_name"], int(episode))] = {
                "pool_indices": sampled_pool,
                "rho50": rho50,
                "ids50": ids50,
            }

    for run in runs:
        for ep_a, ep_b in zip(episodes[:-1], episodes[1:]):
            left = snapshot_cache[(run["exp_name"], int(ep_a))]
            right = snapshot_cache[(run["exp_name"], int(ep_b))]
            common = np.intersect1d(left["pool_indices"], right["pool_indices"])
            if common.size == 0:
                continue
            common = _sample_indices(common, max_points, seed=run["seed"] * 10000 + ep_a + ep_b)
            left_pos = {int(v): i for i, v in enumerate(left["pool_indices"])}
            right_pos = {int(v): i for i, v in enumerate(right["pool_indices"])}
            left_idx = np.asarray([left_pos[int(v)] for v in common], dtype=np.int64)
            right_idx = np.asarray([right_pos[int(v)] for v in common], dtype=np.int64)
            details.append({
                "metric": "temporal stability",
                "comparison": f"R{ep_a} vs R{ep_b}",
                "seed": int(run["seed"]),
                "episode": f"{ep_a}->{ep_b}",
                "spearman": _safe_spearman(left["rho50"][left_idx], right["rho50"][right_idx]),
                "pearson": _safe_pearson(left["rho50"][left_idx], right["rho50"][right_idx]),
            })

    for i, run_a in enumerate(runs):
        for run_b in runs[i + 1:]:
            for episode in episodes:
                left = snapshot_cache[(run_a["exp_name"], int(episode))]
                right = snapshot_cache[(run_b["exp_name"], int(episode))]
                common = np.intersect1d(left["pool_indices"], right["pool_indices"])
                if common.size == 0:
                    continue
                common = _sample_indices(common, max_points, seed=episode * 10000 + run_a["seed"] * 10 + run_b["seed"])
                left_pos = {int(v): idx for idx, v in enumerate(left["pool_indices"])}
                right_pos = {int(v): idx for idx, v in enumerate(right["pool_indices"])}
                left_idx = np.asarray([left_pos[int(v)] for v in common], dtype=np.int64)
                right_idx = np.asarray([right_pos[int(v)] for v in common], dtype=np.int64)
                details.append({
                    "metric": "seed stability",
                    "comparison": f"seed {run_a['seed']} vs seed {run_b['seed']} @ R{episode}",
                    "seed": f"{run_a['seed']}-{run_b['seed']}",
                    "episode": int(episode),
                    "spearman": _safe_spearman(left["rho50"][left_idx], right["rho50"][right_idx]),
                    "pearson": _safe_pearson(left["rho50"][left_idx], right["rho50"][right_idx]),
                })

    details_df = pd.DataFrame(details)
    summary_rows = []
    summary_specs = [
        ("k-sensitivity", "k=20 vs k=50"),
        ("k-sensitivity", "k=75 vs k=50"),
        ("normalized rho", "k=20 vs k=50"),
        ("normalized rho", "k=75 vs k=50"),
        ("temporal stability", "across rounds"),
        ("seed stability", "across seeds"),
        ("optional estimator", "local 2-NN vs MLE k=50"),
    ]
    for metric, comparison_label in summary_specs:
        if metric == "temporal stability":
            subset = details_df[details_df["metric"] == metric]
        elif metric == "seed stability":
            subset = details_df[details_df["metric"] == metric]
        else:
            subset = details_df[(details_df["metric"] == metric) & (details_df["comparison"] == comparison_label)]
        summary_rows.append({
            "metric": metric,
            "comparison": comparison_label,
            "n": int(len(subset)),
            "spearman_mean": _safe_mean(subset["spearman"]),
            "spearman_se": _safe_se(subset["spearman"]),
            "pearson_mean": _safe_mean(subset["pearson"]),
            "pearson_se": _safe_se(subset["pearson"]),
            "spearman_mean_pm_se": _mean_pm_se(_safe_mean(subset["spearman"]), _safe_se(subset["spearman"])),
        })
    summary_df = pd.DataFrame(summary_rows)

    details_path = report_dir / "id_stability_details.csv"
    summary_path = report_dir / "id_stability_summary.csv"
    tex_path = report_dir / "id_stability_table.tex"
    details_df.to_csv(details_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    tex_rows = [
        {
            "Metric": row["metric"],
            "Comparison": row["comparison"],
            "Spearman correlation": row["spearman_mean_pm_se"],
        }
        for _, row in summary_df.iterrows()
        if row["metric"] != "optional estimator"
    ]
    _write_tex_table(
        tex_path,
        ["Metric", "Comparison", "Spearman correlation"],
        tex_rows,
        caption="Stability of local ID estimates on CIFAR-100.",
        label="tab:id_stability",
    )
    return {
        "summary_path": str(summary_path),
        "details_path": str(details_path),
        "tex_path": str(tex_path),
        "runs_used": [run["exp_name"] for run in runs],
    }


def _per_run_timing_from_episode_records(csv_path, method):
    df = pd.read_csv(csv_path)
    df = df[(df["method"] == method) & (df["has_sampling"] == True) & (df["episode"] >= 0)]
    grouped = df.groupby(["exp_name", "seed"], dropna=False)
    return pd.DataFrame({
        "acquisition_mean_sec": grouped["acquisition_time_sec"].mean(),
        "round_mean_sec": grouped["round_time_sec"].mean(),
        "acquisition_cumulative_sec": grouped["acquisition_time_sec"].sum(),
        "round_cumulative_sec": grouped["round_time_sec"].sum(),
    }).reset_index()


def compute_runtime_tables(report_dir, source_reports):
    cold_specs = {
        "CIFAR-100": Path(source_reports["cifar100_cold"]),
        "TinyImageNet": Path(source_reports["tiny_cold"]),
    }
    cold_components = [
        ("Feature load + normalize", ("load_features_sec", "l2_normalize_sec")),
        ("Local ID computation", ("compute_local_id_sec",)),
        ("kNN/adaptive graph construction", ("compute_knn_sec",)),
        ("Total cold preprocessing", ("total_cold_preprocessing_sec",)),
    ]
    cold_rows = []
    cold_jsons = {name: _load_json(path) for name, path in cold_specs.items()}
    for component, keys in cold_components:
        row = {"Component": component}
        for dataset_name, payload in cold_jsons.items():
            row[dataset_name] = sum(float(payload.get(k, 0.0)) for k in keys)
        cold_rows.append(row)
    cold_df = pd.DataFrame(cold_rows)
    cold_df.to_csv(report_dir / "runtime_cold_preprocess.csv", index=False)
    _write_tex_table(
        report_dir / "runtime_cold_table.tex",
        ["Component", "CIFAR-100", "TinyImageNet"],
        [
            {
                "Component": row["Component"],
                "CIFAR-100": f"{row['CIFAR-100']:.2f}",
                "TinyImageNet": f"{row['TinyImageNet']:.2f}",
            }
            for _, row in cold_df.iterrows()
        ],
        caption="Cold preprocessing time for IDProbCover.",
        label="tab:runtime_cold",
    )

    warm_specs = [
        ("CIFAR-100", Path(source_reports["cifar100_episode_records"]), "probcover"),
        ("CIFAR-100", Path(source_reports["cifar100_episode_records"]), "idprobcover"),
        ("TinyImageNet", Path(source_reports["tiny_episode_records"]), "probcover"),
        ("TinyImageNet", Path(source_reports["tiny_episode_records"]), "idprobcover"),
    ]
    warm_rows = []
    per_dataset_method = {}
    for dataset_name, csv_path, method in warm_specs:
        per_run = _per_run_timing_from_episode_records(csv_path, method)
        per_dataset_method[(dataset_name, method)] = per_run
        warm_rows.append({
            "dataset": dataset_name,
            "method": method,
            "runs": int(len(per_run)),
            "acquisition_time_mean_sec": _safe_mean(per_run["acquisition_mean_sec"]),
            "acquisition_time_se_sec": _safe_se(per_run["acquisition_mean_sec"]),
            "total_round_time_mean_sec": _safe_mean(per_run["round_mean_sec"]),
            "total_round_time_se_sec": _safe_se(per_run["round_mean_sec"]),
            "cumulative_acquisition_time_mean_sec": _safe_mean(per_run["acquisition_cumulative_sec"]),
            "cumulative_round_time_mean_sec": _safe_mean(per_run["round_cumulative_sec"]),
        })
    warm_df = pd.DataFrame(warm_rows)
    overhead_rows = []
    for dataset_name in ["CIFAR-100", "TinyImageNet"]:
        prob = warm_df[(warm_df["dataset"] == dataset_name) & (warm_df["method"] == "probcover")]
        idpc = warm_df[(warm_df["dataset"] == dataset_name) & (warm_df["method"] == "idprobcover")]
        if prob.empty or idpc.empty:
            continue
        p_acq = float(prob["acquisition_time_mean_sec"].iloc[0])
        i_acq = float(idpc["acquisition_time_mean_sec"].iloc[0])
        p_round = float(prob["total_round_time_mean_sec"].iloc[0])
        i_round = float(idpc["total_round_time_mean_sec"].iloc[0])
        overhead_rows.append({
            "dataset": dataset_name,
            "method": "idprobcover",
            "absolute_acquisition_overhead_sec": i_acq - p_acq,
            "relative_acquisition_overhead": i_acq / p_acq if p_acq else float("nan"),
            "end_to_end_overhead_percent": 100.0 * (i_round - p_round) / p_round if p_round else float("nan"),
        })
    overhead_df = pd.DataFrame(overhead_rows)
    warm_df = warm_df.merge(overhead_df, on=["dataset", "method"], how="left")
    warm_df.to_csv(report_dir / "runtime_warm_round.csv", index=False)
    _write_tex_table(
        report_dir / "runtime_warm_table.tex",
        ["Dataset", "Method", "Acquisition time / round", "Total round time"],
        [
            {
                "Dataset": row["dataset"],
                "Method": row["method"],
                "Acquisition time / round": _mean_pm_se(row["acquisition_time_mean_sec"], row["acquisition_time_se_sec"], digits=2),
                "Total round time": _mean_pm_se(row["total_round_time_mean_sec"], row["total_round_time_se_sec"], digits=2),
            }
            for _, row in warm_df.iterrows()
        ],
        caption="Warm-cache per-round runtime.",
        label="tab:runtime_warm",
    )

    cache_rows = []
    for dataset_name, payload in cold_jsons.items():
        cache_rows.append({
            "dataset": dataset_name,
            "feature_shape": " x ".join(str(v) for v in payload.get("feature_shape", [])),
            "faiss_backend": payload.get("knn_backend", "unknown"),
            "id_cache_mb": float(payload.get("ids_file_mb", float("nan"))),
            "knn_cache_mb": float(payload.get("knn_file_mb", float("nan"))),
            "total_cache_mb": float(payload.get("ids_file_mb", 0.0)) + float(payload.get("knn_file_mb", 0.0)),
        })
    cache_df = pd.DataFrame(cache_rows)
    cache_df.to_csv(report_dir / "cache_sizes.csv", index=False)
    _write_tex_table(
        report_dir / "cache_size_table.tex",
        ["Dataset", "Feature shape", "ID cache (MB)", "kNN cache (MB)", "Total cache (MB)"],
        [
            {
                "Dataset": row["dataset"],
                "Feature shape": row["feature_shape"],
                "ID cache (MB)": f"{row['id_cache_mb']:.2f}",
                "kNN cache (MB)": f"{row['knn_cache_mb']:.2f}",
                "Total cache (MB)": f"{row['total_cache_mb']:.2f}",
            }
            for _, row in cache_df.iterrows()
        ],
        caption="Cache sizes for IDProbCover preprocessing.",
        label="tab:cache_sizes",
    )

    return {
        "cold_df": cold_df,
        "warm_df": warm_df,
        "cache_df": cache_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate paper-ready CSV/LaTeX tables for IDProbCover stability and runtime.")
    parser.add_argument("--paper_results_dir", type=str, default="/scratch/s219110279/paper_results")
    parser.add_argument("--cifar100_output_root", type=str, default="/scratch/s219110279/TypiClust/output")
    parser.add_argument("--cifar100_report_dir", type=str, default="/scratch/s219110279/TypiClust/reviewer_reports/cifar100_resnet18")
    parser.add_argument("--tiny_report_dir", type=str, default="/scratch/s219110279/TypiClust/reviewer_reports/tinyimagenet_resnet18")
    parser.add_argument("--cifar100_id_stability_dir", type=str, default="/scratch/s219110279/TypiClust/reviewer_reports/cifar100_id_stability")
    args = parser.parse_args()

    paper_dir = Path(args.paper_results_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)

    cifar100_output_root = _resolve_existing_path(
        args.cifar100_output_root,
        "/scratch/s219110279/typirusults/output",
    )
    if not discover_runs(cifar100_output_root, "CIFAR100", "resnet18", "idprobcover"):
        alt_output_root = "/scratch/s219110279/typirusults/output"
        if discover_runs(alt_output_root, "CIFAR100", "resnet18", "idprobcover"):
            cifar100_output_root = alt_output_root
    cifar100_report_dir = _resolve_existing_path(
        args.cifar100_report_dir,
        "/scratch/s219110279/typirusults/reviewer_reports/cifar100_resnet18",
    )
    tiny_report_dir = _resolve_existing_path(
        args.tiny_report_dir,
        "/scratch/s219110279/typirusults/reviewer_reports/tinyimagenet_resnet18",
    )

    source_reports = {
        "cifar100_cold": _resolve_existing_path(
            "/scratch/s219110279/TypiClust/reviewer_reports/cifar100_resnet18/idprobcover_cold_cache_timing.json",
            "/scratch/s219110279/typirusults/reviewer_reports/cifar100_resnet18/idprobcover_cold_cache_timing.json",
        ),
        "tiny_cold": _resolve_existing_path(
            "/scratch/s219110279/TypiClust/reviewer_reports/tinyimagenet_resnet18/idprobcover_cold_cache_timing.json",
            "/scratch/s219110279/typirusults/reviewer_reports/tinyimagenet_resnet18/idprobcover_cold_cache_timing.json",
        ),
        "cifar100_episode_records": str(Path(cifar100_report_dir) / "cifar100_resnet18_episode_records.csv"),
        "tiny_episode_records": str(Path(tiny_report_dir) / "tinyimagenet_resnet18_episode_records.csv"),
    }

    id_result = compute_id_stability(cifar100_output_root, paper_dir)
    runtime_result = compute_runtime_tables(paper_dir, source_reports)

    hardware = _hardware_context()
    manifest = {
        "commands_used": [
            "python deep-al/tools/generate_paper_results.py --paper_results_dir /scratch/s219110279/paper_results",
            "internal reuse of analyze_id_stability.py functions for CIFAR-100 stability recomputation",
        ],
        "files_read": [
            "/scratch/s219110279/TypiClust/output/CIFAR100/resnet18/*/benchmark_summary.json",
            "/scratch/s219110279/TypiClust/output/CIFAR100/resnet18/*/episode_*/uSet.npy",
            source_reports["cifar100_episode_records"],
            source_reports["tiny_episode_records"],
            source_reports["cifar100_cold"],
            source_reports["tiny_cold"],
            "/scratch/s219110279/TypiClust/scan/results/cifar-100/pretext/features_seed1.npy",
        ],
        "number_sources": {
            "id_stability": "newly recomputed from saved pool snapshots and frozen features",
            "runtime_warm": "extracted from existing reviewer episode_records CSVs",
            "runtime_cold": "extracted from existing cold-cache timing JSON measurements",
            "cache_sizes": "extracted from existing cold-cache timing JSON measurements",
        },
        "missing_data": [
            "TinyImageNet standard IDProbCover currently has only one completed seed, so warm-runtime SE for IDProbCover is based on n=1.",
            "TinyImageNet significance vs ProbCover is not robust because paired_n=1 in the current reviewer report.",
            "Optional local 2-NN estimator comparison is available only for CIFAR-100 stability analysis in this package.",
        ],
        "hardware_context": hardware,
    }
    with open(paper_dir / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nID Stability Summary")
    print(pd.read_csv(id_result["summary_path"]).to_string(index=False))
    print("\nRuntime Warm-Round Summary")
    print(runtime_result["warm_df"].to_string(index=False))
    print("\nCold Preprocessing Summary")
    print(runtime_result["cold_df"].to_string(index=False))
    print("\nCache Size Summary")
    print(runtime_result["cache_df"].to_string(index=False))
    print("\nManifest written to", paper_dir / "manifest.json")


if __name__ == "__main__":
    main()
