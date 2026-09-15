#!/usr/bin/env python3
"""Generate read-only, budget-corrected AULC replacement tables.

Every metric is reconstructed from ``benchmark_summary.json`` episode records.
Accuracy is associated with ``labeled_count_before_sampling`` (the labelled set
on which that episode's classifier was trained).  The primary corrected metric
is normalized trapezoidal area over the observed labelled-budget interval.

Existing experiment outputs and thesis tables are never modified.  All files
are written below a separate output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

try:
    import yaml
except ImportError:  # Configuration metadata is optional; trajectories are not.
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "TypiClust" / "output"
DEFAULT_OUTPUT = ROOT / "outputs" / "aulc_trapezoidal_corrected"
EXPECTED_EPISODES = list(range(101))
BASELINE_METHODS = {
    "random", "uncertainty", "entropy", "margin", "dbal", "bald",
    "coreset", "core_set", "probcover", "prob_cover", "typiclust",
    "maxherding",
}
BASELINE_DISPLAY = {
    "random": "Random", "uncertainty": "Least Confidence",
    "entropy": "Entropy", "margin": "Margin", "dbal": "DBAL",
    "bald": "BALD", "coreset": "CoreSet", "core_set": "CoreSet",
    "probcover": "ProbCover", "prob_cover": "ProbCover",
    "typiclust": "TypiClust", "maxherding": "MaxHerding",
}
METRICS = (
    "final_accuracy", "simple_mean", "aulc_trapezoid_observed",
    "aulc_trapezoid_50_5050",
)
CONFIG_FIELDS = (
    "initial_delta", "idpc_alpha", "idpc_k_id", "idpc_k_knn",
    "idpc_mode", "mstc_alpha", "mstc_delta_coarse", "mstc_delta_fine",
)


def finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def clean_number(value):
    value = finite(value)
    return "" if value is None else value


def read_yaml(path):
    if yaml is None or not path.is_file():
        return {}
    try:
        with path.open() as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        return {}


def family_name(exp_name, seed, budget):
    suffix = re.search(r"_(\d+)_(\d+)b$", exp_name)
    if suffix and int(suffix.group(1)) == seed:
        return exp_name[:suffix.start()] + "_{}b".format(int(suffix.group(2)))
    return re.sub(r"(?i)(?:seed|_s)(\d+)", lambda m: m.group(0)[:-len(m.group(1))] + "SEED", exp_name)


def classify_family(method, exp_name):
    text = "{} {}".format(method, exp_name).lower()
    if "multiscale_topocover" in text or exp_name.upper().startswith("MSTC_"):
        return "mstc"
    if "topocover" in text or "topo" in exp_name.lower():
        return "topocover"
    if method.lower() in BASELINE_METHODS and "lidcover" not in text:
        return "baseline"
    lid_tokens = ("lidcover", "idprobcover", "adaptive_cover", "distance_", "density_", "knn_distance")
    if any(token in text for token in lid_tokens):
        return "lidcover"
    return "other"


def is_quarantined(path):
    return any("quarantine" in part.lower() for part in path.parts)


def discover_summaries(input_root, max_files=None):
    """Walk run directories without descending into their episode subtrees."""
    found = []
    for directory, child_dirs, files in os.walk(str(input_root)):
        if "benchmark_summary.json" in files:
            found.append(Path(directory) / "benchmark_summary.json")
            child_dirs[:] = []
            if max_files is not None and len(found) >= max_files:
                break
            continue
        child_dirs[:] = [name for name in child_dirs
                         if not name.startswith("episode_") and name not in {"_logs", "logs"}]
    return sorted(found)


def deduplicate_by_budget(points):
    # Points are (episode, budget, accuracy); the latest episode is authoritative.
    by_budget = {}
    for point in sorted(points, key=lambda item: (item[0], item[1])):
        by_budget[point[1]] = point
    return sorted(by_budget.values(), key=lambda item: item[1])


def trapezoid(points):
    clean = deduplicate_by_budget(points)
    if len(clean) < 2 or clean[-1][1] <= clean[0][1]:
        return None
    area = sum(
        0.5 * (left[2] + right[2]) * (right[1] - left[1])
        for left, right in zip(clean, clean[1:])
    )
    return area / float(clean[-1][1] - clean[0][1])


def interpolate(points, budget):
    clean = deduplicate_by_budget(points)
    if not clean or budget < clean[0][1] or budget > clean[-1][1]:
        return None
    for point in clean:
        if point[1] == budget:
            return point[2]
    for left, right in zip(clean, clean[1:]):
        if left[1] < budget < right[1]:
            weight = (budget - left[1]) / float(right[1] - left[1])
            return left[2] + weight * (right[2] - left[2])
    return None


def trapezoid_interval(points, first_budget, last_budget):
    if last_budget <= first_budget:
        return None
    first_accuracy = interpolate(points, first_budget)
    last_accuracy = interpolate(points, last_budget)
    if first_accuracy is None or last_accuracy is None:
        return None
    selected = [(point[0], point[1], point[2]) for point in deduplicate_by_budget(points)
                if first_budget < point[1] < last_budget]
    selected.insert(0, (-1, first_budget, first_accuracy))
    selected.append((10 ** 9, last_budget, last_accuracy))
    return trapezoid(selected)


def extract_run(summary_path):
    run_dir = summary_path.parent
    try:
        with summary_path.open() as handle:
            payload = json.load(handle)
    except Exception as exc:
        return {"source_file": str(summary_path.resolve()), "run_directory": str(run_dir.resolve()),
                "status": "json_error", "warning": str(exc), "excluded": is_quarantined(summary_path)}

    config = read_yaml(run_dir / "config.yaml")
    al = config.get("ACTIVE_LEARNING", {}) if isinstance(config, dict) else {}
    records = payload.get("episode_records") or []
    dataset = str(payload.get("dataset") or config.get("DATASET", {}).get("NAME") or run_dir.parent.parent.name)
    backbone = str(payload.get("model") or config.get("MODEL", {}).get("TYPE") or run_dir.parent.name)
    method = str(payload.get("sampling_fn") or al.get("SAMPLING_FN") or "unknown").lower()
    seed = int(payload.get("seed", config.get("RNG_SEED", -1)))
    budget_per_round = int(payload.get("budget_per_round") or al.get("BUDGET_SIZE") or 0)
    exp_name = str(payload.get("exp_name") or run_dir.name)
    initial = payload.get("initial_sampling") or {}
    initial_size = finite(initial.get("labeled_count_after_sampling"))
    if initial_size is None:
        initial_size = float(budget_per_round) if budget_per_round else None

    points = []
    warnings = []
    missing_before = 0
    for record in records:
        episode = finite(record.get("episode"))
        accuracy = finite(record.get("test_accuracy"))
        if episode is None or accuracy is None:
            continue
        episode = int(episode)
        before = finite(record.get("labeled_count_before_sampling"))
        if before is None:
            missing_before += 1
            before = initial_size + episode * budget_per_round if initial_size is not None and budget_per_round else None
        if before is None:
            continue
        points.append((episode, int(before), accuracy))

    points.sort(key=lambda item: item[0])
    episodes = [point[0] for point in points]
    budgets = [point[1] for point in points]
    unique_budget_points = deduplicate_by_budget(points)
    simple_mean = mean([point[2] for point in points]) if points else None
    stored_test_auc = finite(payload.get("test_auc"))
    expected_schedule = bool(
        budget_per_round and episodes == EXPECTED_EPISODES
        and budgets == [budget_per_round * (episode + 1) for episode in EXPECTED_EPISODES]
    )
    complete_101 = episodes == EXPECTED_EPISODES
    if missing_before:
        warnings.append("derived {} missing pre-sampling budgets".format(missing_before))
    if len(set(episodes)) != len(episodes):
        warnings.append("duplicate episode IDs")
    if len(unique_budget_points) != len(points):
        warnings.append("duplicate pre-sampling budgets")
    if not complete_101:
        warnings.append("not exactly episodes 0-100")
    if complete_101 and not expected_schedule:
        warnings.append("101 episodes but nonstandard labelled-budget schedule")
    if stored_test_auc is not None and simple_mean is not None and abs(stored_test_auc - simple_mean) > 1e-9:
        warnings.append("stored test_auc differs from trajectory mean")

    row = {
        "source_file": str(summary_path.resolve()),
        "run_directory": str(run_dir.resolve()),
        "excluded": is_quarantined(summary_path),
        "dataset": dataset, "backbone": backbone, "method": method,
        "method_family": classify_family(method, exp_name),
        "experiment_name": exp_name,
        "experiment_family": family_name(exp_name, seed, budget_per_round),
        "seed": seed, "budget_per_round": budget_per_round,
        "n_records": len(records), "n_valid_points": len(points),
        "n_unique_episodes": len(set(episodes)),
        "n_unique_budgets": len(unique_budget_points),
        "first_episode": episodes[0] if episodes else "",
        "last_episode": episodes[-1] if episodes else "",
        "first_budget": unique_budget_points[0][1] if unique_budget_points else "",
        "last_budget": unique_budget_points[-1][1] if unique_budget_points else "",
        "complete_101_episodes": complete_101,
        "expected_budget_schedule": expected_schedule,
        "final_accuracy": points[-1][2] if points else "",
        "stored_test_auc": "" if stored_test_auc is None else stored_test_auc,
        "simple_mean": "" if simple_mean is None else simple_mean,
        "aulc_trapezoid_observed": clean_number(trapezoid(points)),
        "aulc_trapezoid_50_5050": clean_number(trapezoid_interval(points, 50, 5050)),
        "delta_trapezoid_minus_simple": (
            "" if simple_mean is None or trapezoid(points) is None else trapezoid(points) - simple_mean
        ),
        "status": "valid" if len(unique_budget_points) >= 2 else "insufficient_points",
        "warning": "; ".join(warnings),
    }
    for field, key in (
        ("initial_delta", "INITIAL_DELTA"), ("idpc_alpha", "IDPC_ALPHA"),
        ("idpc_k_id", "IDPC_K_ID"), ("idpc_k_knn", "IDPC_K_KNN"),
        ("idpc_mode", "IDPC_MODE"), ("mstc_alpha", "MSTC_ALPHA"),
        ("mstc_delta_coarse", "MSTC_DELTA_COARSE"),
        ("mstc_delta_fine", "MSTC_DELTA_FINE"),
    ):
        row[field] = al.get(key, "")
    row["_points"] = points
    return row


def csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if not key.startswith("_") and key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fields})


def latex_escape(value):
    text = str(value)
    return (text.replace("\\", r"\textbackslash{}")
                .replace("_", r"\_").replace("%", r"\%")
                .replace("&", r"\&").replace("#", r"\#"))


def write_latex(path, rows, fields, caption):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
             r"\begin{tabular}{" + "l" * len(fields) + "}", r"\toprule",
             " & ".join(latex_escape(field) for field in fields) + r" \\", r"\midrule"]
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                cells.append("{:.4f}".format(value))
            else:
                cells.append(latex_escape(value))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{" + caption + "}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def stat_values(values):
    clean = [float(value) for value in values if finite(value) is not None]
    if not clean:
        return "", "", ""
    avg = mean(clean)
    sd = stdev(clean) if len(clean) > 1 else 0.0
    return avg, sd, sd / math.sqrt(len(clean))


def aggregate_runs(rows, group_fields):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("excluded") or row.get("status") != "valid":
            continue
        grouped[tuple(str(row.get(field, "")) for field in group_fields)].append(row)
    output = []
    for key, group in grouped.items():
        record = dict(zip(group_fields, key))
        record.update({
            "n_runs": len(group), "n_seeds": len(set(row.get("seed") for row in group)),
            "min_points": min(int(row["n_valid_points"]) for row in group),
            "max_points": max(int(row["n_valid_points"]) for row in group),
            "first_budget_min": min(int(row["first_budget"]) for row in group),
            "first_budget_max": max(int(row["first_budget"]) for row in group),
            "last_budget_min": min(int(row["last_budget"]) for row in group),
            "last_budget_max": max(int(row["last_budget"]) for row in group),
            "all_complete_101": all(row["complete_101_episodes"] for row in group),
            "all_expected_schedule": all(row["expected_budget_schedule"] for row in group),
        })
        for metric in METRICS:
            avg, sd, sem = stat_values([row.get(metric) for row in group])
            record[metric + "_mean"] = avg
            record[metric + "_sd"] = sd
            record[metric + "_sem"] = sem
        output.append(record)
    return sorted(output, key=lambda row: tuple(str(row.get(field, "")) for field in group_fields))


def canonical_baseline_table(rows):
    selected = []
    pattern = re.compile(r"^(random|uncertainty|entropy|margin|dbal|bald|coreset|core_set|probcover|prob_cover|typiclust|maxherding)_(\d+)_50b$")
    for row in rows:
        match = pattern.match(Path(row["run_directory"]).name.lower())
        if row.get("excluded") or not match or row.get("backbone") != "resnet18":
            continue
        record = dict(row)
        record["canonical_method"] = match.group(1)
        selected.append(record)
    grouped = aggregate_runs(selected, ("dataset", "backbone", "canonical_method"))
    for row in grouped:
        row["method"] = BASELINE_DISPLAY.get(row["canonical_method"], row["canonical_method"])
    return grouped


def corrected_cited_table(rows):
    lookup = {(row.get("dataset"), row.get("backbone"), Path(row.get("run_directory", "")).name): row
              for row in rows}
    specifications = []
    old = {
        ("CIFAR10", "MultiScaleTopoCover"): 46.87, ("CIFAR10", "ProbCover"): 62.35,
        ("CIFAR100", "MultiScaleTopoCover"): 21.12, ("CIFAR100", "ProbCover"): 27.09,
        ("TINYIMAGENET", "MultiScaleTopoCover"): 6.12, ("TINYIMAGENET", "ProbCover"): 11.86,
    }
    for dataset in ("CIFAR10", "CIFAR100", "TINYIMAGENET"):
        for method, pattern in (("MultiScaleTopoCover", "MSTC_C070_F050_A070_{}_50b"),
                                ("ProbCover", "probcover_{}_50b")):
            group = [lookup[(dataset, "resnet18", pattern.format(seed))] for seed in range(1, 6)
                     if (dataset, "resnet18", pattern.format(seed)) in lookup]
            simple, _, simple_sem = stat_values([row["simple_mean"] for row in group])
            trap, _, trap_sem = stat_values([row["aulc_trapezoid_observed"] for row in group])
            specifications.append({
                "dataset": dataset, "method": method, "n": len(group),
                "n_points": min([row["n_valid_points"] for row in group] or [0]),
                "first_budget": min([row["first_budget"] for row in group] or [0]),
                "last_budget": max([row["last_budget"] for row in group] or [0]),
                "historical_reported_aulc": old[(dataset, method)],
                "historical_simple_mean_recomputed": simple,
                "historical_simple_mean_sem": simple_sem,
                "corrected_trapezoid_aulc": trap,
                "corrected_trapezoid_aulc_sem": trap_sem,
                "change_corrected_minus_reported": trap - old[(dataset, method)] if trap != "" else "",
            })
    return specifications


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mstc_provenance_tables(metric_by_dir, output):
    source = ROOT / "evidence_pack" / "seed_level_metrics.csv"
    rows = []
    for original in read_csv(source):
        metric = metric_by_dir.get(str(Path(original["run_directory"]).resolve()))
        if metric is None:
            continue
        row = dict(original)
        row.update({
            "historical_simple_aulc": original.get("aulc", ""),
            "simple_mean_recomputed": metric["simple_mean"],
            "corrected_trapezoid_aulc": metric["aulc_trapezoid_observed"],
            "correction_delta": float(metric["aulc_trapezoid_observed"]) - float(original["aulc"]),
            "n_points": metric["n_valid_points"], "first_budget": metric["first_budget"],
            "last_budget": metric["last_budget"], "complete_101": metric["complete_101_episodes"],
        })
        row["_points"] = metric["_points"]
        rows.append(row)
    write_csv(output / "per_seed" / "mstc_and_baselines_per_seed_corrected.csv", rows)
    dims = ("dataset", "backbone", "method", "alpha", "delta_coarse", "delta_fine", "batch_size")
    renamed = []
    for row in rows:
        item = dict(row)
        item["final_accuracy"] = clean_number(row.get("final_accuracy"))
        item["simple_mean"] = clean_number(row.get("simple_mean_recomputed"))
        item["aulc_trapezoid_observed"] = clean_number(row.get("corrected_trapezoid_aulc"))
        item["aulc_trapezoid_50_5050"] = clean_number(
            trapezoid_interval(row["_points"], 50, 5050))
        item["excluded"] = False
        item["status"] = "valid"
        item["n_valid_points"] = int(row["n_points"])
        item["complete_101_episodes"] = str(row["complete_101"]).lower() == "true"
        item["expected_budget_schedule"] = item["complete_101_episodes"]
        renamed.append(item)
    summary = aggregate_runs(renamed, dims)
    write_csv(output / "tables" / "mstc_all_configurations_corrected.csv", summary)

    def number(row, field):
        return finite(row.get(field))

    table_5_1 = []
    for row in summary:
        if row["backbone"] != "resnet18" or number(row, "batch_size") != 50:
            continue
        method = row["method"]
        if method == "MSTC" and number(row, "delta_coarse") == 0.7 and number(row, "delta_fine") == 0.5:
            alpha = number(row, "alpha")
            if alpha not in (0.0, 0.7, 1.0):
                continue
            label = {0.0: "Fine-only MSTC", 0.7: "MultiScaleTopoCover", 1.0: "Coarse-only MSTC"}[alpha]
        elif method in ("Random", "CoreSet", "Entropy", "Margin", "Least Confidence", "ProbCover"):
            label = method
        else:
            continue
        item = dict(row)
        item["method_display"] = label
        table_5_1.append(item)
    table_5_1.sort(key=lambda row: (row["dataset"], row["method_display"]))
    write_csv(output / "tables" / "table5_1_main_resnet18_corrected.csv", table_5_1)
    write_latex(output / "tables" / "table5_1_main_resnet18_corrected.tex", table_5_1,
                ["dataset", "method_display", "n_seeds", "final_accuracy_mean",
                 "aulc_trapezoid_observed_mean", "aulc_trapezoid_observed_sem"],
                "Corrected main ResNet-18 results using normalized-trapezoidal AULC.")

    table_5_2 = [dict(row, active_scales={0.0: "fine only", 0.7: "coarse + fine", 1.0: "coarse only"}.get(number(row, "alpha"), ""))
                 for row in summary if row["method"] == "MSTC" and number(row, "delta_coarse") == 0.7
                 and number(row, "delta_fine") == 0.5 and number(row, "batch_size") == 50
                 and number(row, "alpha") in (0.0, 0.7, 1.0)]
    write_csv(output / "tables" / "table5_2_single_vs_multiscale_corrected.csv", table_5_2)
    write_latex(output / "tables" / "table5_2_single_vs_multiscale_corrected.tex", table_5_2,
                ["dataset", "backbone", "active_scales", "n_seeds", "final_accuracy_mean",
                 "aulc_trapezoid_observed_mean", "aulc_trapezoid_observed_sem"],
                "Corrected single-scale and multiscale MSTC comparison.")

    table_5_3 = [dict(row, scale_pair="C{:03d}/F{:03d}".format(
        int(number(row, "delta_coarse") * 100), int(number(row, "delta_fine") * 100)))
        for row in summary if row["backbone"] == "resnet18" and row["method"] == "MSTC"
        and number(row, "alpha") == 0.7 and number(row, "batch_size") == 50
        and number(row, "delta_coarse") in (0.6, 0.7, 0.8)]
    write_csv(output / "tables" / "table5_3_scale_sensitivity_corrected.csv", table_5_3)
    write_latex(output / "tables" / "table5_3_scale_sensitivity_corrected.tex", table_5_3,
                ["dataset", "scale_pair", "n_seeds", "final_accuracy_mean",
                 "aulc_trapezoid_observed_mean", "aulc_trapezoid_observed_sem"],
                "Corrected ResNet-18 MSTC scale sensitivity.")

    budget_seed = [row for row in rows if row["dataset"] == "CIFAR100" and row["backbone"] == "resnet18"
                   and row["method"] == "MSTC" and number(row, "alpha") == 0.7
                   and number(row, "delta_coarse") == 0.7 and number(row, "delta_fine") == 0.5]
    # Mirror the historical table's exclusion of incomplete B=500 configurations.
    counts = defaultdict(set)
    for row in budget_seed:
        counts[row["batch_size"]].add(row["seed"])
    budget_seed = [row for row in budget_seed if not (number(row, "batch_size") == 500 and len(counts[row["batch_size"]]) < 5)]
    common_first = max(int(row["first_budget"]) for row in budget_seed) if budget_seed else 0
    common_last = min(int(row["last_budget"]) for row in budget_seed) if budget_seed else 0
    grouped = defaultdict(list)
    for row in budget_seed:
        grouped[row["batch_size"]].append(row)
    table_5_4 = []
    for batch, group in sorted(grouped.items(), key=lambda item: float(item[0])):
        observed, _, observed_sem = stat_values([row["corrected_trapezoid_aulc"] for row in group])
        matched_values = [trapezoid_interval(row["_points"], common_first, common_last) for row in group]
        matched, _, matched_sem = stat_values(matched_values)
        table_5_4.append({
            "batch_size": batch, "n": len(group),
            "observed_first_budget": min(int(row["first_budget"]) for row in group),
            "observed_last_budget": max(int(row["last_budget"]) for row in group),
            "observed_range_trapezoid_aulc": observed,
            "observed_range_trapezoid_sem": observed_sem,
            "matched_first_budget": common_first, "matched_last_budget": common_last,
            "matched_range_trapezoid_aulc": matched,
            "matched_range_trapezoid_sem": matched_sem,
        })
    write_csv(output / "tables" / "table5_4_budget_sensitivity_corrected.csv", table_5_4)
    write_latex(output / "tables" / "table5_4_budget_sensitivity_corrected.tex", table_5_4,
                ["batch_size", "n", "matched_first_budget", "matched_last_budget",
                 "matched_range_trapezoid_aulc", "matched_range_trapezoid_sem"],
                "Corrected CIFAR-100 acquisition-batch sensitivity on a common labelled-budget interval.")
    return summary


def same_value(left, right):
    if left in (None, ""):
        return True
    if right in (None, ""):
        return False
    lf, rf = finite(left), finite(right)
    if lf is not None and rf is not None:
        return abs(lf - rf) <= 1e-8
    return str(left).strip().lower() == str(right).strip().lower()


def lidcover_corrections(metric_by_dir, output):
    source = ROOT / "outputs" / "lidcover_chapter5_matched_radius" / "evidence_pack" / "per_seed_summary_metrics.csv"
    per_seed = []
    for original in read_csv(source):
        metric = metric_by_dir.get(str(Path(original["source_path"]).resolve()))
        if metric is None:
            continue
        row = dict(original)
        row.update({
            "historical_matched_radius_aulc": original.get("aulc", ""),
            "historical_simple_mean": metric["simple_mean"],
            "corrected_trapezoid_aulc": metric["aulc_trapezoid_observed"],
            "correction_delta_vs_historical_table": (
                float(metric["aulc_trapezoid_observed"]) - float(original["aulc"])
                if finite(original.get("aulc")) is not None else ""
            ),
            "n_points": metric["n_valid_points"], "first_budget": metric["first_budget"],
            "last_budget": metric["last_budget"], "complete_101": metric["complete_101_episodes"],
        })
        per_seed.append(row)
    write_csv(output / "per_seed" / "lidcover_matched_radius_per_seed_corrected.csv", per_seed)

    dims = ("dataset", "backbone", "method", "method_key", "variant_key", "delta0",
            "alpha", "k_lid", "batch_size", "selection_mode")
    groups = defaultdict(list)
    for row in per_seed:
        groups[tuple(row.get(field, "") for field in dims)].append(row)
    aggregate = []
    for key, group in groups.items():
        row = dict(zip(dims, key))
        row["n"] = len(set(item["seed"] for item in group))
        for source_metric, target in (
            ("historical_matched_radius_aulc", "historical_aulc"),
            ("historical_simple_mean", "simple_mean"),
            ("corrected_trapezoid_aulc", "corrected_aulc"),
            ("final_accuracy", "final_accuracy"),
        ):
            avg, sd, sem = stat_values([item.get(source_metric) for item in group])
            row[target + "_mean"] = avg
            row[target + "_sd"] = sd
            row[target + "_sem"] = sem
        row["first_budget"] = min(int(item["first_budget"]) for item in group)
        row["last_budget"] = max(int(item["last_budget"]) for item in group)
        row["n_points"] = min(int(item["n_points"]) for item in group)
        aggregate.append(row)
    write_csv(output / "tables" / "lidcover_matched_radius_all_configurations_corrected.csv", aggregate)

    source_tables = ROOT / "outputs" / "lidcover_chapter5_matched_radius" / "tables"
    mapping_report = []
    match_dims = set(dims)
    for table_path in sorted(source_tables.glob("*.csv")):
        source_rows = read_csv(table_path)
        if not source_rows or "aulc_mean" not in source_rows[0]:
            continue
        corrected_rows = []
        for source_row in source_rows:
            comparable = [field for field in match_dims if field in source_row and source_row[field] != ""]
            candidates = [candidate for candidate in aggregate
                          if all(same_value(source_row[field], candidate.get(field)) for field in comparable)]
            out_row = dict(source_row)
            out_row["historical_aulc_mean"] = source_row.get("aulc_mean", "")
            out_row["historical_aulc_sem"] = source_row.get("aulc_sem", "")
            if len(candidates) == 1:
                candidate = candidates[0]
                out_row.update({
                    "corrected_aulc_mean": candidate["corrected_aulc_mean"],
                    "corrected_aulc_sem": candidate["corrected_aulc_sem"],
                    "corrected_minus_historical": (
                        float(candidate["corrected_aulc_mean"]) - float(source_row["aulc_mean"])
                    ),
                    "corrected_first_budget": candidate["first_budget"],
                    "corrected_last_budget": candidate["last_budget"],
                    "corrected_n_points": candidate["n_points"],
                    "mapping_status": "unique",
                })
            else:
                out_row["mapping_status"] = "unresolved_{}_candidates".format(len(candidates))
            corrected_rows.append(out_row)
            mapping_report.append({"source_table": table_path.name, "mapping_status": out_row["mapping_status"]})
        destination = output / "tables" / "lidcover_matched_radius" / (table_path.stem + "_corrected.csv")
        write_csv(destination, corrected_rows)
        preferred = ["dataset", "backbone", "method", "configuration", "analysis", "setting",
                     "control", "base_radius", "alpha_setting", "n", "final_accuracy_mean",
                     "final_accuracy_sem", "corrected_aulc_mean", "corrected_aulc_sem",
                     "corrected_minus_historical"]
        latex_fields = [field for field in preferred if any(field in row for row in corrected_rows)]
        write_latex(destination.with_suffix(".tex"), corrected_rows, latex_fields,
                    "Corrected normalized-trapezoidal AULC replacement for {}.".format(table_path.stem))
    write_csv(output / "audit" / "lidcover_table_mapping_report.csv", mapping_report)
    return aggregate


def write_readme(output, counts):
    text = """# Corrected normalized-trapezoidal AULC tables

Generated from raw `benchmark_summary.json` trajectories. Existing result and
thesis files are not modified.

## Definition

For each seed, episode accuracy is paired with
`labeled_count_before_sampling`. After sorting by labelled budget, corrected
AULC is

`trapz(test_accuracy, labelled_budget) / (last_budget - first_budget)`.

The historical arithmetic mean and stored `test_auc` are retained for audit.
SEM is sample SD across per-seed metrics divided by `sqrt(n)`.

## Main files

- `all_runs_aulc.csv`: every discovered raw run, including quarantined runs
  flagged with `excluded=True`.
- `all_configurations_aulc.csv`: non-quarantined runs aggregated without mixing
  experiment families.
- `tables/table4_1_baselines_resnet18_corrected.csv`
- `tables/table5_1_main_resnet18_corrected.csv`
- `tables/table5_2_single_vs_multiscale_corrected.csv`
- `tables/table5_3_scale_sensitivity_corrected.csv`
- `tables/table5_4_budget_sensitivity_corrected.csv`
- `tables/mstc_all_configurations_corrected.csv`
- `tables/lidcover_matched_radius_all_configurations_corrected.csv`
- `tables/lidcover_matched_radius/*_corrected.csv`
- `tables/cited_values_corrected.csv`
- `audit/invalid_or_incomplete_runs.csv`
- `audit/quarantined_runs.csv`

Budget-sensitivity rows include both each run's observed-range AULC and a
matched-common-budget AULC. Do not compare observed-range AULCs when their
budget ranges differ.

## Scan summary

- summaries discovered: {discovered}
- non-quarantined valid runs: {valid}
- quarantined runs: {quarantined}
- invalid or incomplete/nonstandard runs: {invalid}
""".format(**counts)
    (output / "README.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-files", type=int, default=None,
                        help="Testing only: stop after this many summaries")
    args = parser.parse_args()
    input_root = args.input_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    paths = discover_summaries(input_root, args.max_files)
    rows = []
    for index, path in enumerate(paths, 1):
        if index == 1 or index % 100 == 0:
            print("[{}/{}] {}".format(index, len(paths), path), flush=True)
        rows.append(extract_run(path))

    run_fields = [
        "source_file", "run_directory", "excluded", "dataset", "backbone", "method",
        "method_family", "experiment_name", "experiment_family", "seed", "budget_per_round",
        *CONFIG_FIELDS, "n_records", "n_valid_points", "n_unique_episodes", "n_unique_budgets",
        "first_episode", "last_episode", "first_budget", "last_budget", "complete_101_episodes",
        "expected_budget_schedule", "final_accuracy", "stored_test_auc", "simple_mean",
        "aulc_trapezoid_observed", "aulc_trapezoid_50_5050", "delta_trapezoid_minus_simple",
        "status", "warning",
    ]
    write_csv(output / "all_runs_aulc.csv", rows, run_fields)
    group_fields = ("dataset", "backbone", "method_family", "method", "experiment_family",
                    "budget_per_round", *CONFIG_FIELDS)
    aggregate = aggregate_runs(rows, group_fields)
    write_csv(output / "all_configurations_aulc.csv", aggregate)
    for family in ("baseline", "lidcover", "mstc", "topocover", "other"):
        write_csv(output / "tables" / "families" / (family + "_all_configurations.csv"),
                  [row for row in aggregate if row.get("method_family") == family])

    baselines = canonical_baseline_table(rows)
    write_csv(output / "tables" / "table4_1_baselines_resnet18_corrected.csv", baselines)
    latex_fields = ["dataset", "method", "n_seeds", "first_budget_min", "last_budget_max",
                    "simple_mean_mean", "aulc_trapezoid_observed_mean",
                    "aulc_trapezoid_observed_sem"]
    write_latex(output / "tables" / "table4_1_baselines_resnet18_corrected.tex", baselines,
                latex_fields, "Corrected ResNet-18 baseline AULC using normalized trapezoidal integration.")

    cited = corrected_cited_table(rows)
    write_csv(output / "tables" / "cited_values_corrected.csv", cited)
    write_latex(output / "tables" / "cited_values_corrected.tex", cited,
                ["dataset", "method", "n", "historical_reported_aulc", "corrected_trapezoid_aulc",
                 "corrected_trapezoid_aulc_sem", "change_corrected_minus_reported"],
                "Historical reported trajectory means and corrected normalized-trapezoidal AULC.")

    metric_by_dir = {row["run_directory"]: row for row in rows if row.get("status") == "valid"}
    mstc_provenance_tables(metric_by_dir, output)
    lidcover_corrections(metric_by_dir, output)

    invalid = [row for row in rows if row.get("status") != "valid" or row.get("warning")]
    quarantined = [row for row in rows if row.get("excluded")]
    write_csv(output / "audit" / "invalid_or_incomplete_runs.csv", invalid, run_fields)
    write_csv(output / "audit" / "quarantined_runs.csv", quarantined, run_fields)
    counts = {
        "discovered": len(rows),
        "valid": sum(row.get("status") == "valid" and not row.get("excluded") for row in rows),
        "quarantined": len(quarantined), "invalid": len(invalid),
    }
    write_readme(output, counts)
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()), "input_root": str(input_root),
        "output_dir": str(output), "definition": "normalized trapezoid over labeled_count_before_sampling",
        "counts": counts,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    print("Wrote corrected AULC tables to {}".format(output))


if __name__ == "__main__":
    main()
