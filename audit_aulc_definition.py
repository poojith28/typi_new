#!/usr/bin/env python3
"""Read-only verification of the AULC values cited in thesis Tables 4.1/5.1.

The script reads the original ``benchmark_summary.json`` episode records for
the five ResNet-18 seeds of ProbCover and the principal MultiScaleTopoCover
configuration.  It does not write experiment data or generated table files.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev


REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "TypiClust" / "output"
SEEDS = range(1, 6)
EXPECTED_EPISODES = list(range(101))
EXPECTED_BUDGETS = list(range(50, 5051, 50))


@dataclass(frozen=True)
class Configuration:
    dataset: str
    method: str
    run_pattern: str
    table_5_1_value: float
    table_4_1_value: float | None = None


CONFIGURATIONS = (
    Configuration("CIFAR10", "MultiScaleTopoCover", "MSTC_C070_F050_A070_{seed}_50b", 46.87),
    Configuration("CIFAR10", "ProbCover", "probcover_{seed}_50b", 62.35, 62.4),
    Configuration("CIFAR100", "MultiScaleTopoCover", "MSTC_C070_F050_A070_{seed}_50b", 21.12),
    Configuration("CIFAR100", "ProbCover", "probcover_{seed}_50b", 27.09, 27.1),
    Configuration("TINYIMAGENET", "MultiScaleTopoCover", "MSTC_C070_F050_A070_{seed}_50b", 6.12),
    Configuration("TINYIMAGENET", "ProbCover", "probcover_{seed}_50b", 11.86, 11.9),
)


def normalized_trapezoid(accuracies: list[float], budgets: list[int]) -> float:
    if len(accuracies) != len(budgets) or len(budgets) < 2:
        raise ValueError("AULC requires equally sized accuracy/budget vectors with at least two points")
    width = budgets[-1] - budgets[0]
    if width <= 0 or any(right <= left for left, right in zip(budgets, budgets[1:])):
        raise ValueError("Budgets must be strictly increasing")
    area = sum(
        0.5 * (left_acc + right_acc) * (right_budget - left_budget)
        for left_acc, right_acc, left_budget, right_budget in zip(
            accuracies, accuracies[1:], budgets, budgets[1:]
        )
    )
    return area / width


def load_seed(config: Configuration, seed: int) -> dict:
    path = (
        OUTPUT_ROOT
        / config.dataset
        / "resnet18"
        / config.run_pattern.format(seed=seed)
        / "benchmark_summary.json"
    )
    payload = json.loads(path.read_text())
    records = payload.get("episode_records") or []
    episodes = [int(record["episode"]) for record in records]
    budgets = [int(record["labeled_count_before_sampling"]) for record in records]
    accuracies = [float(record["test_accuracy"]) for record in records]

    if episodes != EXPECTED_EPISODES:
        raise AssertionError(f"{path}: expected episodes 0--100, got {episodes[:2]}...{episodes[-2:]}")
    if budgets != EXPECTED_BUDGETS:
        raise AssertionError(f"{path}: expected budgets 50--5050 by 50, got {budgets[:2]}...{budgets[-2:]}")

    simple_mean = fmean(accuracies)
    stored_test_auc = float(payload["test_auc"])
    if not math.isclose(simple_mean, stored_test_auc, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(
            f"{path}: stored test_auc {stored_test_auc} != trajectory mean {simple_mean}"
        )
    return {
        "path": path,
        "episodes": episodes,
        "budgets": budgets,
        "accuracies": accuracies,
        "simple_mean": simple_mean,
        "normalized_trapezoid": normalized_trapezoid(accuracies, budgets),
    }


def audit_configuration(config: Configuration, reported_value: float) -> tuple[dict, dict]:
    seeds = [load_seed(config, seed) for seed in SEEDS]
    seed_means = [run["simple_mean"] for run in seeds]
    seed_trapezoids = [run["normalized_trapezoid"] for run in seeds]

    mean_curve = [fmean(values) for values in zip(*(run["accuracies"] for run in seeds))]
    mean_after_seed = fmean(seed_means)
    trapezoid_after_seed = fmean(seed_trapezoids)
    mean_after_curve = fmean(mean_curve)
    trapezoid_after_curve = normalized_trapezoid(mean_curve, seeds[0]["budgets"])
    if not math.isclose(mean_after_seed, mean_after_curve, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("Mean aggregation order unexpectedly changed the result")
    if not math.isclose(trapezoid_after_seed, trapezoid_after_curve, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("Trapezoid aggregation order unexpectedly changed the result")

    row = {
        "dataset": config.dataset,
        "method": config.method,
        "number_of_evaluated_points": len(seeds[0]["episodes"]),
        "first_budget": seeds[0]["budgets"][0],
        "last_budget": seeds[0]["budgets"][-1],
        "reported_thesis_value": reported_value,
        "simple_mean": mean_after_seed,
        "normalized_trapezoid": trapezoid_after_seed,
        "difference_reported_vs_mean": reported_value - mean_after_seed,
        "difference_reported_vs_trapezoid": reported_value - trapezoid_after_seed,
    }
    detail = {
        "mean_sem": stdev(seed_means) / math.sqrt(len(seed_means)),
        "trapezoid_sem": stdev(seed_trapezoids) / math.sqrt(len(seed_trapezoids)),
        "mean_order_difference": mean_after_curve - mean_after_seed,
        "trapezoid_order_difference": trapezoid_after_curve - trapezoid_after_seed,
        "paths": [str(run["path"]) for run in seeds],
    }
    return row, detail


COLUMNS = (
    "dataset",
    "method",
    "number_of_evaluated_points",
    "first_budget",
    "last_budget",
    "reported_thesis_value",
    "simple_mean",
    "normalized_trapezoid",
    "difference_reported_vs_mean",
    "difference_reported_vs_trapezoid",
)


def print_markdown(rows: list[dict]) -> None:
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "|".join("---" for _ in COLUMNS) + "|")
    for row in rows:
        cells = []
        for column in COLUMNS:
            value = row[column]
            cells.append(f"{value:.8f}" if isinstance(value, float) else str(value))
        print("| " + " | ".join(cells) + " |")


def main() -> None:
    table_5_rows = []
    table_4_rows = []
    details = {}
    for config in CONFIGURATIONS:
        row, detail = audit_configuration(config, config.table_5_1_value)
        table_5_rows.append(row)
        details[(config.dataset, config.method)] = detail
        if config.table_4_1_value is not None:
            table_4_row = dict(row)
            table_4_row["reported_thesis_value"] = config.table_4_1_value
            table_4_row["difference_reported_vs_mean"] = config.table_4_1_value - row["simple_mean"]
            table_4_row["difference_reported_vs_trapezoid"] = (
                config.table_4_1_value - row["normalized_trapezoid"]
            )
            table_4_rows.append(table_4_row)

    print("Table 4.1 examples (one-decimal reported values)\n")
    print_markdown(table_4_rows)
    print("\nTable 5.1 (two-decimal reported values)\n")
    print_markdown(table_5_rows)
    print("\nAggregation and SEM checks\n")
    for config in CONFIGURATIONS:
        detail = details[(config.dataset, config.method)]
        print(
            f"{config.dataset}/{config.method}: "
            f"mean_SEM={detail['mean_sem']:.8f}, "
            f"trapezoid_SEM={detail['trapezoid_sem']:.8f}, "
            f"mean_order_difference={detail['mean_order_difference']:.3g}, "
            f"trapezoid_order_difference={detail['trapezoid_order_difference']:.3g}"
        )


if __name__ == "__main__":
    main()
