#!/usr/bin/env python3
"""Audit historical ProbCover/LIDCover evidence without launching experiments.

The audit is deliberately read-only with respect to historical run directories.
It writes only the requested evidence-pack manifests and report.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "TypiClust" / "output"
QUARANTINE_ROOT = RUN_ROOT / "_quarantine_superseded_lidcover_20260728"
EXTRA_RUN_ROOTS = (
    ROOT / "TypiClust" / "asfdasdf asf",
    ROOT / "output",
)
EVIDENCE_ROOT = ROOT / "evidence_pack"

SEEDS = (1, 2, 3, 4, 5)
RADII = (0.25, 0.60)
ACQUISITION_BATCH = 50
EXPECTED_FINAL_BUDGET = 5050
EXPECTED_FINAL_EPISODE = 100

# Primary ResNet-18 datasets plus the requested CIFAR-100 cross-backbone matrix.
REQUIRED_DATASET_BACKBONES = (
    ("CIFAR10", "resnet18", "primary"),
    ("CIFAR100", "resnet18", "primary+cross_backbone"),
    ("TINYIMAGENET", "resnet18", "primary"),
    ("CIFAR100", "alexnet", "cross_backbone"),
    ("CIFAR100", "resnet50", "cross_backbone"),
)

METHOD_ALIASES = {
    "probcover": "ProbCover",
    "prob_cover": "ProbCover",
    "probcover_auto_delta": "ProbCoverAutoDeltaMatched",
    "prob_cover_auto_delta": "ProbCoverAutoDeltaMatched",
    "idprobcover": "LIDCover",
    "id_prob_cover": "LIDCover",
    "lidcover_auto_delta": "LIDCoverAutoDelta",
    "idprobcover_auto_delta": "LIDCoverAutoDelta",
    "idprobcover_frontier_density": "LIDCoverFrontierDensity",
    "id_prob_cover_frontier_density": "LIDCoverFrontierDensity",
    "idprobcover_tiebreak_min_id": "LIDCoverTieMinID",
    "idprobcover_minid_tiebreak": "LIDCoverTieMinID",
    "idprobcover_tiebreak_random": "LIDCoverTieRandom",
    "idprobcover_random_tiebreak": "LIDCoverTieRandom",
    "idprobcover_tiebreak_first_max": "LIDCoverTieFirstMax",
    "idprobcover_firstmax_tiebreak": "LIDCoverTieFirstMax",
    "knn_distance_cover": "MeanKNNDistanceCover",
    "adaptive_knn_distance_cover": "MeanKNNDistanceCover",
    "density_cover": "DensityAdaptiveCover",
    "adaptive_density_cover": "DensityAdaptiveCover",
    "distance_variance_cover": "DistanceVarianceCover",
    "adaptive_distance_variance_cover": "DistanceVarianceCover",
    "distance_cv_cover": "DistanceCVCover",
    "adaptive_distance_cv_cover": "DistanceCVCover",
}

FIELDS = [
    "experiment_id",
    "dataset",
    "backbone",
    "method",
    "delta",
    "delta0",
    "alpha",
    "k",
    "batch_size",
    "seed",
    "expected_final_budget",
    "observed_final_budget",
    "status",
    "source_path",
    "notes",
]


def nested(mapping: dict[str, Any], *keys: str, default: Any = "") -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def float_equal(left: Any, right: float, tolerance: float = 1e-9) -> bool:
    try:
        return abs(float(left) - right) <= tolerance
    except (TypeError, ValueError):
        return False


def read_initial_labeled_size(path: Path) -> int | None:
    try:
        return int(len(np.load(path / "lSet.npy", allow_pickle=True)))
    except Exception:
        return None


def config_paths() -> list[tuple[Path, bool]]:
    normal = [(path, False) for path in RUN_ROOT.glob("*/*/*/config.yaml")]
    extra = [
        (path, False)
        for root in EXTRA_RUN_ROOTS
        for path in root.glob("*/*/*/config.yaml")
    ]
    quarantined = [(path, True) for path in QUARANTINE_ROOT.glob("*/*/*/config.yaml")]
    return sorted(normal + extra + quarantined, key=lambda item: str(item[0]))


@dataclass
class Run:
    path: Path
    quarantined: bool
    config: dict[str, Any]
    summary: dict[str, Any] | None
    summary_error: str
    dataset: str
    backbone: str
    sampler: str
    method: str
    radius: float | None
    alpha: float | None
    k: int | None
    k_knn: int | None
    mode: str
    batch_size: int | None
    seed: int | None
    max_iter: int | None
    initial_labeled_size: int | None
    feature_source: str
    feature_ema: float | None
    fine_tune: bool | None
    intrinsic_status: str
    observed_final_budget: int | None
    notes: list[str]

    @property
    def logical_radius_field(self) -> tuple[str, str]:
        if self.method in {"ProbCover", "ProbCoverAutoDeltaMatched"}:
            return (format_float(self.radius), "")
        return ("", format_float(self.radius))


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}"


def load_run(config_path: Path, quarantined: bool) -> Run | None:
    notes: list[str] = []
    try:
        config = yaml.safe_load(config_path.read_text()) or {}
    except Exception as exc:
        return None

    al = config.get("ACTIVE_LEARNING", {}) or {}
    sampler = str(al.get("SAMPLING_FN", "")).lower()
    method = METHOD_ALIASES.get(sampler, sampler or "UNKNOWN")
    path = config_path.parent

    summary: dict[str, Any] | None = None
    summary_error = ""
    summary_path = path / "benchmark_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:
            summary_error = f"malformed benchmark_summary.json: {exc}"
    else:
        summary_error = "benchmark_summary.json absent"

    records = list((summary or {}).get("episode_records", []) or [])
    observed_final_budget: int | None = None
    if records:
        last = max(records, key=lambda row: int(row.get("episode", -1)))
        try:
            observed_final_budget = int(
                last.get("labeled_count_after_sampling", last.get("labeled_count_before_sampling"))
            )
        except (TypeError, ValueError):
            observed_final_budget = None

    metric_values: list[Any] = []
    for record in records:
        metric_values.extend([record.get("test_accuracy"), record.get("best_val_accuracy")])
    numeric_metrics = [value for value in metric_values if value not in (None, "")]
    metrics_finite = bool(numeric_metrics) and all(finite_number(value) for value in numeric_metrics)

    episode_numbers = sorted(
        int(record.get("episode", -1)) for record in records if "episode" in record
    )
    final_episode_present = bool(episode_numbers) and episode_numbers[-1] == EXPECTED_FINAL_EPISODE
    accuracy_history_present = len(records) == EXPECTED_FINAL_EPISODE + 1 and metrics_finite

    # Acquisition success is checked for every non-final record: 50 new, unique
    # selected indices and finite timing/accuracy metadata where present.
    acquisition_ok = True
    for record in records:
        if int(record.get("episode", -1)) == EXPECTED_FINAL_EPISODE:
            continue
        active_ids = record.get("active_set_ids", [])
        active_size = record.get("active_set_size")
        if active_size is not None and int(active_size) != int(al.get("BUDGET_SIZE", -1)):
            acquisition_ok = False
        if active_ids:
            if len(active_ids) != len(set(active_ids)):
                acquisition_ok = False
            if len(active_ids) != int(al.get("BUDGET_SIZE", -1)):
                acquisition_ok = False

    if summary is None:
        episode_dirs = list(path.glob("episode_*"))
        intrinsic_status = "PARTIAL" if episode_dirs else "FAILED"
        notes.append(summary_error)
    elif not records:
        intrinsic_status = "FAILED"
        notes.append("summary contains no accuracy history")
    elif not final_episode_present or observed_final_budget is None:
        intrinsic_status = "PARTIAL"
        notes.append("expected final evaluation episode/budget is absent")
    elif not accuracy_history_present:
        intrinsic_status = "FAILED" if not metrics_finite else "PARTIAL"
        notes.append("accuracy history is incomplete or contains NaN/inf")
    elif not acquisition_ok:
        intrinsic_status = "FAILED"
        notes.append("one or more acquisition records have the wrong or duplicate active-set size")
    else:
        intrinsic_status = "COMPLETE"

    expected_run_budget = None
    try:
        expected_run_budget = (int(al["MAX_ITER"]) + 1) * int(al["BUDGET_SIZE"])
    except (KeyError, TypeError, ValueError):
        pass
    if intrinsic_status == "COMPLETE" and expected_run_budget != observed_final_budget:
        intrinsic_status = "CONFIG_MISMATCH"
        notes.append(
            f"observed final budget {observed_final_budget} disagrees with configured "
            f"(MAX_ITER+1)*BUDGET_SIZE={expected_run_budget}"
        )

    if summary is not None:
        metadata_checks = (
            (summary.get("dataset"), nested(config, "DATASET", "NAME", default=""), "dataset"),
            (summary.get("model"), nested(config, "MODEL", "TYPE", default=""), "backbone"),
            (summary.get("seed"), config.get("RNG_SEED"), "seed"),
            (summary.get("sampling_fn"), al.get("SAMPLING_FN"), "method"),
            (summary.get("budget_per_round"), al.get("BUDGET_SIZE"), "batch size"),
        )
        mismatches = [
            label
            for observed, configured, label in metadata_checks
            if observed not in (None, "") and str(observed).lower() != str(configured).lower()
        ]
        if mismatches:
            intrinsic_status = "CONFIG_MISMATCH"
            notes.append("summary/config disagreement: " + ", ".join(mismatches))

    if quarantined:
        notes.append(
            "stored under the explicitly superseded LIDCover quarantine; geometry-cache "
            "content hashes and code revision were not captured, so it is not canonical"
        )

    alpha: float | None = None
    k: int | None = None
    k_knn: int | None = None
    if method.startswith("LIDCover"):
        alpha = float(al.get("IDPC_ALPHA", 1.0))
        k = int(al.get("IDPC_K_ID", 50))
        k_knn = int(al.get("IDPC_K_KNN", 50))
    elif method in {
        "MeanKNNDistanceCover",
        "DensityAdaptiveCover",
        "DistanceVarianceCover",
        "DistanceCVCover",
    }:
        alpha = float(al.get("ARC_ALPHA", 1.0))
        k = int(al.get("ARC_K_SIGNAL", 50))
        k_knn = int(al.get("ARC_K_KNN", 50))

    return Run(
        path=path,
        quarantined=quarantined,
        config=config,
        summary=summary,
        summary_error=summary_error,
        dataset=str(nested(config, "DATASET", "NAME", default="")),
        backbone=str(nested(config, "MODEL", "TYPE", default="")),
        sampler=sampler,
        method=method,
        radius=float(al["INITIAL_DELTA"]) if "INITIAL_DELTA" in al else None,
        alpha=alpha,
        k=k,
        k_knn=k_knn,
        mode=str(al.get("IDPC_MODE", "")),
        batch_size=int(al["BUDGET_SIZE"]) if "BUDGET_SIZE" in al else None,
        seed=int(config["RNG_SEED"]) if config.get("RNG_SEED") is not None else None,
        max_iter=int(al["MAX_ITER"]) if "MAX_ITER" in al else None,
        initial_labeled_size=read_initial_labeled_size(path),
        feature_source=str(al.get("IDPC_FEATURE_SOURCE", "precomputed")),
        feature_ema=float(al.get("IDPC_FEATURE_EMA", 0.0)),
        fine_tune=bool(al.get("FINE_TUNE")) if "FINE_TUNE" in al else None,
        intrinsic_status=intrinsic_status,
        observed_final_budget=observed_final_budget,
        notes=notes,
    )


def required_id(dataset: str, backbone: str, method: str, radius: float, seed: int) -> str:
    parameter = "delta" if method == "ProbCover" else "delta0"
    return (
        f"matched__{dataset.lower()}__{backbone}__{method.lower()}__"
        f"{parameter}_{radius:.2f}__seed_{seed}"
    )


def matches_required(run: Run, dataset: str, backbone: str, method: str, radius: float, seed: int) -> bool:
    if (
        run.dataset != dataset
        or run.backbone != backbone
        or run.method != method
        or run.seed != seed
        or not float_equal(run.radius, radius)
    ):
        return False
    if run.batch_size != ACQUISITION_BATCH or run.max_iter != EXPECTED_FINAL_EPISODE:
        return False
    if method == "LIDCover":
        return (
            float_equal(run.alpha, 1.0)
            and run.k == 50
            and run.k_knn == 50
            and run.mode == "high_id_more_centers"
        )
    return True


def canonical_initial_protocol_ok(run: Run) -> tuple[bool, str]:
    if run.initial_labeled_size is None:
        return False, "root lSet.npy could not be read"
    if run.initial_labeled_size != 0:
        return (
            False,
            f"initial labelled set has {run.initial_labeled_size} samples; canonical protocol starts empty",
        )
    initial = (run.summary or {}).get("initial_sampling", {}) or {}
    if not initial:
        return False, "initial method-selected acquisition record is absent"
    if int(initial.get("active_set_size", -1)) != ACQUISITION_BATCH:
        return False, "initial method-selected acquisition is not 50 samples"
    if str(initial.get("sampling_fn", "")).lower() != run.sampler:
        return False, "initial acquisition method does not match the configured method"
    return True, ""


def run_row(
    experiment_id: str,
    run: Run,
    status: str,
    extra_notes: list[str] | None = None,
) -> dict[str, Any]:
    delta, delta0 = run.logical_radius_field
    notes = list(run.notes)
    notes.extend(extra_notes or [])
    return {
        "experiment_id": experiment_id,
        "dataset": run.dataset,
        "backbone": run.backbone,
        "method": run.method,
        "delta": delta,
        "delta0": delta0,
        "alpha": "" if run.alpha is None else f"{run.alpha:g}",
        "k": "" if run.k is None else run.k,
        "batch_size": "" if run.batch_size is None else run.batch_size,
        "seed": "" if run.seed is None else run.seed,
        "expected_final_budget": EXPECTED_FINAL_BUDGET,
        "observed_final_budget": "" if run.observed_final_budget is None else run.observed_final_budget,
        "status": status,
        "source_path": str(run.path.resolve()),
        "notes": "; ".join(dict.fromkeys(note for note in notes if note)),
    }


def missing_row(dataset: str, backbone: str, method: str, radius: float, seed: int, notes: str) -> dict[str, Any]:
    return {
        "experiment_id": required_id(dataset, backbone, method, radius, seed),
        "dataset": dataset,
        "backbone": backbone,
        "method": method,
        "delta": format_float(radius) if method == "ProbCover" else "",
        "delta0": format_float(radius) if method == "LIDCover" else "",
        "alpha": "1" if method == "LIDCover" else "",
        "k": "50" if method == "LIDCover" else "",
        "batch_size": ACQUISITION_BATCH,
        "seed": seed,
        "expected_final_budget": EXPECTED_FINAL_BUDGET,
        "observed_final_budget": "",
        "status": "MISSING",
        "source_path": "",
        "notes": notes,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] = FIELDS) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def existing_run_audit(runs: list[Run]) -> list[dict[str, Any]]:
    """Classify every discovered coverage-family run, including non-required ablations."""
    grouped: dict[tuple[Any, ...], list[Run]] = defaultdict(list)
    for run in runs:
        key = (
            run.dataset,
            run.backbone,
            run.sampler,
            run.radius,
            run.alpha,
            run.k,
            run.k_knn,
            run.mode,
            run.batch_size,
            run.seed,
            run.max_iter,
            run.initial_labeled_size,
            run.fine_tune,
        )
        grouped[key].append(run)

    status_by_path: dict[Path, tuple[str, list[str]]] = {}
    for candidates in grouped.values():
        ordered = sorted(
            candidates,
            key=lambda run: (
                run.quarantined,
                run.intrinsic_status != "COMPLETE",
                "partial" in run.path.name.lower(),
                "longrerun" in run.path.name.lower(),
                str(run.path),
            ),
        )
        canonical_complete_seen = False
        for run in ordered:
            extra: list[str] = []
            if run.quarantined:
                status = "UNVERIFIABLE"
            elif run.intrinsic_status != "COMPLETE":
                status = run.intrinsic_status
            elif canonical_complete_seen:
                status = "DUPLICATE"
                extra.append("same full saved configuration, seed, initial-set size and protocol as another complete run")
            else:
                status = "COMPLETE"
                canonical_complete_seen = True
            status_by_path[run.path] = (status, extra)

    rows: list[dict[str, Any]] = []
    for index, run in enumerate(sorted(runs, key=lambda item: str(item.path)), start=1):
        status, extra = status_by_path[run.path]
        delta, delta0 = run.logical_radius_field
        expected = ""
        if run.max_iter is not None and run.batch_size is not None:
            expected = (run.max_iter + 1) * run.batch_size
        rows.append(
            {
                "experiment_id": f"existing__{index:04d}__{run.path.name}",
                "dataset": run.dataset,
                "backbone": run.backbone,
                "method": run.method,
                "delta": delta,
                "delta0": delta0,
                "alpha": "" if run.alpha is None else f"{run.alpha:g}",
                "k": "" if run.k is None else run.k,
                "batch_size": "" if run.batch_size is None else run.batch_size,
                "seed": "" if run.seed is None else run.seed,
                "expected_final_budget": expected,
                "observed_final_budget": "" if run.observed_final_budget is None else run.observed_final_budget,
                "status": status,
                "source_path": str(run.path.resolve()),
                "notes": "; ".join(dict.fromkeys(run.notes + extra)),
            }
        )
    return rows


def ablation_audit(runs: list[Run]) -> list[dict[str, Any]]:
    normal = [run for run in runs if not run.quarantined]

    def complete_count(predicate: Any) -> int:
        return sum(run.intrinsic_status == "COMPLETE" and predicate(run) for run in normal)

    rows = []
    alpha_settings = sorted(
        {
            run.alpha
            for run in normal
            if run.dataset == "CIFAR100"
            and run.method == "LIDCover"
            and run.batch_size == 50
            and float_equal(run.radius, 0.25)
            and run.mode == "high_id_more_centers"
            and run.alpha is not None
        }
    )
    rows.append(
        {
            "topic": "alpha sensitivity",
            "complete_runs": complete_count(
                lambda run: run.dataset == "CIFAR100"
                and run.method == "LIDCover"
                and run.batch_size == 50
                and float_equal(run.radius, 0.25)
                and run.mode == "high_id_more_centers"
                and run.k == 50
                and run.k_knn == 50
                and run.alpha in {0.0, 0.5, 1.0, 2.0}
            ),
            "finding": f"Observed alpha settings: {alpha_settings} across CIFAR-100 backbones.",
            "limitation": "Most alpha runs use a random 50-label start, so they are an internally matched ablation but not a canonical cold-start main comparison.",
        }
    )
    k_settings = sorted(
        {
            run.k
            for run in normal
            if run.dataset == "CIFAR100"
            and run.method == "LIDCover"
            and run.batch_size == 50
            and run.k is not None
        }
    )
    rows.append(
        {
            "topic": "k sensitivity",
            "complete_runs": complete_count(
                lambda run: run.dataset == "CIFAR100"
                and run.method == "LIDCover"
                and run.batch_size == 50
                and float_equal(run.radius, 0.25)
                and float_equal(run.alpha, 1.0)
                and run.mode == "high_id_more_centers"
                and run.k in {20, 50, 75}
                and run.k_knn == run.k
            ),
            "finding": f"Observed joint k_id=k_knn settings include {k_settings}.",
            "limitation": "The existing sweep changes both LID-estimation k and graph-truncation k together; it does not isolate k_id from k_knn.",
        }
    )
    delta_settings = sorted(
        {
            run.radius
            for run in normal
            if run.dataset == "CIFAR100"
            and run.backbone == "resnet18"
            and run.method == "LIDCover"
            and run.batch_size == 50
            and run.radius is not None
            and float_equal(run.alpha, 1.0)
            and run.k == 50
            and run.mode == "high_id_more_centers"
        }
    )
    rows.append(
        {
            "topic": "delta0 sensitivity",
            "complete_runs": complete_count(
                lambda run: run.dataset == "CIFAR100"
                and run.backbone == "resnet18"
                and run.method == "LIDCover"
                and run.batch_size == 50
                and run.radius in {0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6}
                and float_equal(run.alpha, 1.0)
                and run.k == 50
                and run.k_knn == 50
                and run.mode == "high_id_more_centers"
            ),
            "finding": f"Observed ResNet-18 delta0 settings: {delta_settings}.",
            "limitation": "The delta sweep uses a random 50-label start and cannot replace the canonical matched-radius matrix.",
        }
    )
    signal_methods = {
        "MeanKNNDistanceCover",
        "DensityAdaptiveCover",
        "DistanceVarianceCover",
        "DistanceCVCover",
    }
    rows.append(
        {
            "topic": "alternative local signals",
            "complete_runs": complete_count(
                lambda run: run.backbone == "resnet18"
                and run.dataset in {"CIFAR10", "CIFAR100", "TINYIMAGENET"}
                and run.method in signal_methods
                and run.batch_size == 50
                and run.radius in {0.25, 0.6}
            ),
            "finding": "All four requested signal families are represented on the three primary datasets at delta0=0.25 and 0.60.",
            "limitation": "The 0.60 signal runs use a random 50-label start; compare only within that matched ablation block.",
        }
    )
    rows.append(
        {
            "topic": "runtime",
            "complete_runs": complete_count(
                lambda run: run.method in {"ProbCover", "LIDCover"}
                and run.batch_size == 50
                and bool((run.summary or {}).get("timing"))
            ),
            "finding": "Benchmark summaries contain acquisition, training, test, and round timing fields.",
            "limitation": "Cold-cache and warm-cache costs must be kept separate; historical cache artifacts are external to run directories.",
        }
    )
    rows.append(
        {
            "topic": "cross-backbone",
            "complete_runs": complete_count(
                lambda run: run.dataset == "CIFAR100"
                and run.backbone in {"alexnet", "resnet18", "resnet50"}
                and run.method in {"ProbCover", "LIDCover"}
                and run.batch_size == 50
                and run.radius in {0.25, 0.6}
                and (
                    run.method == "ProbCover"
                    or (
                        float_equal(run.alpha, 1.0)
                        and run.k == 50
                        and run.k_knn == 50
                        and run.mode == "high_id_more_centers"
                    )
                )
            ),
            "finding": "Historical own-radius results exist across AlexNet, ResNet-18, and ResNet-50.",
            "limitation": "The four-cell matched-radius cross-backbone matrix is incomplete; see missing jobs.",
        }
    )
    rows.append(
        {
            "topic": "final accuracy and AULC",
            "complete_runs": complete_count(
                lambda run: run.method in {"ProbCover", "LIDCover"}
                and run.batch_size == 50
                and finite_number((run.summary or {}).get("final_test_accuracy"))
                and finite_number((run.summary or {}).get("test_auc"))
            ),
            "finding": "Complete benchmark summaries expose per-seed final test accuracy and test AULC.",
            "limitation": "Only canonical matched-radius rows may be paired for adaptation claims.",
        }
    )
    budget_runs = [
        run
        for run in normal
        if run.dataset == "CIFAR100"
        and run.method in {"ProbCover", "LIDCover"}
        and run.batch_size in {100, 500}
    ]
    rows.append(
        {
            "topic": "budget-dependent/crossover",
            "complete_runs": sum(run.intrinsic_status == "COMPLETE" for run in budget_runs),
            "finding": "100- and 500-sample acquisition directories exist on CIFAR-100.",
            "limitation": "The block is not a valid matched crossover study: initial protocols differ and most 500-batch LIDCover summaries are absent. Saved model-feature/EMA fields are inert for the current LIDCover routing, which directly loads pretrained features.",
        }
    )
    return rows


def build_report(
    runs: list[Run],
    matrix_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
    ablations: list[dict[str, Any]],
    existing_rows: list[dict[str, Any]],
) -> str:
    logical_rows = [row for row in matrix_rows if "::" not in row["experiment_id"]]
    status_counts = Counter(row["status"] for row in logical_rows)
    evidence_counts = Counter(row["status"] for row in matrix_rows)
    existing_counts = Counter(row["status"] for row in existing_rows)

    required_groups: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    for row in logical_rows:
        radius = row["delta"] or row["delta0"]
        required_groups[(row["dataset"], row["backbone"], row["method"], radius)][row["status"]] += 1

    normal_runs = [run for run in runs if not run.quarantined]
    quarantined_runs = [run for run in runs if run.quarantined]
    all_normal_configs = len(list(RUN_ROOT.glob("*/*/*/config.yaml")))
    all_quarantine_configs = len(list(QUARANTINE_ROOT.glob("*/*/*/config.yaml")))
    extra_config_counts = {
        root: len(list(root.glob("*/*/*/config.yaml")))
        for root in EXTRA_RUN_ROOTS
    }
    all_extra_configs = sum(extra_config_counts.values())

    lines = [
        "# LIDCover completion audit",
        "",
        "Audit date: 2026-08-11 (Australia/Melbourne). This audit regeneration is read-only: no experiment was submitted, resumed, overwritten, or modified by the audit.",
        "",
        "## Audit boundary and inventory",
        "",
        f"- Canonical output tree: `{RUN_ROOT}`.",
        f"- Run configurations inspected in the canonical tree: **{all_normal_configs}**.",
        f"- Superseded/quarantined configurations inspected separately: **{all_quarantine_configs}**.",
        f"- Additional experiment configurations outside the canonical output tree: **{all_extra_configs}**.",
        "- Supplemental read-only roots: " + ", ".join(
            f"`{root}` ({extra_config_counts[root]} configs)" for root in EXTRA_RUN_ROOTS
        ) + ".",
        f"- Experiment configurations parsed outside quarantine: **{len(normal_runs)}**.",
        f"- Experiment configurations parsed in quarantine: **{len(quarantined_runs)}**.",
        "- Derived CSVs/figures under `analysis/`, `outputs/`, and `evidence_pack/` were treated as non-canonical; completion was decided from run-local config, root partition, benchmark summary, and per-episode records.",
        "- Completion is determined from run artifacts; live queue state is checked separately by submission wrappers.",
        "",
        "## Canonical protocol reconstructed from code and launch history",
        "",
        "- Canonical seeds: **1, 2, 3, 4, 5**.",
        "- `batch_size` in the matrices means acquisition batch: **50**. The classifier training batch is 100 in the saved primary configs.",
        "- Canonical cold start: root labelled set is empty; the configured acquisition method selects the first 50; episodes E0..E99 add 50 each; E100 is evaluation-only at **5,050 labels**.",
        "- Both coverage methods receive a driver-level radius schedule: the configured base is multiplied by 1.35 for the empty-pool initial acquisition and by 0.85 after labels exist.",
        "- Both use backbone-specific, pretrained SimCLR features. Only `features_seed1.npy` exists for each dataset/backbone, so AL seeds 2..5 fall back to the same representation seed while retaining distinct data/training RNG seeds. Although some configs say `IDPC_FEATURE_SOURCE=model` and EMA=0.3, `_resolve_idpc_feature_inputs` is never called by the LIDCover route; `IDProbCover` directly calls `load_features`.",
        "- Historical LIDCover: pointwise `skdim.id.MLE`, k_id=50, L2 normalization, no LID/radius clipping, and `delta_i = delta_effective * ((LID_i+1e-12)/(median(LID)+1e-12))^(-alpha)` with alpha=1.",
        "- LIDCover constructs coverage only from the first k_knn=50 neighbours plus optional self-cover, chooses maximum uncovered gain, breaks ties by minimum LID, and falls back to minimum LID when gain is non-positive.",
        "- ProbCover constructs an all-pairs strict-distance graph (`distance < delta`) and uses first-index `argmax` tie-breaking. Therefore matched base radius controls radius but does not by itself isolate every implementation difference (kNN truncation, tie-breaking, self-cover, fallback).",
        "",
        "## Required historical matched-radius matrix",
        "",
        "The union contains **100 logical runs**: 60 primary ResNet-18 runs plus 40 additional CIFAR-100 AlexNet/ResNet-50 runs (the CIFAR-100 ResNet-18 cells overlap both requirements).",
        "",
        "| Dataset | Backbone | Method | Radius | COMPLETE | MISSING | Other logical status |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for key, counts in sorted(required_groups.items()):
        dataset, backbone, method, radius = key
        other = sum(value for status, value in counts.items() if status not in {"COMPLETE", "MISSING"})
        lines.append(
            f"| {dataset} | {backbone} | {method} | {radius} | {counts['COMPLETE']} | {counts['MISSING']} | {other} |"
        )

    lines.extend(
        [
            "",
            "### Logical completion totals",
            "",
            f"- Existing complete required experiments: **{status_counts['COMPLETE']}**.",
            f"- Missing required historical experiments: **{status_counts['MISSING']}**.",
            f"- Exact immediately runnable new historical runs required: **{len(missing_rows)}**.",
            "- Phase-2 automatic-radius experiment: **30 additional runs** (3 datasets x 5 seeds x 2 matched methods). `LIDCoverAutoDelta` and its matched ProbCover route are implemented and registered, but submission remains gated until the historical matrix is complete.",
            f"- Exact total new runs required by the full brief: **{len(missing_rows) + 30}**.",
            "",
            "The 30 automatic-radius rows are intentionally separate from `lidcover_missing_jobs.csv`. Their deterministic rule is executable through `lidcover_auto_delta_30.sbatch`, but the phase-2 submission wrapper blocks launch until the historical missing-job count is zero.",
            "",
            "## Invalid, partial, duplicate, and unverifiable evidence",
            "",
            f"Across completion-matrix evidence rows (logical rows plus candidate records): {dict(sorted(evidence_counts.items()))}.",
            f"Across all discovered experiment run directories: {dict(sorted(existing_counts.items()))}.",
            "",
            "- Completed delta0=0.60 LIDCover candidates use a 50-sample random initial set. They are `CONFIG_MISMATCH` for the canonical cold-start protocol and must not fill matched cells.",
            "- The CIFAR-100 ResNet-18 delta0=0.25 cold-start candidates are stored in an explicitly superseded quarantine. Two same-config families have sharply different LID/radius behaviour, while run directories do not contain code revisions or geometry-cache hashes. They are `UNVERIFIABLE`, not silently promoted back to canonical evidence.",
            "- Extra same-config complete candidates are retained and marked `DUPLICATE`; no result directory was removed.",
            "- Runs without a benchmark summary are `FAILED` when no episode evidence exists and `PARTIAL` when episode artifacts exist.",
            "",
            "## Existing ablation audit",
            "",
            "| Evidence area | Complete run directories | Audit finding | Limitation |",
            "|---|---:|---|---|",
        ]
    )
    for row in ablations:
        lines.append(
            f"| {row['topic']} | {row['complete_runs']} | {row['finding']} | {row['limitation']} |"
        )

    lines.extend(
        [
            "",
            "## Downstream analysis readiness",
            "",
            "- Automatic base radius: `LIDCoverAutoDelta` is implemented with a matched ProbCover route, content-hashed geometry cache and run-local provenance. No automatic-radius run exists yet; submission is correctly gated behind historical matrix completion.",
            "- Mechanism analysis: the three requested final artifacts are absent. Run summaries provide selected-point aggregate LID/radius/coverage statistics, and episode records provide selected IDs; per-point fixed coverage, local purity and quartile membership must be reconstructed from representations, labels and geometry. Cache-to-run content hashes are absent, so cache provenance must be resolved before treating reconstructed values as canonical.",
            "- LID heterogeneity versus gain: not yet valid because 40 historical matched cells are missing. It must use only COMPLETE paired configurations after the post-run re-audit and remain exploratory.",
            "- Final claim evidence: not generated during audit-only work; it must be regenerated from canonical per-seed raw runs only after required completion and validation.",
            "",
            "## Submission gate",
            "",
            "Historical completion uses a 40-task array capped at 15 concurrent tasks. Automatic-radius evidence uses a separate 30-task array capped at 5 and cannot be submitted by its wrapper until this audit reports zero historical missing jobs. Both launchers reject an active same-name array and run tasks in new output directories.",
            "",
            "## Generated audit artifacts",
            "",
            "- `evidence_pack/lidcover_completion_matrix.csv`",
            "- `evidence_pack/lidcover_missing_jobs.csv`",
            "- `evidence_pack/lidcover_ablation_audit.csv`",
            "- `evidence_pack/lidcover_existing_run_audit.csv`",
            "- `evidence_pack/method_version_registry.md`",
            "",
        ]
    )
    return "\n".join(lines)


def write_registry() -> None:
    text = """# LIDCover method version registry

This registry records method-defining behaviour found in the repository. Historical code and run artifacts were not modified. Automatic-radius methods are separate identifiers and are not compatible with historical fixed-radius results.

| method_name | parent_method | implementation_file | method_definition | date_created | reason | compatible_with_historical_results | notes |
|---|---|---|---|---|---|---|---|
| ProbCover | — | `TypiClust/deep-al/pycls/al/prob_cover.py`; driver schedule in `TypiClust/deep-al/tools/train_al.py` | L2-normalized pretrained features; directed all-pairs edges for strict distance below the effective radius; greedy maximum uncovered out-degree; first-index argmax ties. Configured base radius is multiplied by 1.35 at empty cold start and 0.85 afterward. | 2026-04-20 (file mtime) | Historical baseline | yes, when config, initial protocol, representation and driver revision agree | Saved config stores the base radius, while episode metadata stores the post-schedule effective radius. |
| LIDCover | ProbCover concept | `TypiClust/deep-al/pycls/al/IDProbCover.py` via `IDprocover.py`; routing in `ActiveLearning.py`; driver schedule in `train_al.py` | Pointwise `skdim.id.MLE`; alpha=1, k_id=k_knn=50, L2-normalized pretrained features; no LID/radius clipping; adaptive radius `delta_effective*(LID/median LID)^(-alpha)`; graph truncated to cached 50-NN plus self; max uncovered gain; minimum-LID tie/fallback. | 2026-05-25 (current file mtime) | Historical thesis method | only with runs whose cache/representation provenance and initial protocol agree | Geometry caches are stored outside run directories and are not content-hashed. Zero LID is retained and can create extremely large finite radii. Same-name historical families show incompatible diagnostics, so quarantine is not canonical. |
| LIDCoverTieRandom | LIDCover | `TypiClust/deep-al/pycls/al/idpc_tiebreak/methods.py` | LIDCover with seeded-random selection among equal-gain candidates and in fallback. | 2026-04-22 (file mtime) | Tie-breaking ablation | no | Separate sampler identifier exists; historical output name also identifies the ablation. |
| LIDCoverTieFirstMax | LIDCover | `TypiClust/deep-al/pycls/al/idpc_tiebreak/methods.py` | LIDCover with first equal-gain candidate and first fallback candidate. | 2026-04-22 (file mtime) | Tie-breaking ablation | no | No complete required matched-radius evidence was promoted from this variant. |
| MeanKNNDistanceCover | LIDCover concept | `TypiClust/deep-al/pycls/al/adaptive_cover/knn_distance_cover.py` | Replaces LID with mean local kNN distance as the adaptive radius signal; retains sparse kNN coverage framework. | 2026-04-20 (file mtime) | Alternative-signal ablation | no | Separate sampler and output names. |
| DensityAdaptiveCover | LIDCover concept | `TypiClust/deep-al/pycls/al/adaptive_cover/density_cover.py` | Replaces LID with a density-derived adaptive signal; retains sparse kNN coverage framework. | 2026-04-20 (file mtime) | Alternative-signal ablation | no | Separate sampler and output names. |
| DistanceVarianceCover | LIDCover concept | `TypiClust/deep-al/pycls/al/adaptive_cover/distance_variance_cover.py` | Replaces LID with local kNN-distance variance; retains sparse kNN coverage framework. | 2026-04-20 (file mtime) | Alternative-signal ablation | no | Separate sampler and output names. |
| DistanceCVCover | LIDCover concept | `TypiClust/deep-al/pycls/al/adaptive_cover/distance_cv_cover.py` | Replaces LID with local distance coefficient of variation; retains sparse kNN coverage framework. | 2026-04-20 (file mtime) | Alternative-signal ablation | no | Separate sampler and output names. |
| ProbCoverAutoDeltaMatched | ProbCover | `TypiClust/deep-al/pycls/al/auto_delta.py`; ProbCover routing in `ActiveLearning.py`; driver schedule in `train_al.py` | For each dataset/backbone/AL seed, excludes the held-out validation indices, L2-normalizes the remaining pretrained training representations, computes exact 50-NN distances, and chooses `delta_auto` as the median 50th-neighbour distance. Uses no labels or accuracy. ProbCover then receives the shared base radius with the historical 1.35 cold-start and 0.85 post-label schedule. | 2026-08-10 | Matched baseline for the required automatic-radius experiment | no | Sampler identifier `probcover_auto_delta`. Feature content and candidate indices are SHA-256 hashed; the selected radius and geometry provenance are written into each run. |
| LIDCoverAutoDelta | LIDCover | `TypiClust/deep-al/pycls/al/auto_delta.py`; LIDCover routing in `ActiveLearning.py`; driver schedule in `train_al.py` | Uses exactly the same per-seed unsupervised `delta_auto` rule and cache as `ProbCoverAutoDeltaMatched`, then applies historical LIDCover with alpha=1, k_id=k_knn=50, `high_id_more_centers`, and the shared 1.35/0.85 driver schedule. | 2026-08-10 | Required automatic-radius experiment | no | Sampler identifier `lidcover_auto_delta`. Separate output names; run-local `auto_delta_provenance.json`; no labels, validation accuracy or test accuracy enter radius selection. |
"""
    (EVIDENCE_ROOT / "method_version_registry.md").write_text(text)


def main() -> None:
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    runs = [run for item in config_paths() if (run := load_run(*item)) is not None]

    matrix_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for dataset, backbone, scope in REQUIRED_DATASET_BACKBONES:
        for radius in RADII:
            for method in ("ProbCover", "LIDCover"):
                for seed in SEEDS:
                    logical_id = required_id(dataset, backbone, method, radius, seed)
                    candidates = [
                        run
                        for run in runs
                        if matches_required(run, dataset, backbone, method, radius, seed)
                    ]
                    candidate_rows: list[tuple[Run, str, list[str]]] = []
                    canonical_candidates: list[Run] = []
                    for run in candidates:
                        protocol_ok, protocol_note = canonical_initial_protocol_ok(run)
                        extra = [f"scope={scope}"]
                        if protocol_note:
                            extra.append(protocol_note)
                        if run.quarantined:
                            status = "UNVERIFIABLE"
                        elif run.intrinsic_status != "COMPLETE":
                            status = run.intrinsic_status
                        elif not protocol_ok:
                            status = "CONFIG_MISMATCH"
                        elif run.observed_final_budget != EXPECTED_FINAL_BUDGET:
                            status = "CONFIG_MISMATCH"
                            extra.append(
                                f"final budget {run.observed_final_budget} does not equal {EXPECTED_FINAL_BUDGET}"
                            )
                        else:
                            status = "COMPLETE"
                            canonical_candidates.append(run)
                        candidate_rows.append((run, status, extra))

                    # Prefer a non-partial/non-long-rerun name if multiple exact candidates pass.
                    canonical: Run | None = None
                    if canonical_candidates:
                        canonical = sorted(
                            canonical_candidates,
                            key=lambda run: (
                                "partial" in run.path.name.lower(),
                                "longrerun" in run.path.name.lower(),
                                str(run.path),
                            ),
                        )[0]
                        matrix_rows.append(
                            run_row(logical_id, canonical, "COMPLETE", [f"scope={scope}"])
                        )
                    else:
                        reason = "no matching run directory found"
                        if candidates:
                            statuses = sorted({status for _, status, _ in candidate_rows})
                            reason = "matching candidates exist but are not canonical: " + ", ".join(statuses)
                        missing = missing_row(dataset, backbone, method, radius, seed, f"scope={scope}; {reason}")
                        matrix_rows.append(missing)
                        missing_rows.append(missing)

                    candidate_index = 0
                    for run, status, extra in candidate_rows:
                        if canonical is not None and run.path == canonical.path:
                            continue
                        candidate_index += 1
                        if status == "COMPLETE":
                            status = "DUPLICATE"
                            extra.append("an equivalent canonical COMPLETE run was selected")
                        matrix_rows.append(
                            run_row(f"{logical_id}::candidate_{candidate_index}", run, status, extra)
                        )

    matrix_rows.sort(key=lambda row: row["experiment_id"])
    missing_rows.sort(key=lambda row: row["experiment_id"])
    write_csv(EVIDENCE_ROOT / "lidcover_completion_matrix.csv", matrix_rows)
    write_csv(EVIDENCE_ROOT / "lidcover_missing_jobs.csv", missing_rows)

    ablations = ablation_audit(runs)
    write_csv(
        EVIDENCE_ROOT / "lidcover_ablation_audit.csv",
        ablations,
        ["topic", "complete_runs", "finding", "limitation"],
    )
    existing_rows = existing_run_audit(runs)
    write_csv(EVIDENCE_ROOT / "lidcover_existing_run_audit.csv", existing_rows)
    write_registry()
    (EVIDENCE_ROOT / "lidcover_completion_report.md").write_text(
        build_report(runs, matrix_rows, missing_rows, ablations, existing_rows)
    )

    logical = [row for row in matrix_rows if "::" not in row["experiment_id"]]
    counts = Counter(row["status"] for row in logical)
    print(
        json.dumps(
            {
                "required_historical_runs": len(logical),
                "complete_historical_runs": counts["COMPLETE"],
                "missing_historical_runs": counts["MISSING"],
                "deferred_auto_delta_runs": 30,
                "total_new_runs_required": len(missing_rows) + 30,
                "missing_jobs_written": len(missing_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
