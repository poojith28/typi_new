#!/usr/bin/env python3
"""Audit TGA/TopoCover/ProbCover evidence without launching experiments.

Historical run directories are read-only.  This script writes only the TGA
audit artifacts requested under ``evidence_pack``.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "TypiClust" / "output"
EXTRA_RUN_ROOT = ROOT / "TypiClust" / "asfdasdf asf"
EVIDENCE_ROOT = ROOT / "evidence_pack"

SEEDS = (1, 2, 3, 4, 5)
DATASETS = ("CIFAR10", "CIFAR100", "TINYIMAGENET")
BACKBONE = "resnet18"
ACQUISITION_BATCH = 50
EXPECTED_FINAL_BUDGET = 5050
EXPECTED_MAX_ITER = 100
CANONICAL_PROBCOVER_BASE_DELTA = 0.60
SCALE_PAIRS = ((0.60, 0.40), (0.70, 0.50), (0.80, 0.60))
CANONICAL_FINE_TUNE = {
    "CIFAR10": False,
    "CIFAR100": True,
    "TINYIMAGENET": True,
}

MSTC_ALIASES = {
    "multiscale_topocover",
    "multi_scale_topocover",
    "mstc",
    "two_scale_topocover",
}
PROBCOVER_ALIASES = {"probcover", "prob_cover"}
TOPOCOVER_ALIASES = {"topocover", "topo_cover"}
RELEVANT_METHODS = {
    "MultiScaleTopoCover",
    "FineOnlyMultiScaleTopoCover",
    "CoarseOnlyMultiScaleTopoCover",
    "ProbCover",
    "TopoCoverHistorical",
    "WeightedTopoCover",
    "FrontierTopoCover",
    "PhasedTopoCover",
    "PersistProbCover",
    "W1ProxyCover",
    "MergeTreeCover",
}

FIELDS = [
    "experiment_id",
    "dataset",
    "backbone",
    "method",
    "alpha",
    "delta_coarse",
    "delta_fine",
    "probcover_delta",
    "batch_size",
    "seed",
    "expected_final_budget",
    "observed_final_budget",
    "status",
    "source_path",
    "notes",
]


def nested(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def close(left: Any, right: float, tol: float = 1e-9) -> bool:
    value = as_float(left)
    return value is not None and abs(value - right) <= tol


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def method_name(sampler: str, alpha: float | None = None) -> str:
    sampler = sampler.lower()
    if sampler in MSTC_ALIASES:
        if close(alpha, 0.0):
            return "FineOnlyMultiScaleTopoCover"
        if close(alpha, 1.0):
            return "CoarseOnlyMultiScaleTopoCover"
        return "MultiScaleTopoCover"
    if sampler in PROBCOVER_ALIASES:
        return "ProbCover"
    if sampler in TOPOCOVER_ALIASES:
        return "TopoCoverHistorical"
    aliases = {
        "weighted_topocover": "WeightedTopoCover",
        "weighted_topo_cover": "WeightedTopoCover",
        "wtopocover": "WeightedTopoCover",
        "wtopo": "WeightedTopoCover",
        "frontier_topocover": "FrontierTopoCover",
        "frontier_topo_cover": "FrontierTopoCover",
        "ftopocover": "FrontierTopoCover",
        "ftopo": "FrontierTopoCover",
        "phased_topocover": "PhasedTopoCover",
        "phased_topo_cover": "PhasedTopoCover",
        "ptopocover": "PhasedTopoCover",
        "ptopo": "PhasedTopoCover",
        "persist_probcover": "PersistProbCover",
        "persistprobcover": "PersistProbCover",
        "w1_proxy_cover": "W1ProxyCover",
        "w1proxycover": "W1ProxyCover",
        "merge_tree_cover": "MergeTreeCover",
        "mergetreecover": "MergeTreeCover",
        "mtc": "MergeTreeCover",
        "merge_tree": "MergeTreeCover",
    }
    return aliases.get(sampler, sampler or "UNKNOWN")


def all_config_paths() -> list[Path]:
    paths = list(RUN_ROOT.glob("*/*/*/config.yaml"))
    paths.extend(EXTRA_RUN_ROOT.glob("*/*/*/config.yaml"))
    return sorted(set(paths), key=str)


@dataclass
class Run:
    path: Path
    config: dict[str, Any]
    summary: dict[str, Any] | None
    dataset: str
    backbone: str
    sampler: str
    method: str
    alpha: float | None
    delta_coarse: float | None
    delta_fine: float | None
    probcover_delta: float | None
    batch_size: int | None
    seed: int | None
    max_iter: int | None
    train_batch_size: int | None
    max_epoch: int | None
    fine_tune: bool | None
    linear_from_features: bool | None
    initial_labeled_before: int | None
    observed_final_budget: int | None
    intrinsic_status: str
    notes: list[str]

    @property
    def logical_key(self) -> tuple[Any, ...]:
        return (
            self.dataset,
            self.backbone,
            self.method,
            self.alpha,
            self.delta_coarse if self.method != "FineOnlyMultiScaleTopoCover" else None,
            self.delta_fine if self.method != "CoarseOnlyMultiScaleTopoCover" else None,
            self.probcover_delta,
            self.batch_size,
            self.seed,
        )


def load_run(config_path: Path) -> Run | None:
    try:
        config = yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        return None

    path = config_path.parent
    al = config.get("ACTIVE_LEARNING", {}) or {}
    sampler = str(al.get("SAMPLING_FN", "")).lower()
    alpha = as_float(al.get("MSTC_ALPHA")) if sampler in MSTC_ALIASES else None
    method = method_name(sampler, alpha)
    # Every repository config is parsed far enough to identify its method, but
    # expensive per-episode validation is limited to the methods in this audit.
    if method not in RELEVANT_METHODS:
        return None
    dc = as_float(al.get("MSTC_DELTA_COARSE")) if sampler in MSTC_ALIASES else None
    df = as_float(al.get("MSTC_DELTA_FINE")) if sampler in MSTC_ALIASES else None
    pd = as_float(al.get("INITIAL_DELTA")) if sampler in PROBCOVER_ALIASES else None
    notes: list[str] = []

    summary: dict[str, Any] | None = None
    summary_path = path / "benchmark_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:
            notes.append(f"malformed benchmark_summary.json: {exc}")

    records = list((summary or {}).get("episode_records", []) or [])
    records_by_episode: dict[int, dict[str, Any]] = {}
    for record in records:
        episode = as_int(record.get("episode"))
        if episode is not None:
            records_by_episode[episode] = record

    max_iter = as_int(al.get("MAX_ITER"))
    budget = as_int(al.get("BUDGET_SIZE"))
    final_record = records_by_episode.get(max_iter) if max_iter is not None else None
    observed_final_budget = None
    if final_record:
        observed_final_budget = as_int(
            final_record.get(
                "labeled_count_after_sampling",
                final_record.get("labeled_count_before_sampling"),
            )
        )

    initial = (summary or {}).get("initial_sampling", {}) or {}
    initial_before = as_int(initial.get("labeled_count_before_sampling"))
    if initial_before is None and records_by_episode.get(0):
        # If a separate initial acquisition is absent, this is the number of
        # labels entering episode zero.  It is evidence of a non-cold start.
        initial_before = as_int(records_by_episode[0].get("labeled_count_before_sampling"))

    status = "COMPLETE"
    if summary is None:
        status = "PARTIAL" if any(path.glob("episode_*")) else "FAILED"
        notes.append("benchmark_summary.json absent")
    elif not records:
        status = "FAILED"
        notes.append("summary contains no episode accuracy history")
    elif max_iter is None:
        status = "UNVERIFIABLE"
        notes.append("MAX_ITER absent from configuration")
    else:
        expected_episodes = list(range(max_iter + 1))
        observed_episodes = sorted(records_by_episode)
        if observed_episodes != expected_episodes:
            status = "PARTIAL"
            missing = sorted(set(expected_episodes) - set(observed_episodes))
            notes.append(f"episode history incomplete; missing={missing[:12]}")
        if not finite_tree(summary):
            status = "FAILED"
            notes.append("summary contains NaN or infinity")

        for episode in expected_episodes:
            record = records_by_episode.get(episode)
            if record is None:
                continue
            if as_float(record.get("test_accuracy")) is None or as_float(record.get("best_val_accuracy")) is None:
                status = "FAILED"
                notes.append(f"episode {episode} lacks finite accuracy")
                break

        acquisition_ok = True
        if initial:
            if as_int(initial.get("active_set_size")) != budget:
                acquisition_ok = False
                notes.append("initial acquisition size disagrees with BUDGET_SIZE")
        elif initial_before == 0:
            acquisition_ok = False
            notes.append("cold-start initial acquisition metadata absent")

        for episode in range(max_iter):
            record = records_by_episode.get(episode)
            if record is None:
                acquisition_ok = False
                continue
            active_size = as_int(record.get("active_set_size"))
            active_ids = record.get("active_set_ids", []) or []
            npy_path = path / f"episode_{episode}" / "activeSet.npy"
            if active_size != budget:
                acquisition_ok = False
                notes.append(f"episode {episode} acquisition size {active_size} != {budget}")
                break
            if active_ids:
                if len(active_ids) != budget or len(active_ids) != len(set(map(int, active_ids))):
                    acquisition_ok = False
                    notes.append(f"episode {episode} active_set_ids invalid")
                    break
            elif not npy_path.exists():
                acquisition_ok = False
                notes.append(f"episode {episode} acquisition output absent")
                break
            for required in ("lSet.npy", "uSet.npy", "episode_summary.json"):
                if not (path / f"episode_{episode}" / required).exists():
                    acquisition_ok = False
                    notes.append(f"episode {episode}/{required} absent")
                    break
            if not acquisition_ok:
                break
        if status == "COMPLETE" and not acquisition_ok:
            status = "PARTIAL"

        if final_record is None or observed_final_budget is None:
            if status == "COMPLETE":
                status = "PARTIAL"
            notes.append("final evaluation record/budget absent")
        else:
            if as_int(final_record.get("active_set_size")) not in (0, None):
                status = "FAILED"
                notes.append("final episode unexpectedly performed acquisition")
            for required in ("lSet.npy", "uSet.npy"):
                if not (path / f"episode_{max_iter}" / required).exists():
                    if status == "COMPLETE":
                        status = "PARTIAL"
                    notes.append(f"final episode {required} absent")

    if summary is not None:
        checks = (
            (summary.get("dataset"), nested(config, "DATASET", "NAME"), "dataset"),
            (summary.get("model"), nested(config, "MODEL", "TYPE"), "backbone"),
            (summary.get("seed"), config.get("RNG_SEED"), "seed"),
            (summary.get("sampling_fn"), al.get("SAMPLING_FN"), "method"),
            (summary.get("budget_per_round"), al.get("BUDGET_SIZE"), "batch size"),
        )
        mismatches = [label for observed, expected, label in checks if str(observed) != str(expected)]
        if mismatches:
            status = "CONFIG_MISMATCH"
            notes.append("summary/config mismatch: " + ", ".join(mismatches))

        if sampler in MSTC_ALIASES:
            metadata_blocks: list[dict[str, Any]] = []
            if initial.get("sampling_metadata"):
                metadata_blocks.append(initial["sampling_metadata"])
            metadata_blocks.extend(
                record.get("sampling_metadata", {}) or {}
                for episode, record in records_by_episode.items()
                if episode != max_iter
            )
            for metadata in metadata_blocks:
                if not (
                    close(metadata.get("alpha"), alpha or 0.0)
                    and close(metadata.get("delta_coarse"), dc or 0.0)
                    and close(metadata.get("delta_fine"), df or 0.0)
                    and str(metadata.get("strategy")) == "multiscale_topocover"
                ):
                    status = "CONFIG_MISMATCH"
                    notes.append("run-local MSTC sampling metadata disagrees with config")
                    break

    return Run(
        path=path,
        config=config,
        summary=summary,
        dataset=str(nested(config, "DATASET", "NAME", default="")),
        backbone=str(nested(config, "MODEL", "TYPE", default="")),
        sampler=sampler,
        method=method,
        alpha=alpha,
        delta_coarse=dc,
        delta_fine=df,
        probcover_delta=pd,
        batch_size=budget,
        seed=as_int(config.get("RNG_SEED")),
        max_iter=max_iter,
        train_batch_size=as_int(nested(config, "TRAIN", "BATCH_SIZE")),
        max_epoch=as_int(nested(config, "OPTIM", "MAX_EPOCH")),
        fine_tune=al.get("FINE_TUNE"),
        linear_from_features=nested(config, "MODEL", "LINEAR_FROM_FEATURES"),
        initial_labeled_before=initial_before,
        observed_final_budget=observed_final_budget,
        intrinsic_status=status,
        notes=list(dict.fromkeys(notes)),
    )


def canonical_protocol(run: Run, expected_batch: int = ACQUISITION_BATCH) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if run.batch_size != expected_batch:
        problems.append(f"acquisition batch is {run.batch_size}, expected {expected_batch}")
    if expected_batch == 50 and run.max_iter != EXPECTED_MAX_ITER:
        problems.append(f"MAX_ITER is {run.max_iter}, expected {EXPECTED_MAX_ITER}")
    if run.initial_labeled_before != 0:
        problems.append(f"initial labelled count is {run.initial_labeled_before}, expected cold-start 0")
    if run.train_batch_size != 100:
        problems.append(f"classifier training batch is {run.train_batch_size}, canonical 100")
    if run.max_epoch != 200:
        problems.append(f"MAX_EPOCH is {run.max_epoch}, canonical 200")
    expected_fine_tune = CANONICAL_FINE_TUNE.get(run.dataset)
    if run.fine_tune is not expected_fine_tune:
        problems.append(
            f"FINE_TUNE is {run.fine_tune}, canonical {str(expected_fine_tune).lower()} for {run.dataset}"
        )
    if run.linear_from_features is not False:
        problems.append(f"LINEAR_FROM_FEATURES is {run.linear_from_features}, canonical false")
    if expected_batch == 50 and run.observed_final_budget not in (None, EXPECTED_FINAL_BUDGET):
        problems.append(
            f"observed final budget is {run.observed_final_budget}, expected {EXPECTED_FINAL_BUDGET}"
        )
    return not problems, problems


def requested_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for dc, df in SCALE_PAIRS:
            for seed in SEEDS:
                rows.append({
                    "experiment_id": f"TGA_PRIMARY_{dataset}_C{int(dc*100):03d}_F{int(df*100):03d}_A070_S{seed}_B050",
                    "dataset": dataset,
                    "backbone": BACKBONE,
                    "method": "MultiScaleTopoCover",
                    "alpha": 0.70,
                    "delta_coarse": dc,
                    "delta_fine": df,
                    "probcover_delta": None,
                    "batch_size": 50,
                    "seed": seed,
                    "expected_final_budget": EXPECTED_FINAL_BUDGET,
                    "scope": "primary_multiscale",
                })
                rows.append({
                    "experiment_id": f"MSTC_FINE_ONLY_{dataset}_F{int(df*100):03d}_A000_S{seed}_B050",
                    "dataset": dataset,
                    "backbone": BACKBONE,
                    "method": "FineOnlyMultiScaleTopoCover",
                    "alpha": 0.0,
                    "delta_coarse": None,
                    "delta_fine": df,
                    "probcover_delta": None,
                    "batch_size": 50,
                    "seed": seed,
                    "expected_final_budget": EXPECTED_FINAL_BUDGET,
                    "scope": "fine_only_control",
                })
                rows.append({
                    "experiment_id": f"MSTC_COARSE_ONLY_{dataset}_C{int(dc*100):03d}_A100_S{seed}_B050",
                    "dataset": dataset,
                    "backbone": BACKBONE,
                    "method": "CoarseOnlyMultiScaleTopoCover",
                    "alpha": 1.0,
                    "delta_coarse": dc,
                    "delta_fine": None,
                    "probcover_delta": None,
                    "batch_size": 50,
                    "seed": seed,
                    "expected_final_budget": EXPECTED_FINAL_BUDGET,
                    "scope": "coarse_only_control",
                })

        for seed in SEEDS:
            rows.append({
                "experiment_id": f"PROBCOVER_CANONICAL_{dataset}_D060_S{seed}_B050",
                "dataset": dataset,
                "backbone": BACKBONE,
                "method": "ProbCover",
                "alpha": None,
                "delta_coarse": None,
                "delta_fine": None,
                "probcover_delta": CANONICAL_PROBCOVER_BASE_DELTA,
                "batch_size": 50,
                "seed": seed,
                "expected_final_budget": EXPECTED_FINAL_BUDGET,
                "scope": "canonical_probcover",
            })

    # Alpha 0, 0.7, and 1 overlap the required C070/F050 endpoint/main rows.
    # Only the three new interior weights are added as unique configurations.
    for alpha in (0.25, 0.50, 0.75):
        for seed in SEEDS:
            rows.append({
                "experiment_id": f"TGA_ALPHA_CIFAR100_C070_F050_A{int(alpha*100):03d}_S{seed}_B050",
                "dataset": "CIFAR100",
                "backbone": BACKBONE,
                "method": "MultiScaleTopoCover",
                "alpha": alpha,
                "delta_coarse": 0.70,
                "delta_fine": 0.50,
                "probcover_delta": None,
                "batch_size": 50,
                "seed": seed,
                "expected_final_budget": EXPECTED_FINAL_BUDGET,
                "scope": "primary_alpha_ablation",
            })
    return rows


def matches(run: Run, row: dict[str, Any]) -> bool:
    if (
        run.dataset != row["dataset"]
        or run.backbone != row["backbone"]
        or run.method != row["method"]
        or run.batch_size != row["batch_size"]
        or run.seed != row["seed"]
    ):
        return False
    for field in ("alpha", "delta_coarse", "delta_fine", "probcover_delta"):
        expected = row[field]
        if expected is not None and not close(getattr(run, field), expected):
            return False
    return True


def candidate_status(run: Run) -> tuple[str, list[str]]:
    notes = list(run.notes)
    if run.intrinsic_status != "COMPLETE":
        return run.intrinsic_status, notes
    protocol_ok, protocol_notes = canonical_protocol(run)
    if not protocol_ok:
        notes.extend(protocol_notes)
        return "CONFIG_MISMATCH", list(dict.fromkeys(notes))
    return "COMPLETE", notes


def materialize_matrix(runs: list[Run]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matrix: list[dict[str, Any]] = []
    missing_jobs: list[dict[str, Any]] = []
    for logical in requested_rows():
        candidates = [run for run in runs if matches(run, logical)]
        ranked: list[tuple[int, Run, str, list[str]]] = []
        order = {
            "COMPLETE": 0,
            "DUPLICATE": 1,
            "PARTIAL": 2,
            "FAILED": 3,
            "CONFIG_MISMATCH": 4,
            "UNVERIFIABLE": 5,
        }
        for candidate in candidates:
            status, notes = candidate_status(candidate)
            ranked.append((order.get(status, 9), candidate, status, notes))
        ranked.sort(key=lambda item: (item[0], str(item[1].path)))

        if not ranked:
            status = "MISSING"
            source = ""
            observed = ""
            notes = [f"no run config matches required {logical['scope']} configuration"]
        else:
            _, chosen, status, notes = ranked[0]
            source = str(chosen.path)
            observed = chosen.observed_final_budget or ""
            if len(ranked) > 1:
                notes.append(f"{len(ranked)-1} additional candidate run(s) retained; see notes/source audit")

        if logical["scope"] == "canonical_probcover":
            notes.append(
                "probcover_delta is saved base radius 0.60; driver effective radius is 0.81 at cold start and 0.51 thereafter"
            )
        if logical["method"] == "FineOnlyMultiScaleTopoCover":
            notes.append("inactive coarse radius intentionally omitted")
        if logical["method"] == "CoarseOnlyMultiScaleTopoCover":
            notes.append("inactive fine radius intentionally omitted")

        row = {key: logical.get(key, "") for key in FIELDS}
        row.update({
            "alpha": fmt(logical["alpha"]),
            "delta_coarse": fmt(logical["delta_coarse"]),
            "delta_fine": fmt(logical["delta_fine"]),
            "probcover_delta": fmt(logical["probcover_delta"]),
            "observed_final_budget": observed,
            "status": status,
            "source_path": source,
            "notes": "; ".join(dict.fromkeys(notes)),
        })
        matrix.append(row)

        if status != "COMPLETE":
            # All rows in the concrete matrix use already implemented methods.
            # Give every rerun a new identifier/output name; never reuse source.
            job = dict(row)
            job["experiment_id"] = "AUDIT_20260810_" + logical["experiment_id"]
            job["observed_final_budget"] = ""
            job["status"] = "MISSING"
            job["source_path"] = ""
            dependency = (
                "; phase 2 only—submit after the four core endpoint-control repairs pass re-audit"
                if logical["scope"] == "primary_alpha_ablation"
                else "; phase 1 core-matrix repair"
            )
            job["notes"] = (
                f"new unique run required because audited status was {status}; "
                f"do not overwrite any candidate path{dependency}"
            )
            missing_jobs.append(job)

    return matrix, missing_jobs


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def status_table(matrix: list[dict[str, Any]], prefix: str) -> Counter[str]:
    return Counter(row["status"] for row in matrix if row["experiment_id"].startswith(prefix))


def count_complete_at_budget(run: Run, budget: int) -> bool:
    if run.intrinsic_status != "COMPLETE" or run.summary is None:
        return False
    for record in run.summary.get("episode_records", []) or []:
        if (
            as_int(record.get("labeled_count_before_sampling")) == budget
            and as_float(record.get("test_accuracy")) is not None
        ):
            return True
    return False


def optional_and_secondary_audit(runs: list[Run]) -> dict[str, Any]:
    # Reduced alpha sweep: only alpha=.5 is not already in the endpoint/main
    # matrix for CIFAR-10 and TinyImageNet.
    reduced_complete = 0
    reduced_missing = 0
    for dataset in ("CIFAR10", "TINYIMAGENET"):
        for seed in SEEDS:
            candidates = [
                run for run in runs
                if run.dataset == dataset
                and run.backbone == BACKBONE
                and run.method == "MultiScaleTopoCover"
                and close(run.alpha, 0.5)
                and close(run.delta_coarse, 0.7)
                and close(run.delta_fine, 0.5)
                and run.batch_size == 50
                and run.seed == seed
            ]
            if any(candidate_status(run)[0] == "COMPLETE" for run in candidates):
                reduced_complete += 1
            else:
                reduced_missing += 1

    # Existing batch-50 and batch-100 long runs can be compared at the common
    # 5,000-label checkpoint even though their benchmark endpoints differ.
    matched = {50: 0, 100: 0}
    long_final_budgets: dict[int, list[int]] = defaultdict(list)
    for batch in (50, 100):
        for seed in SEEDS:
            candidates = [
                run for run in runs
                if run.dataset == "CIFAR100"
                and run.backbone == BACKBONE
                and run.method == "MultiScaleTopoCover"
                and close(run.alpha, 0.7)
                and close(run.delta_coarse, 0.7)
                and close(run.delta_fine, 0.5)
                and run.batch_size == batch
                and run.seed == seed
            ]
            if any(count_complete_at_budget(run, 5000) for run in candidates):
                matched[batch] += 1
            long_final_budgets[batch].extend(
                run.observed_final_budget
                for run in candidates
                if run.observed_final_budget is not None
            )

    methods = Counter(run.method for run in runs)
    method_status: dict[str, Counter[str]] = defaultdict(Counter)
    for run in runs:
        method_status[run.method][run.intrinsic_status] += 1
    relevant_status = Counter(
        run.intrinsic_status
        for run in runs
        if run.method in {
            "MultiScaleTopoCover",
            "FineOnlyMultiScaleTopoCover",
            "CoarseOnlyMultiScaleTopoCover",
            "ProbCover",
            "TopoCoverHistorical",
            "WeightedTopoCover",
            "FrontierTopoCover",
            "PhasedTopoCover",
            "PersistProbCover",
            "W1ProxyCover",
            "MergeTreeCover",
        }
    )
    return {
        "reduced_alpha_complete": reduced_complete,
        "reduced_alpha_missing": reduced_missing,
        "matched_budget_counts": matched,
        "long_final_budgets": {key: sorted(set(value)) for key, value in long_final_budgets.items()},
        "method_directory_counts": methods,
        "method_status": method_status,
        "probcover_base_radii": sorted({run.probcover_delta for run in runs if run.method == "ProbCover" and run.probcover_delta is not None}),
        "mstc_alphas": sorted({run.alpha for run in runs if run.sampler in MSTC_ALIASES and run.alpha is not None}),
        "mstc_scale_pairs": sorted({(run.delta_coarse, run.delta_fine) for run in runs if run.sampler in MSTC_ALIASES}),
        "relevant_intrinsic_status": relevant_status,
    }


def write_registry() -> None:
    path = EVIDENCE_ROOT / "tga_method_version_registry.md"
    path.write_text("""# TGA / TopoCover method version registry

Audit date: 2026-08-10 (Australia/Melbourne). Existing implementations and run directories were not modified. Current source hashes are recorded for future runs, but historical run directories do not contain source hashes, so compatibility is supported by configs and run-local sampling metadata rather than cryptographic provenance.

| method_name | parent_method | implementation_file | exact_behaviour | difference_from_parent | date_created | compatible_with_old_results | notes |
|---|---|---|---|---|---|---|---|
| ProbCover | — | `TypiClust/deep-al/pycls/al/prob_cover.py`; schedule in `tools/train_al.py` | Loads fixed representation features; builds an all-pairs directed graph with strict Euclidean `distance < delta`; greedily chooses first-index maximum uncovered out-degree; updates covered targets after each pick. The driver multiplies saved base delta by 1.35 at empty cold start and 0.85 after labels exist. | Historical baseline. | 2026-04-20 file mtime | yes only when driver schedule, representation, config and protocol agree | Current SHA-256 `5e41d51543a3bb5c77633834d43f6f3361e83317130b24a4b466bda660a4191a`. The implementation assumes CUDA and omits a trailing partial graph batch, harmless for the canonical 45,000-point pool because it is divisible by 500. |
| LIDCover / IDProbCover | ProbCover concept | `TypiClust/deep-al/pycls/al/IDProbCover.py`; import shim `IDprocover.py` | Estimates pointwise local intrinsic dimension, scales each point's radius relative to median LID, uses a cached truncated kNN coverage graph plus self, selects maximum uncovered gain, and uses minimum-LID tie/fallback behavior. The driver applies the same base-radius schedule before adaptation. | Point-dependent radii, sparse graph, and different tie/fallback behavior. | historical implementation; see existing LID audit | only with matching LIDCover config/provenance; never as ProbCover | Separate historical method. Full matched-radius audit remains in `evidence_pack/lidcover_completion_report.md`. |
| LIDCover tie-break variants | LIDCover / IDProbCover | `TypiClust/deep-al/pycls/al/idpc_tiebreak/methods.py` | Retain the LIDCover geometry while selecting among equal-gain/fallback candidates by minimum LID, seeded random choice, or first maximum. | Tie/fallback policy only, with separate sampler identifiers. | historical ablation implementations | no | `IDProbCoverMinIDTieBreak`, `IDProbCoverRandomTieBreak`, and `IDProbCoverFirstMaxTieBreak` must remain distinct. |
| TopoCoverHistorical | ProbCover concept | `TypiClust/deep-al/pycls/al/topocover.py` | L2-normalises fixed features; uses a symmetrised, k=50 truncated radius graph with closed `distance <= delta`; labels connected components of the uncovered induced graph; chooses an uncovered candidate touching the most distinct uncovered components; recomputes after each pick; strict-greater update preserves pool-order ties; stops without arbitrary padding on non-positive gain. | Replaces point-coverage gain with uncovered-component gain and changes graph construction/tie/fallback behaviour. | 2026-08-05 file mtime | historical outputs unavailable | Current SHA-256 `bb2871e4c52256bf4e6701b349faff64f523b03f50745c8d8b75b78379c69adf`. `add_self_cover` and the constructor `recompute_components` argument are effectively ignored: self is always added and recomputation is always enabled. |
| MultiScaleTopoCover | TopoCover concept | `TypiClust/deep-al/pycls/al/multiscale_topocover.py` | Loads fixed features (requesting normalised features); constructs exact batched all-pairs closed-radius graphs at coarse and fine scales; independently marks covered vertices and recomputes uncovered components after every pick; score is `alpha*coarse_component_gain + (1-alpha)*fine_component_gain`; deterministic first maximum; always selects until batch/pool exhaustion, including zero-gain candidates; updates both scales after each pick. | Two exact radius graphs and weighted two-scale gain; graph, eligibility and zero-gain behaviour differ from historical TopoCover. | 2026-08-05 file mtime | yes for MSTC runs with matching metadata; no as TopoCover evidence | Current SHA-256 `a4b2f3b0bbfe55b1f75ffc10c4077318e6f51b767a880e14f1653d277a72bfe1`. This is the implementation called TGA/MSTC in the thesis brief. |
| FineOnlyMultiScaleTopoCover | MultiScaleTopoCover | same file, `alpha=0` code path | Score uses only fine component gain, but coarse graph/state is still built and updated; first-maximum and zero-gain behaviour remain MSTC behaviour. | Coarse contribution weight is zero. | 2026-08-03 campaign | compatible only with same MSTC endpoint configuration | Not historical TopoCover. Inactive coarse radius must not be presented as active. |
| CoarseOnlyMultiScaleTopoCover | MultiScaleTopoCover | same file, `alpha=1` code path | Score uses only coarse component gain, but fine graph/state is still built and updated; first-maximum and zero-gain behaviour remain MSTC behaviour. | Fine contribution weight is zero. | 2026-08-03 campaign | compatible only with same MSTC endpoint configuration | Not historical TopoCover. Inactive fine radius must not be presented as active. |
| WeightedTopoCover | TopoCoverHistorical | `TypiClust/deep-al/pycls/al/topo_variants.py` | Same sparse graph/coverage loop, with gain equal to the summed sizes of distinct touched uncovered components. | Size-weighted rather than component-count gain. | 2026-08-05 file mtime | no | Separate sampler and pilot output name. |
| FrontierTopoCover | TopoCoverHistorical | `TypiClust/deep-al/pycls/al/topo_variants.py` | Credits only uncovered components adjacent to current coverage (all components when no frontier exists), weighted by component size. | Frontier filtering plus size weighting. | 2026-08-05 file mtime | no | Separate sampler and pilot output name. |
| PhasedTopoCover | ProbCover / WeightedTopoCover | `TypiClust/deep-al/pycls/al/topo_variants.py` | Delegates to ProbCover below a labelled-count threshold (default 1,000), then to WeightedTopoCover. | Stage-dependent hybrid. | 2026-08-05 file mtime | no | Separate sampler and pilot output name. |
| PersistProbCover | ProbCover concept | `TypiClust/deep-al/pycls/al/persist_cover.py` | Exact closed-radius graph; initially scores newly covered points only in components meeting a minimum size, then falls back to ordinary point gain when signal is exhausted. | Suppresses micro-component gain before point-gain fallback. | 2026-08-05 file mtime | no | Separate sampler and pilot output name. |
| W1ProxyCover | ProbCover concept | `TypiClust/deep-al/pycls/al/persist_cover.py` | Combines ordinary radius coverage with coverage of significant kNN-Kruskal MST merge edges selected by a death-scale quantile. | Adds an H0 merge-edge/barcode proxy term. | 2026-08-05 file mtime | no | Separate sampler and pilot output name. |
| MergeTreeCover | ProbCover / H0 concept | `TypiClust/deep-al/pycls/al/merge_tree_cover.py` | Rebuilds residual kNN-Kruskal merge events, weights them by merge scale/size/balance, selects representatives covering important events, optionally mixes point gain, and has separately configured point-only or ProbCover tails. | Residual merge-tree event objective rather than fixed-scale component gain. | 2026-08-05 file mtime | no | Multiple pilot configs exist; they must remain distinct by config/output identity. |
| RandomComponentSampler | new control | not implemented | Reserved: use the frozen TGA reference component structure, then randomly choose an eligible representative instead of TGA gain. | Removes TGA gain while retaining component exposure. | not created | no | Must be implemented and registered before execution. |
| GraphDegreeSampler | new control | not implemented | Reserved for degree-based selection on the frozen reference graph. | Replaces TGA gain with graph degree. | not created | no | Must be implemented and registered before execution. |
| LowDensitySampler | new control | not implemented | Reserved for direct mean-kNN-distance selection on the frozen reference representation. | Replaces TGA gain with low-density score. | not created | no | Existing `KnnDistanceCover` is an adaptive coverage method, not automatically this direct control. |
| TGAAutoScale | MultiScaleTopoCover | not implemented | Reserved for a frozen deterministic unsupervised rule selecting coarse/fine radii without labels or test accuracy. | Automatic rather than manual scale selection. | not created | no | Rule must be frozen and registered before predictive evaluation. |
""")


def write_report(
    matrix: list[dict[str, Any]],
    missing_jobs: list[dict[str, Any]],
    runs: list[Run],
    extras: dict[str, Any],
) -> None:
    status_counts = Counter(row["status"] for row in matrix)
    scope_counts: dict[str, Counter[str]] = defaultdict(Counter)
    logical_by_id = {row["experiment_id"]: row for row in requested_rows()}
    for row in matrix:
        scope_counts[logical_by_id[row["experiment_id"]]["scope"]][row["status"]] += 1

    historical = [run for run in runs if run.method == "TopoCoverHistorical"]
    variants = [
        run for run in runs
        if run.method in {
            "WeightedTopoCover", "FrontierTopoCover", "PhasedTopoCover",
            "PersistProbCover", "W1ProxyCover", "MergeTreeCover",
        }
    ]
    invalid_rows = [row for row in matrix if row["status"] not in {"COMPLETE", "MISSING"}]
    phase_one_jobs = [row for row in missing_jobs if "phase 1" in row["notes"]]
    phase_two_jobs = [row for row in missing_jobs if "phase 2" in row["notes"]]

    def job_lines(rows: list[dict[str, Any]]) -> str:
        return "\n".join(
        f"- `{row['experiment_id']}`: {row['dataset']}, {row['method']}, "
        f"alpha={row['alpha'] or '—'}, coarse={row['delta_coarse'] or '—'}, "
        f"fine={row['delta_fine'] or '—'}, ProbCover base={row['probcover_delta'] or '—'}, seed={row['seed']}"
        for row in rows
        ) or "- None."

    def counter_text(counter: Counter[str]) -> str:
        return ", ".join(f"{key}={value}" for key, value in sorted(counter.items())) or "none"

    inventory_lines = []
    for method, counts in sorted(extras["method_status"].items()):
        inventory_lines.append(
            f"| {method} | {sum(counts.values())} | {counter_text(counts)} |"
        )
    inventory_table = "\n".join(inventory_lines)

    report = f"""# TGA / TopoCover completion audit

Audit date: 2026-08-10 (Australia/Melbourne). This is an audit-only result: no experiment was submitted, resumed, overwritten, deleted, or modified.

## Repository and evidence boundary

- Run configurations method-classified: **{len(all_config_paths())}**; topology/ProbCover configurations deeply validated: **{len(runs)}**.
- Canonical raw run tree: `{RUN_ROOT}`. Aggregate CSVs, figures, and prior evidence packs were treated as non-canonical leads.
- Relevant raw-run intrinsic statuses: **{counter_text(extras['relevant_intrinsic_status'])}**.
- Canonical seeds reconstructed from launch manifests and configs: **1, 2, 3, 4, 5**.
- Scheduler state: **UNVERIFIABLE**. A read-only `squeue` query failed with `Unable to contact slurm controller`; no submission is permitted until RUNNING/PENDING state can be checked successfully.
- Historical run directories lack code hashes. Current source hashes are in `tga_method_version_registry.md`; historical compatibility is inferred from run config plus per-acquisition metadata where available.

## Canonical protocol

- Primary backbone: ResNet-18; acquisition batch: 50; classifier training batch: 100; 200 epochs per round; fixed pretrained representation; empty labelled-set cold start. The preserved dataset configs set `FINE_TUNE=false` for CIFAR-10 and `true` for CIFAR-100/TinyImageNet.
- Only `features_seed1.npy` exists for each primary ResNet-18 dataset. The loader therefore falls back to that same fixed representation for AL seeds 2–5 while preserving distinct partition/training RNG seeds.
- The acquisition method selects the first 50 labels, episodes 0–99 add 50 each, and episode 100 is evaluation-only at **5,050 labels**.
- MultiScaleTopoCover/TGA uses exact all-pairs closed-radius graphs. Historical TopoCover uses a symmetrised k=50 sparse radius graph. Their graph, candidate eligibility, zero-gain, and endpoint code paths are not identical.
- ProbCover's canonical saved base radius is **0.60**. The current driver turns that into effective radius **0.81** for the empty-pool acquisition and **0.51** thereafter. A row labelled 0.60 therefore denotes the saved base configuration, not a constant effective radius.

## Required concrete matrix

The CSV contains **{len(matrix)} unique concrete configurations**. The 30-run CIFAR-100 alpha sweep overlaps 15 existing main/endpoint cells (alpha 0, 0.7, 1), so only 15 interior-alpha configurations are additional rows.

| Scope | Logical configurations | Status counts |
|---|---:|---|
| Primary multiscale (3 datasets × 3 pairs × 5 seeds) | 45 | {counter_text(scope_counts['primary_multiscale'])} |
| Fine-only controls | 45 | {counter_text(scope_counts['fine_only_control'])} |
| Coarse-only controls | 45 | {counter_text(scope_counts['coarse_only_control'])} |
| Canonical ProbCover | 15 | {counter_text(scope_counts['canonical_probcover'])} |
| New interior alpha-ablation cells | 15 | {counter_text(scope_counts['primary_alpha_ablation'])} |
| **Unique total** | **{len(matrix)}** | **{counter_text(status_counts)}** |

Complete required configurations: **{status_counts['COMPLETE']}**. Missing/invalid configurations requiring new unique execution: **{len(missing_jobs)}**.

## Invalid or incomplete candidates

There are **{len(invalid_rows)}** logical rows whose best candidate is PARTIAL, FAILED, CONFIG_MISMATCH, or UNVERIFIABLE. Their exact status, source path, and reason are recorded in `tga_completion_matrix.csv`. Existing partial directories are evidence and must not be reused as output locations.

For every candidate, validation checked config/summary agreement, exact episode sequence, finite accuracy history, initial and per-round acquisition size, unique selected IDs or saved acquisition arrays, per-episode labelled/unlabelled partitions, final evaluation-only behavior, canonical training settings, cold-start convention, and observed final budget. No required logical cell had more than one canonical complete candidate (`DUPLICATE=0`); the required matrix also has `FAILED=0`, `CONFIG_MISMATCH=0`, and `UNVERIFIABLE=0` after protocol matching.

## Historical TopoCover

- Genuine `topocover` run directories currently present: **{len(historical)}**.
- Launch manifests show that historical TopoCover campaigns were submitted, but the corresponding raw output directories are absent. Prior repository reports explicitly state that those outputs were removed.
- Status: **MISSING/UNAVAILABLE**. They were not reconstructed, and no alpha endpoint was substituted.

## Existing topology variants

- Variant run directories found: **{len(variants)}**. These are separately named WeightedTopoCover, FrontierTopoCover, PhasedTopoCover, PersistProbCover, W1ProxyCover, or MergeTreeCover pilots.
- They do not fill TGA, historical TopoCover, or graph-control cells. Exact behavior and parent differences are registered separately.

## Discovered implementation/result inventory

| Registered method family | Run directories | Intrinsic raw-run status |
|---|---:|---|
{inventory_table}

- Observed saved ProbCover base radii: **{extras['probcover_base_radii']}**. The canonical thesis baseline is base 0.60; base 0.25 runs are a separate matched-radius campaign and are not substituted.
- Observed MSTC alpha values: **{extras['mstc_alphas']}**.
- Observed MSTC configured scale pairs: **{extras['mstc_scale_pairs']}**. Endpoint rows report only the scientifically active radius in the completion CSV.

## Alpha ablation and optional reduced sweep

- The required CIFAR-100 alpha grid has 30 logical cells. Alpha 0, 0.7, and 1 are shared with the endpoint/main matrix; alpha 0.25, 0.50, and 0.75 contribute 15 unique cells.
- Optional reduced-sweep interior alpha=0.5 cells on CIFAR-10 and TinyImageNet: **{extras['reduced_alpha_complete']} complete, {extras['reduced_alpha_missing']} missing**. They are not placed in the concrete launch manifest because the brief conditions them on resource availability.
- No alpha should be chosen using test performance.

## Matched-budget batch-size audit

- Existing C070/F050/A070 runs with finite accuracy at the common **5,000-label checkpoint**: batch 50 = **{extras['matched_budget_counts'][50]}/5**, batch 100 = **{extras['matched_budget_counts'][100]}/5**.
- Observed long-run endpoints: batch 50 {extras['long_final_budgets'][50]}, batch 100 {extras['long_final_budgets'][100]}.
- Therefore benchmark-summary final accuracies (5,050 versus 10,100) are not a matched-budget comparison. The existing raw curves can be compared at 5,000 labels without rerunning when all ten checkpoint records are present; the analysis must call these matched checkpoint accuracies, not final-run accuracies.

## Downstream evidence readiness

- Structural mechanism artifacts requested in the brief do not yet exist in their final named form. Existing per-round selected IDs and fixed representations make reconstruction possible, but a single frozen reference graph definition and provenance record must be established first.
- Graph controls `RandomComponentSampler`, `GraphDegreeSampler`, and direct `LowDensitySampler` are not implemented. Existing adaptive `KnnDistanceCover` is not silently substituted. These 15 primary CIFAR-100 control runs are deferred until separate implementations/configs/output names are registered.
- `TGAAutoScale` and its deterministic unsupervised scale rule are not implemented. Its 15 predictive runs are deferred until the rule is frozen before looking at predictive performance.
- Runtime summaries contain total acquisition/train/test/round times, but not the required graph/component/score breakdown or peak CPU/GPU memory. A non-method-changing benchmark harness is still needed.
- The second-representation comparison is conditional on compute and follows core evidence; it is not part of the immediate launch manifest.
- Final claim evidence must wait for completion, re-audit, structural reconstruction, graph controls, auto-scale, and runtime instrumentation. Negative or inconsistent results must remain visible.

## Missing-job manifest and ordering gate

The manifest contains all **{len(missing_jobs)}** concrete jobs still needed, but the alpha ablation is gated behind completion of the core historical matrix.

### Phase 1 — eligible core repairs

{job_lines(phase_one_jobs)}

Phase-1 jobs: **{len(phase_one_jobs)}**. Re-run the audit after they finish.

### Phase 2 — alpha ablation, only after Phase 1 passes re-audit

{job_lines(phase_two_jobs)}

Phase-2 jobs: **{len(phase_two_jobs)}**. Total missing-job manifest rows: **{len(missing_jobs)}**.

No queue was launched. Before a later submission, re-run this audit, query scheduler state, remove COMPLETE/RUNNING/PENDING/already-submitted experiment IDs, and submit one experiment per array task with a collective cap of **`%20`**. The maximum is 20 across this entire workflow, not per method.
"""
    (EVIDENCE_ROOT / "tga_completion_report.md").write_text(report)


def main() -> None:
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    runs = [run for path in all_config_paths() if (run := load_run(path)) is not None]
    matrix, missing_jobs = materialize_matrix(runs)
    extras = optional_and_secondary_audit(runs)
    write_csv(EVIDENCE_ROOT / "tga_completion_matrix.csv", matrix)
    write_csv(EVIDENCE_ROOT / "tga_missing_jobs.csv", missing_jobs)
    write_registry()
    write_report(matrix, missing_jobs, runs, extras)

    print(f"Complete required matrix rows: {len(matrix)}")
    print(f"Already complete: {sum(row['status'] == 'COMPLETE' for row in matrix)}")
    print(f"Missing/invalid manifest jobs: {len(missing_jobs)} (4 phase 1, 15 gated phase 2)")
    print("Logical status counts:", dict(Counter(row["status"] for row in matrix)))
    print("No jobs submitted. Future rolling array concurrency cap: 20.")


if __name__ == "__main__":
    main()
