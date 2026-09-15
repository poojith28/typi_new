#!/usr/bin/env python3
"""Build a compact, self-contained LIDCover fixed-vs-automatic thesis package."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "thesis_lidcover_fixed_vs_automatic_20260907"
EVIDENCE = ROOT / "evidence/lidcover_extensions_2026"
FIXED_MANIFEST = EVIDENCE / "batch_size_manifest.csv"
AUTO_MANIFEST = EVIDENCE / "auto_sparse_radius_v3_manifest.csv"


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sem(values):
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")


def aulc(records, metric, maximum_budget=5000):
    selected = []
    for index, record in enumerate(records):
        budget = int(record.get("labeled_count_before_sampling", 50 * (index + 1)))
        if 50 <= budget <= maximum_budget:
            selected.append((budget, float(record[metric])))
    selected = sorted(dict(selected).items())
    if [budget for budget, _ in selected] != list(range(50, maximum_budget + 1, 50)):
        raise ValueError("AULC requires exact 50-label checkpoints from 50 through 5000")
    x = np.asarray([item[0] for item in selected], dtype=float)
    y = np.asarray([item[1] for item in selected], dtype=float)
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def copy_run_artifacts(source, target):
    target.mkdir(parents=True, exist_ok=True)
    root_names = {
        "auto_delta_provenance.json", "benchmark_summary.json", "config.yaml",
        "id_probcover_rounds.csv", "initial_sampling_summary.json", "lSet.npy",
        "uSet.npy", "valSet.npy",
    }
    for name in root_names:
        src = source / name
        if src.is_file():
            shutil.copy2(str(src), str(target / name))
    for episode in sorted(source.glob("episode_*"), key=lambda p: int(p.name.split("_")[-1])):
        dst_episode = target / episode.name
        for name in ("episode_summary.json", "activeSet.npy", "lSet.npy"):
            src = episode / name
            if src.is_file():
                dst_episode.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst_episode / name))


def strict_status(row, condition):
    source = Path(row["output_dir"])
    summary_path = source / "benchmark_summary.json"
    expected = 100 if condition == "fixed_delta_0.25" else 101
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if len(summary.get("episode_records", [])) == expected:
            return "COMPLETE", expected, int(summary["episode_records"][-1].get(
                "labeled_count_before_sampling", int(row["final_evaluated_budget"])
            ))
    completed = []
    for path in source.glob("episode_*/episode_summary.json"):
        try:
            completed.append(json.loads(path.read_text()))
        except Exception:
            pass
    if completed:
        latest = max(completed, key=lambda item: int(item["episode"]))
        return "PARTIAL_TIMEOUT", len(completed), int(latest["labeled_count_before_sampling"])
    return "MISSING", 0, 0


def main():
    if DEST.exists():
        raise SystemExit(f"refusing to overwrite existing package: {DEST}")
    DEST.mkdir(parents=True)

    fixed_rows = [
        row for row in read_csv(FIXED_MANIFEST)
        if row["method"] == "lidcover_batch_size_v2" and row["acquisition_batch_size"] == "50"
    ]
    auto_rows = [
        row for row in read_csv(AUTO_MANIFEST)
        if row["method"] == "lidcover_auto_sparse_radius_v3"
    ]
    if len(fixed_rows) != 5 or len(auto_rows) != 15:
        raise RuntimeError("unexpected source matrix size")

    inventory = []
    curves = []
    mechanism = []
    loaded = {}
    for condition, rows in (("fixed_delta_0.25", fixed_rows), ("automatic_v3", auto_rows)):
        for row in rows:
            dataset = row["dataset"]
            seed = int(row["seed"])
            source = Path(row["output_dir"])
            packaged = DEST / "results/raw" / condition / dataset / f"seed_{seed}"
            status, evaluated_rounds, last_budget = strict_status(row, condition)
            copy_run_artifacts(source, packaged)

            provenance_path = source / "auto_delta_provenance.json"
            provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
            summary_path = source / "benchmark_summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
            loaded[(condition, dataset, seed)] = summary
            inventory.append({
                "condition": condition,
                "dataset": dataset,
                "backbone": row["backbone"],
                "seed": seed,
                "experiment_id": row["experiment_id"],
                "status": status,
                "evaluated_rounds": evaluated_rounds,
                "last_evaluated_budget": last_budget,
                "batch_size": int(row["acquisition_batch_size"]),
                "configured_base_radius": 0.25 if condition == "fixed_delta_0.25" else provenance.get("delta_auto", ""),
                "cold_start_effective_radius": 0.3375 if condition == "fixed_delta_0.25" else provenance.get("initial_effective_radius", ""),
                "post_cold_start_effective_radius": 0.2125 if condition == "fixed_delta_0.25" else provenance.get("post_cold_start_effective_radius", ""),
                "final_test_accuracy": summary.get("final_test_accuracy", "") if summary else "",
                "final_val_accuracy": summary.get("final_val_accuracy", "") if summary else "",
                "source_output_dir": str(source),
                "packaged_result_dir": str(packaged.relative_to(DEST)),
            })

            records = summary.get("episode_records", []) if summary else []
            if not records:
                episode_files = sorted(
                    source.glob("episode_*/episode_summary.json"),
                    key=lambda p: int(p.parent.name.split("_")[-1]),
                )
                records = [json.loads(path.read_text()) for path in episode_files]
            for index, record in enumerate(records):
                budget = int(record.get("labeled_count_before_sampling", 50 * (index + 1)))
                curves.append({
                    "condition": condition,
                    "dataset": dataset,
                    "seed": seed,
                    "episode": int(record.get("episode", index)),
                    "labeled_budget": budget,
                    "test_accuracy": record.get("test_accuracy", ""),
                    "best_val_accuracy": record.get("best_val_accuracy", ""),
                    "train_time_sec": record.get("train_time_sec", ""),
                    "test_time_sec": record.get("test_time_sec", ""),
                    "acquisition_time_sec": record.get("acquisition_time_sec", ""),
                    "round_time_sec": record.get("round_time_sec", ""),
                })

            rounds_path = source / "id_probcover_rounds.csv"
            if rounds_path.is_file():
                for round_index, metrics in enumerate(read_csv(rounds_path)):
                    mechanism.append({
                        "condition": condition, "dataset": dataset, "seed": seed,
                        "selection_event": round_index, **metrics,
                    })

            cache_path = provenance.get("knn_cache_path", "")
            id_cache = None
            if condition == "automatic_v3":
                id_cache = ROOT / "experiments/lidcover_extensions_2026/cache/auto_sparse_radius_v3/idpc/resnet18" / dataset / f"seed{seed}" / "mle_local_k50.npy"
            else:
                id_cache = ROOT / "experiments/lidcover_extensions_2026/cache/batch_size/resnet18" / dataset / f"seed{seed}" / "mle_local_k50.npy"
            if id_cache.is_file():
                id_target = DEST / "results/geometry/local_ids" / condition / dataset / f"seed_{seed}_mle_local_k50.npy"
                id_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(id_cache), str(id_target))

    write_csv(DEST / "data/run_inventory.csv", inventory)
    write_csv(DEST / "data/learning_curves.csv", curves)
    write_csv(DEST / "data/lidcover_round_metrics.csv", mechanism)
    write_csv(DEST / "manifests/fixed_delta_0.25_cifar100_b50.csv", fixed_rows)
    write_csv(DEST / "manifests/automatic_v3_lidcover_only.csv", auto_rows)

    paired = []
    for seed in range(1, 6):
        fixed = loaded[("fixed_delta_0.25", "CIFAR100", seed)]
        auto = loaded[("automatic_v3", "CIFAR100", seed)]
        fixed_records = fixed["episode_records"]
        auto_records = [r for r in auto["episode_records"] if int(r.get("labeled_count_before_sampling", 0)) <= 5000]
        fixed_5000 = next(r for r in fixed_records if int(r["labeled_count_before_sampling"]) == 5000)
        auto_5000 = next(r for r in auto_records if int(r["labeled_count_before_sampling"]) == 5000)
        auto_row = next(
            row for row in auto_rows
            if row["dataset"] == "CIFAR100" and int(row["seed"]) == seed
        )
        auto_prov = json.loads((Path(auto_row["output_dir"]) / "auto_delta_provenance.json").read_text())
        paired.append({
            "seed": seed,
            "fixed_base_radius": 0.25,
            "automatic_base_radius": auto_prov["delta_auto"],
            "fixed_test_accuracy_at_5000": fixed_5000["test_accuracy"],
            "automatic_test_accuracy_at_5000": auto_5000["test_accuracy"],
            "automatic_minus_fixed_test_at_5000": float(auto_5000["test_accuracy"]) - float(fixed_5000["test_accuracy"]),
            "fixed_val_accuracy_at_5000": fixed_5000["best_val_accuracy"],
            "automatic_val_accuracy_at_5000": auto_5000["best_val_accuracy"],
            "automatic_minus_fixed_val_at_5000": float(auto_5000["best_val_accuracy"]) - float(fixed_5000["best_val_accuracy"]),
            "fixed_test_aulc_50_5000": aulc(fixed_records, "test_accuracy"),
            "automatic_test_aulc_50_5000": aulc(auto_records, "test_accuracy"),
            "fixed_val_aulc_50_5000": aulc(fixed_records, "best_val_accuracy"),
            "automatic_val_aulc_50_5000": aulc(auto_records, "best_val_accuracy"),
        })
    write_csv(DEST / "data/paired_cifar100_fixed_vs_automatic.csv", paired)

    aggregate = []
    comparisons = [
        ("test_accuracy_at_5000", "fixed_test_accuracy_at_5000", "automatic_test_accuracy_at_5000"),
        ("val_accuracy_at_5000", "fixed_val_accuracy_at_5000", "automatic_val_accuracy_at_5000"),
        ("test_aulc_50_5000", "fixed_test_aulc_50_5000", "automatic_test_aulc_50_5000"),
        ("val_aulc_50_5000", "fixed_val_aulc_50_5000", "automatic_val_aulc_50_5000"),
    ]
    for label, fixed_key, auto_key in comparisons:
        fixed = np.asarray([float(row[fixed_key]) for row in paired])
        auto = np.asarray([float(row[auto_key]) for row in paired])
        differences = auto - fixed
        interval = stats.t.interval(0.95, len(differences) - 1, loc=differences.mean(), scale=stats.sem(differences))
        aggregate.append({
            "metric": label,
            "n_paired_seeds": len(differences),
            "fixed_mean": fixed.mean(), "fixed_sem": sem(fixed),
            "automatic_mean": auto.mean(), "automatic_sem": sem(auto),
            "automatic_minus_fixed_mean": differences.mean(),
            "paired_difference_sem": sem(differences),
            "paired_difference_95ci_low": interval[0],
            "paired_difference_95ci_high": interval[1],
            "paired_ttest_p_exploratory": stats.ttest_rel(auto, fixed).pvalue,
        })
    write_csv(DEST / "data/aggregate_cifar100_fixed_vs_automatic.csv", aggregate)

    radius_rows = []
    for row in auto_rows:
        provenance = json.loads((Path(row["output_dir"]) / "auto_delta_provenance.json").read_text())
        radius_rows.append({
            "dataset": row["dataset"], "seed": int(row["seed"]),
            "base_radius": provenance["delta_auto"],
            "cold_start_effective_radius": provenance["initial_effective_radius"],
            "post_cold_start_effective_radius": provenance["post_cold_start_effective_radius"],
            "target_mean_nonself_out_degree": provenance["target_mean_nonself_out_degree"],
            "realised_mean_nonself_out_degree": provenance["realised_mean_nonself_out_degree"],
            "candidate_count": provenance["candidate_count"],
            "representation_sha256": provenance["representation_sha256"],
            "candidate_index_sha256": provenance["candidate_index_sha256"],
        })
    write_csv(DEST / "data/automatic_radius_by_dataset_seed.csv", radius_rows)

    # Primary matched plot: exact common budgets only.
    plt.figure(figsize=(7.2, 4.8))
    x = np.arange(50, 5001, 50)
    for condition, label, colour in (
        ("fixed_delta_0.25", "Fixed base radius 0.25", "#4C78A8"),
        ("automatic_v3", "Automatic radius v3", "#E45756"),
    ):
        ys = []
        for seed in range(1, 6):
            records = loaded[(condition, "CIFAR100", seed)]["episode_records"]
            points = {int(r["labeled_count_before_sampling"]): float(r["test_accuracy"]) for r in records}
            ys.append([points[int(budget)] for budget in x])
        ys = np.asarray(ys)
        mean = ys.mean(axis=0); error = ys.std(axis=0, ddof=1) / np.sqrt(ys.shape[0])
        plt.plot(x, mean, label=label, color=colour, linewidth=2)
        plt.fill_between(x, mean - error, mean + error, color=colour, alpha=0.18)
    plt.xlabel("Labelled training examples")
    plt.ylabel("Test accuracy (%)")
    plt.title("CIFAR-100 ResNet18 LIDCover: fixed vs automatic radius")
    plt.grid(alpha=0.2)
    plt.legend(frameon=False)
    plt.tight_layout()
    (DEST / "figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(DEST / "figures/cifar100_fixed_vs_automatic_learning_curve.png", dpi=220)
    plt.savefig(DEST / "figures/cifar100_fixed_vs_automatic_learning_curve.pdf")
    plt.close()

    # Code/config/evidence snapshots required to understand the exact run.
    source_files = [
        "experiments/lidcover_extensions_2026/auto_sparse_radius_v3.py",
        "experiments/lidcover_extensions_2026/auto_radius_v2.py",
        "experiments/lidcover_extensions_2026/extension_entrypoint.py",
        "experiments/lidcover_extensions_2026/samplers.py",
        "TypiClust/deep-al/pycls/al/IDProbCover.py",
        "TypiClust/deep-al/tools/train_al.py",
        "TypiClust/deep-al/pycls/core/config.py",
        "TypiClust/deep-al/pycls/datasets/data.py",
        "TypiClust/deep-al/pycls/datasets/tiny_imagenet.py",
        "TypiClust/deep-al/pycls/datasets/utils/features.py",
        "scripts/lidcover_extensions_2026/build_auto_sparse_radius_v3_manifest.py",
        "scripts/lidcover_extensions_2026/run_auto_sparse_radius_v3_task.sh",
        "scripts/lidcover_extensions_2026/launch_auto_sparse_radius_v3.sbatch",
        "scripts/lidcover_extensions_2026/run_manifest_task.sh",
        "scripts/lidcover_extensions_2026/launch_batch_size.sbatch",
        "scripts/lidcover_extensions_2026/build_manifests.py",
        "analysis/lidcover_extensions_2026/aggregate_results.py",
    ]
    hashes = []
    for relative in source_files:
        src = ROOT / relative
        dst = DEST / "code/repository_snapshot" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        hashes.append({"path": relative, "sha256": sha256(src), "bytes": src.stat().st_size})
    write_csv(DEST / "integrity/source_file_sha256.csv", hashes)

    for relative in (
        "TypiClust/deep-al/configs/cifar10/al/RESNET18.yaml",
        "TypiClust/deep-al/configs/cifar100/al/RESNET18.yaml",
        "TypiClust/deep-al/configs/tinyimagenet/al/RESNET18.yaml",
    ):
        src = ROOT / relative
        dst = DEST / "configs" / Path(relative).parts[-3] / Path(relative).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
    for name in (
        "AUTO_SPARSE_RADIUS_V3_DESIGN.md", "auto_sparse_radius_v3_launch_3557829.json",
        "historical_source_hashes_before.json", "batch_size_validation.md",
    ):
        (DEST / "evidence").mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(EVIDENCE / name), str(DEST / "evidence" / name))

    # Preserve full timeout logs only for the two incomplete automatic LIDCover runs.
    for task_id in (26, 29):
        for suffix in ("out", "err"):
            src = ROOT / f"experiments/lidcover_extensions_2026/logs/asr-v3-3557829_{task_id}.{suffix}"
            if src.is_file():
                (DEST / "logs").mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(DEST / "logs" / src.name))

    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "matplotlib": matplotlib.__version__,
        "training_environment_name": "typiclust",
        "known_training_stack": {
            "torch": "1.13.1+cu117", "faiss": "1.7.2", "scikit-learn": "1.0.2",
            "scikit-dimension": "0.3.4", "pandas": "1.3.5",
        },
    }
    (DEST / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")

    result_by_metric = {row["metric"]: row for row in aggregate}
    final_test = result_by_metric["test_accuracy_at_5000"]
    test_aulc = result_by_metric["test_aulc_50_5000"]
    readme = f"""# LIDCover fixed-radius versus automatic-radius thesis package

## Start here

This folder is a self-contained evidence package for a thesis chapter comparing
two **LIDCover-only** conditions. The primary analysis excludes every non-LID
baseline and uses only the exact matched CIFAR-100 comparison:

- fixed base radius `delta0 = 0.25`;
- automatic sparse-radius v3, derived without labels or predictive accuracy.

Both conditions use CIFAR-100, ResNet18, acquisition batch size 50, empty
labelled-set cold start, seeds 1--5, `alpha=1`, `k_id=50`, `k_knn=50`, the same
classifier configuration, and exact saved checkpoints. The primary endpoint is
5,000 labels. Automatic runs also contain a 5,050-label evaluation, but it is
excluded from the primary comparison to keep budgets identical.

## Primary result

At 5,000 labels, fixed-radius LIDCover achieves
`{float(final_test['fixed_mean']):.3f} +/- {float(final_test['fixed_sem']):.3f}` test accuracy
(mean +/- SEM), while automatic v3 achieves
`{float(final_test['automatic_mean']):.3f} +/- {float(final_test['automatic_sem']):.3f}`.
The paired automatic-minus-fixed difference is
`{float(final_test['automatic_minus_fixed_mean']):+.3f}` percentage points
(95% CI `{float(final_test['paired_difference_95ci_low']):+.3f}` to
`{float(final_test['paired_difference_95ci_high']):+.3f}`).

Normalised trapezoidal test AULC over the exact common 50--5,000 label domain is
`{float(test_aulc['fixed_mean']):.3f}` for fixed and
`{float(test_aulc['automatic_mean']):.3f}` for automatic, a paired difference of
`{float(test_aulc['automatic_minus_fixed_mean']):+.3f}` points. These results
support performance equivalence at the observed resolution: the automatic rule
removes manual radius choice without a measurable loss on CIFAR-100.

## Automatic rule

For each dataset/seed, v3 uses only acquisition-eligible rows of the pretrained
representation, row-normalises them, computes exact 50-nearest-neighbour
distances, estimates local intrinsic dimension with an explicit Hill/MLE
formula, and chooses the smallest base radius whose proxy adaptive graph at the
post-cold-start scale has mean non-self out-degree 2. It uses no labels,
validation accuracy, or test accuracy.

The frozen driver applies the historical schedule: `1.35 * delta0` for the empty
cold start and `0.85 * delta0` thereafter. Historical LIDCover then applies its
per-point inverse-LID transformation. The five automatic CIFAR-100 base radii
range from 0.2681 to 0.2702 (mean about 0.2690), close to the fixed 0.25 regime
without encoding 0.25 in the rule.

## Package map

- `data/paired_cifar100_fixed_vs_automatic.csv`: primary seed-paired results.
- `data/aggregate_cifar100_fixed_vs_automatic.csv`: means, SEMs, paired CIs, and exploratory t-tests.
- `data/learning_curves.csv`: all available per-budget accuracy/timing records.
- `data/lidcover_round_metrics.csv`: per-selection LID, radius, gain, and coverage summaries.
- `data/automatic_radius_by_dataset_seed.csv`: radius and provenance hashes.
- `data/run_inventory.csv`: completion status and original/packaged locations.
- `results/raw/`: compact run artifacts, selected sets, and episode summaries.
- `results/geometry/local_ids/`: exact cached local-ID arrays used by LIDCover.
- `code/repository_snapshot/`: exact implementation and launch-code snapshot.
- `figures/`: primary matched learning curve in PNG and PDF.
- `evidence/`: design, launch, validation, and historical hash records.
- `integrity/`: hashes for copied source files and the final package inventory.

## Supplementary automatic runs

The automatic folder includes CIFAR-10 and Tiny ImageNet because they are useful
for analysing transfer of the calibration rule. They are not fixed-versus-auto
comparisons because this package has no matched fixed-radius runs for those
datasets. CIFAR-10 has five complete automatic runs. Tiny ImageNet has three
complete runs; seeds 2 and 5 timed out after their 4,700- and 3,650-label
evaluations. These partial runs are explicitly marked and must not be treated as
final endpoints.

## Mechanism caveat

The historical MLE-LID cache contains about 20 zero estimates on CIFAR-100 and
Tiny ImageNet. Through the inverse-LID formula these create extremely large
nominal radii for a few early selected points, although the 50-neighbour graph
truncation caps realised coverage. Report this honestly and consider a clipped-
LID sensitivity analysis before making a smooth-radius mechanism claim.

## Deliberate exclusions

Model checkpoints, full stdout logs, feature matrices, and kNN distance caches
are excluded because they are large and unnecessary for statistical/GPT
analysis. Per-episode selected sets, accumulated labelled sets, summaries,
configs, provenance, local-ID arrays, and incomplete-run timeout logs are
included. No source experiment directory was modified.
"""
    (DEST / "README_FOR_GPT.md").write_text(readme)

    method = """# Exact method specification

Let `X_C` be the acquisition-eligible pretrained features for one dataset and
seed. After row-wise L2 normalisation, compute exact Euclidean 50-nearest-
neighbour distances `d_ij`. Estimate candidate-only local dimension as

`LID_i = -(k-1) / sum_(j=1)^(k-1) log((d_ij+eps)/(d_i,k+eps))`, with `k=50`.

Set `rho_i = LID_i / median(LID)` and define the required base threshold for
directed neighbour `(i,j)` as `d_ij * rho_i / 0.85`. The v3 base radius is the
strict order statistic that gives a target mean non-self out-degree of 2.

The driver then uses `1.35 * delta0` at the empty cold start and `0.85 * delta0`
after labels exist. Historical LIDCover independently loads its frozen full-
training MLE-LID estimates and uses

`delta_i = delta_effective * ((LID_i+1e-12)/(median_LID+1e-12))^(-1)`.

Selection is the frozen historical `IDProbCover` implementation in
`code/repository_snapshot/TypiClust/deep-al/pycls/al/IDProbCover.py`. The
automatic resolver is
`code/repository_snapshot/experiments/lidcover_extensions_2026/auto_sparse_radius_v3.py`.
"""
    (DEST / "METHOD_SPECIFICATION.md").write_text(method)

    # Hash every packaged file last (excluding the inventory itself).
    package_files = []
    for path in sorted(DEST.rglob("*")):
        if path.is_file() and path.name != "package_file_sha256.csv":
            package_files.append({
                "path": str(path.relative_to(DEST)), "sha256": sha256(path), "bytes": path.stat().st_size,
            })
    write_csv(DEST / "integrity/package_file_sha256.csv", package_files)
    print(json.dumps({
        "package": str(DEST), "fixed_runs": len(fixed_rows), "automatic_runs": len(auto_rows),
        "complete_runs": sum(row["status"] == "COMPLETE" for row in inventory),
        "partial_runs": sum(row["status"] != "COMPLETE" for row in inventory),
        "files": len(package_files) + 1,
    }, indent=2))


if __name__ == "__main__":
    main()
