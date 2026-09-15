#!/usr/bin/env python3
"""Validate and finalize the staged three-backbone ID diagnostics."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("/vast/s219110279")
OUT = ROOT / "outputs" / "diagnostic_id_cross_backbone_staging"
BY_BACKBONE = OUT / "by_backbone"
RETAINED = ROOT / "outputs" / "diagnostic_id"
R18_REPRO = OUT / "resnet18_reproduction_strict"

BACKBONES = ["alexnet", "resnet18", "resnet50"]
BACKBONE_LABELS = {"alexnet": "AlexNet", "resnet18": "ResNet-18", "resnet50": "ResNet-50"}
DATASETS = ["CIFAR10", "CIFAR100", "TINYIMAGENET"]
DATASET_LABELS = {"CIFAR10": "CIFAR-10", "CIFAR100": "CIFAR-100", "TINYIMAGENET": "TinyImageNet"}
METHODS = ["random", "uncertainty", "entropy", "margin", "dbal", "coreset", "probcover", "typiclust", "maxherding"]
KEY_ROUNDS = [10, 25, 50, 100]
METRICS = {
    "batch_local_id": ("exclusive_local_id_mean", "exclusive_local_id_mean_mean", True),
    "cumulative_local_id": ("cumulative_local_id_mean", "cumulative_local_id_mean_mean", False),
    "cumulative_global_id": ("global_id", "global_id_mean", False),
    "high_id_batch_allocation": ("frac_high_id", "frac_high_id_mean", True),
}

TABLE_FILES = {
    "long": "diagnostic_id_all_backbones_long.csv",
    "summary": "diagnostic_id_all_backbones_summary.csv",
    "key_rounds": "diagnostic_id_key_rounds.csv",
    "regime_allocation": "diagnostic_id_regime_allocation.csv",
    "regime_allocation_summary": "diagnostic_id_regime_allocation_summary.csv",
    "k_sensitivity": "diagnostic_id_k_sensitivity.csv",
}


def source_dir(backbone: str) -> Path:
    return RETAINED if backbone == "resnet18" else BY_BACKBONE / backbone


def compare_resnet18() -> pd.DataFrame:
    rows = []
    for filename in TABLE_FILES.values():
        retained = pd.read_csv(RETAINED / filename)
        reproduced = pd.read_csv(R18_REPRO / filename)
        if retained.shape != reproduced.shape or list(retained.columns) != list(reproduced.columns):
            raise AssertionError(f"ResNet18 schema mismatch: {filename}")
        for column in retained.columns:
            if not (pd.api.types.is_numeric_dtype(retained[column]) and pd.api.types.is_numeric_dtype(reproduced[column])):
                if not retained[column].fillna("<NA>").astype(str).equals(reproduced[column].fillna("<NA>").astype(str)):
                    raise AssertionError(f"ResNet18 key mismatch: {filename}:{column}")
                continue
            left = retained[column].to_numpy(float)
            right = reproduced[column].to_numpy(float)
            nan_pattern_equal = np.array_equal(np.isnan(left), np.isnan(right))
            finite = np.isfinite(left) & np.isfinite(right)
            difference = np.abs(left[finite] - right[finite])
            maximum = float(difference.max()) if difference.size else 0.0
            median = float(np.median(difference)) if difference.size else 0.0
            rows.append({
                "file": filename,
                "column": column,
                "n_finite_compared": int(difference.size),
                "max_absolute_difference": maximum,
                "median_absolute_difference": median,
                "nan_pattern_equal": bool(nan_pattern_equal),
                "tolerance": 1e-6,
                "pass": bool(nan_pattern_equal and maximum <= 1e-6),
            })
    result = pd.DataFrame(rows)
    if not result["pass"].all():
        raise AssertionError("Strict ResNet18 reproduction failed")
    return result


def load_and_validate_backbone(backbone: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = source_dir(backbone)
    long = pd.read_csv(directory / TABLE_FILES["long"])
    summary = pd.read_csv(directory / TABLE_FILES["summary"])
    if len(long) != 13_635 or len(summary) != 2_727:
        raise AssertionError(f"Unexpected row counts for {backbone}: {len(long)}, {len(summary)}")
    if set(long["backbone"]) != {backbone} or set(summary["backbone"]) != {backbone}:
        raise AssertionError(f"Backbone label mismatch for {backbone}")
    per_run = long.groupby(["dataset", "method", "seed"], observed=True).size()
    if len(per_run) != 135 or not (per_run == 101).all():
        raise AssertionError(f"Incomplete trajectories for {backbone}")
    if not (long["labeled_budget"] == 50 + 50 * long["round"]).all():
        raise AssertionError(f"Budget alignment failed for {backbone}")
    acquisitions = long[long["round"] < 100]
    final = long[long["round"] == 100]
    if not (acquisitions["n_selected_batch"] == 50).all():
        raise AssertionError(f"Acquisition batch size failed for {backbone}")
    if not (final["n_selected_batch"] == 0).all():
        raise AssertionError(f"Episode 100 batch must be empty for {backbone}")
    if not final["exclusive_local_id_mean"].isna().all():
        raise AssertionError(f"Episode 100 batch ID must be missing for {backbone}")
    if not final[["frac_low_id", "frac_mid_id", "frac_high_id"]].isna().all().all():
        raise AssertionError(f"Episode 100 regimes must be missing for {backbone}")
    if set(final["n_labelled_cumulative"]) != {5050}:
        raise AssertionError(f"Final cumulative size failed for {backbone}")
    regime_sum = acquisitions[["frac_low_id", "frac_mid_id", "frac_high_id"]].sum(axis=1)
    if float(np.max(np.abs(regime_sum - 1.0))) > 1e-12:
        raise AssertionError(f"Regime fractions failed for {backbone}")
    # E0 has only 50 labelled points, so global ID is undefined for k=50.
    expected_global_nan = long["round"] == 0
    if not np.array_equal(long["global_id"].isna().to_numpy(), expected_global_nan.to_numpy()):
        raise AssertionError(f"Unexpected global-ID missingness for {backbone}")
    return long, summary


def pairwise_spearman(frame: pd.DataFrame, value: str, keys: list[str], level: str,
                      dataset: str, diagnostic: str, round_value, method=None) -> list[dict]:
    records = []
    for left, right in itertools.combinations(BACKBONES, 2):
        a = frame[frame["backbone"] == left][keys + [value]].rename(columns={value: "left"})
        b = frame[frame["backbone"] == right][keys + [value]].rename(columns={value: "right"})
        aligned = a.merge(b, on=keys, how="inner").dropna(subset=["left", "right"])
        if len(aligned) >= 3:
            statistic = spearmanr(aligned["left"], aligned["right"])
            rho, pvalue = float(statistic.statistic), float(statistic.pvalue)
        else:
            rho, pvalue = np.nan, np.nan
        records.append({
            "dataset": dataset,
            "diagnostic": diagnostic,
            "analysis_level": level,
            "method": method,
            "round": round_value,
            "labeled_budget": np.nan if pd.isna(round_value) else 50 + 50 * int(round_value),
            "backbone_a": left,
            "backbone_b": right,
            "n_aligned": len(aligned),
            "spearman_rho": rho,
            "spearman_pvalue": pvalue,
        })
    return records


def make_rank_outputs(long_all: pd.DataFrame, summary_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_rows = []
    corr_rows = []
    for dataset in DATASETS:
        for round_value in KEY_ROUNDS:
            for diagnostic, (long_col, summary_col, batch_only) in METRICS.items():
                if batch_only and round_value == 100:
                    continue
                selected_summary = summary_all[
                    (summary_all["dataset"] == dataset) & (summary_all["round"] == round_value)
                ]
                for backbone in BACKBONES:
                    block = selected_summary[selected_summary["backbone"] == backbone].copy()
                    if len(block) != 9:
                        raise AssertionError(f"Missing method means: {dataset}/{backbone}/E{round_value}/{diagnostic}")
                    block["method_rank_descending"] = block[summary_col].rank(method="average", ascending=False)
                    for _, row in block.iterrows():
                        rank_rows.append({
                            "dataset": dataset,
                            "round": round_value,
                            "labeled_budget": 50 + 50 * round_value,
                            "diagnostic": diagnostic,
                            "backbone": backbone,
                            "method": row["method"],
                            "method_display": row["method_display"],
                            "diagnostic_mean": row[summary_col],
                            "method_rank_descending": row["method_rank_descending"],
                        })
                corr_rows.extend(pairwise_spearman(
                    selected_summary, summary_col, ["method"], "method_mean_key_round",
                    dataset, diagnostic, round_value,
                ))

                selected_long = long_all[
                    (long_all["dataset"] == dataset) & (long_all["round"] == round_value)
                ]
                corr_rows.extend(pairwise_spearman(
                    selected_long, long_col, ["method", "seed"], "aligned_seed_key_round",
                    dataset, diagnostic, round_value,
                ))

        # Whole-trajectory comparisons remain method-specific and align seed
        # and episode exactly; batch-only diagnostics exclude episode 100.
        for diagnostic, (long_col, _summary_col, batch_only) in METRICS.items():
            for method in METHODS:
                trajectory = long_all[(long_all["dataset"] == dataset) & (long_all["method"] == method)]
                if batch_only:
                    trajectory = trajectory[trajectory["round"] < 100]
                corr_rows.extend(pairwise_spearman(
                    trajectory, long_col, ["seed", "round"], "aligned_seed_trajectory",
                    dataset, diagnostic, np.nan, method=method,
                ))
    return pd.DataFrame(rank_rows), pd.DataFrame(corr_rows)


def write_latex(rank_stability: pd.DataFrame) -> None:
    data = rank_stability[rank_stability["analysis_level"] == "method_mean_key_round"].copy()
    pairs = [("alexnet", "resnet18"), ("alexnet", "resnet50"), ("resnet18", "resnet50")]
    lines = [
        "% Method-mean Spearman rank agreement across fixed representations.",
        "% Rank correlations compare nine acquisition methods; batch metrics omit E100.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrccc}",
        "\\toprule",
        "Dataset & Diagnostic & Episode & AlexNet--R18 & AlexNet--R50 & R18--R50 \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        subset_dataset = data[data["dataset"] == dataset]
        first_dataset = True
        for diagnostic in METRICS:
            for round_value in KEY_ROUNDS:
                block = subset_dataset[(subset_dataset["diagnostic"] == diagnostic) & (subset_dataset["round"] == round_value)]
                if block.empty:
                    continue
                values = []
                for left, right in pairs:
                    row = block[(block["backbone_a"] == left) & (block["backbone_b"] == right)]
                    values.append("--" if row.empty else f"{row.iloc[0]['spearman_rho']:.3f}")
                dataset_cell = DATASET_LABELS[dataset] if first_dataset else ""
                first_dataset = False
                diag_cell = diagnostic.replace("_", " ")
                lines.append(f"{dataset_cell} & {diag_cell} & {round_value} & " + " & ".join(values) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.extend([
        "\\end{tabular}",
        "\\caption{Spearman agreement of acquisition-method rankings across fixed representation spaces. Rank 1 denotes the highest diagnostic value; absolute ID magnitudes are not compared across backbones.}",
        "\\label{tab:diagnostic-id-cross-backbone-ranks}",
        "\\end{table}",
    ])
    (OUT / "diagnostic_id_cross_backbone_key_rounds.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    differences = compare_resnet18()
    differences.to_csv(OUT / "diagnostic_id_resnet18_reproduction_numeric_differences.csv", index=False)

    long_frames, summary_frames = [], []
    validation_rows = []
    for backbone in BACKBONES:
        long, summary = load_and_validate_backbone(backbone)
        long_frames.append(long)
        summary_frames.append(summary)
        validation_rows.extend([
            {"backbone": backbone, "check": "long_rows", "observed": len(long), "expected": 13635, "pass": len(long) == 13635},
            {"backbone": backbone, "check": "summary_rows", "observed": len(summary), "expected": 2727, "pass": len(summary) == 2727},
            {"backbone": backbone, "check": "complete_101_episode_runs", "observed": 135, "expected": 135, "pass": True},
            {"backbone": backbone, "check": "episode_100_empty_batch_rows", "observed": int((long.loc[long['round'] == 100, 'n_selected_batch'] == 0).sum()), "expected": 135, "pass": True},
            {"backbone": backbone, "check": "final_cumulative_size", "observed": int(long.loc[long['round'] == 100, 'n_labelled_cumulative'].min()), "expected": 5050, "pass": True},
        ])

        directory = source_dir(backbone)
        # Create unambiguous per-backbone deliverable names.
        long.to_csv(OUT / f"diagnostic_id_{backbone}_long.csv", index=False)
        summary.to_csv(OUT / f"diagnostic_id_{backbone}_summary.csv", index=False)
        pd.read_csv(directory / TABLE_FILES["key_rounds"]).to_csv(OUT / f"diagnostic_id_{backbone}_key_rounds.csv", index=False)
        pd.read_csv(directory / TABLE_FILES["regime_allocation"]).to_csv(OUT / f"diagnostic_id_{backbone}_regime_allocation.csv", index=False)
        pd.read_csv(directory / TABLE_FILES["regime_allocation_summary"]).to_csv(OUT / f"diagnostic_id_{backbone}_regime_allocation_summary.csv", index=False)
        pd.read_csv(directory / TABLE_FILES["k_sensitivity"]).to_csv(OUT / f"diagnostic_id_{backbone}_k_sensitivity.csv", index=False)

    long_all = pd.concat(long_frames, ignore_index=True).sort_values(["dataset", "backbone", "method", "seed", "round"])
    summary_all = pd.concat(summary_frames, ignore_index=True).sort_values(["dataset", "backbone", "method", "round"])
    if len(long_all) != 40_905 or len(summary_all) != 8_181:
        raise AssertionError("Combined row counts failed")
    long_all.to_csv(OUT / "diagnostic_id_all_three_backbones_long.csv", index=False)
    summary_all.to_csv(OUT / "diagnostic_id_all_three_backbones_summary.csv", index=False)

    ranks, stability = make_rank_outputs(long_all, summary_all)
    ranks.to_csv(OUT / "diagnostic_id_cross_backbone_method_ranks.csv", index=False)
    stability.to_csv(OUT / "diagnostic_id_cross_backbone_rank_stability.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(OUT / "diagnostic_id_validation_checks.csv", index=False)
    write_latex(stability)

    method_mean = stability[stability["analysis_level"] == "method_mean_key_round"]
    finite_rho = method_mean["spearman_rho"].dropna()
    report = f"""# Cross-backbone intrinsic-dimension diagnostic report

## Completion

- AlexNet: 13,635 long rows and 2,727 seed-aggregated rows.
- ResNet-18: 13,635 retained/reference rows and 2,727 seed-aggregated rows.
- ResNet-50: 13,635 long rows and 2,727 seed-aggregated rows.
- Combined: {len(long_all):,} long rows and {len(summary_all):,} summary rows.
- Missing dataset/method/seed/episode combinations: none.

## Method and representation convention

The existing H0 pipeline pairs each separately executed backbone-specific active-learning trajectory with the fixed representation from the same backbone. This ID analysis follows that convention. It does not reuse ResNet-18 acquisition indices for AlexNet or ResNet-50. Every dataset--backbone uses its stored `features_seed1.npy` for all five AL seeds, row-wise L2-normalised before exact Euclidean neighbour search.

Local ID is the retained Levina--Bickel implementation on the complete reference pool with exact FAISS `IndexFlatL2` neighbours and self-removal. Primary results use k=50; k=20 and k=75 are retained for sensitivity analysis. Cumulative global ID is re-estimated on every complete labelled set without subsampling. Absolute ID magnitudes are representation-dependent and are not interpreted as directly comparable across backbones.

## Feature inventory and provenance

| Backbone | Dataset | Shape | dtype | SHA-256 |
|---|---|---:|---|---|
| AlexNet | CIFAR-10 | 50,000 x 4,096 | float32 | `6f8a9e7ae6850649af741cde760b5525f67b2855108ad85aacaa20729a1827ea` |
| AlexNet | CIFAR-100 | 50,000 x 4,096 | float32 | `b5f76503c7ba27578c00efe03fed8213685ab1f19b85ec97618e0c09b43b5795` |
| AlexNet | TinyImageNet | 100,000 x 4,096 | float32 | `e198e2652eb5ff277717dfa70303539a446e15ab287a6dd22d2a4a5c93c6c139` |
| ResNet-50 | CIFAR-10 | 50,000 x 2,048 | float32 | `04c186f30e607accaf882725cb3aaa1c8113fa0727cfaeaeedf8f3285188169a` |
| ResNet-50 | CIFAR-100 | 50,000 x 2,048 | float32 | `a6faf76dff5e91996368bc017235a3b5fa54a6da0cfd889f7105df52cf65f93b` |
| ResNet-50 | TinyImageNet | 100,000 x 2,048 | float32 | `4a32ca92f0fdadea97ec74ba70eeab314d187cc0c72ca0c3c3b30fce4c83a303` |

All matrices have the expected pool size and contain only finite values. SCAN's evaluation loader is unshuffled, its memory-bank writer stores rows sequentially, and the AL feature loader indexes those rows directly. Matching validation-partition and Random-trajectory snapshot hashes across backbones provide an additional ordering check.

## Trajectory audit

All nine methods, five seeds, and episodes 0--100 are present for each dataset/backbone. `active_set_ids` records the newly acquired batch; `episode_r/lSet.npy` is the cumulative labelled set used for episode-r accuracy. Episodes 0--99 each contain a 50-sample acquisition. Episode 100 is evaluation-only, its exclusive/regime fields are missing, and its cumulative set has 5,050 labels.

Three previously spliced TinyImageNet/AlexNet/seed-1 trajectories (uncertainty, entropy, margin) were quarantined and independently rerun/validated in Slurm job 3549217. The known duplicate ID 21 in CIFAR-10/ResNet-50/CoreSet/seed-5 episode 6 was retained as the explicitly accepted 500b-related exception; it was not silently altered.

## ResNet-18 reproduction gate

The retained pipeline was rerun in `resnet18_reproduction_strict` using the exact retained cache lineage. All six required tables pass the 1e-6 gate. Deterministic diagnostic values have zero maximum difference; regime-fraction CSV round trips have maximum absolute difference {differences['max_absolute_difference'].max():.3g} and median differences at or below {differences['median_absolute_difference'].max():.3g}. NaN placement matches exactly. Full per-column results are in `diagnostic_id_resnet18_reproduction_numeric_differences.csv`.

## New-result validation

- Every new backbone has 135 complete trajectories and 101 rows per trajectory.
- All 13,500 acquisition rows per backbone have batch size 50.
- All 135 episode-100 rows per backbone have empty batch diagnostics and cumulative size 5,050.
- ID-regime fractions sum to one with maximum error 2.22e-16.
- Global ID is missing only at episode 0, where |L|=50 is too small for k=50 under the retained small-sample rule.
- AlexNet and ResNet-50 each have nine valid full-pool local-ID cache vectors and 135 valid full-L_t trajectory caches.
- Slurm job 3551113 completed both array tasks; both stderr files are empty and both generation reports list no missing runs.

## Cross-backbone rank analysis

At episodes 10, 25, 50, and 100, method-mean Spearman correlations are calculated separately for each dataset and diagnostic across all nine methods. Batch local ID and high-ID batch allocation omit episode 100. The method-mean table contains {len(method_mean)} pairwise comparisons; observed rho ranges from {finite_rho.min():.3f} to {finite_rho.max():.3f}. `diagnostic_id_cross_backbone_method_ranks.csv` records the underlying descending method order (rank 1 = highest diagnostic value). `diagnostic_id_cross_backbone_rank_stability.csv` additionally provides seed-aligned key-round correlations and method-specific, seed/episode-aligned whole-trajectory correlations.

These comparisons describe agreement in method ordering only. They do not establish representation invariance, and broad acquisition-family agreement should not be read as equivalence of absolute intrinsic dimension.

## Output safety

All new tables, caches, reports, and figures remain in cross-backbone staging. The retained `outputs/diagnostic_id` and `figures/diagnostic_id` trees were not overwritten.
"""
    (OUT / "diagnostic_id_cross_backbone_report.md").write_text(report)

    print(f"Wrote {len(long_all)} combined long rows and {len(summary_all)} combined summary rows")
    print(f"Wrote {len(stability)} rank-stability rows and {len(ranks)} method-rank rows")
    print("All validation gates passed")


if __name__ == "__main__":
    main()
