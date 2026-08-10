#!/usr/bin/env python3
"""
Cross-backbone method rank consistency (Spearman).

For each dataset, correlate ResNet-18 method rankings with ResNet-50 and
AlexNet rankings on late structural / accuracy metrics.

Does not recompute embeddings or AL runs. Uses existing diagnostic tables.
ID metrics exist only for ResNet-18 in current outputs — those comparisons
are reported as unavailable (not invented).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path("/vast/s219110279")
ID_LONG = REPO / "outputs" / "diagnostic_id" / "diagnostic_id_all_backbones_long.csv"
H0_LONG = REPO / "outputs" / "diagnostic_h0" / "diagnostic_h0_all_backbones_long.csv"
ACC_KEY = REPO / "outputs" / "diagnostic_accuracy" / "diagnostic_accuracy_key_rounds.csv"

OUT = REPO / "outputs"
FIG = REPO / "figures" / "joint_diagnostics"
FIG.mkdir(parents=True, exist_ok=True)

DATASETS = ["CIFAR10", "CIFAR100", "TINYIMAGENET"]
DATASET_LABELS = {
    "CIFAR10": "CIFAR-10",
    "CIFAR100": "CIFAR-100",
    "TINYIMAGENET": "TinyImageNet",
}
REF = "resnet18"
COMPARE = ["resnet50", "alexnet"]
BB_LABELS = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "alexnet": "AlexNet",
}
METHOD_ORDER = [
    "random", "uncertainty", "entropy", "margin", "dbal",
    "coreset", "probcover", "typiclust", "maxherding",
]
METHOD_DISPLAY = {
    "random": "Random",
    "uncertainty": "Uncertainty (LC)",
    "entropy": "Entropy",
    "margin": "Margin",
    "dbal": "DBAL (BALD)",
    "coreset": "CoreSet",
    "probcover": "ProbCover",
    "typiclust": "TypiClust",
    "maxherding": "MaxHerding",
}
EXCL_ROUNDS = list(range(10, 91))


def _mean_by_method(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Seed-averaged method means; index = method."""
    g = (
        df.groupby("method", as_index=True)[value_col]
        .mean()
        .reindex(METHOD_ORDER)
        .dropna()
    )
    return g.astype(float)


def build_method_scores() -> tuple[pd.DataFrame, dict]:
    """One row per dataset × backbone × method with late metric means."""
    id_df = pd.read_csv(ID_LONG)
    h0_df = pd.read_csv(H0_LONG)
    acc = pd.read_csv(ACC_KEY)

    id_bbs = sorted(id_df.backbone.unique())
    h0_bbs = sorted(h0_df.backbone.unique())
    acc_bbs = sorted(acc.backbone.unique())
    coverage = {
        "id_backbones": id_bbs,
        "h0_backbones": h0_bbs,
        "accuracy_backbones": acc_bbs,
        "id_metrics_available_for_cross_backbone": False,
        "note": (
            "Final global ID and mean exclusive local ID are available only for "
            f"{id_bbs}; Spearman vs other backbones is not computed for those metrics."
        ),
    }

    rows = []
    all_bbs = sorted(set(h0_bbs) | set(acc_bbs) | set(id_bbs))
    for ds in DATASETS:
        for bb in all_bbs:
            # Final accuracy @ E100 from accuracy key table
            a = acc[(acc.dataset == ds) & (acc.backbone == bb)]
            acc_map = {
                r.method: float(r.final_accuracy)
                for _, r in a.iterrows()
                if pd.notna(r.final_accuracy)
            }

            # H0 @ E100
            h = h0_df[
                (h0_df.dataset == ds)
                & (h0_df.backbone == bb)
                & (h0_df["round"] == 100)
            ]
            w1 = _mean_by_method(h, "cumulative_wasserstein") if len(h) else pd.Series(dtype=float)
            b0 = _mean_by_method(h, "beta0_auc_normalised") if len(h) else pd.Series(dtype=float)

            # ID metrics (resnet18 only in current outputs)
            i = id_df[(id_df.dataset == ds) & (id_df.backbone == bb)]
            if len(i):
                gid = _mean_by_method(i[i["round"] == 100], "global_id")
                excl = _mean_by_method(
                    i[i["round"].isin(EXCL_ROUNDS)], "exclusive_local_id_mean"
                )
            else:
                gid = pd.Series(dtype=float)
                excl = pd.Series(dtype=float)

            methods = sorted(
                set(acc_map)
                | set(w1.index)
                | set(b0.index)
                | set(gid.index)
                | set(excl.index)
            )
            for m in methods:
                if m not in METHOD_ORDER:
                    continue
                rows.append({
                    "dataset": ds,
                    "backbone": bb,
                    "method": m,
                    "method_display": METHOD_DISPLAY.get(m, m),
                    "final_global_id": float(gid[m]) if m in gid.index else np.nan,
                    "mean_exclusive_local_id_E10_E90": (
                        float(excl[m]) if m in excl.index else np.nan
                    ),
                    "final_cumulative_wasserstein": (
                        float(w1[m]) if m in w1.index else np.nan
                    ),
                    "final_beta0_auc_normalised": (
                        float(b0[m]) if m in b0.index else np.nan
                    ),
                    "final_accuracy": acc_map.get(m, np.nan),
                })

    scores = pd.DataFrame(rows)
    return scores, coverage


METRICS = [
    ("final_global_id", "Final global ID"),
    ("mean_exclusive_local_id_E10_E90", "Mean exclusive local ID (E10–E90)"),
    ("final_cumulative_wasserstein", "Final cumulative Wasserstein"),
    ("final_beta0_auc_normalised", "Final normalised β0 AUC"),
    ("final_accuracy", "Final accuracy"),
]


def spearman_table(scores: pd.DataFrame, coverage: dict) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        ref = scores[(scores.dataset == ds) & (scores.backbone == REF)].set_index("method")
        for other in COMPARE:
            oth = scores[(scores.dataset == ds) & (scores.backbone == other)].set_index("method")
            for col, label in METRICS:
                rec = {
                    "dataset": ds,
                    "dataset_label": DATASET_LABELS[ds],
                    "metric": col,
                    "metric_label": label,
                    "ref_backbone": REF,
                    "compare_backbone": other,
                    "pair_label": f"{BB_LABELS[REF]} vs {BB_LABELS[other]}",
                    "spearman_rho": np.nan,
                    "n_methods": 0,
                    "methods_used": "",
                    "status": "ok",
                    "note": "",
                }
                # ID metrics unavailable off ResNet-18
                if col in ("final_global_id", "mean_exclusive_local_id_E10_E90"):
                    if other not in coverage["id_backbones"] or REF not in coverage["id_backbones"]:
                        rec["status"] = "unavailable"
                        rec["note"] = (
                            "ID diagnostics currently exist only for ResNet-18; "
                            "no ranking to correlate."
                        )
                        rows.append(rec)
                        continue

                if col not in ref.columns or col not in oth.columns:
                    rec["status"] = "unavailable"
                    rec["note"] = "metric column missing"
                    rows.append(rec)
                    continue

                a = ref[col].dropna()
                b = oth[col].dropna()
                common = sorted(set(a.index) & set(b.index), key=lambda m: METHOD_ORDER.index(m))
                if len(common) < 3:
                    rec["status"] = "insufficient"
                    rec["n_methods"] = len(common)
                    rec["note"] = f"fewer than 3 shared methods with finite values (n={len(common)})"
                    rows.append(rec)
                    continue

                rho = spearmanr(a.loc[common], b.loc[common]).correlation
                rec["spearman_rho"] = float(rho) if rho is not None and np.isfinite(rho) else np.nan
                rec["n_methods"] = len(common)
                rec["methods_used"] = ",".join(common)
                rows.append(rec)
    return pd.DataFrame(rows)


def write_latex(corr: pd.DataFrame, path: Path):
    """Compact table: datasets as column groups, metrics as rows, two compare backbones."""
    lines = [
        "% Auto-generated cross-backbone rank consistency (Spearman).",
        "% Method rankings: ResNet-18 vs ResNet-50 / AlexNet. Descriptive only.",
        "% --- = ID metrics unavailable (ResNet-18 only in current ID outputs).",
        "\\begin{tabular}{l" + "rr" * len(DATASETS) + "}",
        "\\toprule",
        " & "
        + " & ".join(
            f"\\multicolumn{{2}}{{c}}{{{DATASET_LABELS[ds]}}}" for ds in DATASETS
        )
        + " \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "Metric & "
        + " & ".join(["R50", "Alex"] * len(DATASETS))
        + " \\\\",
        "\\midrule",
    ]
    for col, label in METRICS:
        cells = [label]
        for ds in DATASETS:
            for other in COMPARE:
                sub = corr[
                    (corr.dataset == ds)
                    & (corr.compare_backbone == other)
                    & (corr.metric == col)
                ]
                if sub.empty or sub.iloc[0].status != "ok" or not np.isfinite(sub.iloc[0].spearman_rho):
                    cells.append("---")
                else:
                    cells.append(f"{sub.iloc[0].spearman_rho:.2f}")
        lines.append(" & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "",
        "% Columns R50 / Alex = Spearman $\\rho$ of method rankings vs ResNet-18.",
        "% $n{=}9$ methods wherever a finite value is shown.",
        "",
    ]
    path.write_text("\n".join(lines))


def write_findings(corr: pd.DataFrame, coverage: dict, scores: pd.DataFrame, path: Path):
    lines = [
        "# Cross-backbone rank consistency",
        "",
        "Spearman correlation between **ResNet-18 method rankings** and "
        "ResNet-50 / AlexNet rankings, computed **separately per dataset** "
        "from seed-averaged method means. Descriptive only — not causal.",
        "",
        "## Data coverage",
        "",
        f"- ID long backbones: {coverage['id_backbones']}",
        f"- H0 long backbones: {coverage['h0_backbones']}",
        f"- Accuracy key backbones: {coverage['accuracy_backbones']}",
        f"- {coverage['note']}",
        "",
        "## Metrics",
        "",
        "- Final global ID: ID table, round 100",
        "- Mean exclusive local ID: ID table, mean over acquisition rounds E10–E90",
        "- Final cumulative Wasserstein / normalised β0 AUC: H0 table, round 100",
        "- Final accuracy: diagnostic accuracy key table (`final_accuracy`)",
        "",
        "## Results (Spearman ρ)",
        "",
    ]

    for ds in DATASETS:
        lines.append(f"### {DATASET_LABELS[ds]}")
        lines.append("")
        for col, label in METRICS:
            bits = []
            for other in COMPARE:
                r = corr[
                    (corr.dataset == ds)
                    & (corr.metric == col)
                    & (corr.compare_backbone == other)
                ].iloc[0]
                if r.status != "ok":
                    bits.append(f"{BB_LABELS[other]}: unavailable")
                else:
                    bits.append(f"{BB_LABELS[other]}: ρ={r.spearman_rho:.3f} (n={r.n_methods})")
            lines.append(f"- **{label}**: " + "; ".join(bits))
        lines.append("")

    # Short interpretation from available metrics only
    avail = corr[corr.status == "ok"].copy()
    lines += [
        "## Architecture robustness (from available cells only)",
        "",
    ]
    if avail.empty:
        lines.append("- No finite Spearman values available.")
    else:
        for col, label in METRICS:
            sub = avail[avail.metric == col]
            if sub.empty:
                lines.append(
                    f"- **{label}**: cannot assess (ID only on ResNet-18)."
                    if "global_id" in col or "exclusive" in col
                    else f"- **{label}**: no finite values."
                )
                continue
            rhos = sub.spearman_rho.astype(float)
            lines.append(
                f"- **{label}**: median ρ={rhos.median():.2f} "
                f"(min={rhos.min():.2f}, max={rhos.max():.2f}) across "
                f"{len(sub)} dataset×backbone pairs."
            )
            # per-dataset floor
            for ds in DATASETS:
                dsub = sub[sub.dataset == ds]
                if dsub.empty:
                    continue
                weak = dsub[dsub.spearman_rho < 0.5]
                if len(weak):
                    pairs = ", ".join(
                        f"{BB_LABELS[r.compare_backbone]} ({r.spearman_rho:.2f})"
                        for _, r in weak.iterrows()
                    )
                    lines.append(
                        f"  - {DATASET_LABELS[ds]} pairs with ρ<0.5: {pairs}"
                    )
                else:
                    lines.append(
                        f"  - {DATASET_LABELS[ds]}: all available pairs ρ≥0.5"
                    )
        lines.append("")
        lines.append(
            "High ρ means method **ordering** is similar to ResNet-18; "
            "it does not imply equal metric values."
        )

    # Missing inventory
    miss = corr[corr.status != "ok"]
    lines += [
        "",
        "## Missing / unavailable comparisons",
        "",
        f"- Unavailable or insufficient cells: **{len(miss)}** / {len(corr)}",
    ]
    for _, r in miss.iterrows():
        lines.append(
            f"- {r.dataset_label} · {r.metric_label} · {r.pair_label}: {r.note}"
        )
    lines += [
        "",
        "## Files",
        "",
        "- `outputs/joint_structural_cross_backbone_ranks.csv` (method scores)",
        "- `outputs/joint_structural_cross_backbone_rank_consistency.csv`",
        "- `outputs/joint_structural_cross_backbone_rank_consistency.tex`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    scores, coverage = build_method_scores()
    scores_path = OUT / "joint_structural_cross_backbone_ranks.csv"
    scores.to_csv(scores_path, index=False)
    print(f"[write] {scores_path} ({len(scores)} rows)")
    print("[coverage]", coverage)

    corr = spearman_table(scores, coverage)
    corr_path = OUT / "joint_structural_cross_backbone_rank_consistency.csv"
    corr.to_csv(corr_path, index=False)
    print(f"[write] {corr_path} ({len(corr)} rows)")

    tex_path = OUT / "joint_structural_cross_backbone_rank_consistency.tex"
    write_latex(corr, tex_path)
    print(f"[write] {tex_path}")

    md_path = OUT / "joint_structural_cross_backbone_rank_consistency.md"
    write_findings(corr, coverage, scores, md_path)
    print(f"[write] {md_path}")

    # Preview
    show = corr[corr.status == "ok"][
        ["dataset", "metric_label", "compare_backbone", "spearman_rho", "n_methods"]
    ]
    print(show.to_string(index=False))
    print("[DONE]")


if __name__ == "__main__":
    main()
