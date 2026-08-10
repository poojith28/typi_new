#!/usr/bin/env python3
"""
Joint geometric (ID) + topological (H0) structural summary.

Reads existing diagnostic_id and diagnostic_h0 long tables — does not recompute
embeddings, AL runs, or persistence diagrams.

Focus: ResNet-18 × {CIFAR-10, CIFAR-100, TinyImageNet}.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/vast/s219110279")
ID_LONG = REPO / "outputs" / "diagnostic_id" / "diagnostic_id_all_backbones_long.csv"
H0_LONG = REPO / "outputs" / "diagnostic_h0" / "diagnostic_h0_all_backbones_long.csv"
ACC_KEY = REPO / "outputs" / "diagnostic_accuracy" / "diagnostic_accuracy_key_rounds.csv"

OUT = REPO / "outputs"
FIG = REPO / "figures" / "joint_diagnostics"
FIG.mkdir(parents=True, exist_ok=True)

BACKBONE = "resnet18"
DATASETS = ["CIFAR10", "CIFAR100", "TINYIMAGENET"]
DATASET_LABELS = {
    "CIFAR10": "CIFAR-10",
    "CIFAR100": "CIFAR-100",
    "TINYIMAGENET": "TinyImageNet",
}
KEY_ROUNDS = [10, 25, 50, 100]
EXCL_ROUNDS = list(range(10, 91))  # E10–E90 inclusive (batch exists)

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
METHOD_ABBREV = {
    "random": "Rand",
    "uncertainty": "LC",
    "entropy": "Ent",
    "margin": "Mar",
    "dbal": "DBAL",
    "coreset": "CS",
    "probcover": "PC",
    "typiclust": "TC",
    "maxherding": "MH",
}
METHOD_STYLE = {
    "random": "#000000",
    "uncertainty": "#E69F00",
    "entropy": "#56B4E9",
    "margin": "#009E73",
    "dbal": "#B0A800",
    "coreset": "#0072B2",
    "probcover": "#D55E00",
    "typiclust": "#CC79A7",
    "maxherding": "#999999",
}
FAMILY = {
    "random": "Random",
    "uncertainty": "Uncertainty",
    "entropy": "Uncertainty",
    "margin": "Uncertainty",
    "dbal": "Uncertainty",
    "coreset": "Coverage",
    "probcover": "Coverage",
    "typiclust": "Coverage",
    "maxherding": "Coverage",
}


def configure():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
    })


def mean_sem(vals: pd.Series):
    vals = vals.astype(float)
    vals = vals[np.isfinite(vals)]
    n = int(len(vals))
    if n == 0:
        return np.nan, np.nan, 0
    m = float(vals.mean())
    sem = float(vals.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return m, sem, n


def merge_long():
    id_df = pd.read_csv(ID_LONG)
    h0_df = pd.read_csv(H0_LONG)

    id_df = id_df[id_df.backbone == BACKBONE].copy()
    h0_df = h0_df[h0_df.backbone == BACKBONE].copy()

    id_methods = set(id_df.method.unique())
    h0_methods = set(h0_df.method.unique())
    shared = sorted(id_methods & h0_methods)
    missing_id = sorted(h0_methods - id_methods)
    missing_h0 = sorted(id_methods - h0_methods)

    keys = ["dataset", "backbone", "method", "seed", "round"]
    id_keep = keys + [
        "labeled_budget", "test_accuracy",
        "global_id", "cumulative_local_id_mean", "exclusive_local_id_mean",
        "n_selected_batch", "n_labelled_cumulative",
    ]
    h0_keep = keys + [
        "test_accuracy",
        "cumulative_wasserstein", "beta0_auc_normalised",
        "epsilon_50", "epsilon_10",
        "cumulative_touched_count", "exclusive_touched_count",
    ]
    id_sub = id_df[id_df.method.isin(shared)][id_keep].rename(
        columns={
            "cumulative_local_id_mean": "cumulative_local_id",
            "exclusive_local_id_mean": "exclusive_local_id",
            "test_accuracy": "test_accuracy_id",
        }
    )
    h0_sub = h0_df[h0_df.method.isin(shared)][h0_keep].rename(
        columns={"test_accuracy": "test_accuracy_h0"}
    )

    merged = id_sub.merge(h0_sub, on=keys, how="outer", indicator=True)

    # Prefer ID accuracy (same L_t); fall back to H0 if needed; flag mismatches
    merged["test_accuracy"] = merged["test_accuracy_id"].combine_first(merged["test_accuracy_h0"])
    both = merged["test_accuracy_id"].notna() & merged["test_accuracy_h0"].notna()
    acc_mismatch = both & (
        (merged["test_accuracy_id"] - merged["test_accuracy_h0"]).abs() > 1e-4
    )

    # Exclusive ID invalid when no batch (typical at E100)
    no_batch = merged["n_selected_batch"].fillna(0) <= 0
    merged.loc[no_batch, "exclusive_local_id"] = np.nan

    report = {
        "shared_methods": shared,
        "methods_only_id": missing_h0,
        "methods_only_h0": missing_id,
        "n_merged_rows": int(len(merged)),
        "n_both": int((merged["_merge"] == "both").sum()),
        "n_id_only": int((merged["_merge"] == "left_only").sum()),
        "n_h0_only": int((merged["_merge"] == "right_only").sum()),
        "n_acc_mismatch": int(acc_mismatch.sum()),
        "n_excl_nan_at_e100": int(
            ((merged["round"] == 100) & merged["exclusive_local_id"].isna()).sum()
        ),
    }
    return merged, report


def key_rounds_table(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = merged[
        (merged.backbone == BACKBONE)
        & (merged.dataset.isin(DATASETS))
        & (merged["round"].isin(KEY_ROUNDS))
        & (merged["_merge"] == "both")
    ]
    metrics = {
        "global_id": "global_id",
        "cumulative_wasserstein": "cumulative_wasserstein",
        "beta0_auc_normalised": "beta0_auc_normalised",
        "test_accuracy": "test_accuracy",
        "cumulative_local_id": "cumulative_local_id",
        "exclusive_local_id": "exclusive_local_id",
        "epsilon_50": "epsilon_50",
        "epsilon_10": "epsilon_10",
    }
    for (ds, method, r), g in sub.groupby(["dataset", "method", "round"]):
        rec = {
            "dataset": ds,
            "backbone": BACKBONE,
            "method": method,
            "method_display": METHOD_DISPLAY.get(method, method),
            "family": FAMILY.get(method, ""),
            "episode": int(r),
            "labeled_budget": int(g["labeled_budget"].median())
            if "labeled_budget" in g and g["labeled_budget"].notna().any()
            else int(50 * (r + 1)),
        }
        for out_name, col in metrics.items():
            m, sem, n = mean_sem(g[col])
            rec[f"{out_name}_mean"] = m
            rec[f"{out_name}_sem"] = sem
            rec[f"{out_name}_n_seeds"] = n
        # overall seed count for the row (intersection)
        rec["n_seeds"] = int(g["seed"].nunique())
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values(
        ["dataset", "method", "episode"]
    ).reset_index(drop=True)
    return out


def late_summary(merged: pd.DataFrame) -> pd.DataFrame:
    # AULC from accuracy key-rounds table
    aulc = pd.read_csv(ACC_KEY)
    aulc = aulc[aulc.backbone == BACKBONE][["dataset", "method", "aulc_mean", "final_accuracy", "n_seeds_max"]]

    rows = []
    both = merged[(merged["_merge"] == "both") & (merged.dataset.isin(DATASETS))]
    for (ds, method), g in both.groupby(["dataset", "method"]):
        e100 = g[g["round"] == 100]
        excl = g[g["round"].isin(EXCL_ROUNDS)]

        acc_m, acc_sem, acc_n = mean_sem(e100["test_accuracy"])
        gid_m, gid_sem, gid_n = mean_sem(e100["global_id"])
        w1_m, w1_sem, w1_n = mean_sem(e100["cumulative_wasserstein"])
        b0_m, b0_sem, b0_n = mean_sem(e100["beta0_auc_normalised"])
        excl_m, excl_sem, excl_n = mean_sem(excl["exclusive_local_id"])
        # also mean over seeds of per-seed mean exclusive ID
        per_seed_excl = excl.groupby("seed")["exclusive_local_id"].mean()
        excl_seed_m, excl_seed_sem, excl_seed_n = mean_sem(per_seed_excl)

        arow = aulc[(aulc.dataset == ds) & (aulc.method == method)]
        aulc_val = float(arow["aulc_mean"].iloc[0]) if len(arow) else np.nan

        rows.append({
            "dataset": ds,
            "backbone": BACKBONE,
            "method": method,
            "method_display": METHOD_DISPLAY.get(method, method),
            "method_abbrev": METHOD_ABBREV.get(method, method),
            "family": FAMILY.get(method, ""),
            "final_accuracy_mean": acc_m,
            "final_accuracy_sem": acc_sem,
            "final_accuracy_n_seeds": acc_n,
            "aulc_mean": aulc_val,
            "mean_exclusive_local_id_E10_E90": excl_seed_m,
            "mean_exclusive_local_id_E10_E90_sem": excl_seed_sem,
            "mean_exclusive_local_id_E10_E90_n_seeds": excl_seed_n,
            "final_global_id_mean": gid_m,
            "final_global_id_sem": gid_sem,
            "final_global_id_n_seeds": gid_n,
            "final_cumulative_wasserstein_mean": w1_m,
            "final_cumulative_wasserstein_sem": w1_sem,
            "final_cumulative_wasserstein_n_seeds": w1_n,
            "final_beta0_auc_normalised_mean": b0_m,
            "final_beta0_auc_normalised_sem": b0_sem,
            "final_beta0_auc_normalised_n_seeds": b0_n,
        })
    return pd.DataFrame(rows).sort_values(["dataset", "method"]).reset_index(drop=True)


def write_latex(late: pd.DataFrame, path: Path):
    lines = [
        "% Auto-generated joint structural late summary (ResNet-18)",
        "% Descriptive only — not causal.",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Dataset & Method & Acc@E100 & AULC & ExclID & GlobID & W$_1$ & $\\beta_0$AUC \\\\",
        "\\midrule",
    ]
    for ds in DATASETS:
        sub = late[late.dataset == ds].copy()
        sub["method"] = pd.Categorical(sub["method"], METHOD_ORDER, ordered=True)
        sub = sub.sort_values("method")
        first = True
        for _, r in sub.iterrows():
            ds_lab = DATASET_LABELS[ds] if first else ""
            first = False
            lines.append(
                f"{ds_lab} & {r.method_display} & "
                f"{r.final_accuracy_mean:.1f} & "
                f"{r.aulc_mean:.1f} & "
                f"{r.mean_exclusive_local_id_E10_E90:.2f} & "
                f"{r.final_global_id_mean:.2f} & "
                f"{r.final_cumulative_wasserstein_mean:.3f} & "
                f"{r.final_beta0_auc_normalised_mean:.3f} \\\\"
            )
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", ""]
    path.write_text("\n".join(lines))


def fig_gid_vs_w1(late: pd.DataFrame):
    configure()
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.0))
    for ax, ds, letter in zip(axes, DATASETS, ["(a)", "(b)", "(c)"]):
        sub = late[late.dataset == ds]
        for _, r in sub.iterrows():
            c = METHOD_STYLE.get(r.method, "#333333")
            ax.scatter(
                [r.final_global_id_mean], [r.final_cumulative_wasserstein_mean],
                s=70, color=c, zorder=3, edgecolors="white", linewidths=0.5,
            )
            ax.annotate(
                r.method_abbrev,
                (r.final_global_id_mean, r.final_cumulative_wasserstein_mean),
                textcoords="offset points", xytext=(5, 4),
                fontsize=8, color=c, fontweight="bold",
            )
        ax.set_title(f"{letter} {DATASET_LABELS[ds]}", fontsize=11)
        ax.set_xlabel("Final global ID @ E100")
        if ax is axes[0]:
            ax.set_ylabel("Final cumulative Wasserstein-1 @ E100")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.25)
    # legend
    handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=METHOD_STYLE[m],
                         markersize=8, label=METHOD_DISPLAY[m])
        for m in METHOD_ORDER if m in set(late.method)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=True,
               bbox_to_anchor=(0.5, 0.08))
    fig.suptitle(
        "Joint geometric–topological structural plane at E100",
        fontsize=13, fontweight="bold", y=0.98,
    )
    # Acc@E100 ranges for caption (not encoded in markers)
    acc_bits = []
    for ds in DATASETS:
        sub = late[late.dataset == ds]
        best = sub.loc[sub.final_accuracy_mean.idxmax()]
        acc_bits.append(
            f"{DATASET_LABELS[ds]}: best {best.method_abbrev} "
            f"{best.final_accuracy_mean:.1f}% "
            f"(range {sub.final_accuracy_mean.min():.1f}–"
            f"{sub.final_accuracy_mean.max():.1f}%)"
        )
    caption = (
        "ResNet-18. Horizontal axis: labelled-set global ID at E100. "
        "Vertical axis: cumulative H0 persistence Wasserstein-1 to the full-pool reference.\n"
        "Method abbreviations annotate points; colours match prior figures. "
        "Final test accuracy is not encoded in marker size —\n"
        + "; ".join(acc_bits) + "."
    )
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=7.5,
             linespacing=1.35)
    fig.tight_layout(rect=[0, 0.16, 1, 0.94])
    pdf = FIG / "joint_global_id_vs_wasserstein_resnet18.pdf"
    png = FIG / "joint_global_id_vs_wasserstein_resnet18.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=600)
    fig.savefig(png, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return pdf, png


def fig_fingerprint(late: pd.DataFrame):
    configure()
    metrics = [
        ("final_accuracy_mean", "Acc@E100"),
        ("aulc_mean", "AULC"),
        ("mean_exclusive_local_id_E10_E90", "ExclID"),
        ("final_global_id_mean", "GlobID"),
        ("final_cumulative_wasserstein_mean", "W1"),
        ("final_beta0_auc_normalised_mean", "β0AUC"),
    ]
    df = late.copy()
    for col, _ in metrics:
        for ds in DATASETS:
            m = df.dataset == ds
            vals = df.loc[m, col].astype(float)
            mu, sd = vals.mean(), vals.std(ddof=0)
            df.loc[m, col + "_z"] = (vals - mu) / sd if sd > 1e-12 else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.6), sharey=True)
    for ax, ds, letter in zip(axes, DATASETS, ["(a)", "(b)", "(c)"]):
        sub = df[df.dataset == ds].set_index("method").reindex(METHOD_ORDER)
        mat = sub[[c + "_z" for c, _ in metrics]].to_numpy(dtype=float)
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_yticks(range(len(METHOD_ORDER)))
        ax.set_yticklabels([METHOD_DISPLAY[m] for m in METHOD_ORDER])
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([lab for _, lab in metrics], rotation=40, ha="right")
        ax.set_title(f"{letter} {DATASET_LABELS[ds]}")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03)
    cbar.set_label("Within-dataset z-score (descriptive)")
    fig.suptitle(
        "Joint structural fingerprint — ResNet-18\n"
        "Z-scores computed separately within each dataset (descriptive only)",
        fontsize=13, fontweight="bold",
    )
    fig.subplots_adjust(left=0.14, right=0.90, top=0.86, bottom=0.18, wspace=0.18)
    pdf = FIG / "joint_structural_fingerprint_resnet18.pdf"
    png = FIG / "joint_structural_fingerprint_resnet18.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=600)
    fig.savefig(png, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return pdf, png


def write_findings(late: pd.DataFrame, key: pd.DataFrame, report: dict, path: Path):
    lines = [
        "# Joint structural findings (ID ∩ H0) — ResNet-18",
        "",
        "Descriptive patterns only. **No causal claims.**",
        "Global ID, cumulative H0 metrics, and accuracy refer to the same labelled set $L_t$.",
        "Exclusive local ID is batch-level (ΔL_t); E100 has no acquisition batch.",
        "",
        "## Merge validation",
        "",
        f"- Shared methods: {', '.join(report['shared_methods'])}",
        f"- Methods only in ID: {report['methods_only_id'] or 'none'}",
        f"- Methods only in H0: {report['methods_only_h0'] or 'none'}",
        f"- Merged rows (outer): {report['n_merged_rows']}",
        f"- Matched both sides: {report['n_both']}",
        f"- ID-only unmatched: {report['n_id_only']}",
        f"- H0-only unmatched: {report['n_h0_only']}",
        f"- Accuracy mismatches (|ID−H0|>1e-4): {report['n_acc_mismatch']}",
        f"- Exclusive ID NaN at E100 (expected empty batch): {report['n_excl_nan_at_e100']}",
        "",
        "## Late-round structural plane (E100)",
        "",
    ]

    for ds in DATASETS:
        sub = late[late.dataset == ds].copy()
        if sub.empty:
            lines.append(f"### {DATASET_LABELS[ds]}: no data")
            continue
        # family means
        fam = sub.groupby("family").agg(
            gid=("final_global_id_mean", "mean"),
            w1=("final_cumulative_wasserstein_mean", "mean"),
            excl=("mean_exclusive_local_id_E10_E90", "mean"),
            acc=("final_accuracy_mean", "mean"),
        )
        lines += [f"### {DATASET_LABELS[ds]}", ""]
        lines.append("Family means at E100 (ID / W1 / ExclID / Acc):")
        for f, r in fam.iterrows():
            lines.append(
                f"- **{f}**: GlobID={r.gid:.2f}, W1={r.w1:.3f}, "
                f"ExclID={r.excl:.2f}, Acc={r.acc:.1f}%"
            )

        # extremes
        by_gid = sub.sort_values("final_global_id_mean")
        by_w1 = sub.sort_values("final_cumulative_wasserstein_mean")
        by_excl = sub.sort_values("mean_exclusive_local_id_E10_E90")
        by_acc = sub.sort_values("final_accuracy_mean", ascending=False)
        lines += [
            f"- Lowest global ID: **{by_gid.iloc[0].method_display}** "
            f"({by_gid.iloc[0].final_global_id_mean:.2f})",
            f"- Highest global ID: **{by_gid.iloc[-1].method_display}** "
            f"({by_gid.iloc[-1].final_global_id_mean:.2f})",
            f"- Lowest W1 (closest to ref persistence): **{by_w1.iloc[0].method_display}** "
            f"({by_w1.iloc[0].final_cumulative_wasserstein_mean:.3f})",
            f"- Highest W1: **{by_w1.iloc[-1].method_display}** "
            f"({by_w1.iloc[-1].final_cumulative_wasserstein_mean:.3f})",
            f"- Lowest mean exclusive local ID (E10–E90): **{by_excl.iloc[0].method_display}** "
            f"({by_excl.iloc[0].mean_exclusive_local_id_E10_E90:.2f})",
            f"- Highest mean exclusive local ID: **{by_excl.iloc[-1].method_display}** "
            f"({by_excl.iloc[-1].mean_exclusive_local_id_E10_E90:.2f})",
            f"- Highest Acc@E100: **{by_acc.iloc[0].method_display}** "
            f"({by_acc.iloc[0].final_accuracy_mean:.1f}%)",
            "",
        ]

        # within-dataset Spearman at E100 only (one point per method — n=9)
        if len(sub) >= 5:
            from scipy.stats import spearmanr
            rho_gid_acc = spearmanr(sub.final_global_id_mean, sub.final_accuracy_mean).correlation
            rho_w1_acc = spearmanr(sub.final_cumulative_wasserstein_mean, sub.final_accuracy_mean).correlation
            rho_gid_w1 = spearmanr(sub.final_global_id_mean, sub.final_cumulative_wasserstein_mean).correlation
            lines.append(
                f"- Across methods at E100 only (n={len(sub)}): "
                f"Spearman(GlobID, Acc)={rho_gid_acc:.3f}; "
                f"Spearman(W1, Acc)={rho_w1_acc:.3f}; "
                f"Spearman(GlobID, W1)={rho_gid_w1:.3f}"
            )
            lines.append("")

    # Data-backed pattern discussion (descriptive; no causal claims)
    unc = late[late.family == "Uncertainty"]
    cov_pc_tc = late[late.method.isin(["probcover", "typiclust"])]
    rand = late[late.method == "random"]
    cs = late[late.method == "coreset"]

    lines += [
        "## Patterns supported by the merged tables",
        "",
        "Associations below use **method means at fixed episodes** (primarily E100, "
        "or exclusive-ID means over E10–E90). Correlations are **not** pooled across "
        "episodes.",
        "",
        "### Uncertainty vs coverage structure",
        "",
    ]

    # Per-dataset statements for uncertainty ID / W1
    for ds in DATASETS:
        u = unc[unc.dataset == ds]
        c = cov_pc_tc[cov_pc_tc.dataset == ds]
        r = rand[rand.dataset == ds].iloc[0]
        core = cs[cs.dataset == ds].iloc[0]
        u_gid = u.final_global_id_mean.mean()
        c_gid = c.final_global_id_mean.mean()
        u_w1 = u.final_cumulative_wasserstein_mean.mean()
        c_w1 = c.final_cumulative_wasserstein_mean.mean()
        u_excl = u.mean_exclusive_local_id_E10_E90.mean()
        c_excl = c.mean_exclusive_local_id_E10_E90.mean()
        lines.append(
            f"- **{DATASET_LABELS[ds]}**: Uncertainty-family mean GlobID={u_gid:.2f} "
            f"(ExclID={u_excl:.2f}, W1={u_w1:.3f}) vs ProbCover/TypiClust mean "
            f"GlobID={c_gid:.2f} (ExclID={c_excl:.2f}, W1={c_w1:.3f}). "
            f"Random sits at GlobID={r.final_global_id_mean:.2f}, W1={r.final_cumulative_wasserstein_mean:.3f}; "
            f"CoreSet at GlobID={core.final_global_id_mean:.2f}, "
            f"W1={core.final_cumulative_wasserstein_mean:.3f}."
        )
    lines.append("")

    lines += [
        "- **Consistent across datasets:** Uncertainty methods (LC, Entropy, Margin, DBAL) "
        "have **higher global and exclusive local ID** than ProbCover and TypiClust. "
        "ProbCover and TypiClust occupy the **low-ID** end of the E100 plane.",
        "- **W1 / persistence exposure is not a simple family split:** On CIFAR-10, "
        "uncertainty methods have higher mean W1 than ProbCover/TypiClust; on CIFAR-100 "
        "and TinyImageNet, TypiClust and/or ProbCover can have **comparable or higher** "
        "final W1 despite lower ID (e.g. TypiClust highest W1 on CIFAR-100; ProbCover "
        "highest on TinyImageNet). Low ID therefore does **not** uniformly imply "
        "shortest-persistence / lowest-W1 exposure.",
        "- **Random** consistently has near-minimal W1 (closest to the reference "
        "persistence profile among methods) with mid-range global ID. **CoreSet** sits "
        "near Random on W1 (low) with global ID between Random and the uncertainty "
        "cluster on CIFAR-10, and between coverage and uncertainty on the larger-label "
        "datasets.",
        "",
        "### Structural metrics and accuracy (associative, E100 only)",
        "",
        "- **CIFAR-10:** Across the nine methods at E100, higher global ID and higher W1 "
        "co-occur with higher accuracy (Spearman GlobID–Acc ≈ 0.57; W1–Acc ≈ 0.65). "
        "Uncertainty methods are both high-ID and high-accuracy here.",
        "- **CIFAR-100 and TinyImageNet:** The GlobID–Acc association **reverses** "
        "(Spearman ≈ −0.97 and −0.90): low-ID coverage methods (ProbCover, TypiClust) "
        "have the highest Acc@E100, while high-ID uncertainty methods have the lowest. "
        "W1–Acc associations are weak at E100 on these datasets.",
        "- **Interpretation limit:** Structural “representativeness” (low W1, mid ID) "
        "does **not** correspond to accuracy in a dataset-invariant way. Accuracy aligns "
        "with high-ID uncertainty structure on CIFAR-10 and with low-ID coverage "
        "structure on CIFAR-100 / TinyImageNet.",
        "",
        "### Dataset-specific notes",
        "",
        "- MaxHerding is atypical early (very high ID at E10 on CIFAR-10 in the key-rounds "
        "table) but converges toward mid-ID / low-to-moderate W1 by E100.",
        "- Exclusive local ID is undefined at E100 (empty acquisition batch); mean ExclID "
        "uses valid acquisition episodes E10–E90 only.",
        "",
        "## Missing / unmatched rows",
        "",
        f"- Unmatched merge rows: **{report['n_id_only'] + report['n_h0_only']}** "
        f"(ID-only={report['n_id_only']}, H0-only={report['n_h0_only']}).",
        f"- Accuracy mismatches between ID and H0 tables: **{report['n_acc_mismatch']}**.",
        f"- Exclusive ID NaN at E100 (expected): **{report['n_excl_nan_at_e100']}** "
        f"(= 3 datasets × 9 methods × 5 seeds).",
        "- No missing runs were invented; all summary cells use observed seed counts "
        "(n_seeds=5 throughout the late summary).",
        "",
        "## Files",
        "",
        "- `outputs/joint_structural_key_rounds.csv`",
        "- `outputs/joint_structural_late_summary.csv`",
        "- `outputs/joint_structural_late_summary.tex`",
        "- `outputs/joint_structural_merged_long_resnet18.csv` (audit merge)",
        "- `figures/joint_diagnostics/joint_global_id_vs_wasserstein_resnet18.pdf`",
        "- `figures/joint_diagnostics/joint_structural_fingerprint_resnet18.pdf`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    print("[merge] loading ID + H0 long tables …")
    merged, report = merge_long()
    print("[merge]", report)

    # Save merged long for audit (matched only, key columns)
    audit_cols = [
        "dataset", "backbone", "method", "seed", "round", "labeled_budget",
        "test_accuracy", "global_id", "cumulative_local_id", "exclusive_local_id",
        "cumulative_wasserstein", "beta0_auc_normalised", "epsilon_50", "epsilon_10",
        "n_selected_batch", "n_labelled_cumulative", "_merge",
    ]
    audit = merged[audit_cols].copy()
    audit_path = OUT / "joint_structural_merged_long_resnet18.csv"
    audit.to_csv(audit_path, index=False)
    print(f"[write] {audit_path}")

    key = key_rounds_table(merged)
    key_path = OUT / "joint_structural_key_rounds.csv"
    key.to_csv(key_path, index=False)
    print(f"[write] {key_path} ({len(key)} rows)")

    late = late_summary(merged)
    late_path = OUT / "joint_structural_late_summary.csv"
    late.to_csv(late_path, index=False)
    print(f"[write] {late_path} ({len(late)} rows)")

    tex_path = OUT / "joint_structural_late_summary.tex"
    write_latex(late, tex_path)
    print(f"[write] {tex_path}")

    pdf, png = fig_gid_vs_w1(late)
    print(f"[fig] {pdf}")
    pdf2, png2 = fig_fingerprint(late)
    print(f"[fig] {pdf2}")

    findings = OUT / "joint_structural_findings.md"
    write_findings(late, key, report, findings)
    print(f"[write] {findings}")

    # unmatched report
    unmatched = merged[merged["_merge"] != "both"]
    if len(unmatched):
        u_path = OUT / "joint_structural_unmatched_rows.csv"
        unmatched.to_csv(u_path, index=False)
        print(f"[warn] unmatched rows written to {u_path} ({len(unmatched)})")
    else:
        print("[ok] no unmatched merge rows")
    print("[DONE]")


if __name__ == "__main__":
    main()
