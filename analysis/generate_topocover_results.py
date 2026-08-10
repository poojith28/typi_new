#!/usr/bin/env python3
"""
Aggregate FINAL TopoCover predictive-performance results (NOT LIDCOVER).

Source of truth: TOPOCOVER_PAPER_* runs under TypiClust/output/
Final main setting: TOPOCOVER_PAPER_D070 (δ=0.70, budget 50/round, seeds 1–5).

Produces CSVs, LaTeX, learning-curve figures, and a write-up report.
Does not invent missing runs.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/vast/s219110279")
OUT_ROOT = REPO / "TypiClust" / "output"
OUT = REPO / "outputs" / "topocover"
FIG = REPO / "figures" / "topocover"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

ACC_LONG = REPO / "outputs" / "diagnostic_accuracy" / "diagnostic_accuracy_all_backbones_long.csv"

DATASETS = ["CIFAR10", "CIFAR100", "TINYIMAGENET"]
DATASET_LABELS = {
    "CIFAR10": "CIFAR-10",
    "CIFAR100": "CIFAR-100",
    "TINYIMAGENET": "TinyImageNet",
}
BACKBONES = ["resnet18", "resnet50", "alexnet"]
BB_LABELS = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "alexnet": "AlexNet",
}
KEY_ROUNDS = [10, 25, 50, 100]
BASELINES = [
    "random", "uncertainty", "entropy", "margin", "dbal",
    "coreset", "probcover", "typiclust", "maxherding",
]
BASELINE_DISPLAY = {
    "random": "Random",
    "uncertainty": "Uncertainty (LC)",
    "entropy": "Entropy",
    "margin": "Margin",
    "dbal": "DBAL (BALD)",
    "coreset": "CoreSet",
    "probcover": "ProbCover",
    "typiclust": "TypiClust",
    "maxherding": "MaxHerding",
    "topocover": "TopoCover",
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
    "topocover": "#E41A1C",
}
EXP_RE = re.compile(
    r"^TOPOCOVER_PAPER_D(?P<d>\d+)_(?P<seed>\d+)_(?P<budget>\d+)b$"
)


def configure():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
    })


def mean_sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = int(len(vals))
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    m = float(vals.mean())
    sem = float(vals.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    sd = float(vals.std(ddof=1)) if n > 1 else 0.0
    return m, sem, sd, n


def discover_runs():
    rows = []
    missing = []
    for ds in DATASETS:
        for bb in BACKBONES:
            root = OUT_ROOT / ds / bb
            if not root.is_dir():
                continue
            for d in sorted(root.iterdir()):
                m = EXP_RE.match(d.name)
                if not m:
                    continue
                delta = int(m.group("d")) / 100.0
                seed = int(m.group("seed"))
                budget = int(m.group("budget"))
                summary = d / "benchmark_summary.json"
                rec = {
                    "dataset": ds,
                    "backbone": bb,
                    "exp_name": d.name,
                    "delta": delta,
                    "seed": seed,
                    "budget_per_round": budget,
                    "run_dir": str(d),
                    "has_summary": summary.is_file(),
                    "n_episodes": 0,
                    "complete": False,
                }
                if not summary.is_file():
                    # count episode dirs if any
                    eps = list(d.glob("episode_*"))
                    rec["n_episodes"] = len(eps)
                    missing.append({**rec, "reason": "no benchmark_summary.json"})
                    rows.append(rec)
                    continue
                data = json.loads(summary.read_text())
                eps = data.get("episode_records") or []
                rec["n_episodes"] = len(eps)
                rec["complete"] = len(eps) >= 101 and abs(
                    float(data.get("final_test_accuracy", np.nan))
                    - float(eps[-1]["test_accuracy"])
                ) < 1e-3 or len(eps) >= 101
                # Prefer explicit complete = 101 episodes
                rec["complete"] = len(eps) >= 101
                rec["final_test_accuracy"] = float(data.get("final_test_accuracy", np.nan))
                timing = data.get("timing") or {}
                acq = timing.get("acquisition_time_sec") or {}
                trn = timing.get("train_time_sec") or {}
                rec["acq_cumulative_sec"] = float(acq.get("cumulative_with_initial_sampling", acq.get("cumulative", np.nan)))
                rec["acq_mean_sec"] = float(acq.get("mean", np.nan))
                rec["train_cumulative_sec"] = float(trn.get("cumulative", np.nan))
                rows.append(rec)
                if not rec["complete"]:
                    missing.append({**rec, "reason": f"only {len(eps)} episodes"})
    return pd.DataFrame(rows), pd.DataFrame(missing)


def load_long(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in inventory.iterrows():
        if not r.has_summary:
            continue
        summary = Path(r.run_dir) / "benchmark_summary.json"
        data = json.loads(summary.read_text())
        for ep in data.get("episode_records") or []:
            rows.append({
                "dataset": r.dataset,
                "backbone": r.backbone,
                "method": "topocover",
                "method_display": "TopoCover",
                "exp_name": r.exp_name,
                "delta": float(r.delta),
                "seed": int(r.seed),
                "budget_per_round": int(r.budget_per_round),
                "round": int(ep["episode"]),
                "labeled_budget": int(ep["labeled_count_before_sampling"]),
                "test_accuracy": float(ep["test_accuracy"]),
                "acquisition_time_sec": float(ep.get("acquisition_time_sec", np.nan)),
                "train_time_sec": float(ep.get("train_time_sec", np.nan)),
                "complete_run": bool(r.complete),
            })
    return pd.DataFrame(rows)


def summarise_curve(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "backbone", "method", "delta", "budget_per_round", "round"]
    for key, g in long.groupby(keys):
        m, sem, sd, n = mean_sem(g["test_accuracy"])
        rows.append({
            "dataset": key[0],
            "backbone": key[1],
            "method": key[2],
            "method_display": "TopoCover",
            "delta": key[3],
            "budget_per_round": key[4],
            "round": key[5],
            "labeled_budget": int(g["labeled_budget"].mode().iloc[0]),
            "mean_accuracy": m,
            "sem_accuracy": sem,
            "std_accuracy": sd,
            "n_seeds": n,
        })
    return pd.DataFrame(rows).sort_values(keys)


def key_rounds_table(summary: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset × backbone × delta × budget; Acc@key rounds + AULC."""
    rows = []
    group_cols = ["dataset", "backbone", "delta", "budget_per_round"]
    for key, gsum in summary.groupby(group_cols):
        ds, bb, delta, budget = key
        # AULC = mean over episodes of seed-mean accuracy (match diagnostic_accuracy)
        aulc = float(gsum["mean_accuracy"].mean())
        # Also seed-level AULC then mean±SEM
        glong = long[
            (long.dataset == ds)
            & (long.backbone == bb)
            & (long.delta == delta)
            & (long.budget_per_round == budget)
        ]
        seed_aulc = glong.groupby("seed")["test_accuracy"].mean()
        aulc_m, aulc_sem, aulc_sd, aulc_n = mean_sem(seed_aulc)
        rec = {
            "dataset": ds,
            "backbone": bb,
            "method": "topocover",
            "method_display": "TopoCover",
            "delta": delta,
            "budget_per_round": budget,
            "aulc_mean": aulc_m,
            "aulc_sem": aulc_sem,
            "aulc_std": aulc_sd,
            "aulc_curve_mean": aulc,
            "n_seeds": int(aulc_n),
        }
        for ep in KEY_ROUNDS:
            sub = gsum[gsum["round"] == ep]
            if sub.empty:
                rec[f"acc_round_{ep}"] = np.nan
                rec[f"sem_round_{ep}"] = np.nan
                rec[f"std_round_{ep}"] = np.nan
                rec[f"budget_round_{ep}"] = np.nan
                rec[f"n_seeds_round_{ep}"] = 0
            else:
                r = sub.iloc[0]
                rec[f"acc_round_{ep}"] = float(r.mean_accuracy)
                rec[f"sem_round_{ep}"] = float(r.sem_accuracy)
                rec[f"std_round_{ep}"] = float(r.std_accuracy)
                rec[f"budget_round_{ep}"] = int(r.labeled_budget)
                rec[f"n_seeds_round_{ep}"] = int(r.n_seeds)
        # final = E100
        rec["final_accuracy"] = rec.get("acc_round_100", np.nan)
        rec["final_sem"] = rec.get("sem_round_100", np.nan)
        rec["final_std"] = rec.get("std_round_100", np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def runtime_table(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = inventory[inventory.complete].copy()
    for key, g in sub.groupby(["dataset", "backbone", "delta", "budget_per_round"]):
        acq_m, acq_sem, acq_sd, n = mean_sem(g["acq_cumulative_sec"])
        trn_m, trn_sem, trn_sd, _ = mean_sem(g["train_cumulative_sec"])
        acq_mean_m, acq_mean_sem, acq_mean_sd, _ = mean_sem(g["acq_mean_sec"])
        rows.append({
            "dataset": key[0],
            "backbone": key[1],
            "delta": key[2],
            "budget_per_round": key[3],
            "n_seeds": n,
            "acq_cumulative_sec_mean": acq_m,
            "acq_cumulative_sec_sem": acq_sem,
            "acq_cumulative_sec_std": acq_sd,
            "acq_mean_sec_mean": acq_mean_m,
            "acq_mean_sec_sem": acq_mean_sem,
            "train_cumulative_sec_mean": trn_m,
            "train_cumulative_sec_sem": trn_sem,
            "train_cumulative_sec_std": trn_sd,
        })
    return pd.DataFrame(rows)


def write_main_latex(key: pd.DataFrame, path: Path):
    main = key[(key.delta == 0.7) & (key.budget_per_round == 50)].copy()
    lines = [
        "% TopoCover final predictive performance (δ=0.70, budget 50/round).",
        "% Mean ± SEM across seeds. Missing seeds are not invented.",
        "\\begin{tabular}{lllccccc}",
        "\\toprule",
        "Dataset & Backbone & $n$ & Acc@E10 & Acc@E25 & Acc@E50 & Acc@E100 & AULC \\\\",
        "\\midrule",
    ]
    for ds in DATASETS:
        first = True
        for bb in BACKBONES:
            sub = main[(main.dataset == ds) & (main.backbone == bb)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            ds_lab = DATASET_LABELS[ds] if first else ""
            first = False

            def cell(ep):
                a = r[f"acc_round_{ep}"]
                s = r[f"sem_round_{ep}"]
                n = int(r[f"n_seeds_round_{ep}"])
                if n == 0 or not np.isfinite(a):
                    return "---"
                return f"{a:.1f}$\\pm${s:.1f}"

            lines.append(
                f"{ds_lab} & {BB_LABELS[bb]} & {int(r.n_seeds)} & "
                f"{cell(10)} & {cell(25)} & {cell(50)} & {cell(100)} & "
                f"{r.aulc_mean:.1f}$\\pm${r.aulc_sem:.1f} \\\\"
            )
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", ""]
    path.write_text("\n".join(lines))


def write_ablation_latex(key: pd.DataFrame, path: Path):
    abl = key[
        (key.backbone == "resnet18")
        & (key.budget_per_round == 50)
        & (key.delta.isin([0.6, 0.7, 0.8]))
    ].copy()
    lines = [
        "% TopoCover δ scale ablation (ResNet-18, budget 50/round).",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Dataset & $\\delta$ & Acc@E25 & Acc@E50 & Acc@E100 & AULC \\\\",
        "\\midrule",
    ]
    for ds in DATASETS:
        first = True
        for delta in [0.6, 0.7, 0.8]:
            sub = abl[(abl.dataset == ds) & (np.isclose(abl.delta, delta))]
            if sub.empty:
                continue
            r = sub.iloc[0]
            ds_lab = DATASET_LABELS[ds] if first else ""
            first = False

            def cell(ep):
                a = r[f"acc_round_{ep}"]
                s = r[f"sem_round_{ep}"]
                if not np.isfinite(a):
                    return "---"
                return f"{a:.1f}$\\pm${s:.1f}"

            lines.append(
                f"{ds_lab} & {delta:.2f} & {cell(25)} & {cell(50)} & {cell(100)} & "
                f"{r.aulc_mean:.1f}$\\pm${r.aulc_sem:.1f} \\\\"
            )
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", ""]
    path.write_text("\n".join(lines))


def fig_learning_curves(summary_tc: pd.DataFrame, backbone: str = "resnet18"):
    """TopoCover (δ=0.7) vs diagnostic baselines for one backbone."""
    configure()
    bas = pd.read_csv(ACC_LONG)
    bas = bas[(bas.backbone == backbone) & (bas.method.isin(BASELINES))]
    tc = summary_tc[
        (summary_tc.backbone == backbone)
        & (np.isclose(summary_tc.delta, 0.7))
        & (summary_tc.budget_per_round == 50)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharey=False)
    for ax, ds, letter in zip(axes, DATASETS, ["(a)", "(b)", "(c)"]):
        for method in BASELINES:
            sub = bas[(bas.dataset == ds) & (bas.method == method)].sort_values("round")
            if sub.empty:
                continue
            # aggregate seeds
            g = sub.groupby("round")["test_accuracy"].agg(["mean", "sem"])
            ax.plot(
                g.index, g["mean"], color=METHOD_STYLE[method],
                linewidth=1.2, alpha=0.85, label=BASELINE_DISPLAY[method],
            )
        sub = tc[tc.dataset == ds].sort_values("round")
        if not sub.empty:
            ax.plot(
                sub["round"], sub["mean_accuracy"],
                color=METHOD_STYLE["topocover"], linewidth=2.2,
                label="TopoCover (δ=0.70)", zorder=5,
            )
            ax.fill_between(
                sub["round"],
                sub["mean_accuracy"] - sub["sem_accuracy"],
                sub["mean_accuracy"] + sub["sem_accuracy"],
                color=METHOD_STYLE["topocover"], alpha=0.18, zorder=4,
            )
        ax.set_title(f"{letter} {DATASET_LABELS[ds]}")
        ax.set_xlabel("Episode")
        if ax is axes[0]:
            ax.set_ylabel("Test accuracy (%)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=True,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"TopoCover learning curves vs baselines — {BB_LABELS[backbone]}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.96])
    stem = f"topocover_learning_curves_{backbone}"
    pdf, png = FIG / f"{stem}.pdf", FIG / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=600)
    fig.savefig(png, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return pdf, png


def fig_delta_ablation(summary_tc: pd.DataFrame):
    configure()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    colors = {0.6: "#1B9E77", 0.7: "#E41A1C", 0.8: "#7570B3"}
    for ax, ds, letter in zip(axes, DATASETS, ["(a)", "(b)", "(c)"]):
        for delta in [0.6, 0.7, 0.8]:
            sub = summary_tc[
                (summary_tc.dataset == ds)
                & (summary_tc.backbone == "resnet18")
                & (summary_tc.budget_per_round == 50)
                & (np.isclose(summary_tc.delta, delta))
            ].sort_values("round")
            if sub.empty:
                continue
            ax.plot(
                sub["round"], sub["mean_accuracy"],
                color=colors[delta], linewidth=2.0,
                label=f"δ={delta:.2f}",
            )
            ax.fill_between(
                sub["round"],
                sub["mean_accuracy"] - sub["sem_accuracy"],
                sub["mean_accuracy"] + sub["sem_accuracy"],
                color=colors[delta], alpha=0.15,
            )
        ax.set_title(f"{letter} {DATASET_LABELS[ds]}")
        ax.set_xlabel("Episode")
        if ax is axes[0]:
            ax.set_ylabel("Test accuracy (%)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.25)
        ax.legend(frameon=True, fontsize=8)
    fig.suptitle(
        "TopoCover δ scale ablation — ResNet-18 (budget 50/round)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    pdf = FIG / "topocover_delta_ablation_resnet18.pdf"
    png = FIG / "topocover_delta_ablation_resnet18.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=600)
    fig.savefig(png, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return pdf, png


def write_report(inventory, missing, key, runtime, paths, path: Path):
    main = key[(key.delta == 0.7) & (key.budget_per_round == 50)]
    lines = [
        "# TopoCover predictive performance — final results",
        "",
        "**Method only: TopoCover** (not LIDCOVER).",
        "",
        "## Final setting",
        "",
        "- Experiment name: `TOPOCOVER_PAPER_D070_*_50b`",
        "- **δ (initial_delta) = 0.70**",
        "- kNN: `topocover_k_knn = 50`",
        "- Budget: **50 labels / round**, episodes 0–100 (labelled budget 50→5050)",
        "- Seeds: 1–5",
        "- Datasets: CIFAR-10, CIFAR-100, TinyImageNet",
        "- Backbones: ResNet-18, ResNet-50, AlexNet",
        "- Campaign: `typiclust_runs/Typiclust_runs/topocover_paper_individual_20260716/`",
        "",
        "## Baselines in the comparison figures",
        "",
        "From `outputs/diagnostic_accuracy/` (same AL protocol):",
        "Random, Uncertainty (LC), Entropy, Margin, DBAL (BALD), CoreSet, "
        "ProbCover, TypiClust, MaxHerding.",
        "",
        "## Completeness (main δ=0.70, 50b)",
        "",
    ]
    inv_main = inventory[
        (np.isclose(inventory.delta, 0.7)) & (inventory.budget_per_round == 50)
    ]
    for ds in DATASETS:
        for bb in BACKBONES:
            sub = inv_main[(inv_main.dataset == ds) & (inv_main.backbone == bb)]
            n_ok = int(sub.complete.sum())
            n_tot = int(len(sub))
            flag = "" if n_ok == 5 else " **INCOMPLETE**"
            lines.append(f"- {DATASET_LABELS[ds]} / {BB_LABELS[bb]}: {n_ok}/{n_tot} seeds{flag}")
    if len(missing):
        lines += ["", "### Missing / incomplete runs", ""]
        for _, r in missing.iterrows():
            lines.append(
                f"- `{r.dataset}/{r.backbone}/{r.exp_name}` — {r.reason}"
            )

    lines += ["", "## Final Acc@E100 and AULC (δ=0.70, mean±SEM)", ""]
    lines.append("| Dataset | Backbone | n | Acc@E100 | AULC |")
    lines.append("|---|---|---:|---:|---:|")
    for ds in DATASETS:
        for bb in BACKBONES:
            sub = main[(main.dataset == ds) & (main.backbone == bb)]
            if sub.empty:
                lines.append(f"| {DATASET_LABELS[ds]} | {BB_LABELS[bb]} | 0 | --- | --- |")
                continue
            r = sub.iloc[0]
            lines.append(
                f"| {DATASET_LABELS[ds]} | {BB_LABELS[bb]} | {int(r.n_seeds)} | "
                f"{r.final_accuracy:.2f}±{r.final_sem:.2f} | "
                f"{r.aulc_mean:.2f}±{r.aulc_sem:.2f} |"
            )

    lines += [
        "",
        "## Early / middle / late budgets (δ=0.70)",
        "",
        "Episodes E10 / E25 / E50 / E100 (labelled budgets typically 550 / 1300 / 2550 / 5050).",
        "See `topocover_key_rounds.csv` and `topocover_main_key_rounds.tex`.",
        "",
        "## δ scale ablation (ResNet-18 only)",
        "",
        "Raw runs: `TOPOCOVER_PAPER_D060`, `D070`, `D080` on CIFAR-10/100/TinyImageNet × ResNet-18.",
        "No TopoCover α / λ / multi-scale ablations exist in this workspace "
        "(those belong to LIDCOVER).",
        "",
        "## Runtime",
        "",
        "Per-run acquisition/train times aggregated in `topocover_runtime_summary.csv`.",
        "",
    ]
    if len(runtime):
        lines.append("| Dataset | Backbone | n | Acq cumulative (s) | Train cumulative (s) |")
        lines.append("|---|---|---:|---:|---:|")
        rt = runtime[(np.isclose(runtime.delta, 0.7)) & (runtime.budget_per_round == 50)]
        for _, r in rt.sort_values(["dataset", "backbone"]).iterrows():
            lines.append(
                f"| {DATASET_LABELS[r.dataset]} | {BB_LABELS[r.backbone]} | {int(r.n_seeds)} | "
                f"{r.acq_cumulative_sec_mean:.0f}±{r.acq_cumulative_sec_sem:.0f} | "
                f"{r.train_cumulative_sec_mean:.0f}±{r.train_cumulative_sec_sem:.0f} |"
            )

    lines += [
        "",
        "## Output files",
        "",
    ]
    for p in paths:
        lines.append(f"- `{p}`")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main():
    print("[discover] scanning TOPOCOVER_PAPER_* runs …")
    inventory, missing = discover_runs()
    inventory.to_csv(OUT / "topocover_run_inventory.csv", index=False)
    missing.to_csv(OUT / "topocover_missing_runs.csv", index=False)
    print(f"[inventory] {len(inventory)} runs; incomplete/missing={len(missing)}")

    long = load_long(inventory)
    # Use only complete runs for main summaries; still keep incomplete in inventory
    long_ok = long[long.complete_run].copy()
    long_ok.to_csv(OUT / "topocover_all_backbones_long.csv", index=False)
    print(f"[long] {len(long_ok)} episode rows from complete runs")

    summary = summarise_curve(long_ok)
    summary.to_csv(OUT / "topocover_all_backbones_summary.csv", index=False)

    key = key_rounds_table(summary, long_ok)
    key.to_csv(OUT / "topocover_key_rounds.csv", index=False)

    main_key = key[(key.delta == 0.7) & (key.budget_per_round == 50)].copy()
    main_key.to_csv(OUT / "topocover_main_d070_key_rounds.csv", index=False)

    abl = key[
        (key.backbone == "resnet18")
        & (key.budget_per_round == 50)
        & (key.delta.isin([0.6, 0.7, 0.8]))
    ].copy()
    abl.to_csv(OUT / "topocover_delta_ablation_resnet18.csv", index=False)

    runtime = runtime_table(inventory)
    runtime.to_csv(OUT / "topocover_runtime_summary.csv", index=False)

    write_main_latex(key, OUT / "topocover_main_key_rounds.tex")
    write_ablation_latex(key, OUT / "topocover_delta_ablation_resnet18.tex")

    pdf1, png1 = fig_learning_curves(summary, "resnet18")
    print(f"[fig] {pdf1}")
    pdf2, png2 = fig_learning_curves(summary, "resnet50")
    print(f"[fig] {pdf2}")
    pdf3, png3 = fig_learning_curves(summary, "alexnet")
    print(f"[fig] {pdf3}")
    pdf4, png4 = fig_delta_ablation(summary)
    print(f"[fig] {pdf4}")

    paths = [
        "outputs/topocover/topocover_run_inventory.csv",
        "outputs/topocover/topocover_missing_runs.csv",
        "outputs/topocover/topocover_all_backbones_long.csv",
        "outputs/topocover/topocover_all_backbones_summary.csv",
        "outputs/topocover/topocover_key_rounds.csv",
        "outputs/topocover/topocover_main_d070_key_rounds.csv",
        "outputs/topocover/topocover_main_key_rounds.tex",
        "outputs/topocover/topocover_delta_ablation_resnet18.csv",
        "outputs/topocover/topocover_delta_ablation_resnet18.tex",
        "outputs/topocover/topocover_runtime_summary.csv",
        "outputs/topocover/topocover_results_report.md",
        "figures/topocover/topocover_learning_curves_resnet18.pdf",
        "figures/topocover/topocover_learning_curves_resnet50.pdf",
        "figures/topocover/topocover_learning_curves_alexnet.pdf",
        "figures/topocover/topocover_delta_ablation_resnet18.pdf",
    ]
    write_report(inventory, missing, key, runtime, paths, OUT / "topocover_results_report.md")
    print(f"[write] {OUT / 'topocover_results_report.md'}")

    # Print main numbers for the chat
    print("\n=== MAIN δ=0.70 Acc@E100 / AULC ===")
    show = main_key[
        ["dataset", "backbone", "n_seeds", "final_accuracy", "final_sem", "aulc_mean", "aulc_sem"]
    ]
    print(show.to_string(index=False))
    print("[DONE]")


if __name__ == "__main__":
    main()
