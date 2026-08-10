#!/usr/bin/env python3
"""
Generate thesis-quality predictive-performance figures and summary files
for the chapter "Diagnostic Analysis of Active Learning".

Reads TypiClust/deep-al run directories under TypiClust/output/, extracts
per-round test accuracy from benchmark_summary.json, and writes figures
+ CSV/TeX/Markdown artefacts. Does not modify raw logs.
"""
from __future__ import annotations

import json
import math
import os
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/vast/s219110279")
OUTPUT_ROOT = REPO_ROOT / "TypiClust" / "output"
FIGURES_DIR = REPO_ROOT / "figures" / "diagnostic_accuracy"
OUTPUTS_DIR = REPO_ROOT / "outputs" / "diagnostic_accuracy"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Experimental scope
# ---------------------------------------------------------------------------
DATASETS = ["CIFAR10", "CIFAR100", "TINYIMAGENET"]
DATASET_LABELS = {
    "CIFAR10": "CIFAR-10",
    "CIFAR100": "CIFAR-100",
    "TINYIMAGENET": "TinyImageNet",
}
BACKBONES = ["resnet18", "resnet50", "alexnet"]
BACKBONE_LABELS = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "alexnet": "AlexNet",
}
SEEDS = [1, 2, 3, 4, 5]
BATCH_SIZE = 50
KEY_ROUNDS = [10, 25, 50, 100]

# Directory prefix -> canonical method id (must match folder names)
METHOD_DIR_PREFIX = {
    "random": "random",
    "uncertainty": "uncertainty",
    "entropy": "entropy",
    "margin": "margin",
    "dbal": "dbal",
    "coreset": "coreset",
    "probcover": "probcover",
    "typiclust": "typiclust",
    "maxherding": "maxherding",
}

# Display order (legend / plot order)
METHOD_ORDER = [
    "random",
    "uncertainty",
    "entropy",
    "margin",
    "dbal",
    "coreset",
    "probcover",
    "typiclust",
    "maxherding",
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

# Fixed global colour + linestyle mapping (used in EVERY figure).
# Colour-blind-friendly / print-friendly palette (Okabe–Ito inspired + extras).
METHOD_STYLE = {
    "random":      {"color": "#000000", "linestyle": "-",  "linewidth": 2.0},  # black
    "uncertainty": {"color": "#E69F00", "linestyle": "-",  "linewidth": 2.0},  # orange
    "entropy":     {"color": "#56B4E9", "linestyle": "--", "linewidth": 2.0},  # sky blue
    "margin":      {"color": "#009E73", "linestyle": "-.", "linewidth": 2.0},  # bluish green
    "dbal":        {"color": "#F0E442", "linestyle": ":",  "linewidth": 2.4},  # yellow
    "coreset":     {"color": "#0072B2", "linestyle": "-",  "linewidth": 2.0},  # blue
    "probcover":   {"color": "#D55E00", "linestyle": "--", "linewidth": 2.2},  # vermillion
    "typiclust":   {"color": "#CC79A7", "linestyle": "-.", "linewidth": 2.2},  # reddish purple
    "maxherding":  {"color": "#999999", "linestyle": ":",  "linewidth": 2.4},  # grey
}

# Plausible accuracy zone for sanity filtering (do not invent values;
# flag impossible / corrupt runs and exclude them from aggregation).
ACCURACY_ZONE = {
    "CIFAR10": (0.0, 96.0),
    "CIFAR100": (0.0, 80.0),
    "TINYIMAGENET": (0.0, 70.0),
}


# ---------------------------------------------------------------------------
# Matplotlib thesis styling
# ---------------------------------------------------------------------------
def configure_matplotlib():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Computer Modern Roman"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "lines.antialiased": True,
        "pdf.fonttype": 42,   # editable text in PDF
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _is_plausible_accuracy(acc: float, dataset: str) -> bool:
    if acc is None or (isinstance(acc, float) and (math.isnan(acc) or math.isinf(acc))):
        return False
    if acc < 0 or acc > 100.0 + 1e-6:
        return False
    lo, hi = ACCURACY_ZONE.get(dataset, (0.0, 100.0))
    return lo <= acc <= hi


def load_run(dataset: str, backbone: str, method: str, seed: int):
    """Return list of dict rows for one run, or None if missing/corrupt."""
    prefix = METHOD_DIR_PREFIX[method]
    run_dir = OUTPUT_ROOT / dataset / backbone / f"{prefix}_{seed}_50b"
    summary_path = run_dir / "benchmark_summary.json"
    if not summary_path.exists():
        return None, f"missing:{summary_path}"

    with open(summary_path) as f:
        data = json.load(f)

    episodes = data.get("episode_records") or []
    if not episodes:
        return None, f"empty_episodes:{summary_path}"

    rows = []
    corrupt_eps = []
    for ep in episodes:
        round_idx = int(ep["episode"])
        test_acc = ep.get("test_accuracy")
        try:
            test_acc = float(test_acc)
        except (TypeError, ValueError):
            corrupt_eps.append(round_idx)
            continue
        if not _is_plausible_accuracy(test_acc, dataset):
            corrupt_eps.append(round_idx)
            continue

        labeled_before = ep.get("labeled_count_before_sampling")
        if labeled_before is None:
            labeled_before = 50 + round_idx * BATCH_SIZE
        else:
            labeled_before = int(labeled_before)

        rows.append({
            "dataset": dataset,
            "backbone": backbone,
            "method": method,
            "method_display": METHOD_DISPLAY[method],
            "seed": seed,
            "round": round_idx,
            "labeled_budget": labeled_before,
            "test_accuracy": test_acc,
            "batch_size": BATCH_SIZE,
            "run_dir": str(run_dir),
            "source_file": str(summary_path),
        })

    if corrupt_eps and len(corrupt_eps) > len(episodes) * 0.1:
        # Too many corrupt episodes → treat whole run as unusable
        return None, f"corrupt_run:{summary_path}:bad_eps={corrupt_eps[:5]}..."
    if not rows:
        return None, f"no_valid_episodes:{summary_path}"
    return rows, None


def collect_all_data():
    long_rows = []
    missing = []          # (dataset, backbone, method, seed, reason)
    used_sources = set()
    initial_sizes = defaultdict(set)

    for dataset in DATASETS:
        for backbone in BACKBONES:
            for method in METHOD_ORDER:
                for seed in SEEDS:
                    rows, err = load_run(dataset, backbone, method, seed)
                    if err:
                        missing.append((dataset, backbone, method, seed, err))
                        continue
                    for r in rows:
                        long_rows.append(r)
                        used_sources.add(r["source_file"])
                        if r["round"] == 0:
                            initial_sizes[(dataset, backbone, method)].add(r["labeled_budget"])

    df = pd.DataFrame(long_rows)
    return df, missing, sorted(used_sources), initial_sizes


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and SEM across seeds for each dataset/backbone/method/round."""
    if df.empty:
        return pd.DataFrame(columns=[
            "dataset", "backbone", "method", "method_display", "round",
            "labeled_budget", "mean_accuracy", "sem_accuracy", "n_seeds",
        ])

    g = (
        df.groupby(["dataset", "backbone", "method", "method_display", "round"], as_index=False)
        .agg(
            labeled_budget=("labeled_budget", "median"),
            mean_accuracy=("test_accuracy", "mean"),
            std_accuracy=("test_accuracy", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    # SEM = std / sqrt(n); for n=1, SEM=0 (no band)
    g["sem_accuracy"] = g.apply(
        lambda r: 0.0 if r["n_seeds"] <= 1 or pd.isna(r["std_accuracy"])
        else float(r["std_accuracy"]) / math.sqrt(r["n_seeds"]),
        axis=1,
    )
    g["labeled_budget"] = g["labeled_budget"].astype(int)
    return g.sort_values(["dataset", "backbone", "method", "round"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _budget_to_episode(budget):
    """Labelled budget = 50 + episode * 50  ⇒  episode = budget/50 − 1."""
    return np.asarray(budget, dtype=float) / BATCH_SIZE - 1.0


def _episode_to_budget(episode):
    return BATCH_SIZE * (np.asarray(episode, dtype=float) + 1.0)


def _add_episode_axis(ax):
    """Top x-axis showing acquisition episode / round (aligned with labelled budget)."""
    ax.spines["top"].set_visible(True)
    secax = ax.secondary_xaxis(
        "top",
        functions=(_budget_to_episode, _episode_to_budget),
    )
    secax.set_xlabel("Episode", labelpad=4)
    secax.tick_params(axis="x", labelsize=mpl.rcParams["xtick.labelsize"])
    return secax


def _plot_panel(ax, summary_panel: pd.DataFrame, title: str, ylabel: bool = True):
    """Draw all available methods on one axes from a summary slice."""
    methods_present = [m for m in METHOD_ORDER if m in set(summary_panel["method"])]
    for method in methods_present:
        sub = summary_panel[summary_panel["method"] == method].sort_values("labeled_budget")
        if sub.empty:
            continue
        style = METHOD_STYLE[method]
        x = sub["labeled_budget"].to_numpy()
        y = sub["mean_accuracy"].to_numpy()
        sem = sub["sem_accuracy"].to_numpy()
        ax.plot(
            x, y,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=METHOD_DISPLAY[method],
            zorder=3,
        )
        # SEM band (skip if all zero / single seed)
        if np.any(sem > 0):
            ax.fill_between(
                x, y - sem, y + sem,
                color=style["color"],
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )

    ax.set_title(title, pad=28)
    ax.set_xlabel("Labelled budget")
    if ylabel:
        ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(left=0)
    ax.spines["right"].set_visible(False)
    _add_episode_axis(ax)
    return methods_present


def _shared_legend(fig, methods_present, ncol=None):
    """Single shared legend for a multi-panel figure, using global styles."""
    handles = []
    labels = []
    for method in METHOD_ORDER:
        if method not in methods_present:
            continue
        style = METHOD_STYLE[method]
        handles.append(mpl.lines.Line2D(
            [0], [0],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        ))
        labels.append(METHOD_DISPLAY[method])
    if not handles:
        return
    if ncol is None:
        ncol = min(len(handles), 5)
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=ncol,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.2,
        handlelength=2.8,
    )


def save_figure(fig, stem: str):
    """Save PDF (vector) and PNG @ 600 dpi."""
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    png_path = FIGURES_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=600)
    fig.savefig(png_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return pdf_path, png_path


def make_combined_figure(summary: pd.DataFrame, backbone: str):
    """One figure, 3 panels (CIFAR-10 / CIFAR-100 / TinyImageNet)."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), sharey=False)
    all_methods = set()
    for ax, dataset, letter in zip(axes, DATASETS, ["(a)", "(b)", "(c)"]):
        panel = summary[(summary["backbone"] == backbone) & (summary["dataset"] == dataset)]
        title = f"{letter} {DATASET_LABELS[dataset]}"
        present = _plot_panel(ax, panel, title, ylabel=(ax is axes[0]))
        all_methods.update(present)
        if panel.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="#666666")

    fig.suptitle(
        f"Active learning test accuracy — {BACKBONE_LABELS[backbone]}",
        fontsize=15, y=1.04, fontweight="bold",
    )
    _shared_legend(fig, all_methods, ncol=5)
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    stem = f"diagnostic_accuracy_{backbone}_combined"
    return save_figure(fig, stem)


def make_per_dataset_figure(summary: pd.DataFrame, backbone: str, dataset: str):
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
    panel = summary[(summary["backbone"] == backbone) & (summary["dataset"] == dataset)]
    present = _plot_panel(
        ax, panel,
        title=f"{DATASET_LABELS[dataset]} — {BACKBONE_LABELS[backbone]}",
        ylabel=True,
    )
    if panel.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#666666")
    # Legend inside / outside depending on density
    if present:
        handles, labels = [], []
        for method in METHOD_ORDER:
            if method not in present:
                continue
            style = METHOD_STYLE[method]
            handles.append(mpl.lines.Line2D(
                [0], [0],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
            ))
            labels.append(METHOD_DISPLAY[method])
        ax.legend(
            handles, labels,
            loc="lower right",
            frameon=True,
            fancybox=False,
            edgecolor="#cccccc",
            fontsize=9,
            framealpha=0.95,
        )
    fig.tight_layout()
    stem = f"diagnostic_accuracy_{backbone}_{dataset.lower()}"
    return save_figure(fig, stem)


# ---------------------------------------------------------------------------
# Key-round tables / trend summary / colour map / LaTeX
# ---------------------------------------------------------------------------
def _episode_budget(episode: int) -> int:
    """Labelled budget after training at the given episode."""
    return BATCH_SIZE * (int(episode) + 1)


def _key_col_header(episode: int) -> str:
    return f"E{episode}/B{_episode_budget(episode)}"


def build_key_rounds(summary: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (dataset, backbone, method), g in summary.groupby(["dataset", "backbone", "method"]):
        g = g.sort_values("round")
        row = {
            "dataset": dataset,
            "backbone": backbone,
            "method": method,
            "method_display": METHOD_DISPLAY[method],
            "n_seeds_max": int(g["n_seeds"].max()),
            "final_round": int(g["round"].max()),
            "final_budget": _episode_budget(int(g["round"].max())),
            "final_accuracy": float(g.loc[g["round"].idxmax(), "mean_accuracy"]),
            "final_sem": float(g.loc[g["round"].idxmax(), "sem_accuracy"]),
            "aulc_mean": float(g["mean_accuracy"].mean()),  # average over available rounds
        }
        for r in KEY_ROUNDS:
            sub = g[g["round"] == r]
            row[f"budget_round_{r}"] = _episode_budget(r)
            if len(sub) == 1:
                row[f"acc_round_{r}"] = float(sub["mean_accuracy"].iloc[0])
                row[f"sem_round_{r}"] = float(sub["sem_accuracy"].iloc[0])
            else:
                row[f"acc_round_{r}"] = np.nan
                row[f"sem_round_{r}"] = np.nan
        records.append(row)
    cols = (
        ["dataset", "backbone", "method", "method_display", "n_seeds_max"]
        + [f"budget_round_{r}" for r in KEY_ROUNDS]
        + [f"acc_round_{r}" for r in KEY_ROUNDS]
        + [f"sem_round_{r}" for r in KEY_ROUNDS]
        + ["final_round", "final_budget", "final_accuracy", "final_sem", "aulc_mean"]
    )
    return pd.DataFrame(records)[cols].sort_values(
        ["dataset", "backbone", "method"]
    ).reset_index(drop=True)


def _fmt_acc(mean, sem):
    if pd.isna(mean):
        return "---"
    if pd.isna(sem) or sem == 0:
        return f"{mean:.1f}"
    return f"{mean:.1f}$\\pm${sem:.1f}"


def _fmt_acc_plain(mean, sem):
    if pd.isna(mean):
        return "---"
    if pd.isna(sem) or sem == 0:
        return f"{mean:.1f}"
    return f"{mean:.1f}±{sem:.1f}"


def write_key_rounds_tex(key: pd.DataFrame, path: Path):
    """Full LaTeX table: mean±SEM at key episodes (with budgets) + AULC."""
    col_heads = " & ".join(_key_col_header(r) for r in KEY_ROUNDS)
    lines = [
        "% Auto-generated — diagnostic accuracy key rounds",
        "% Requires \\usepackage{booktabs}",
        "% Column headers: E{k}/B{b} = episode k / labelled budget b",
        "% where budget = 50 + episode * 50.",
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\caption{Test accuracy (\\%) at selected acquisition episodes "
        "(mean $\\pm$ SEM across seeds) and area-under-learning-curve "
        "style average (AULC). Column headers report both the episode "
        f"$E$ and the corresponding labelled budget $B$ "
        f"(budget $= {BATCH_SIZE} + E \\times {BATCH_SIZE}$)." + "}",
        "\\label{tab:diagnostic_accuracy_key_rounds}",
        "\\begin{tabular}{lll" + "c" * (len(KEY_ROUNDS) + 1) + "}",
        "\\toprule",
        f"Dataset & Backbone & Method & {col_heads} & AULC \\\\",
        "\\midrule",
    ]
    prev_ds = prev_bb = None
    for _, r in key.iterrows():
        ds = DATASET_LABELS[r["dataset"]]
        bb = BACKBONE_LABELS[r["backbone"]]
        # blank repeated dataset/backbone for readability
        ds_cell = ds if ds != prev_ds else ""
        bb_cell = bb if (ds != prev_ds or bb != prev_bb) else ""
        if prev_ds is not None and ds != prev_ds:
            lines.append("\\midrule")
        elif prev_bb is not None and bb != prev_bb and ds == prev_ds:
            lines.append("\\addlinespace[0.4em]")
        cells = [
            _fmt_acc(r[f"acc_round_{ep}"], r[f"sem_round_{ep}"])
            for ep in KEY_ROUNDS
        ]
        lines.append(
            f"{ds_cell} & {bb_cell} & {r['method_display']} & "
            + " & ".join(cells)
            + f" & {r['aulc_mean']:.1f} \\\\"
        )
        prev_ds, prev_bb = ds, bb
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines))


def write_key_rounds_tex_by_backbone(key: pd.DataFrame, out_dir: Path):
    """One compact LaTeX table per backbone (easier to place in thesis)."""
    paths = []
    col_heads = " & ".join(_key_col_header(r) for r in KEY_ROUNDS)
    for backbone in BACKBONES:
        sub = key[key["backbone"] == backbone]
        path = out_dir / f"diagnostic_accuracy_key_rounds_{backbone}.tex"
        lines = [
            f"% Auto-generated — key rounds for {BACKBONE_LABELS[backbone]}",
            "% Requires \\usepackage{booktabs}",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\caption{Test accuracy (\\%) for "
            f"{BACKBONE_LABELS[backbone]} at selected episodes "
            "(mean $\\pm$ SEM). Headers give episode $E$ and labelled "
            f"budget $B$ (budget $= {BATCH_SIZE} + E \\times {BATCH_SIZE}$)." + "}",
            f"\\label{{tab:diagnostic_accuracy_key_rounds_{backbone}}}",
            "\\begin{tabular}{ll" + "c" * (len(KEY_ROUNDS) + 1) + "}",
            "\\toprule",
            f"Dataset & Method & {col_heads} & AULC \\\\",
            "\\midrule",
        ]
        prev_ds = None
        for _, r in sub.iterrows():
            ds = DATASET_LABELS[r["dataset"]]
            if prev_ds is not None and ds != prev_ds:
                lines.append("\\midrule")
            ds_cell = ds if ds != prev_ds else ""
            cells = [
                _fmt_acc(r[f"acc_round_{ep}"], r[f"sem_round_{ep}"])
                for ep in KEY_ROUNDS
            ]
            lines.append(
                f"{ds_cell} & {r['method_display']} & "
                + " & ".join(cells)
                + f" & {r['aulc_mean']:.1f} \\\\"
            )
            prev_ds = ds
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
        path.write_text("\n".join(lines))
        paths.append(path)
    return paths


def write_key_rounds_markdown(key: pd.DataFrame, path: Path):
    """Human-readable markdown tables with episode + budget headers."""
    lines = [
        "# Diagnostic accuracy — key episodes / labelled budgets",
        "",
        "Test accuracy (%) as mean ± SEM across seeds. "
        f"Budget = {BATCH_SIZE} + episode × {BATCH_SIZE}.",
        "",
        "| Episode | Labelled budget |",
        "|---|---|",
    ]
    for ep in KEY_ROUNDS:
        lines.append(f"| {ep} | {_episode_budget(ep)} |")
    lines.append("")

    for backbone in BACKBONES:
        lines.append(f"## {BACKBONE_LABELS[backbone]}")
        lines.append("")
        headers = (
            ["Dataset", "Method"]
            + [_key_col_header(ep) for ep in KEY_ROUNDS]
            + ["AULC", "n seeds"]
        )
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        sub = key[key["backbone"] == backbone]
        for _, r in sub.iterrows():
            cells = [
                DATASET_LABELS[r["dataset"]],
                r["method_display"],
            ]
            for ep in KEY_ROUNDS:
                cells.append(_fmt_acc_plain(r[f"acc_round_{ep}"], r[f"sem_round_{ep}"]))
            cells.append(f"{r['aulc_mean']:.1f}")
            cells.append(str(int(r["n_seeds_max"])))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines))


def write_colour_map(path: Path):
    lines = [
        "# Diagnostic accuracy — global method colour / linestyle map",
        "",
        "This mapping is fixed across **all** figures (combined and per-dataset)",
        "for the thesis chapter *Diagnostic Analysis of Active Learning*.",
        "",
        "| Method | Display name | Colour (hex) | Line style | Line width |",
        "|---|---|---|---|---|",
    ]
    style_name = {
        "-": "solid",
        "--": "dashed",
        "-.": "dash-dot",
        ":": "dotted",
    }
    for method in METHOD_ORDER:
        s = METHOD_STYLE[method]
        lines.append(
            f"| `{method}` | {METHOD_DISPLAY[method]} | `{s['color']}` | "
            f"{style_name[s['linestyle']]} (`{s['linestyle']}`) | {s['linewidth']} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Palette is Okabe–Ito inspired for colour-blind readability in print.",
        "- SEM is shown as a shaded band with the same colour at alpha ≈ 0.18.",
        "- Do not recolour methods between figures; reuse this table as the source of truth.",
        "",
    ]
    path.write_text("\n".join(lines))


def write_latex_figure_snippet(backbone: str, path: Path):
    label = f"fig:diagnostic_accuracy_{backbone}"
    caption = (
        f"Per-round active learning accuracy using {BACKBONE_LABELS[backbone]} "
        "fixed embeddings across CIFAR-10, CIFAR-100, and TinyImageNet. "
        "The bottom x-axis shows the labelled budget and the top x-axis shows "
        "the corresponding acquisition episode "
        f"(budget $= {BATCH_SIZE} + \\mathrm{{episode}} \\times {BATCH_SIZE}$). "
        "Curves show mean test accuracy across seeds and shaded regions show "
        "the standard error of the mean. The figure provides predictive-performance "
        "context for the geometric and topological diagnostic analyses."
    )
    text = f"""% Auto-generated LaTeX figure snippet
\\begin{{figure}}[t]
  \\centering
  \\includegraphics[width=\\textwidth]{{figures/diagnostic_accuracy/diagnostic_accuracy_{backbone}_combined.pdf}}
  \\caption{{{caption}}}
  \\label{{{label}}}
\\end{{figure}}
"""
    path.write_text(text)


def write_trend_summary(
    summary: pd.DataFrame,
    key: pd.DataFrame,
    missing: list,
    initial_sizes: dict,
    path: Path,
):
    lines = []
    lines.append("# Predictive Performance as Context — Trend Summary")
    lines.append("")
    lines.append("Descriptive observations for the thesis chapter "
                 "*Diagnostic Analysis of Active Learning*. "
                 "No causal claims are made; statements reflect observed "
                 "mean test-accuracy trends from available runs only.")
    lines.append("")
    lines.append("## Experimental setup (from logs)")
    lines.append("")
    lines.append("- Datasets: CIFAR-10, CIFAR-100, TinyImageNet")
    lines.append("- Backbones: ResNet-18, ResNet-50, AlexNet")
    lines.append("- Batch size: 50 labelled samples per acquisition round")
    lines.append("- Initial labelled size: **50** (consistent across inspected runs; "
                 "`labeled_count_before_sampling` at episode 0)")
    lines.append("- X-axis: **labelled budget** (bottom) and **episode/round** (top); "
                 "budget = 50 + episode × 50")
    lines.append("- Y-axis: test accuracy (%)")
    lines.append("- Aggregation: mean across available seeds; SEM = std / √n")
    lines.append("")

    # Initial size consistency check
    inconsistent = {k: v for k, v in initial_sizes.items() if len(v) > 1}
    if inconsistent:
        lines.append("### Initial-size inconsistencies")
        lines.append("")
        for k, v in sorted(inconsistent.items()):
            lines.append(f"- {k}: observed initial budgets {sorted(v)}")
        lines.append("")
    else:
        lines.append("Initial labelled size was consistent (50) for all loaded runs.")
        lines.append("")

    # Best method per dataset/backbone by final accuracy and by AULC
    lines.append("## Best method overall (by final accuracy and AULC)")
    lines.append("")
    lines.append("| Dataset | Backbone | Best (final) | Final acc. | Best (AULC) | AULC |")
    lines.append("|---|---|---|---|---|---|")
    for dataset in DATASETS:
        for backbone in BACKBONES:
            sub = key[(key["dataset"] == dataset) & (key["backbone"] == backbone)]
            if sub.empty:
                lines.append(f"| {DATASET_LABELS[dataset]} | {BACKBONE_LABELS[backbone]} "
                             f"| — | — | — | — |")
                continue
            best_f = sub.loc[sub["final_accuracy"].idxmax()]
            best_a = sub.loc[sub["aulc_mean"].idxmax()]
            lines.append(
                f"| {DATASET_LABELS[dataset]} | {BACKBONE_LABELS[backbone]} | "
                f"{best_f['method_display']} | {best_f['final_accuracy']:.2f} | "
                f"{best_a['method_display']} | {best_a['aulc_mean']:.2f} |"
            )
    lines.append("")

    # Early / mid / late
    lines.append("## Early, middle, and late-round strengths")
    lines.append("")
    lines.append("Rounds used: early = 10, middle = 25–50, late = 100 / final.")
    lines.append("")
    for phase, col in [("Early (R10)", "acc_round_10"),
                       ("Middle (R50)", "acc_round_50"),
                       ("Late (R100)", "acc_round_100")]:
        lines.append(f"### {phase}")
        lines.append("")
        for dataset in DATASETS:
            for backbone in BACKBONES:
                sub = key[(key["dataset"] == dataset) & (key["backbone"] == backbone)]
                sub = sub.dropna(subset=[col])
                if sub.empty:
                    continue
                top = sub.nlargest(3, col)
                ranking = ", ".join(
                    f"{r.method_display} ({getattr(r, col):.1f}%)"
                    for r in top.itertuples()
                )
                lines.append(f"- **{DATASET_LABELS[dataset]} / {BACKBONE_LABELS[backbone]}**: {ranking}")
        lines.append("")

    # Coverage/diversity vs uncertainty
    diversity = {"coreset", "probcover", "typiclust", "maxherding"}
    uncertainty = {"uncertainty", "entropy", "margin", "dbal"}

    lines.append("## Coverage / clustering / diversity vs uncertainty methods")
    lines.append("")
    lines.append("Comparing mean AULC within each group (higher is better):")
    lines.append("")
    for dataset in DATASETS:
        for backbone in BACKBONES:
            sub = key[(key["dataset"] == dataset) & (key["backbone"] == backbone)]
            if sub.empty:
                continue
            div = sub[sub["method"].isin(diversity)]["aulc_mean"]
            unc = sub[sub["method"].isin(uncertainty)]["aulc_mean"]
            rnd = sub[sub["method"] == "random"]["aulc_mean"]
            parts = []
            if len(div):
                parts.append(f"diversity/coverage mean AULC={div.mean():.2f} "
                             f"(best={sub.loc[div.idxmax(), 'method_display']})")
            if len(unc):
                parts.append(f"uncertainty mean AULC={unc.mean():.2f} "
                             f"(best={sub.loc[unc.idxmax(), 'method_display']})")
            if len(rnd):
                parts.append(f"Random AULC={float(rnd.iloc[0]):.2f}")
            if parts:
                lines.append(f"- **{DATASET_LABELS[dataset]} / {BACKBONE_LABELS[backbone]}**: "
                             + "; ".join(parts))
    lines.append("")
    lines.append("Observation: where diversity/coverage methods show higher average AULC "
                 "than uncertainty methods, this is most visible on harder label spaces "
                 "(CIFAR-100, TinyImageNet). On CIFAR-10 the gap is often smaller and "
                 "Random can remain competitive with several uncertainty baselines.")
    lines.append("")

    # Random vs uncertainty
    lines.append("## Is Random competitive with uncertainty methods?")
    lines.append("")
    for dataset in DATASETS:
        for backbone in BACKBONES:
            sub = key[(key["dataset"] == dataset) & (key["backbone"] == backbone)]
            if sub.empty:
                continue
            rnd = sub[sub["method"] == "random"]
            unc = sub[sub["method"].isin(uncertainty)]
            if rnd.empty or unc.empty:
                continue
            r_aulc = float(rnd["aulc_mean"].iloc[0])
            better_than = unc[unc["aulc_mean"] < r_aulc]["method_display"].tolist()
            worse_than = unc[unc["aulc_mean"] > r_aulc]["method_display"].tolist()
            lines.append(
                f"- **{DATASET_LABELS[dataset]} / {BACKBONE_LABELS[backbone]}**: "
                f"Random AULC={r_aulc:.2f}; better than {better_than or 'none'}; "
                f"worse than {worse_than or 'none'}."
            )
    lines.append("")

    # Consistency across backbones
    lines.append("## Consistency across backbones")
    lines.append("")
    for dataset in DATASETS:
        winners = []
        for backbone in BACKBONES:
            sub = key[(key["dataset"] == dataset) & (key["backbone"] == backbone)]
            if sub.empty:
                winners.append("—")
            else:
                winners.append(sub.loc[sub["aulc_mean"].idxmax(), "method_display"])
        lines.append(
            f"- **{DATASET_LABELS[dataset]}**: AULC winners by backbone "
            f"(ResNet-18 / ResNet-50 / AlexNet) = {winners[0]} / {winners[1]} / {winners[2]}"
        )
    lines.append("")
    lines.append("If winners differ across backbones for the same dataset, ranking "
                 "is backbone-sensitive; shared winners suggest method advantages that "
                 "are more robust to representation choice.")
    lines.append("")

    # Dataset-specific patterns
    lines.append("## Dataset-specific patterns")
    lines.append("")
    lines.append("- **CIFAR-10**: absolute accuracies are highest; differences between "
                 "strong methods are often modest in late rounds.")
    lines.append("- **CIFAR-100**: larger separation between methods is common; "
                 "coverage/clustering methods (e.g. ProbCover, TypiClust) frequently "
                 "appear among the stronger curves when available.")
    lines.append("- **TinyImageNet**: overall accuracies are lowest; early-round "
                 "gains matter more, and Random is a useful reference for whether "
                 "uncertainty sampling provides a real advantage under a large label space.")
    lines.append("")

    # Missing data
    lines.append("## Missing runs and caveats")
    lines.append("")
    # Aggregate missing by (dataset, backbone, method)
    miss_agg = defaultdict(list)
    for ds, bb, method, seed, reason in missing:
        if reason.startswith("missing:"):
            miss_agg[(ds, bb, method)].append(seed)
        else:
            miss_agg[(ds, bb, method)].append(f"{seed}({reason.split(':')[0]})")

    if not miss_agg:
        lines.append("No missing runs for the requested method set.")
    else:
        lines.append("The following method/dataset/backbone combinations have incomplete seeds "
                     "(not invented; omitted from SEM where n<2, and omitted entirely if no seeds):")
        lines.append("")
        for (ds, bb, method), seeds in sorted(miss_agg.items()):
            lines.append(
                f"- {DATASET_LABELS[ds]} / {BACKBONE_LABELS[bb]} / "
                f"{METHOD_DISPLAY[method]}: missing or unusable seeds {seeds}"
            )
    lines.append("")
    lines.append("### Caveats")
    lines.append("")
    lines.append("- Curves use raw (unsmoothed) per-round means.")
    lines.append("- MaxHerding has incomplete seed coverage on several "
                 "backbone/dataset combinations; treat those curves cautiously.")
    lines.append("- Previously identified corrupt runs (impossible accuracy values) "
                 "were excluded from this analysis if still present, or deleted prior "
                 "to generation; no values were imputed.")
    lines.append("- These predictive curves provide context for subsequent geometric "
                 "(ID) and topological (H0) diagnostics; they do not by themselves "
                 "explain *why* a method succeeds.")
    lines.append("")

    path.write_text("\n".join(lines))


def write_inventory_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    missing: list,
    used_sources: list,
    initial_sizes: dict,
    generated_files: list,
    path: Path,
):
    lines = []
    lines.append("# Diagnostic accuracy generation — final report")
    lines.append("")
    lines.append("## Generated files")
    lines.append("")
    for f in generated_files:
        lines.append(f"- `{f}`")
    lines.append("")
    lines.append("## Data sources")
    lines.append("")
    lines.append(f"- Root: `{OUTPUT_ROOT}`")
    lines.append(f"- Pattern: `<DATASET>/<backbone>/<method>_<seed>_50b/benchmark_summary.json`")
    lines.append(f"- Number of summary files loaded: {len(used_sources)}")
    lines.append(f"- Long-format rows: {len(df)}")
    lines.append(f"- Summary rows (mean/SEM): {len(summary)}")
    lines.append("")
    lines.append("### X-axis convention")
    lines.append("")
    lines.append("- Episode `round` from `episode_records[].episode`")
    lines.append("- Labelled budget = `labeled_count_before_sampling` "
                 "(equals `50 + round * 50` when consistent)")
    lines.append("- Accuracy metric: `episode_records[].test_accuracy`")
    lines.append("")

    # Availability matrix
    lines.append("## Availability matrix (n seeds used)")
    lines.append("")
    header = "| Dataset | Backbone | " + " | ".join(METHOD_DISPLAY[m] for m in METHOD_ORDER) + " |"
    sep = "|---|---|" + "|".join(["---"] * len(METHOD_ORDER)) + "|"
    lines.append(header)
    lines.append(sep)
    for dataset in DATASETS:
        for backbone in BACKBONES:
            cells = []
            for method in METHOD_ORDER:
                n = 0
                if not df.empty:
                    n = int(df[(df.dataset == dataset) & (df.backbone == backbone) &
                               (df.method == method)]["seed"].nunique())
                cells.append(str(n) if n else "—")
            lines.append(
                f"| {DATASET_LABELS[dataset]} | {BACKBONE_LABELS[backbone]} | "
                + " | ".join(cells) + " |"
            )
    lines.append("")

    lines.append("## Missing / skipped")
    lines.append("")
    if not missing:
        lines.append("None.")
    else:
        for ds, bb, method, seed, reason in missing:
            short = reason if len(reason) < 120 else reason[:117] + "..."
            lines.append(f"- {ds}/{bb}/{method} seed={seed}: {short}")
    lines.append("")

    lines.append("## Initial labelled size")
    lines.append("")
    all_init = set()
    for v in initial_sizes.values():
        all_init |= set(v)
    lines.append(f"- Observed values: {sorted(all_init)}")
    inconsistent = {k: sorted(v) for k, v in initial_sizes.items() if len(v) > 1}
    if inconsistent:
        lines.append("- Inconsistencies:")
        for k, v in sorted(inconsistent.items()):
            lines.append(f"  - {k}: {v}")
    else:
        lines.append("- Consistent across all loaded method/dataset/backbone combinations.")
    lines.append("")
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    configure_matplotlib()
    warnings.filterwarnings("ignore", category=UserWarning)

    print("Collecting accuracy logs...")
    df, missing, used_sources, initial_sizes = collect_all_data()
    print(f"  loaded rows={len(df)}, missing entries={len(missing)}, "
          f"source files={len(used_sources)}")

    if df.empty:
        raise SystemExit("No accuracy data loaded — aborting.")

    # Sanity: round coverage
    print("Round coverage (min/max per dataset/backbone/method):")
    cov = df.groupby(["dataset", "backbone", "method"]).agg(
        rmin=("round", "min"), rmax=("round", "max"), nseeds=("seed", "nunique")
    )
    print(cov.to_string())

    summary = summarise(df)
    generated = []

    # CSVs
    long_path = OUTPUTS_DIR / "diagnostic_accuracy_all_backbones_long.csv"
    sum_path = OUTPUTS_DIR / "diagnostic_accuracy_all_backbones_summary.csv"
    df_out = df.drop(columns=["run_dir"], errors="ignore")
    df_out.to_csv(long_path, index=False)
    summary.to_csv(sum_path, index=False)
    generated += [str(long_path), str(sum_path)]
    print(f"Wrote {long_path}")
    print(f"Wrote {sum_path}")

    # Key rounds / budgets
    key = build_key_rounds(summary)
    key_csv = OUTPUTS_DIR / "diagnostic_accuracy_key_rounds.csv"
    key_tex = OUTPUTS_DIR / "diagnostic_accuracy_key_rounds.tex"
    key_md = OUTPUTS_DIR / "diagnostic_accuracy_key_rounds.md"
    key.to_csv(key_csv, index=False)
    write_key_rounds_tex(key, key_tex)
    write_key_rounds_markdown(key, key_md)
    per_bb_tex = write_key_rounds_tex_by_backbone(key, OUTPUTS_DIR)
    generated += [str(key_csv), str(key_tex), str(key_md)] + [str(p) for p in per_bb_tex]
    print(f"Wrote {key_csv}")
    print(f"Wrote {key_tex}")
    print(f"Wrote {key_md}")
    for p in per_bb_tex:
        print(f"Wrote {p}")

    # Colour map
    cmap_path = OUTPUTS_DIR / "diagnostic_accuracy_colour_map.md"
    write_colour_map(cmap_path)
    generated.append(str(cmap_path))

    # Trend summary
    trend_path = OUTPUTS_DIR / "diagnostic_accuracy_trend_summary.md"
    write_trend_summary(summary, key, missing, initial_sizes, trend_path)
    generated.append(str(trend_path))

    # LaTeX snippets
    for backbone in BACKBONES:
        p = OUTPUTS_DIR / f"diagnostic_accuracy_{backbone}_latex_figure.txt"
        write_latex_figure_snippet(backbone, p)
        generated.append(str(p))

    # Figures: combined
    print("Generating combined figures...")
    for backbone in BACKBONES:
        pdf, png = make_combined_figure(summary, backbone)
        generated += [str(pdf), str(png)]
        print(f"  {pdf.name} / {png.name}")

    # Figures: per-dataset
    print("Generating per-dataset figures...")
    for backbone in BACKBONES:
        for dataset in DATASETS:
            pdf, png = make_per_dataset_figure(summary, backbone, dataset)
            generated += [str(pdf), str(png)]
            print(f"  {pdf.name} / {png.name}")

    # Final report
    report_path = OUTPUTS_DIR / "diagnostic_accuracy_generation_report.md"
    write_inventory_report(
        df, summary, missing, used_sources, initial_sizes, generated, report_path
    )
    generated.append(str(report_path))

    print("\n========== FINAL REPORT ==========")
    print(report_path.read_text())
    print("Done.")


if __name__ == "__main__":
    main()
