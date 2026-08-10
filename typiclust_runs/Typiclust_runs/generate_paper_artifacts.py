from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


ROOT = Path("/scratch/s219110279")
OUTPUT_ROOT = ROOT / "TypiClust/output"
PAPER_ROOT = ROOT / "paper_results"
FIG_ROOT = PAPER_ROOT / "figures"


DATASETS = ["CIFAR10", "CIFAR100", "TINYIMAGENET"]
DATASET_LABELS = {
    "CIFAR10": "CIFAR-10",
    "CIFAR100": "CIFAR-100",
    "TINYIMAGENET": "TinyImageNet",
}

MAIN_METHODS = {
    "random": "Random",
    "uncertainty": "Least Confidence",
    "margin": "Margin",
    "entropy": "Entropy",
    "coreset": "CoreSet",
    "probcover": "ProbCover",
    "LIDCOVER": "LID-Cover",
}

OPTIONAL_METHODS = {
    "dbal": "DBAL",
}

ADAPTIVE_SIGNAL_METHODS = {
    "LIDCOVER": "LID-Cover",
    "density_cover": "Density-adaptive cover",
    "knn_distance_cover": "Mean kNN-distance cover",
    "distance_variance_cover": "Distance-variance cover",
    "distance_cv_cover": "Distance-CV cover",
}

ABLATION_GROUPS = {
    "alpha": {
        "LIDCOVER_alpha0": r"$\alpha=0$",
        "LIDCOVER_alpha05": r"$\alpha=0.5$",
        "LIDCOVER_alpha1": r"$\alpha=1$",
        "LIDCOVER_alpha2": r"$\alpha=2$",
        "LIDCOVER": r"default",
    },
    "k": {
        "LIDCOVER_k20": r"$k=20$",
        "LIDCOVER": r"$k=50$",
        "LIDCOVER_k75": r"$k=75$",
    },
    "delta": {
        "LIDCOVER_d020": r"$\delta_0=0.20$",
        "LIDCOVER": r"$\delta_0=0.25$",
        "LIDCOVER_d030": r"$\delta_0=0.30$",
    },
    "selection": {
        "LIDCOVER": r"LID-Cover",
        "LIDCOVER_lowID": r"Low-ID score",
        "LIDCOVER_tie_random": r"Random tie-break",
    },
}

BUDGET100_METHODS = {
    "random": "Random",
    "entropy": "Entropy",
    "coreset": "CoreSet",
    "probcover": "ProbCover",
    "LIDCOVER": "LID-Cover",
}

METHOD_LABELS = {
    **MAIN_METHODS,
    **OPTIONAL_METHODS,
    **ADAPTIVE_SIGNAL_METHODS,
    **{
        "LIDCOVER_alpha0": r"LID-Cover $\alpha=0$",
        "LIDCOVER_alpha05": r"LID-Cover $\alpha=0.5$",
        "LIDCOVER_alpha1": r"LID-Cover $\alpha=1$",
        "LIDCOVER_alpha2": r"LID-Cover $\alpha=2$",
        "LIDCOVER_k20": r"LID-Cover $k=20$",
        "LIDCOVER_k75": r"LID-Cover $k=75$",
        "LIDCOVER_d020": r"LID-Cover $\delta_0=0.20$",
        "LIDCOVER_d030": r"LID-Cover $\delta_0=0.30$",
        "LIDCOVER_lowID": "Low-ID score",
        "LIDCOVER_tie_random": "Random tie-break",
    },
}

CHECKPOINTS = [25, 50, 75, 100]


@dataclass
class Run:
    dataset: str
    method_key: str
    method_label: str
    exp_name: str
    seed: int
    budget: int
    records: list[dict]
    summary: dict


def sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return stdev(values) / math.sqrt(len(values))


def fmt_mean_se(values: list[float], bold: bool = False) -> str:
    if not values:
        return "--"
    text = f"{mean(values):.2f} $\\pm$ {sem(values):.2f}"
    return rf"\textbf{{{text}}}" if bold else text


def parse_exp_name(name: str) -> tuple[str, int | None, int | None]:
    m = re.match(r"(.+)_([0-9]+)_([0-9]+)b$", name)
    if not m:
        return name, None, None
    return m.group(1), int(m.group(2)), int(m.group(3))


def read_runs() -> list[Run]:
    runs: list[Run] = []
    for path in OUTPUT_ROOT.glob("*/resnet18/*/benchmark_summary.json"):
        with path.open() as f:
            summary = json.load(f)
        dataset = summary.get("dataset") or path.parts[-4]
        exp_name = summary.get("exp_name") or path.parent.name
        method_key, seed_from_name, budget_from_name = parse_exp_name(exp_name)
        seed = int(summary.get("seed") or seed_from_name or 0)
        budget = int(summary.get("budget_per_round") or budget_from_name or 0)
        records = summary.get("episode_records", [])
        if not records:
            continue
        method_label = (
            MAIN_METHODS.get(method_key)
            or OPTIONAL_METHODS.get(method_key)
            or ADAPTIVE_SIGNAL_METHODS.get(method_key)
            or method_key
        )
        runs.append(
            Run(
                dataset=dataset,
                method_key=method_key,
                method_label=method_label,
                exp_name=exp_name,
                seed=seed,
                budget=budget,
                records=records,
                summary=summary,
            )
        )
    return runs


def record_metric(record: dict) -> float | None:
    value = record.get("test_accuracy")
    return float(value) if value is not None else None


def curve_stats(runs: list[Run]) -> dict[int, tuple[float, float, int, float]]:
    by_episode: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for run in runs:
        for rec in run.records:
            acc = record_metric(rec)
            if acc is None:
                continue
            labels = rec.get("labeled_count_before_sampling")
            if labels is None:
                labels = 50 + int(rec["episode"]) * run.budget
            by_episode[int(rec["episode"])].append((float(labels), acc))
    stats = {}
    for ep, pairs in by_episode.items():
        labels = [p[0] for p in pairs]
        vals = [p[1] for p in pairs]
        stats[ep] = (mean(vals), sem(vals), len(vals), mean(labels))
    return stats


def values_at(runs: list[Run], episode: int) -> list[float]:
    vals = []
    for run in runs:
        for rec in run.records:
            if int(rec.get("episode", -1)) == episode:
                acc = record_metric(rec)
                if acc is not None:
                    vals.append(acc)
                break
    return vals


def group_runs(runs: list[Run], dataset: str, methods: dict[str, str], budget: int) -> dict[str, list[Run]]:
    grouped = {}
    for key in methods:
        selected = [
            r
            for r in runs
            if r.dataset == dataset and r.method_key == key and r.budget == budget
        ]
        if selected:
            grouped[key] = selected
    return grouped


def plot_curves(
    grouped: dict[str, list[Run]],
    labels: dict[str, str],
    title: str,
    out_path: Path,
    min_episode: int = 0,
    max_episode: int = 100,
) -> None:
    plt.figure(figsize=(6.4, 4.2))
    for key, runs in grouped.items():
        stats = curve_stats(runs)
        episodes = [e for e in sorted(stats) if min_episode <= e <= max_episode]
        xs = [stats[e][3] for e in episodes]
        ys = [stats[e][0] for e in episodes]
        es = [stats[e][1] for e in episodes]
        plt.plot(xs, ys, linewidth=1.9, label=labels.get(key, key))
        plt.fill_between(xs, [y - e for y, e in zip(ys, es)], [y + e for y, e in zip(ys, es)], alpha=0.13)
    plt.title(title)
    plt.xlabel("Labeled samples")
    plt.ylabel("Test accuracy (%)")
    plt.grid(True, linewidth=0.4, alpha=0.35)
    plt.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_main_tables(runs: list[Run]) -> None:
    rows = []
    for dataset in DATASETS:
        for key, label in MAIN_METHODS.items():
            selected = group_runs(runs, dataset, {key: label}, 50).get(key, [])
            if not selected:
                continue
            row = {
                "dataset": DATASET_LABELS[dataset],
                "method": label,
                "seeds": len(selected),
                "final_mean": mean(values_at(selected, 100)),
                "final_se": sem(values_at(selected, 100)),
                "auc_mean": mean([float(r.summary.get("test_auc", 0.0)) for r in selected]),
                "auc_se": sem([float(r.summary.get("test_auc", 0.0)) for r in selected]),
            }
            rows.append(row)
    write_csv(
        PAPER_ROOT / "main_final_results.csv",
        rows,
        ["dataset", "method", "seeds", "final_mean", "final_se", "auc_mean", "auc_se"],
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Final test accuracy at round 100 for the main budget-50 experiments. Results are mean $\pm$ standard error over completed seeds.}",
        r"\label{tab:main_final_results}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Dataset & Method & Seeds & R100 test accuracy \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        drows = [r for r in rows if r["dataset"] == DATASET_LABELS[dataset]]
        if not drows:
            continue
        best = max(drows, key=lambda r: r["final_mean"])["method"]
        for i, row in enumerate(drows):
            dataset_cell = row["dataset"] if i == 0 else ""
            value = f"{row['final_mean']:.2f} $\\pm$ {row['final_se']:.2f}"
            if row["method"] == best:
                value = rf"\textbf{{{value}}}"
            lines.append(f"{dataset_cell} & {row['method']} & {row['seeds']} & {value} \\\\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}", ""]
    (PAPER_ROOT / "main_final_results_table.tex").write_text("\n".join(lines))


def write_checkpoint_table(
    runs: list[Run],
    dataset: str,
    methods: dict[str, str],
    budget: int,
    path: Path,
    caption: str,
    label: str,
) -> None:
    grouped = group_runs(runs, dataset, methods, budget)
    table_rows = []
    for key, selected in grouped.items():
        table_rows.append((methods[key], {ep: values_at(selected, ep) for ep in CHECKPOINTS}, len(selected)))
    best_by_ep = {}
    for ep in CHECKPOINTS:
        candidates = [(label, mean(vals[ep])) for label, vals, _ in table_rows if vals[ep]]
        if candidates:
            best_by_ep[ep] = max(candidates, key=lambda x: x[1])[0]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Seeds & R25 & R50 & R75 & R100 \\",
        r"\midrule",
    ]
    for label_name, vals, n in table_rows:
        cells = []
        for ep in CHECKPOINTS:
            cells.append(fmt_mean_se(vals[ep], bold=best_by_ep.get(ep) == label_name))
        lines.append(f"{label_name} & {n} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def write_adaptive_signal_table(runs: list[Run]) -> None:
    rows = []
    for dataset in ["CIFAR100", "TINYIMAGENET"]:
        for key, label in ADAPTIVE_SIGNAL_METHODS.items():
            selected = group_runs(runs, dataset, {key: label}, 50).get(key, [])
            if selected:
                vals = values_at(selected, 100)
                rows.append(
                    {
                        "dataset": DATASET_LABELS[dataset],
                        "method": label,
                        "seeds": len(selected),
                        "r100_mean": mean(vals),
                        "r100_se": sem(vals),
                    }
                )
    write_csv(PAPER_ROOT / "adaptive_signal_comparison.csv", rows, ["dataset", "method", "seeds", "r100_mean", "r100_se"])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Comparison between LID-Cover and alternative adaptive-radius signals at round 100. Results are mean $\pm$ standard error over completed seeds.}",
        r"\label{tab:adaptive_signal_comparison}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Seeds & CIFAR-100 R100 & TinyImageNet R100 \\",
        r"\midrule",
    ]
    by_method = defaultdict(dict)
    for row in rows:
        by_method[row["method"]][row["dataset"]] = row
    best_by_dataset = {}
    for dataset in ["CIFAR-100", "TinyImageNet"]:
        candidates = [r for r in rows if r["dataset"] == dataset]
        if candidates:
            best_by_dataset[dataset] = max(candidates, key=lambda r: r["r100_mean"])["method"]
    for key, label in ADAPTIVE_SIGNAL_METHODS.items():
        c100 = by_method[label].get("CIFAR-100")
        tiny = by_method[label].get("TinyImageNet")
        c100_text = "--" if not c100 else f"{c100['r100_mean']:.2f} $\\pm$ {c100['r100_se']:.2f}"
        tiny_text = "--" if not tiny else f"{tiny['r100_mean']:.2f} $\\pm$ {tiny['r100_se']:.2f}"
        seeds = max([r["seeds"] for r in (c100, tiny) if r] or [0])
        if best_by_dataset.get("CIFAR-100") == label:
            c100_text = rf"\textbf{{{c100_text}}}"
        if best_by_dataset.get("TinyImageNet") == label:
            tiny_text = rf"\textbf{{{tiny_text}}}"
        lines.append(f"{label} & {seeds} & {c100_text} & {tiny_text} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (PAPER_ROOT / "adaptive_signal_comparison_table.tex").write_text("\n".join(lines))


def write_ablation_tables(runs: list[Run]) -> None:
    csv_rows = []
    for group_name, methods in ABLATION_GROUPS.items():
        for key, label in methods.items():
            selected = group_runs(runs, "CIFAR100", {key: label}, 50).get(key, [])
            if not selected:
                continue
            row = {"group": group_name, "setting": label, "seeds": len(selected)}
            for ep in CHECKPOINTS:
                vals = values_at(selected, ep)
                row[f"r{ep}_mean"] = mean(vals)
                row[f"r{ep}_se"] = sem(vals)
            csv_rows.append(row)
    write_csv(
        PAPER_ROOT / "cifar100_lidcover_ablation.csv",
        csv_rows,
        ["group", "setting", "seeds"] + [f"r{ep}_{stat}" for ep in CHECKPOINTS for stat in ("mean", "se")],
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{LID-Cover ablations on CIFAR-100 with budget 50. Results are mean $\pm$ standard error over completed seeds.}",
        r"\label{tab:lid_cover_ablation_cifar100_resnet18}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Setting & Seeds & R25 & R50 & R75 & R100 \\",
        r"\midrule",
    ]
    for group_name, methods in ABLATION_GROUPS.items():
        group_rows = [r for r in csv_rows if r["group"] == group_name]
        if not group_rows:
            continue
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{group_name.capitalize()} ablation}}}} \\")
        for row in group_rows:
            cells = []
            for ep in CHECKPOINTS:
                cells.append(f"{row[f'r{ep}_mean']:.2f} $\\pm$ {row[f'r{ep}_se']:.2f}")
            lines.append(f"{row['setting']} & {row['seeds']} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}", ""]
    (PAPER_ROOT / "lidcover_ablation_table.tex").write_text("\n".join(lines))


def write_results_text(runs: list[Run]) -> None:
    main_rows = list(csv.DictReader((PAPER_ROOT / "main_final_results.csv").open()))
    lines = [
        r"\subsection{Main Accuracy Comparison}",
        "",
        r"Figure~\ref{fig:main_results_all} reports the budget-50 active learning trajectories from the completed run summaries. "
        r"Table~\ref{tab:main_final_results} summarizes round-100 test accuracy over completed seeds.",
        "",
    ]
    for dataset in ["CIFAR-10", "CIFAR-100", "TinyImageNet"]:
        drows = [r for r in main_rows if r["dataset"] == dataset]
        if not drows:
            continue
        best = max(drows, key=lambda r: float(r["final_mean"]))
        lid = next((r for r in drows if r["method"] == "LID-Cover"), None)
        if lid:
            lines.append(
                f"On {dataset}, LID-Cover reaches {float(lid['final_mean']):.2f} $\\pm$ {float(lid['final_se']):.2f}\\% "
                f"at round 100, while the strongest completed method is {best['method']} "
                f"with {float(best['final_mean']):.2f} $\\pm$ {float(best['final_se']):.2f}\\%."
            )
    lines += [
        "",
        r"\subsection{Adaptive-Radius Signal Comparison}",
        "",
        r"Table~\ref{tab:adaptive_signal_comparison} compares LID-Cover with alternative adaptive-radius signals. "
        r"In the current completed runs, LID-Cover is strongest on TinyImageNet, while mean kNN-distance adaptation is slightly higher on CIFAR-100 at round 100. "
        r"Accordingly, the adaptive-signal results should be described as mixed rather than as uniformly favoring LID across datasets.",
        "",
        r"\subsection{Additional Budget Comparison}",
        "",
        r"Table~\ref{tab:cifar100_budget100} and Figure~\ref{fig:cifar100_budget100} report the CIFAR-100 budget-100 setting. "
        r"Under this budget, ProbCover is strongest at rounds 25, 50, and 75, while CoreSet is strongest at round 100. "
        r"LID-Cover remains competitive but does not dominate this setting.",
        "",
        r"\paragraph{Budget-500 handling.}",
        r"The CIFAR-100 budget-500 jobs are excluded from the paper tables and figures because the protocol exhausts the unlabeled pool before 100 acquisition rounds. "
        r"Those runs terminate at the next sampling step with the expected assertion that the budget must be strictly smaller than the remaining unlabeled set.",
        "",
    ]
    (PAPER_ROOT / "updated_results_text.tex").write_text("\n".join(lines))


def write_figure_snippets() -> None:
    lines = [
        r"\begin{figure*}[t]",
        r"    \centering",
        r"    \includegraphics[width=\textwidth]{figures/main_results_all_budget50.png}",
        r"    \caption{Mean test accuracy over 100 active learning rounds under query budget 50. Shaded regions indicate standard error over completed seeds.}",
        r"    \label{fig:main_results_all}",
        r"\end{figure*}",
        "",
        r"\begin{figure*}[t]",
        r"    \centering",
        r"    \begin{subfigure}[t]{0.48\textwidth}",
        r"        \centering",
        r"        \includegraphics[width=\textwidth]{figures/cifar100_adaptive_signals.png}",
        r"        \caption{CIFAR-100}",
        r"    \end{subfigure}",
        r"    \hfill",
        r"    \begin{subfigure}[t]{0.48\textwidth}",
        r"        \centering",
        r"        \includegraphics[width=\textwidth]{figures/tinyimagenet_adaptive_signals.png}",
        r"        \caption{TinyImageNet}",
        r"    \end{subfigure}",
        r"    \caption{Adaptive-radius signal comparisons under the same uncovered-coverage framework.}",
        r"    \label{fig:adaptive_signal_curves}",
        r"\end{figure*}",
        "",
        r"\begin{figure*}[t]",
        r"    \centering",
        r"    \begin{subfigure}[t]{0.32\textwidth}",
        r"        \centering",
        r"        \includegraphics[width=\textwidth]{figures/cifar100_ablation_alpha.png}",
        r"        \caption{Adaptation strength}",
        r"    \end{subfigure}",
        r"    \hfill",
        r"    \begin{subfigure}[t]{0.32\textwidth}",
        r"        \centering",
        r"        \includegraphics[width=\textwidth]{figures/cifar100_ablation_k.png}",
        r"        \caption{Neighborhood size}",
        r"    \end{subfigure}",
        r"    \hfill",
        r"    \begin{subfigure}[t]{0.32\textwidth}",
        r"        \centering",
        r"        \includegraphics[width=\textwidth]{figures/cifar100_ablation_delta.png}",
        r"        \caption{Base radius}",
        r"    \end{subfigure}",
        r"    \caption{CIFAR-100 LID-Cover ablation curves under query budget 50.}",
        r"    \label{fig:lidcover_ablation_curves}",
        r"\end{figure*}",
        "",
        r"\begin{figure}[t]",
        r"    \centering",
        r"    \includegraphics[width=0.72\textwidth]{figures/cifar100_budget100_comparison.png}",
        r"    \caption{CIFAR-100 active learning comparison under query budget 100.}",
        r"    \label{fig:cifar100_budget100}",
        r"\end{figure}",
        "",
    ]
    (PAPER_ROOT / "figure_snippets.tex").write_text("\n".join(lines))


def make_plots(runs: list[Run]) -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        grouped = group_runs(runs, dataset, MAIN_METHODS, 50)
        plot_curves(
            grouped,
            MAIN_METHODS,
            f"{DATASET_LABELS[dataset]} budget 50",
            FIG_ROOT / f"{dataset.lower()}_main_budget50.png",
        )
    # Combined main figure.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
    for ax, dataset in zip(axes, DATASETS):
        grouped = group_runs(runs, dataset, MAIN_METHODS, 50)
        for key, selected in grouped.items():
            stats = curve_stats(selected)
            episodes = sorted(e for e in stats if 0 <= e <= 100)
            xs = [stats[e][3] for e in episodes]
            ys = [stats[e][0] for e in episodes]
            es = [stats[e][1] for e in episodes]
            ax.plot(xs, ys, linewidth=1.6, label=MAIN_METHODS[key])
            ax.fill_between(xs, [y - e for y, e in zip(ys, es)], [y + e for y, e in zip(ys, es)], alpha=0.10)
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("Labeled samples")
        ax.grid(True, linewidth=0.4, alpha=0.35)
    axes[0].set_ylabel("Test accuracy (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(FIG_ROOT / "main_results_all_budget50.png", dpi=240)
    plt.close(fig)

    # Adaptive signal curves.
    for dataset in ["CIFAR100", "TINYIMAGENET"]:
        grouped = group_runs(runs, dataset, ADAPTIVE_SIGNAL_METHODS, 50)
        plot_curves(
            grouped,
            ADAPTIVE_SIGNAL_METHODS,
            f"{DATASET_LABELS[dataset]} adaptive-radius signals",
            FIG_ROOT / f"{dataset.lower()}_adaptive_signals.png",
        )

    # Ablation curves.
    for group_name, methods in ABLATION_GROUPS.items():
        grouped = group_runs(runs, "CIFAR100", methods, 50)
        plot_curves(
            grouped,
            methods,
            f"CIFAR-100 LID-Cover {group_name} ablation",
            FIG_ROOT / f"cifar100_ablation_{group_name}.png",
        )

    # Budget 100 only.
    grouped = group_runs(runs, "CIFAR100", BUDGET100_METHODS, 100)
    plot_curves(
        grouped,
        BUDGET100_METHODS,
        "CIFAR-100 budget 100",
        FIG_ROOT / "cifar100_budget100_comparison.png",
    )


def write_manifest(runs: list[Run]) -> None:
    counts = defaultdict(int)
    for r in runs:
        if r.budget != 500:
            counts[(r.dataset, r.method_key, r.budget)] += 1
    manifest = {
        "generated_from": str(OUTPUT_ROOT),
        "budget500_excluded": True,
        "run_counts": [
            {"dataset": k[0], "method": k[1], "budget": k[2], "completed_seeds": v}
            for k, v in sorted(counts.items())
        ],
        "figures_dir": str(FIG_ROOT),
    }
    (PAPER_ROOT / "paper_artifacts_manifest.json").write_text(json.dumps(manifest, indent=2))


def nested_timing(summary: dict, key: str, subkey: str, default: float = 0.0) -> float:
    value = summary.get("timing", {}).get(key, {})
    if isinstance(value, dict):
        return float(value.get(subkey, default) or default)
    return default


def run_timing_row(run: Run) -> dict:
    timing = run.summary.get("timing", {})
    init = float(timing.get("initial_sampling_time_sec", 0.0) or 0.0)
    acq_mean = nested_timing(run.summary, "acquisition_time_sec", "mean")
    acq_median = nested_timing(run.summary, "acquisition_time_sec", "median")
    acq_cum = nested_timing(run.summary, "acquisition_time_sec", "cumulative")
    train_mean = nested_timing(run.summary, "train_time_sec", "mean")
    train_cum = nested_timing(run.summary, "train_time_sec", "cumulative")
    test_mean = nested_timing(run.summary, "test_time_sec", "mean")
    test_cum = nested_timing(run.summary, "test_time_sec", "cumulative")
    round_mean = nested_timing(run.summary, "round_time_sec", "mean")
    round_cum = nested_timing(run.summary, "round_time_sec", "cumulative")
    total = nested_timing(run.summary, "round_time_sec", "cumulative_with_initial_sampling", round_cum + init)
    return {
        "dataset": DATASET_LABELS.get(run.dataset, run.dataset),
        "method_key": run.method_key,
        "method": METHOD_LABELS.get(run.method_key, run.method_label),
        "budget": run.budget,
        "seed": run.seed,
        "sampled_rounds": int(timing.get("sampled_rounds", len(run.records) - 1) or 0),
        "cold_initial_sampling_sec": init,
        "warm_acquisition_mean_sec": acq_mean,
        "warm_acquisition_median_sec": acq_median,
        "cumulative_acquisition_sec": acq_cum,
        "train_mean_sec": train_mean,
        "cumulative_train_sec": train_cum,
        "test_mean_sec": test_mean,
        "cumulative_test_sec": test_cum,
        "round_mean_sec": round_mean,
        "cumulative_round_sec": round_cum,
        "total_runtime_sec": total,
        "total_runtime_hours": total / 3600.0,
        "acquisition_runtime_fraction": (acq_cum + init) / total if total > 0 else 0.0,
    }


def summarize_timing(rows: list[dict], method_filter: set[str] | None = None) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        if method_filter is not None and row["method_key"] not in method_filter:
            continue
        grouped[(row["dataset"], row["method_key"], int(row["budget"]))].append(row)

    metrics = [
        "cold_initial_sampling_sec",
        "warm_acquisition_mean_sec",
        "cumulative_acquisition_sec",
        "train_mean_sec",
        "test_mean_sec",
        "round_mean_sec",
        "total_runtime_sec",
        "total_runtime_hours",
        "acquisition_runtime_fraction",
    ]
    summary_rows = []
    for (dataset, method_key, budget), items in sorted(grouped.items()):
        out = {
            "dataset": dataset,
            "method_key": method_key,
            "method": METHOD_LABELS.get(method_key, method_key),
            "budget": budget,
            "seeds": len(items),
        }
        for metric in metrics:
            vals = [float(item[metric]) for item in items]
            out[f"{metric}_mean"] = mean(vals)
            out[f"{metric}_se"] = sem(vals)
        summary_rows.append(out)
    return summary_rows


def write_timing_artifacts(runs: list[Run]) -> None:
    rows = [run_timing_row(run) for run in runs if run.budget != 500]
    raw_fields = [
        "dataset",
        "method_key",
        "method",
        "budget",
        "seed",
        "sampled_rounds",
        "cold_initial_sampling_sec",
        "warm_acquisition_mean_sec",
        "warm_acquisition_median_sec",
        "cumulative_acquisition_sec",
        "train_mean_sec",
        "cumulative_train_sec",
        "test_mean_sec",
        "cumulative_test_sec",
        "round_mean_sec",
        "cumulative_round_sec",
        "total_runtime_sec",
        "total_runtime_hours",
        "acquisition_runtime_fraction",
    ]
    write_csv(PAPER_ROOT / "timing_runs_all_non500.csv", rows, raw_fields)

    summary = summarize_timing(rows)
    summary_fields = ["dataset", "method_key", "method", "budget", "seeds"]
    for metric in [
        "cold_initial_sampling_sec",
        "warm_acquisition_mean_sec",
        "cumulative_acquisition_sec",
        "train_mean_sec",
        "test_mean_sec",
        "round_mean_sec",
        "total_runtime_sec",
        "total_runtime_hours",
        "acquisition_runtime_fraction",
    ]:
        summary_fields += [f"{metric}_mean", f"{metric}_se"]
    write_csv(PAPER_ROOT / "timing_summary_all_non500.csv", summary, summary_fields)

    main_summary = [
        row
        for row in summary
        if int(row["budget"]) == 50 and row["method_key"] in MAIN_METHODS
    ]
    write_csv(PAPER_ROOT / "timing_summary_main_budget50.csv", main_summary, summary_fields)

    all_budget50_summary = [
        row
        for row in summary
        if int(row["budget"]) == 50 and row["method_key"] in METHOD_LABELS
    ]
    write_csv(PAPER_ROOT / "timing_summary_all_methods_budget50.csv", all_budget50_summary, summary_fields)

    budget100_summary = [
        row
        for row in summary
        if row["dataset"] == "CIFAR-100" and int(row["budget"]) == 100 and row["method_key"] in BUDGET100_METHODS
    ]
    write_csv(PAPER_ROOT / "timing_summary_cifar100_budget100.csv", budget100_summary, summary_fields)

    def cell(row: dict, metric: str, scale: float = 1.0, suffix: str = "") -> str:
        return f"{row[f'{metric}_mean'] / scale:.2f} $\\pm$ {row[f'{metric}_se'] / scale:.2f}{suffix}"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Runtime summary for main budget-50 experiments. Cold time is the recorded initial sampling or preprocessing step. Warm acquisition, training, testing, and round times are per active-learning round. Total time includes all recorded rounds and the initial cold step.}",
        r"\label{tab:runtime_main_budget50}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Dataset & Method & Cold (s) & Warm acq. (s/rnd) & Train (s/rnd) & Round (s/rnd) & Total (h) \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        dlabel = DATASET_LABELS[dataset]
        drows = [r for r in main_summary if r["dataset"] == dlabel]
        for i, row in enumerate(drows):
            dataset_cell = dlabel if i == 0 else ""
            lines.append(
                f"{dataset_cell} & {row['method']} & "
                f"{cell(row, 'cold_initial_sampling_sec')} & "
                f"{cell(row, 'warm_acquisition_mean_sec')} & "
                f"{cell(row, 'train_mean_sec')} & "
                f"{cell(row, 'round_mean_sec')} & "
                f"{cell(row, 'total_runtime_hours')} \\\\"
            )
        if drows:
            lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}", ""]
    (PAPER_ROOT / "runtime_main_budget50_table.tex").write_text("\n".join(lines))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Runtime summary for all completed budget-50 methods. Cold time is the recorded initial sampling or preprocessing step. Warm acquisition and round time are per active-learning round; total time is the full recorded run time including the cold step.}",
        r"\label{tab:runtime_all_methods_budget50}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Dataset & Method & Seeds & Cold (s) & Warm acq. (s/rnd) & Round (s/rnd) & Total (h) \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        dlabel = DATASET_LABELS[dataset]
        drows = [r for r in all_budget50_summary if r["dataset"] == dlabel]
        for i, row in enumerate(drows):
            dataset_cell = dlabel if i == 0 else ""
            lines.append(
                f"{dataset_cell} & {row['method']} & {row['seeds']} & "
                f"{cell(row, 'cold_initial_sampling_sec')} & "
                f"{cell(row, 'warm_acquisition_mean_sec')} & "
                f"{cell(row, 'round_mean_sec')} & "
                f"{cell(row, 'total_runtime_hours')} \\\\"
            )
        if drows:
            lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}", ""]
    (PAPER_ROOT / "runtime_all_methods_budget50_table.tex").write_text("\n".join(lines))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{CIFAR-100 budget-100 runtime summary. Times are mean $\pm$ standard error over completed seeds.}",
        r"\label{tab:runtime_cifar100_budget100}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Cold (s) & Warm acq. (s/rnd) & Round (s/rnd) & Total (h) \\",
        r"\midrule",
    ]
    for row in budget100_summary:
        lines.append(
            f"{row['method']} & "
            f"{cell(row, 'cold_initial_sampling_sec')} & "
            f"{cell(row, 'warm_acquisition_mean_sec')} & "
            f"{cell(row, 'round_mean_sec')} & "
            f"{cell(row, 'total_runtime_hours')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (PAPER_ROOT / "runtime_cifar100_budget100_table.tex").write_text("\n".join(lines))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Detailed timing artifacts written by the result generator.}",
        r"\label{tab:runtime_artifact_files}",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Artifact & Contents \\",
        r"\midrule",
        r"timing\_runs\_all\_non500.csv & Per-seed timing for all completed non-budget-500 runs \\",
        r"timing\_summary\_all\_non500.csv & Mean and standard error by dataset, method, and budget \\",
        r"timing\_summary\_main\_budget50.csv & Main-method timing summary for budget 50 \\",
        r"timing\_summary\_all\_methods\_budget50.csv & Full budget-50 timing summary for all completed methods \\",
        r"timing\_summary\_cifar100\_budget100.csv & Budget-100 timing summary for CIFAR-100 \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    (PAPER_ROOT / "runtime_artifact_files_table.tex").write_text("\n".join(lines))

    lines = [
        r"\subsection{Runtime Analysis}",
        "",
        r"Table~\ref{tab:runtime_main_budget50} reports runtime statistics from the completed budget-50 runs. "
        r"The cold column is the recorded initial sampling or preprocessing step in the active-learning summary, while warm acquisition is the mean acquisition time per subsequent active-learning round. "
        r"Total runtime includes all recorded rounds and the cold step.",
        "",
        r"The comprehensive per-seed timing data are provided in \texttt{timing\_runs\_all\_non500.csv}; the corresponding aggregate table for every completed budget-50 method is Table~\ref{tab:runtime_all_methods_budget50}. "
        r"Budget-500 runs are excluded for the same pool-exhaustion reason described in the results section.",
        "",
        r"Table~\ref{tab:runtime_cold} is a separate component-level cold-preprocessing measurement for the LID-based preprocessing cache. "
        r"It should not be read as the total job runtime; it decomposes feature loading, local-ID computation, and nearest-neighbor/adaptive-graph construction.",
        "",
    ]
    (PAPER_ROOT / "updated_runtime_text.tex").write_text("\n".join(lines))


def main() -> None:
    PAPER_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    runs = [r for r in read_runs() if r.budget != 500]
    write_main_tables(runs)
    write_checkpoint_table(
        runs,
        "CIFAR100",
        BUDGET100_METHODS,
        100,
        PAPER_ROOT / "cifar100_budget100_table.tex",
        "Additional CIFAR-100 results under query budget 100. Results are mean $\\pm$ standard error over completed seeds.",
        "tab:cifar100_budget100",
    )
    write_adaptive_signal_table(runs)
    write_ablation_tables(runs)
    write_timing_artifacts(runs)
    make_plots(runs)
    write_results_text(runs)
    write_figure_snippets()
    write_manifest(runs)
    print(f"Wrote paper artifacts to {PAPER_ROOT}")
    print(f"Wrote figures to {FIG_ROOT}")


if __name__ == "__main__":
    main()
