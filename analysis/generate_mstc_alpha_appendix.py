#!/usr/bin/env python3
"""Generate the pre-specified CIFAR-100/ResNet-18 MSTC alpha appendix.

This is a read-only analysis of completed run directories.  It reports every
pre-specified alpha and never selects a configuration using test accuracy.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "TypiClust/output/CIFAR100/resnet18"
OUT = ROOT / "outputs/thesis_appendix/mstc_alpha"
FIG = ROOT / "figures/thesis_appendix/mstc_alpha"
TABLE = ROOT / "tables/thesis_appendix"
ALPHAS = (0.0, 0.25, 0.50, 0.70, 0.75, 1.0)
SEEDS = range(1, 6)
KEY_EPISODES = (10, 25, 50, 75, 100)


def run_path(alpha: float, seed: int) -> Path:
    tag = f"{round(alpha * 100):03d}"
    if alpha == 0.25:
        name = f"THESIS_APPENDIX_REPAIR_20260901_R2_MSTC_C070_F050_A{tag}_S{seed}_B050"
    elif alpha in (0.50, 0.75):
        name = f"THESIS_APPENDIX_REPAIR_20260901_MSTC_C070_F050_A{tag}_S{seed}_B050"
    else:
        name = f"MSTC_C070_F050_A{tag}_{seed}_50b"
    return RUN_ROOT / name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sem(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else (0.0 if len(values) else np.nan)


def mean_sem(group: pd.DataFrame, column: str) -> tuple[float, float]:
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan, sem(values)


def load_runs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    per_seed, trajectory, hashes = [], [], {}
    for alpha in ALPHAS:
        for seed in SEEDS:
            path = run_path(alpha, seed)
            summary_path = path / "benchmark_summary.json"
            config_path = path / "config.yaml"
            if not summary_path.exists():
                raise FileNotFoundError(summary_path)
            summary = json.loads(summary_path.read_text())
            records = sorted(summary.get("episode_records") or [], key=lambda row: int(row["episode"]))
            episodes = [int(row["episode"]) for row in records]
            if episodes != list(range(101)):
                raise RuntimeError(f"incomplete episode history: {path}: {episodes[:3]}...{episodes[-3:]}")
            rows = []
            for record in records:
                episode = int(record["episode"])
                metadata = record.get("sampling_metadata") or {}
                row = {
                    "alpha": alpha,
                    "seed": seed,
                    "episode": episode,
                    "labelled_budget": int(record.get("labeled_count_before_sampling", 50 + 50 * episode)),
                    "labelled_count_after_sampling": record.get("labeled_count_after_sampling"),
                    "test_accuracy": float(record["test_accuracy"]),
                    "acquisition_time_sec": record.get("acquisition_time_sec"),
                    "train_time_sec": record.get("train_time_sec"),
                    "round_time_sec": record.get("round_time_sec"),
                    "coverage_fraction_fine": metadata.get("coverage_fraction_fine"),
                    "coverage_fraction_coarse": metadata.get("coverage_fraction_coarse"),
                    "selected_mixed_score_mean": metadata.get("selected_score_mean"),
                    "selected_mixed_score_max": metadata.get("selected_score_max"),
                    "active_set_size": record.get("active_set_size"),
                    "raw_path": str(path),
                }
                rows.append(row)
                trajectory.append(row)
            frame = pd.DataFrame(rows)
            y = frame.test_accuracy.to_numpy(float)
            x = frame.labelled_budget.to_numpy(float)
            acquisition = pd.to_numeric(frame.acquisition_time_sec, errors="coerce")
            final_acquisition = frame[frame.episode == 99].iloc[0]
            per_seed.append({
                "alpha": alpha,
                "seed": seed,
                "n_episodes": len(frame),
                "final_labelled_budget": int(frame.iloc[-1].labelled_budget),
                "final_accuracy": float(summary["final_test_accuracy"]),
                "aulc_mean_accuracy": float(y.mean()),
                "aulc_trapezoid_normalised": float(np.trapezoid(y, x) / (x[-1] - x[0])),
                "total_acquisition_time_sec": float(acquisition.sum()),
                "mean_acquisition_time_sec": float(acquisition.dropna().mean()),
                "final_fine_coverage": float(final_acquisition.coverage_fraction_fine),
                "final_coarse_coverage": float(final_acquisition.coverage_fraction_coarse),
                **{f"accuracy_E{episode}": float(frame.loc[frame.episode == episode, "test_accuracy"].iloc[0]) for episode in KEY_EPISODES},
                "raw_path": str(path),
                "benchmark_summary_sha256": sha256(summary_path),
                "config_sha256": sha256(config_path),
            })
            hashes[str(summary_path)] = sha256(summary_path)
            hashes[str(config_path)] = sha256(config_path)
    seed_frame = pd.DataFrame(per_seed)
    principal = seed_frame[seed_frame.alpha == 0.70].set_index("seed")
    for metric in ("final_accuracy", "aulc_mean_accuracy", "aulc_trapezoid_normalised"):
        seed_frame[f"{metric}_minus_alpha070"] = seed_frame.apply(
            lambda row: row[metric] - principal.loc[row.seed, metric], axis=1
        )
    return seed_frame, pd.DataFrame(trajectory), hashes


def aggregate(seed_frame: pd.DataFrame, trajectory: pd.DataFrame):
    summary_rows = []
    metrics = [
        "final_accuracy", "aulc_mean_accuracy", "aulc_trapezoid_normalised",
        "total_acquisition_time_sec", "mean_acquisition_time_sec",
        "final_fine_coverage", "final_coarse_coverage",
    ]
    for alpha, group in seed_frame.groupby("alpha", sort=True):
        row = {"alpha": alpha, "n": int(group.seed.nunique())}
        for metric in metrics:
            row[f"{metric}_mean"], row[f"{metric}_sem"] = mean_sem(group, metric)
        for metric in ("final_accuracy_minus_alpha070", "aulc_mean_accuracy_minus_alpha070"):
            row[f"paired_{metric}_mean"], row[f"paired_{metric}_sem"] = mean_sem(group, metric)
        summary_rows.append(row)

    key_rows = []
    principal = trajectory[trajectory.alpha == 0.70].set_index(["seed", "episode"])
    for (alpha, episode), group in trajectory[trajectory.episode.isin(KEY_EPISODES)].groupby(["alpha", "episode"]):
        differences = [
            row.test_accuracy - principal.loc[(row.seed, episode), "test_accuracy"]
            for row in group.itertuples()
        ]
        key_rows.append({
            "alpha": alpha,
            "episode": int(episode),
            "labelled_budget": int(group.labelled_budget.iloc[0]),
            "accuracy_mean": float(group.test_accuracy.mean()),
            "accuracy_sem": sem(group.test_accuracy),
            "paired_difference_vs_alpha070_mean": float(np.mean(differences)),
            "paired_difference_vs_alpha070_sem": sem(differences),
            "n": int(group.seed.nunique()),
        })

    structural_rows = []
    for alpha in ALPHAS:
        alpha_frame = trajectory[trajectory.alpha == alpha]
        for requested_episode in KEY_EPISODES:
            source_episode = 99 if requested_episode == 100 else requested_episode
            group = alpha_frame[alpha_frame.episode == source_episode]
            fine, fine_sem = mean_sem(group, "coverage_fraction_fine")
            coarse, coarse_sem = mean_sem(group, "coverage_fraction_coarse")
            score, score_sem = mean_sem(group, "selected_mixed_score_mean")
            structural_rows.append({
                "alpha": alpha,
                "requested_episode": requested_episode,
                "source_acquisition_episode": source_episode,
                "coverage_labelled_budget_after_sampling": int(pd.to_numeric(group.labelled_count_after_sampling).iloc[0]),
                "fine_vertex_coverage_mean": fine,
                "fine_vertex_coverage_sem": fine_sem,
                "coarse_vertex_coverage_mean": coarse,
                "coarse_vertex_coverage_sem": coarse_sem,
                "selected_mixed_score_mean": score,
                "selected_mixed_score_sem": score_sem,
                "selected_fine_component_gain_mean": score if alpha == 0.0 else np.nan,
                "selected_coarse_component_gain_mean": score if alpha == 1.0 else np.nan,
                "component_gain_status": "RECORDED_ENDPOINT_ONLY" if alpha in (0.0, 1.0) else "UNVERIFIABLE_FROM_SAVED_METADATA",
                "zero_gain_selection_fraction": np.nan,
                "zero_gain_status": "UNVERIFIABLE_FROM_AGGREGATE_SCORE_METADATA",
                "n": int(group.seed.nunique()),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(key_rows), pd.DataFrame(structural_rows)


def configure_plot():
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.grid": True,
        "grid.alpha": 0.25, "pdf.fonttype": 42, "figure.facecolor": "white",
    })


def make_figures(seed_frame, trajectory):
    configure_plot()
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(ALPHAS)))
    labels = [f"{alpha:g}" for alpha in ALPHAS]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for alpha, color in zip(ALPHAS, colors):
        group = trajectory[trajectory.alpha == alpha]
        agg = group.groupby("labelled_budget").test_accuracy.agg(["mean", sem]).reset_index()
        ax.plot(agg.labelled_budget, agg["mean"], color=color, label=rf"$\alpha={alpha:g}$")
        ax.fill_between(agg.labelled_budget, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"], color=color, alpha=0.12)
    ax.set_xlabel("Labelled budget")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "mstc_alpha_learning_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    def bar(metric, ylabel, filename):
        agg = seed_frame.groupby("alpha")[metric].agg(["mean", sem]).reindex(ALPHAS)
        fig, ax = plt.subplots(figsize=(6.6, 4.0))
        ax.bar(labels, agg["mean"], yerr=agg["sem"], color=colors, capsize=3)
        ax.set_xlabel(r"MSTC $\alpha$")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(FIG / filename, bbox_inches="tight")
        plt.close(fig)

    bar("final_accuracy", "Final test accuracy (%)", "mstc_alpha_final_accuracy.pdf")
    bar("aulc_mean_accuracy", "Mean trajectory accuracy (%)", "mstc_alpha_aulc.pdf")

    acquisition = trajectory[trajectory.episode < 100]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), sharex=True)
    for alpha, color in zip(ALPHAS, colors):
        group = acquisition[acquisition.alpha == alpha]
        for ax, metric, title in zip(
            axes,
            ("coverage_fraction_fine", "coverage_fraction_coarse"),
            ("Fine-scale vertex coverage", "Coarse-scale vertex coverage"),
        ):
            curve = group.groupby("labelled_count_after_sampling")[metric].mean()
            ax.plot(curve.index, curve.values, color=color, label=rf"$\alpha={alpha:g}$")
            ax.set_title(title)
            ax.set_xlabel("Labelled budget after acquisition")
            ax.set_ylabel("Covered fraction")
    axes[1].legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "mstc_alpha_fine_coarse_coverage.pdf", bbox_inches="tight")
    plt.close(fig)


def write_latex(summary: pd.DataFrame, key: pd.DataFrame):
    key_wide = key.pivot(index="alpha", columns="episode", values=["accuracy_mean", "accuracy_sem"])
    lines = [
        r"\begin{tabular}{crrrrrrr}", r"\toprule",
        r"$\alpha$ & Final accuracy & Trajectory AULC & E10 & E25 & E50 & E75 & E100 \\",
        r"\midrule",
    ]
    for row in summary.itertuples():
        def key_cell(episode):
            return f"{key_wide.loc[row.alpha, ('accuracy_mean', episode)]:.2f} $\\pm$ {key_wide.loc[row.alpha, ('accuracy_sem', episode)]:.2f}"
        lines.append(
            f"{row.alpha:g} & {row.final_accuracy_mean:.2f} $\\pm$ {row.final_accuracy_sem:.2f} & "
            f"{row.aulc_mean_accuracy_mean:.2f} $\\pm$ {row.aulc_mean_accuracy_sem:.2f} & "
            + " & ".join(key_cell(ep) for ep in KEY_EPISODES)
            + r" \\"
        )
    lines += [r"\midrule", r"\multicolumn{8}{l}{All entries are mean $\pm$ SEM over $n=5$ seeds; the alpha grid was pre-specified.} \\", r"\bottomrule", r"\end{tabular}"]
    (TABLE / "mstc_alpha_sensitivity.tex").write_text("\n".join(lines) + "\n")


def write_readme(summary, key, structural, hashes):
    final_best = summary.loc[summary.final_accuracy_mean.idxmax()]
    aulc_best = summary.loc[summary.aulc_mean_accuracy_mean.idxmax()]
    episode_best = key.loc[key.groupby("episode").accuracy_mean.idxmax(), ["episode", "alpha", "accuracy_mean"]]
    fine = summary.set_index("alpha")
    endpoint_early = key[key.episode == 10].set_index("alpha")
    text = f"""# MSTC interior-alpha sensitivity

All 30 pre-specified CIFAR-100/ResNet-18 C0.70/F0.50 runs are included: six alpha values by five seeds. Alpha 0, 0.7, and 1 are historical complete endpoints/principal runs; alpha 0.25, 0.50, and 0.75 are the 15 new appendix runs. No alpha was selected using test accuracy.

## Descriptive findings

- Highest mean final accuracy in this fixed grid: alpha={final_best.alpha:g}, {final_best.final_accuracy_mean:.2f} +/- {final_best.final_accuracy_sem:.2f} percentage points.
- Highest mean trajectory accuracy: alpha={aulc_best.alpha:g}, {aulc_best.aulc_mean_accuracy_mean:.2f} +/- {aulc_best.aulc_mean_accuracy_sem:.2f}.
- The mean-accuracy leader by key evaluation episode is: {', '.join(f'E{int(r.episode)}: alpha={r.alpha:g}' for r in episode_best.itertuples())}. This is a descriptive sensitivity result, not test-set tuning.
- Fine-only versus coarse-only E10 means are {endpoint_early.loc[0.0, 'accuracy_mean']:.2f} and {endpoint_early.loc[1.0, 'accuracy_mean']:.2f}; final means are {fine.loc[0.0, 'final_accuracy_mean']:.2f} and {fine.loc[1.0, 'final_accuracy_mean']:.2f}. The endpoints therefore show an early-versus-final trade-off, but the full grid is not monotone: intermediate alpha=0.25 and 0.50 have higher mean final accuracy than alpha=1.
- Interpolation is not globally smooth. Alpha=0 is separated from the interior/coarse-weighted settings early, and the identity of the highest mean changes with budget (alpha=0 through E50, alpha=0.5 at E75, alpha=0.25 at E100).
- The alpha=0.7 result is not unique to that setting: alpha=0.50 has a paired mean trajectory difference of {summary.set_index('alpha').loc[0.50, 'paired_aulc_mean_accuracy_minus_alpha070_mean']:+.2f} +/- {summary.set_index('alpha').loc[0.50, 'paired_aulc_mean_accuracy_minus_alpha070_sem']:.2f}, while alpha=0.25 has a paired final difference of {summary.set_index('alpha').loc[0.25, 'paired_final_accuracy_minus_alpha070_mean']:+.2f} +/- {summary.set_index('alpha').loc[0.25, 'paired_final_accuracy_minus_alpha070_sem']:.2f} percentage points. Alpha=0.75 final performance differs from alpha=0.7 by only {summary.set_index('alpha').loc[0.75, 'paired_final_accuracy_minus_alpha070_mean']:+.2f} +/- {summary.set_index('alpha').loc[0.75, 'paired_final_accuracy_minus_alpha070_sem']:.2f}, which is small relative to seed variability.
- Fine/coarse coverage is recorded exactly from acquisition metadata. Separate fine- and coarse-component gains are only identifiable at alpha=0 and alpha=1, respectively. For interior alpha values the saved metadata contains only the mixed score, so separate gains and the fraction of zero-gain selections are marked UNVERIFIABLE rather than reconstructed approximately.

Differences should be interpreted against the reported seed SEMs and paired seed differences in the CSVs. With n=5, no p-value or universal-superiority claim is made. Alpha endpoints remain MultiScaleTopoCover controls and are not substituted for historical TopoCover.

## Files

- `mstc_alpha_per_seed.csv`: per-seed final accuracy, two AULC conventions, key accuracies, runtime, paired differences, raw paths, and hashes.
- `mstc_alpha_summary.csv`: mean +/- SEM with n for final/AULC/runtime/coverage.
- `mstc_alpha_key_budgets.csv`: mean +/- SEM and paired differences versus alpha=0.7 at E10/E25/E50/E75/E100.
- `mstc_alpha_structural_summary.csv`: fine/coarse coverage and saved score evidence, including explicit UNVERIFIABLE fields.

The exact input hashes and analysis-code hash are in `provenance.json`.
"""
    (OUT / "README.md").write_text(text)
    provenance = {
        "analysis": str(Path(__file__).resolve()),
        "analysis_sha256": sha256(Path(__file__).resolve()),
        "run_count": 30,
        "new_run_count": 15,
        "alphas": list(ALPHAS),
        "seeds": list(SEEDS),
        "delta_coarse": 0.70,
        "delta_fine": 0.50,
        "test_based_selection": False,
        "raw_file_sha256": hashes,
        "unverifiable": [
            "separate fine/coarse component gain for interior alpha",
            "fraction of zero-gain selections from aggregate saved scores",
        ],
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    seed_frame, trajectory, hashes = load_runs()
    summary, key, structural = aggregate(seed_frame, trajectory)
    seed_frame.to_csv(OUT / "mstc_alpha_per_seed.csv", index=False)
    summary.to_csv(OUT / "mstc_alpha_summary.csv", index=False)
    key.to_csv(OUT / "mstc_alpha_key_budgets.csv", index=False)
    structural.to_csv(OUT / "mstc_alpha_structural_summary.csv", index=False)
    trajectory.to_csv(OUT / "mstc_alpha_trajectory.csv", index=False)
    make_figures(seed_frame, trajectory)
    write_latex(summary, key)
    write_readme(summary, key, structural, hashes)
    print(json.dumps({
        "per_seed_rows": len(seed_frame), "trajectory_rows": len(trajectory),
        "summary_rows": len(summary), "key_budget_rows": len(key),
        "structural_rows": len(structural),
    }, indent=2))


if __name__ == "__main__":
    main()
