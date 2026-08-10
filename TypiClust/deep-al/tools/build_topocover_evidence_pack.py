#!/usr/bin/env python3
"""Build an auditable TopoCover/MSTC thesis evidence pack from completed runs.

This script is deliberately read-only with respect to experiment outputs.  It
does not train models or reconstruct deleted runs.  AULC follows the existing
thesis convention: the arithmetic mean of episode-level test accuracy for each
seed, followed by aggregation across seeds.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/vast/s219110279")
OUTPUT = ROOT / "TypiClust/output"
CAMPAIGNS = ROOT / "typiclust_runs/Typiclust_runs"
PACK = ROOT / "evidence_pack"
FIG = PACK / "figures"
TAB = PACK / "tables"
EXPECTED_CAMPAIGNS = [
    "topocover_50b_20260630", "topocover_d070_50b_20260702",
    "topocover_paper_individual_20260716", "mstc_thesis_individual_20260728",
    "mstc_singlescale_via_alpha_20260803", "topo_variants_pilot_20260804",
    "persist_w1_pilot_20260804", "merge_tree_cover_pilot_20260805",
]
TOPO_TOKENS = ("MSTC_", "TOPOCOVER", "WEIGHTED_TOPO", "FRONTIER_TOPO",
               "PHASED_TOPO", "PERSIST", "W1", "MERGE_TREE", "MERGETREE", "MTC_")
BASE_METHODS = {
    "random": "Random", "coreset": "CoreSet", "core_set": "CoreSet",
    "uncertainty": "Least Confidence", "least_confidence": "Least Confidence",
    "margin": "Margin", "entropy": "Entropy", "probcover": "ProbCover",
    "prob_cover": "ProbCover",
}
TABLEAU = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
COLORS = {
    "Random": TABLEAU[0], "Least Confidence": TABLEAU[1],
    "Entropy": TABLEAU[2], "Margin": TABLEAU[3], "CoreSet": TABLEAU[4],
    "ProbCover": TABLEAU[5], "Fine-only MSTC": TABLEAU[6],
    "Coarse-only MSTC": TABLEAU[7], "MSTC": TABLEAU[9],
}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def sem(x):
    x = pd.Series(x).dropna().astype(float)
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else np.nan


def parse_name(name, sampling=""):
    out = {"method": sampling or name, "alpha": np.nan, "delta_coarse": np.nan,
           "delta_fine": np.nan, "batch_size": np.nan, "seed": np.nan}
    m = re.search(r"MSTC_C(\d{3})_F(\d{3})_A(\d{3})_(\d+)_(\d+)b", name, re.I)
    if m:
        out.update(method="MSTC", delta_coarse=int(m[1])/100,
                   delta_fine=int(m[2])/100, alpha=int(m[3])/100,
                   seed=int(m[4]), batch_size=int(m[5]))
        return out
    m = re.search(r"D(\d{3})_(\d+)_(\d+)b", name, re.I)
    if m:
        out.update(delta_coarse=int(m[1])/100, seed=int(m[2]), batch_size=int(m[3]))
    low = (sampling or name).lower()
    if "weighted" in low: out["method"] = "WeightedTopo"
    elif "frontier" in low: out["method"] = "FrontierTopo"
    elif "phased" in low: out["method"] = "PhasedTopo"
    elif "w1" in low: out["method"] = "W1Proxy"
    elif "persist" in low: out["method"] = "PersistProbCover"
    elif "merge" in low or low.startswith("mtc"): out["method"] = "MergeTreeCover"
    elif "topocover" in low: out["method"] = "Original TopoCover"
    return out


def campaign_for(name):
    if name.startswith("MSTC_"):
        return "mstc_singlescale_via_alpha_20260803" if ("A000" in name or "A100" in name) else "mstc_thesis_individual_20260728"
    if "WEIGHTED" in name or "FRONTIER" in name or "PHASED" in name: return "topo_variants_pilot_20260804"
    if "PERSIST" in name or "W1" in name: return "persist_w1_pilot_20260804"
    if "MERGE" in name or name.startswith("MTC_"): return "merge_tree_cover_pilot_20260805"
    if "PAPER" in name: return "topocover_paper_individual_20260716"
    return "topocover campaign (exact source not encoded in output)"


def read_summary(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def is_topology_dir(p):
    return any(t.lower() in p.name.lower() for t in TOPO_TOKENS)


def scan_runs():
    rows, curves = [], []
    dirs = {p.parent for p in OUTPUT.rglob("benchmark_summary.json") if is_topology_dir(p.parent)}
    dirs |= {p for p in OUTPUT.glob("*/*/*") if p.is_dir() and is_topology_dir(p)}
    for d in sorted(dirs):
        sp = d / "benchmark_summary.json"
        s = read_summary(sp) if sp.exists() else None
        dataset = d.parents[1].name if len(d.parents) > 1 else ""
        backbone = d.parent.name
        if backbone not in {"alexnet", "resnet18", "resnet50"}: continue
        parsed = parse_name(d.name, (s or {}).get("sampling_fn", ""))
        recs = (s or {}).get("episode_records", []) or []
        episodes = [int(r.get("episode", -1)) for r in recs]
        final_ep = max(episodes, default=-1)
        target = 100
        final_budget = max([int(r.get("labeled_count_before_sampling", 0)) for r in recs] + [0])
        has_sel = bool(recs and any(r.get("active_set_ids") for r in recs)) or any(d.glob("episode_*/activeSet.npy"))
        has_runtime = bool((s or {}).get("timing")) or any(fnum(r.get("round_time_sec")) == fnum(r.get("round_time_sec")) for r in recs)
        has_eval = s is not None and np.isfinite(fnum(s.get("final_test_accuracy")))
        complete = has_eval and final_ep >= target
        status = "complete" if complete else ("partial" if s or recs else "missing evaluation output")
        reason = "" if complete else f"{status}: final episode {final_ep}, target {target}, evaluation={has_eval}"
        row = dict(dataset=dataset, backbone=backbone, method=parsed["method"],
                   alpha=parsed["alpha"], delta_coarse=parsed["delta_coarse"],
                   delta_fine=parsed["delta_fine"],
                   batch_size=int((s or {}).get("budget_per_round", parsed["batch_size"])) if np.isfinite(fnum((s or {}).get("budget_per_round", parsed["batch_size"]))) else np.nan,
                   seed=int((s or {}).get("seed", parsed["seed"])) if np.isfinite(fnum((s or {}).get("seed", parsed["seed"]))) else np.nan,
                   campaign=campaign_for(d.name), run_directory=str(d), status=status,
                   final_completed_episode=final_ep, target_episode=target,
                   final_labelled_budget=final_budget, has_benchmark_summary=sp.exists(),
                   has_learning_curve=bool(recs), has_selected_indices=has_sel,
                   has_runtime=has_runtime, modification_time=datetime.fromtimestamp(d.stat().st_mtime, timezone.utc).isoformat(),
                   exclusion_reason=reason, final_test_accuracy=fnum((s or {}).get("final_test_accuracy")),
                   final_val_accuracy=fnum((s or {}).get("final_val_accuracy")))
        rows.append(row)
        if complete:
            for r in recs:
                curves.append({**{k: row[k] for k in ["dataset","backbone","method","alpha","delta_coarse","delta_fine","batch_size","seed","run_directory"]},
                               "episode": int(r.get("episode", -1)),
                               "labelled_budget": int(r.get("labeled_count_before_sampling", 0)),
                               "test_accuracy": fnum(r.get("test_accuracy")),
                               "acquisition_time_sec": fnum(r.get("acquisition_time_sec")),
                               "train_time_sec": fnum(r.get("train_time_sec")),
                               "round_time_sec": fnum(r.get("round_time_sec")),
                               "sampling_metadata": r.get("sampling_metadata", {})})
    return pd.DataFrame(rows), pd.DataFrame(curves)


def scan_baselines():
    runs, curves = [], []
    for sp in OUTPUT.rglob("benchmark_summary.json"):
        if "_quarantine" in str(sp): continue
        if not re.match(r"^(random|coreset|core_set|uncertainty|least_confidence|margin|entropy|probcover|prob_cover)_\d+_\d+b\Z", sp.parent.name.lower()): continue
        s = read_summary(sp)
        if not s: continue
        raw = str(s.get("sampling_fn", "")).lower()
        disp = BASE_METHODS.get(raw)
        if not disp: continue
        recs = s.get("episode_records", []) or []
        if max([int(r.get("episode", -1)) for r in recs] + [-1]) < 100: continue
        base = dict(dataset=s.get("dataset", sp.parents[2].name), backbone=s.get("model", sp.parents[1].name),
                    method=disp, seed=int(s.get("seed", -1)), batch_size=int(s.get("budget_per_round", 50)),
                    run_directory=str(sp.parent))
        runs.append({**base, "final_test_accuracy": fnum(s.get("final_test_accuracy"))})
        for r in recs:
            curves.append({**base, "episode": int(r.get("episode", -1)),
                           "labelled_budget": int(r.get("labeled_count_before_sampling", 0)),
                           "test_accuracy": fnum(r.get("test_accuracy")),
                           "acquisition_time_sec": fnum(r.get("acquisition_time_sec")),
                           "train_time_sec": fnum(r.get("train_time_sec")),
                           "round_time_sec": fnum(r.get("round_time_sec"))})
    return pd.DataFrame(runs), pd.DataFrame(curves)


def label_mstc(a):
    return "Fine-only MSTC" if a == 0 else ("Coarse-only MSTC" if a == 1 else "MSTC")


def aggregate(curves):
    if curves.empty: return pd.DataFrame(), pd.DataFrame()
    keys = ["dataset","backbone","method","alpha","delta_coarse","delta_fine","batch_size","seed","run_directory"]
    seed = curves.groupby(keys, dropna=False).agg(
        final_accuracy=("test_accuracy", "last"), aulc=("test_accuracy", "mean"),
        total_acquisition_time=("acquisition_time_sec", "sum"), mean_acquisition_time=("acquisition_time_sec", "mean"),
        total_training_time=("train_time_sec", "sum"), mean_round_time=("round_time_sec", "mean"),
        total_round_time=("round_time_sec", "sum")).reset_index()
    gkeys = keys[:-2]
    agg = seed.groupby(gkeys, dropna=False).agg(
        final_accuracy=("final_accuracy","mean"), final_std=("final_accuracy","std"), final_sem=("final_accuracy",sem),
        aulc=("aulc","mean"), aulc_std=("aulc","std"), aulc_sem=("aulc",sem), n=("seed","nunique"),
        acquisition_per_round=("mean_acquisition_time","mean"), total_acquisition_time=("total_acquisition_time","mean"),
        training_time=("total_training_time","mean"), complete_round_time=("total_round_time","mean")).reset_index()
    return seed, agg


def latex_table(df, path, caption):
    cols = list(df.columns)
    align = "l" + "r" * (len(cols)-1)
    body = df.to_latex(index=False, escape=True, na_rep="--", float_format=lambda x: f"{x:.2f}")
    text = "\\begin{table}[htbp]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n" + body.replace("\\begin{tabular}", "\\begin{tabular}") + f"\\caption{{{caption}}}\n\\end{{table}}\n"
    path.write_text(text)


def save_table(df, stem, caption):
    df.to_csv(TAB / f"{stem}.csv", index=False)
    latex_table(df, TAB / f"{stem}.tex", caption)


def main_subset(topo_curves, base_curves):
    t = topo_curves[(topo_curves.backbone == "resnet18") & (topo_curves.method == "MSTC") &
                    (topo_curves.delta_coarse == .7) & (topo_curves.delta_fine == .5) &
                    (topo_curves.batch_size == 50) & (topo_curves.alpha.isin([0,.7,1]))].copy()
    t["display"] = t.alpha.map(label_mstc)
    b = base_curves[(base_curves.backbone == "resnet18") & (base_curves.batch_size == 50)].copy()
    b["display"] = b.method
    return pd.concat([t, b], ignore_index=True, sort=False)


def curve_plot(data, methods, path, datasets=("CIFAR10","CIFAR100","TINYIMAGENET")):
    fig, axes = plt.subplots(1, len(datasets), figsize=((14.5, 4.4) if len(datasets) > 1 else (6.8, 4.4)), sharex=False)
    if len(datasets) == 1: axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = data[data.dataset == ds]
        for m in methods:
            q = sub[sub.display == m]; color = COLORS.get(m, TABLEAU[methods.index(m) % len(TABLEAU)])
            if q.empty: continue
            z = q.groupby("labelled_budget").test_accuracy.agg(["mean", sem]).reset_index()
            ax.plot(z.labelled_budget, z["mean"], label=m, color=color, lw=1.5)
            if z["sem"].notna().any(): ax.fill_between(z.labelled_budget, z["mean"]-z["sem"].fillna(0), z["mean"]+z["sem"].fillna(0), color=color, alpha=.14)
        ax.set_xlabel("Labelled samples"); ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.set_title({"CIFAR10":"CIFAR-10","CIFAR100":"CIFAR-100","TINYIMAGENET":"TinyImageNet"}[ds])
    axes[0].set_ylabel("Test accuracy (%)"); handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5,max(1,len(labels))), frameon=False)
    fig.tight_layout(rect=(0,0.12,1,1)); fig.savefig(path, dpi=240, bbox_inches="tight"); fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight"); plt.close(fig)


def bar_plot(df, x, hue, metrics, path):
    fig, axes = plt.subplots(1, len(metrics), figsize=(14.5,4.2))
    if len(metrics)==1: axes=[axes]
    xs=list(df[x].drop_duplicates()); hs=list(df[hue].drop_duplicates()); width=.8/max(1,len(hs))
    for ax, metric in zip(axes,metrics):
        for j,h in enumerate(hs):
            q=df[df[hue]==h].set_index(x).reindex(xs)
            err_col = {"final_accuracy":"final_sem", "aulc":"aulc_sem"}.get(metric); yerr = q[err_col] if err_col in q.columns and not q[err_col].isna().all() else None; ax.bar(np.arange(len(xs))+(j-(len(hs)-1)/2)*width,q[metric],width,yerr=yerr,capsize=3,label=str(h),color=COLORS.get(str(h), TABLEAU[j % len(TABLEAU)]))
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, rotation=25, ha="right", fontsize=8); ax.set_ylabel(metric.replace("_"," ").title()); ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc="lower center",ncol=min(5,len(labels)),frameon=False)
    fig.tight_layout(rect=(0,0.12,1,1)); fig.savefig(path,dpi=240,bbox_inches="tight"); fig.savefig(path.with_suffix(".png"),dpi=240,bbox_inches="tight"); plt.close(fig)


def paired_tests(seed_df):
    rows=[]
    main=seed_df[(seed_df.backbone=="resnet18")&(seed_df.batch_size==50)]
    configs={"MSTC":main[(main.method=="MSTC")&(main.alpha==.7)&(main.delta_coarse==.7)&(main.delta_fine==.5)],
             "Coarse-only MSTC":main[(main.method=="MSTC")&(main.alpha==1)&(main.delta_coarse==.7)],
             "Fine-only MSTC":main[(main.method=="MSTC")&(main.alpha==0)&(main.delta_fine==.5)]}
    for ds in sorted(main.dataset.unique()):
        for a,b in [("MSTC","Coarse-only MSTC"),("MSTC","Fine-only MSTC")]:
            m=configs[a][configs[a].dataset==ds].merge(configs[b][configs[b].dataset==ds],on="seed",suffixes=("_a","_b"))
            for metric in ("final_accuracy","aulc"):
                d=(m[f"{metric}_a"]-m[f"{metric}_b"]).dropna().to_numpy()
                if not len(d): continue
                rng=np.random.default_rng(20260805); boots=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(10000)])
                try: w=stats.wilcoxon(d,mode="exact") if np.any(d) else None
                except ValueError: w=None
                rows.append(dict(dataset=ds,comparison=f"{a} - {b}",metric=metric,n=len(d),mean_paired_difference=d.mean(),
                                 t_stat=stats.ttest_1samp(d,0).statistic if len(d)>1 else np.nan,
                                 t_p=stats.ttest_1samp(d,0).pvalue if len(d)>1 else np.nan,
                                 wilcoxon_stat=w.statistic if w else np.nan,wilcoxon_p=w.pvalue if w else np.nan,
                                 bootstrap_ci_low=np.quantile(boots,.025),bootstrap_ci_high=np.quantile(boots,.975)))
    return pd.DataFrame(rows)


def structural_table(curves):
    rows=[]
    q=curves[(curves.dataset=="CIFAR100")&(curves.backbone=="resnet18")&(curves.batch_size==50)&
             (curves.method=="MSTC")&(curves.delta_coarse==.7)&(curves.delta_fine==.5)&(curves.alpha.isin([0,.7,1]))&
             (curves.episode.isin([0,10,25,50,75,100]))]
    for _,r in q.iterrows():
        md=r.sampling_metadata if isinstance(r.sampling_metadata,dict) else {}
        rows.append(dict(dataset=r.dataset,method=label_mstc(r.alpha),seed=r.seed,episode=r.episode,
                         labelled_budget=r.labelled_budget,n=1,
                         vertex_coverage_coarse=fnum(md.get("coverage_fraction_coarse")),
                         vertex_coverage_fine=fnum(md.get("coverage_fraction_fine")),
                         selected_gain_mean=fnum(md.get("selected_score_mean")),
                         beta_0=np.nan,uncovered_components=np.nan,component_coverage=np.nan,
                         singleton_component_proportion=np.nan,largest_component_fraction=np.nan,
                         mean_component_size=np.nan,median_component_size=np.nan,
                         components_touched_by_batch=np.nan,neighbourhood_redundancy=np.nan))
    return pd.DataFrame(rows)


def write_equivalence():
    (PACK/"single_scale_equivalence_audit.md").write_text("""# Single-scale equivalence audit

The deleted historical TopoCover outputs were not reconstructed or substituted.

## Code-path comparison

Original `TopoCover` builds a sparse k-nearest-neighbour approximation to a fixed-radius graph, marks vertices in labelled neighbourhoods as covered, recomputes connected components of the uncovered induced graph, and scores a candidate by the number of uncovered components touched by its neighbourhood. It uses deterministic candidate ordering/tie-breaking from its greedy implementation, adds each selected neighbourhood to coverage, and falls back to deterministic unused candidates when no positive gain remains.

`MultiScaleTopoCover` independently builds coarse and fine radius graphs. Its score is `alpha * coarse_gain + (1-alpha) * fine_gain`; coverage and component labels are updated at both scales after every selection. Its tie-breaking, batch update, and zero-gain fallback implementation are not the same code path as historical `TopoCover`.

## Finding

- **Coarse-only MSTC (`alpha=1`) is a conceptual stand-in, not verified exact equivalence.** Fine-scale state is still constructed and updated, and the implementation differs in graph construction details, tie-breaking, and fallback behaviour.
- **Fine-only MSTC (`alpha=0`) is likewise a conceptual single-scale control, not verified exact equivalence** to a hypothetical fine-radius historical TopoCover.

Accordingly, the evidence pack consistently calls these runs “Coarse-only MSTC” and “Fine-only MSTC”; neither is relabelled as original TopoCover.
""")


def write_prose(main_tab, missing, pilots, campaigns):
    def row(ds,m):
        q=main_tab[(main_tab.dataset==ds)&(main_tab.method==m)]
        return q.iloc[0] if len(q) else None
    lines=["# TopoCover and MultiScaleTopoCover findings","",
           "TopoCover and MultiScaleTopoCover provide a constructive demonstration that $H_0$ connectivity can be translated from a diagnostic description into an active-learning acquisition signal.","",
           "The principal comparison uses the preselected MultiScaleTopoCover configuration $\\alpha=0.7$, $\\delta_c=0.70$, $\\delta_f=0.50$, and acquisition batch size 50 with a ResNet-18 backbone. No dataset-specific configuration was silently substituted.","","## Main evidence",""]
    for ds in ["CIFAR10","CIFAR100","TINYIMAGENET"]:
        q=main_tab[main_tab.dataset==ds]
        if len(q):
            best=q.loc[q.final_accuracy.idxmax()]
            lines.append(f"- {ds}: among protocol-matched methods present, {best.method} had the highest mean final test accuracy ({best.final_accuracy:.2f}%, SEM {best.final_sem:.2f}, n={int(best.n)}). This is a configuration-specific result, not evidence of universal superiority.")
    lines += ["","Multiscale effects are dataset- and scale-dependent. Single-seed topology variants remain exploratory and are not used for definitive ranking. All means exclude incomplete runs."]
    (ROOT/"THESIS_TOPOCOVER_FINDINGS.md").write_text("\n".join(lines)+"\n")
    lim=["# TopoCover and MultiScaleTopoCover limitations","",
         f"- The audit found {len(missing)} missing or partial topology runs; none were aggregated.",
         "- Historical original TopoCover outputs that were removed were not reconstructed. Alpha endpoints are single-scale MSTC controls, not historical TopoCover results.",
         "- Structural summaries use only quantities preserved in benchmark metadata. Exact $\\beta_0$, uncovered-component counts, component-size distributions, batch component touch counts, and neighbourhood redundancy are left unavailable where saved graph objects were absent.",
         "- The representation used for acquisition is a frozen self-supervised embedding, while the supervised classifier is trained from scratch under the configured active-learning protocol.",
         "- Small seed counts limit power; effect sizes, seed consistency, and bootstrap intervals should be considered alongside p-values.",
         "- Exploratory variants have n=1 and cannot establish inferiority or superiority."]
    (ROOT/"THESIS_TOPOCOVER_LIMITATIONS.md").write_text("\n".join(lim)+"\n")
    tex="""\\section{Topology-guided acquisition results}
TopoCover and MultiScaleTopoCover (MSTC) provide a constructive demonstration that $H_0$ connectivity can be translated from a diagnostic description into an active-learning acquisition signal. The principal MSTC configuration was fixed in advance at $\\alpha=0.7$, $\\delta_c=0.70$, $\\delta_f=0.50$, with acquisition batches of 50. Results are reported only for completed runs and always include the number of completed seeds. The evidence indicates that connectivity-guided acquisition can be competitive in particular settings, but its benefit depends on dataset, scale, backbone, and acquisition batch size. Connectivity alone is therefore an operational acquisition signal rather than a universally sufficient criterion.

The $\\alpha=1$ and $\\alpha=0$ experiments are described as coarse-only and fine-only MSTC controls. They are not substituted for deleted historical TopoCover results because exact code equivalence was not established. Exploratory topology variants use one seed and are interpreted only as pilots.
"""
    (ROOT/"THESIS_TOPOCOVER_RESULTS.tex").write_text(tex)


def main():
    for p in (PACK,FIG,TAB): p.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white","pdf.fonttype":42})
    manifest, tc = scan_runs(); br, bc = scan_baselines()
    manifest.to_csv(PACK/"run_manifest.csv",index=False)
    missing=manifest[manifest.status!="complete"].copy(); missing.to_csv(PACK/"missing_or_partial_runs.csv",index=False)
    excluded=manifest[manifest.exclusion_reason.astype(str)!=""].copy(); excluded.to_csv(PACK/"excluded_runs.csv",index=False)
    matrix=manifest.groupby(["dataset","backbone","method","alpha","delta_coarse","delta_fine","batch_size"],dropna=False).agg(found_runs=("seed","count"),complete_runs=("status",lambda x:(x=="complete").sum()),partial_or_missing=("status",lambda x:(x!="complete").sum())).reset_index()
    matrix.to_csv(PACK/"completion_matrix.csv",index=False)
    seed,agg=aggregate(tc)
    # Add baselines to common aggregate schema.
    if not bc.empty:
        bx=bc.assign(alpha=np.nan,delta_coarse=np.nan,delta_fine=np.nan)
        bseed,bagg=aggregate(bx)
        allseed=pd.concat([seed,bseed],ignore_index=True); allagg=pd.concat([agg,bagg],ignore_index=True)
    else: allseed,allagg=seed,agg
    allseed.to_csv(PACK/"seed_level_metrics.csv",index=False); allagg.to_csv(PACK/"configuration_metrics.csv",index=False)
    mainc=main_subset(tc,bc)
    # Tables
    mt=allagg[(allagg.backbone=="resnet18")&(allagg.batch_size==50)].copy()
    mst=mt[(mt.method=="MSTC")&(mt.delta_coarse==.7)&(mt.delta_fine==.5)&(mt.alpha==.7)].assign(method="MSTC")
    controls=mt[(mt.method=="MSTC")&(mt.delta_coarse==.7)&(mt.delta_fine==.5)&(mt.alpha.isin([0,1]))].copy(); controls["method"]=controls.alpha.map(label_mstc)
    bases=mt[mt.method.isin(BASE_METHODS.values())]
    table1=pd.concat([mst,controls,bases])[['dataset','method','final_accuracy','final_std','final_sem','aulc','aulc_sem','n']].sort_values(['dataset','method'])
    save_table(table1,"table1_main_resnet18","Main ResNet-18 predictive results (mean, standard deviation, SEM, and completed seeds).")
    sv=allagg[(allagg.method=="MSTC")&(allagg.delta_coarse==.7)&(allagg.delta_fine==.5)&(allagg.batch_size==50)&(allagg.alpha.isin([0,.7,1]))].copy(); sv["active_scales"]=sv.alpha.map({0.:"fine only",.7:"coarse + fine",1.:"coarse only"})
    save_table(sv[['dataset','backbone','alpha','active_scales','final_accuracy','final_sem','aulc','aulc_sem','n']],"table2_single_vs_multiscale","Single-scale and multiscale MSTC controls.")
    sc=allagg[(allagg.backbone=="resnet18")&(allagg.method=="MSTC")&(allagg.alpha==.7)&(allagg.batch_size==50)&(allagg.delta_coarse.isin([.6,.7,.8]))].copy(); sc["scale_pair"]=sc.apply(lambda r:f"C{int(r.delta_coarse*100):03d}/F{int(r.delta_fine*100):03d}",axis=1)
    save_table(sc[['dataset','scale_pair','final_accuracy','final_sem','aulc','aulc_sem','n']],"table3_scale_sensitivity_resnet18","ResNet-18 MSTC scale sensitivity; alpha 0.7.")
    bu=allagg[(allagg.dataset=="CIFAR100")&(allagg.backbone=="resnet18")&(allagg.method=="MSTC")&(allagg.alpha==.7)&(allagg.delta_coarse==.7)&(allagg.delta_fine==.5)].copy(); bu=bu[~((bu.batch_size==500)&(bu.n<5))]; bu["labelled_budget"]=bu.batch_size*100+bu.batch_size
    save_table(bu[['batch_size','labelled_budget','final_accuracy','final_sem','aulc','aulc_sem','n']],"table4_budget_sensitivity_cifar100","CIFAR-100 acquisition-batch-size sensitivity; incomplete B=500 excluded.")
    save_table(sv[['dataset','backbone','alpha','active_scales','final_accuracy','final_sem','aulc','aulc_sem','n']],"table5_cross_backbone","Cross-backbone MSTC summary at C070/F050.")
    st=structural_table(tc); save_table(st,"table6_structural_coverage","Stored structural coverage metadata; unavailable graph-derived quantities are shown as dashes.")
    rt=allagg[['dataset','backbone','method','alpha','batch_size','acquisition_per_round','total_acquisition_time','training_time','complete_round_time','n']].dropna(subset=['acquisition_per_round'],how='all')
    save_table(rt,"table7_runtime","Runtime summaries over completed runs.")
    pilots=allagg[allagg.method.isin(["WeightedTopo","FrontierTopo","PhasedTopo","PersistProbCover","W1Proxy","MergeTreeCover"])].copy(); pilots["interpretation_warning"]="Exploratory single-seed pilot; no definitive ranking"
    best_rows=sc.loc[sc.groupby("dataset").final_accuracy.idxmax()].copy(); best_rows["method"]=best_rows["scale_pair"].map(lambda x:f"Best-observed MSTC ({x})")
    best_table=pd.concat([best_rows[["dataset","method","final_accuracy","final_sem","aulc","aulc_sem","n"]],bases[["dataset","method","final_accuracy","final_sem","aulc","aulc_sem","n"]]],ignore_index=True).sort_values(["dataset","method"])
    save_table(best_table,"table9_best_observed_mstc_with_baselines","Best observed MSTC scale per dataset in the evaluated grid, compared with protocol-matched baselines; this is a post-hoc sensitivity summary, not the canonical comparison.")
    save_table(pilots[['dataset','backbone','method','final_accuracy','aulc','n','interpretation_warning']],"table8_exploratory_variants","Exploratory single-seed topology pilots.")
    paired_tests(allseed).to_csv(PACK/"paired_statistical_comparisons.csv",index=False)
    # Figures
    main_methods=["MSTC","Coarse-only MSTC","Fine-only MSTC","ProbCover","Random","CoreSet","Least Confidence","Margin","Entropy"]
    best_parts=[]
    for _,r in best_rows.iterrows():
        q=tc[(tc.dataset==r.dataset)&(tc.backbone=="resnet18")&(tc.method=="MSTC")&(tc.alpha==.7)&(tc.batch_size==50)&(tc.delta_coarse==r.delta_coarse)&(tc.delta_fine==r.delta_fine)].copy(); q["display"]="Best-observed MSTC"; best_parts.append(q)
    bestc=pd.concat(best_parts+[mainc[mainc.display.isin(["ProbCover","Random","CoreSet","Least Confidence","Margin","Entropy"])]],ignore_index=True,sort=False)
    COLORS["Best-observed MSTC"]=TABLEAU[9]
    curve_plot(bestc,["Best-observed MSTC","ProbCover","Random","CoreSet","Least Confidence","Margin","Entropy"],FIG/"10_best_observed_mstc_with_baselines_resnet18.pdf")
    curve_plot(mainc,main_methods,FIG/"01_main_learning_curves_resnet18.pdf")
    bar_plot(table1,"dataset","method",["final_accuracy","aulc"],FIG/"02_main_final_and_aulc_resnet18.pdf")
    curve_plot(mainc,["Fine-only MSTC","MSTC","Coarse-only MSTC"],FIG/"03_single_vs_multiscale_resnet18.pdf")
    scur=tc[(tc.backbone=="resnet18")&(tc.method=="MSTC")&(tc.alpha==.7)&(tc.batch_size==50)].copy(); scur["display"]=scur.apply(lambda r:f"C{int(r.delta_coarse*100):03d}/F{int(r.delta_fine*100):03d}",axis=1)
    curve_plot(scur,["C060/F040","C070/F050","C080/F060"],FIG/"04_scale_sensitivity_resnet18.pdf")
    bcur=tc[(tc.dataset=="CIFAR100")&(tc.backbone=="resnet18")&(tc.method=="MSTC")&(tc.alpha==.7)&(tc.delta_coarse==.7)&(tc.delta_fine==.5)&(tc.batch_size.isin(bu.batch_size))].copy(); bcur["display"]=bcur.batch_size.map(lambda x:f"B={int(x)}")
    curve_plot(bcur,[f"B={int(x)}" for x in sorted(bu.batch_size.unique())],FIG/"05_budget_size_sensitivity_cifar100_resnet18.pdf",datasets=("CIFAR100",))
    # Structural figure only uses preserved metadata.
    if len(st):
        z=st.groupby(['method','episode']).agg(vertex_coverage_coarse=('vertex_coverage_coarse','mean'),vertex_coverage_fine=('vertex_coverage_fine','mean'),selected_gain_mean=('selected_gain_mean','mean')).reset_index()
        fig,axs=plt.subplots(1,3,figsize=(14.5,4.2));
        for ax,col in zip(axs,['vertex_coverage_coarse','vertex_coverage_fine','selected_gain_mean']):
            for m,g in z.groupby('method'): ax.plot(g.episode,g[col],label=m,color=COLORS.get(m),linewidth=1.9);
            ax.set_xlabel("Episode"); ax.set_ylabel(col.replace("_"," ").title()); ax.grid(True,linewidth=0.4,alpha=0.35)
        h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=3,frameon=False); fig.tight_layout(rect=(0,0.12,1,1)); fig.savefig(FIG/"06_structural_coverage_cifar100_resnet18.pdf",dpi=240,bbox_inches="tight"); fig.savefig(FIG/"06_structural_coverage_cifar100_resnet18.png",dpi=240,bbox_inches="tight"); plt.close(fig)
    bar_plot(sv.assign(configuration=sv.alpha.map(label_mstc), dataset_backbone=sv.dataset+"\n"+sv.backbone),"dataset_backbone","configuration",["final_accuracy"],FIG/"07_cross_backbone_summary.pdf")
    if len(pilots): bar_plot(pilots,"method","dataset",["final_accuracy"],FIG/"08_topological_variant_pilots.pdf")
    if len(rt): bar_plot(rt[rt.backbone=="resnet18"].groupby(["dataset","method"],as_index=False)[["acquisition_per_round","complete_round_time"]].mean().rename(columns={"method":"configuration"}),"dataset","configuration",["acquisition_per_round","complete_round_time"],FIG/"09_runtime_comparison.pdf")
    write_equivalence()
    campaign_status={c:{"exists":(CAMPAIGNS/c).exists(),"path":str(CAMPAIGNS/c)} for c in EXPECTED_CAMPAIGNS}
    provenance={"generated_utc":datetime.now(timezone.utc).isoformat(),"generator":str(Path(__file__).resolve()),
                "figure_design_reference":"/vast/s219110279/outputs/lidcover_chapter5 (Tableau palette, 14.5-inch panels, thin grey grid, SEM bands/bars, shared frameless lower legends, 240 dpi)",
                "aulc_definition":"Arithmetic mean of all completed episode-level test accuracies per seed, matching analysis/generate_topocover_results.py lines 217-227.",
                "experiment_output_root":str(OUTPUT),"campaigns":campaign_status,
                "source_run_directories":sorted(manifest.run_directory.tolist()),
                "baseline_run_directories":sorted(br.run_directory.tolist()) if len(br) else [],
                "structural_note":"Only structural quantities explicitly stored in sampling_metadata were reported; missing graph-derived quantities remain NA.",
                "validation":{"partial_runs_excluded":True,"all_tables_include_n":True,"error_bars":"SEM","main_configuration":"alpha=.7,C=.70,F=.50,B=50","pilots_labelled_exploratory":True}}
    (PACK/"provenance.json").write_text(json.dumps(provenance,indent=2))
    write_prose(table1,missing,pilots,campaign_status)
    report={"configurations_found":int(len(matrix)),"complete_runs":int((manifest.status=='complete').sum()),"missing_or_partial_runs":int(len(missing)),"excluded_runs":int(len(excluded)),"figures_pdf":len(list(FIG.glob('*.pdf'))),"figures_png":len(list(FIG.glob('*.png'))),"tables_csv":len(list(TAB.glob('*.csv'))),"tables_tex":len(list(TAB.glob('*.tex'))),"unresolved_issues":["Exact graph-derived structural quantities unavailable without saved graph objects/reconstruction.","Deleted historical TopoCover outputs remain absent."],"supported_claims":["H0 connectivity is operationalised as an acquisition signal.","Performance is configuration- and dataset-dependent."],"unsupported_claims":["Universal topology superiority.","Exact equivalence of alpha endpoints to historical TopoCover.","Definitive pilot rankings."]}
    (PACK/"completion_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
