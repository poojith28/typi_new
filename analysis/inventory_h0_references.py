#!/usr/bin/env python3
"""Inventory / validate H0 reference persistence diagrams for the thesis.

Writes:
  outputs/diagnostic_h0/diagnostic_h0_reference_report.md
  outputs/diagnostic_h0/reference/<DATASET>_<backbone>_h0_reference.csv  (CIFAR copies)
"""
from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/vast/s219110279")
PERS_ROOT = REPO / "TypiClust" / "output"
OUT = REPO / "outputs" / "diagnostic_h0"
REF_DIR = OUT / "reference"

DATASETS = {
    "CIFAR10": "cifar10",
    "CIFAR100": "cifar100",
    "TINYIMAGENET": "tinyimagenet",
}
BACKBONES = ["alexnet", "resnet18", "resnet50"]


def find_full_csv(dataset: str, backbone: str) -> Path | None:
    tag = DATASETS[dataset]
    d = PERS_ROOT / dataset / "analysis" / "persistence"
    cand = d / f"{backbone}_{tag}_simclr_h0_full.csv"
    return cand if cand.exists() else None


def parse_simplex(s):
    if isinstance(s, (list, tuple)):
        return list(s)
    return list(ast.literal_eval(s))


def summarise_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    # Prefer scope==all when present (CIFAR full files include per-class rows)
    if "scope" in df.columns:
        df = df[df["scope"] == "all"]
    if "dim" in df.columns:
        df = df[df["dim"] == 0]
    finite = df[np.isfinite(df["death"])].copy()
    deaths = finite["death"].to_numpy(dtype=np.float64)
    pers = finite["persistence"].to_numpy(dtype=np.float64)
    births = finite["birth_simplex"].map(parse_simplex)
    birth_lens = births.map(len)
    bv = births.map(lambda x: x[0] if x else None)
    meta_path = Path(str(path).replace(".csv", "_meta.json"))
    if not meta_path.exists():
        # our MST naming uses same prefix
        meta_path = path.with_name(path.name.replace(".csv", "_meta.json"))
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    return {
        "path": str(path),
        "meta_path": str(meta_path) if meta_path.exists() else None,
        "n_rows_all_scope": int(len(df)),
        "n_finite_pairs": int(len(finite)),
        "n_infinite": int((~np.isfinite(df["death"])).sum()) if "death" in df else 0,
        "death_min": float(deaths.min()) if len(deaths) else None,
        "death_median": float(np.median(deaths)) if len(deaths) else None,
        "death_mean": float(deaths.mean()) if len(deaths) else None,
        "death_max": float(deaths.max()) if len(deaths) else None,
        "pers_min": float(pers.min()) if len(pers) else None,
        "pers_median": float(np.median(pers)) if len(pers) else None,
        "pers_mean": float(pers.mean()) if len(pers) else None,
        "pers_max": float(pers.max()) if len(pers) else None,
        "birth_simplex_encoding": "string list of vertex indices, e.g. \"[40653]\" (H0 birth = one vertex)",
        "birth_simplex_len_unique": sorted({int(x) for x in birth_lens.unique()}),
        "n_unique_birth_vertices": int(pd.Series(bv).nunique()),
        "columns": list(df.columns),
        "meta": meta,
    }


def construction_block(meta: dict) -> list[str]:
    lines = []
    if not meta:
        lines.append("- Construction details: **unresolved** (no `_meta.json`).")
        return lines
    params = meta.get("params") or {}
    libs = meta.get("library_versions") or {}
    method = meta.get("method") or meta.get("script")
    lines += [
        f"- Script / method: `{method}`",
        f"- Library: GUDHI {libs.get('gudhi', 'unresolved')}"
        + (f" (numpy {libs.get('numpy')})" if libs.get("numpy") else ""),
        f"- Filtration: {params.get('filtration') or ('Vietoris–Rips (gudhi.RipsComplex)' if 'sparse' in params else meta.get('equivalence', 'unresolved'))}",
        f"- Metric: {params.get('metric', 'euclidean (GUDHI points=)')}",
        f"- L2 normalisation: {params.get('normalize', 'unresolved')}",
        f"- Graph / kNN restriction: {params.get('knn_restriction', params.get('sparse'))}",
        f"- Max edge length: {params.get('max_edge_length', 'unresolved')}",
        f"- Features: `{params.get('features') or (meta.get('inputs') or {}).get('features', 'unresolved')}`",
    ]
    if meta.get("equivalence"):
        lines.append(f"- Equivalence note: {meta['equivalence']}")
    return lines


def main():
    REF_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    md = [
        "# H0 reference persistence-diagram report",
        "",
        "Thesis chapter: *Diagnostic Analysis of Active Learning*.",
        "",
        "Policy: **one reference H0 PD per dataset/backbone**, computed on the",
        "complete fixed training embedding pool (`features_seed1.npy`).",
        "Downstream diagnostics map selected indices to reference pairs via",
        "`BirthSimplex(p) ∩ S ≠ ∅` — they do **not** recompute a PD per labelled set.",
        "",
        "## Column naming",
        "",
        "On disk the CSVs use snake_case:",
        "",
        "| Thesis brief | File column |",
        "|---|---|",
        "| Birth | `birth` |",
        "| Death | `death` |",
        "| Persistence | `persistence` |",
        "| Birth Simplex | `birth_simplex` |",
        "",
        "H0 births are `0.0`; for finite pairs `persistence == death`.",
        "`birth_simplex` is a single pool index, e.g. `[40653]`.",
        "Indices are training-pool row indices (same space as AL `lSet` / `active_set_ids`).",
        "",
        "## Coverage",
        "",
    ]

    table = [
        "| Dataset | Backbone | Status | Finite H0 pairs | Source |",
        "|---|---|---|---|---|",
    ]

    for ds in DATASETS:
        for bb in BACKBONES:
            path = find_full_csv(ds, bb)
            if path is None:
                table.append(f"| {ds} | {bb} | **MISSING** | — | — |")
                rows.append({"dataset": ds, "backbone": bb, "status": "missing"})
                continue
            info = summarise_csv(path)
            # Stage a copy under diagnostic_h0/reference
            dest = REF_DIR / f"{ds}_{bb}_h0_reference.csv"
            shutil.copy2(path, dest)
            meta_src = Path(str(path).replace(".csv", "_meta.json"))
            if meta_src.exists():
                shutil.copy2(meta_src, REF_DIR / f"{ds}_{bb}_h0_reference_meta.json")
            inf_src = Path(str(path).replace(".csv", "_with_infinite.csv"))
            if inf_src.exists():
                shutil.copy2(inf_src, REF_DIR / f"{ds}_{bb}_h0_reference_with_infinite.csv")

            status = "OK"
            source = "dense GUDHI VR" if (info["meta"].get("params") or {}).get("sparse") in (None, "null") else "see meta"
            if info["meta"].get("script") == "compute_h0_reference_mst.py":
                source = "exact MST + GUDHI SimplexTree"
            table.append(
                f"| {ds} | {bb} | {status} | {info['n_finite_pairs']} | `{path.name}` ({source}) |"
            )
            rows.append({"dataset": ds, "backbone": bb, "status": status, **{k: v for k, v in info.items() if k != "meta"}})

            md += [f"## {ds} / {bb}", ""]
            md += [
                f"- Path: `{path}`",
                f"- Staged copy: `{dest}`",
                f"- Finite H0 pairs (`scope=all`): **{info['n_finite_pairs']}**",
                f"- Death: min={info['death_min']:.6g}, median={info['death_median']:.6g}, "
                f"mean={info['death_mean']:.6g}, max={info['death_max']:.6g}",
                f"- Persistence: min={info['pers_min']:.6g}, median={info['pers_median']:.6g}, "
                f"mean={info['pers_mean']:.6g}, max={info['pers_max']:.6g}",
                f"- Birth Simplex: {info['birth_simplex_encoding']}",
                f"- Unique birth vertices among finite pairs: {info['n_unique_birth_vertices']}",
                f"- Columns: `{', '.join(info['columns'])}`",
                "",
                "### Construction",
                "",
            ]
            md += construction_block(info["meta"])
            md.append("")

    md[md.index("## Coverage") + 2 : md.index("## Coverage") + 2] = []  # noop safety
    # Insert coverage table after Coverage header
    cov_i = md.index("## Coverage")
    md = md[: cov_i + 2] + table + [""] + md[cov_i + 2 :]

    md += [
        "## Missing / unresolved",
        "",
        "- TinyImageNet full-pool dense GUDHI VR previously **timed out** at 1 day "
        "(jobs 2931137–2931139). Exact MST+GUDHI jobs replace that path.",
        "- Do **not** use `*_h0_n10k.csv` as the chapter reference — only 10k indices "
        "appear in simplices.",
        "- Posthoc `h0_reference_events.csv` (kNN-MST edges) is a **different** "
        "representation and is not interchangeable with these GUDHI/MST PD CSVs.",
        "",
        "## Next step",
        "",
        "Once all nine references are present under `outputs/diagnostic_h0/reference/`, "
        "run the touched-pair diagnostic pipeline (exclusive ΔL_t / cumulative L_t).",
        "",
    ]

    report = OUT / "diagnostic_h0_reference_report.md"
    report.write_text("\n".join(md) + "\n")
    pd.DataFrame(rows).to_csv(OUT / "diagnostic_h0_reference_inventory.csv", index=False)
    print(f"Wrote {report}")
    print(f"Staged references in {REF_DIR}")
    missing = [r for r in rows if r.get("status") == "missing"]
    print(f"Missing full references: {len(missing)}")
    for r in missing:
        print(f"  - {r['dataset']}/{r['backbone']}")


if __name__ == "__main__":
    main()
