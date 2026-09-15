#!/usr/bin/env python3
"""Post-hoc LIDCover mechanism analysis from canonical raw trajectories.

No model training or acquisition is run.  Historical ID/kNN caches and saved
selected-ID order are used.  Per-candidate LIDCover gains are accepted only
when their aggregate statistics reproduce run-local acquisition metadata.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/thesis_appendix/lidcover_mechanism"
FIG = ROOT / "figures/thesis_appendix/lidcover_mechanism"
MATRIX = ROOT / "thesis_appendix_audit/_regenerated_lidcover/lidcover_completion_matrix.csv"
FEATURES = ROOT / "results/results/resnet18/cifar-100/pretext/features_seed1.npy"
KEY_EPISODES = (10, 25, 50, 75, 100)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ids(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=True)
    if values.dtype == object:
        values = np.asarray(values.tolist())
    return np.asarray(values, dtype=np.int64).reshape(-1)


def canonical_runs() -> dict[tuple[str, int], Path]:
    runs = {}
    with MATRIX.open(newline="") as handle:
        for row in csv.DictReader(handle):
            radius = row["delta0"] if row["method"] == "LIDCover" else row["delta"]
            if (
                "candidate" not in row["experiment_id"] and row["status"] == "COMPLETE"
                and row["dataset"] == "CIFAR100" and row["backbone"] == "resnet18"
                and row["method"] in {"LIDCover", "ProbCover"}
                and radius and abs(float(radius) - 0.25) < 1e-9
            ):
                runs[(row["method"], int(row["seed"]))] = Path(row["source_path"])
    if len(runs) != 10:
        raise RuntimeError(f"expected 10 canonical matched runs, found {len(runs)}")
    return runs


def quartile(value: float, boundaries: np.ndarray) -> int:
    return int(np.searchsorted(boundaries, value, side="left") + 1)


def episode_record(run: Path, episode: int) -> dict:
    path = run / f"episode_{episode}/episode_summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def lid_neighbourhood(global_id, eligible_mask, ids, knn_idx, knn_dist, radius):
    keep = eligible_mask[knn_idx[global_id]] & (knn_dist[global_id] < radius)
    neighbours = knn_idx[global_id][keep].astype(np.int64)
    if global_id not in neighbours:
        neighbours = np.concatenate([np.asarray([global_id], dtype=np.int64), neighbours])
    return neighbours


def reconstruct_lidcover(run: Path, seed: int, episode: int, ids, knn_idx, knn_dist) -> tuple[list[dict], dict]:
    if episode == 100:
        return [], {"status": "NOT_APPLICABLE_EVALUATION_ONLY"}
    record = episode_record(run, episode)
    selected = np.asarray(record.get("active_set_ids") or [], dtype=np.int64)
    lset = load_ids(run / f"episode_{episode}/lSet.npy")
    uset = load_ids(run / f"episode_{episode}/uSet.npy")
    eligible = np.sort(np.concatenate([lset, uset]))
    eligible_mask = np.zeros(len(ids), dtype=bool)
    eligible_mask[eligible] = True
    median_id = float(np.median(ids[eligible]))
    effective_delta = float(record.get("sampling_metadata", {}).get("effective_delta", 0.2125))
    radii = effective_delta * ((ids[eligible] + 1e-12) / (median_id + 1e-12)) ** -1.0
    id_bounds = np.quantile(ids[eligible], [0.25, 0.50, 0.75])
    radius_bounds = np.quantile(radii, [0.25, 0.50, 0.75])
    radius_lookup = dict(zip(eligible.tolist(), radii.tolist()))
    covered = np.zeros(len(ids), dtype=bool)
    for global_id in lset:
        radius = radius_lookup[int(global_id)]
        covered[lid_neighbourhood(int(global_id), eligible_mask, ids, knn_idx, knn_dist, radius)] = True
    rows = []
    for order, global_id in enumerate(selected, start=1):
        radius = radius_lookup[int(global_id)]
        neighbours = lid_neighbourhood(int(global_id), eligible_mask, ids, knn_idx, knn_dist, radius)
        gain = int(np.sum(~covered[neighbours]))
        covered[neighbours] = True
        distances = knn_dist[int(global_id)]
        rows.append({
            "method": "LIDCover", "seed": seed, "episode": episode,
            "labelled_budget": len(lset), "selection_order": order,
            "selected_id": int(global_id), "fixed_local_id": float(ids[global_id]),
            "normalised_local_id": float(ids[global_id] / median_id),
            "candidate_radius": float(radius), "knn_distance_min": float(distances.min()),
            "knn_distance_median": float(np.median(distances)),
            "knn_distance_mean": float(distances.mean()), "knn_distance_max": float(distances.max()),
            "uncovered_coverage_gain": gain, "coverage_gain_status": "PENDING_VALIDATION",
            "id_quartile": quartile(float(ids[global_id]), id_bounds),
            "radius_quartile": quartile(float(radius), radius_bounds), "raw_path": str(run),
        })
    metadata = record.get("sampling_metadata", {})
    gains = np.asarray([row["uncovered_coverage_gain"] for row in rows], dtype=float)
    comparisons = {
        "mean": (float(gains.mean()) if len(gains) else np.nan, metadata.get("selected_coverage_gain_mean")),
        "min": (float(gains.min()) if len(gains) else np.nan, metadata.get("selected_coverage_gain_min")),
        "max": (float(gains.max()) if len(gains) else np.nan, metadata.get("selected_coverage_gain_max")),
        "std": (float(gains.std()) if len(gains) else np.nan, metadata.get("selected_coverage_gain_std")),
    }
    validated = all(saved is not None and abs(reconstructed - float(saved)) <= 1e-5 for reconstructed, saved in comparisons.values())
    status = "VERIFIED_AGAINST_RUN_AGGREGATE" if validated else "UNVERIFIABLE"
    for row in rows:
        row["coverage_gain_status"] = status
        if not validated:
            row["uncovered_coverage_gain"] = np.nan
    return rows, {
        "status": status, "reconstructed": {key: value[0] for key, value in comparisons.items()},
        "saved": {key: value[1] for key, value in comparisons.items()},
        "id_cache_path": metadata.get("id_cache_path"), "knn_cache_path": metadata.get("knn_cache_path"),
    }


def posthoc_probcover(run: Path, seed: int, episode: int, ids, knn_dist) -> tuple[list[dict], dict]:
    if episode == 100:
        return [], {"status": "NOT_APPLICABLE_EVALUATION_ONLY"}
    record = episode_record(run, episode)
    selected = np.asarray(record.get("active_set_ids") or [], dtype=np.int64)
    lset = load_ids(run / f"episode_{episode}/lSet.npy")
    uset = load_ids(run / f"episode_{episode}/uSet.npy")
    eligible = np.sort(np.concatenate([lset, uset]))
    median_id = float(np.median(ids[eligible]))
    id_bounds = np.quantile(ids[eligible], [0.25, 0.50, 0.75])
    effective_delta = 0.25 * 0.85
    rows = []
    for order, global_id in enumerate(selected, start=1):
        distances = knn_dist[int(global_id)]
        rows.append({
            "method": "ProbCover", "seed": seed, "episode": episode,
            "labelled_budget": len(lset), "selection_order": order,
            "selected_id": int(global_id), "fixed_local_id": float(ids[global_id]),
            "normalised_local_id": float(ids[global_id] / median_id),
            "candidate_radius": effective_delta, "knn_distance_min": float(distances.min()),
            "knn_distance_median": float(np.median(distances)),
            "knn_distance_mean": float(distances.mean()), "knn_distance_max": float(distances.max()),
            "uncovered_coverage_gain": np.nan, "coverage_gain_status": "UNVERIFIABLE_NO_RUN_GAIN_METADATA_OR_CODE_HASH",
            "id_quartile": quartile(float(ids[global_id]), id_bounds), "radius_quartile": 1,
            "raw_path": str(run),
        })
    return rows, {"status": "UNVERIFIABLE_NO_RUN_GAIN_METADATA_OR_CODE_HASH"}


def summarise(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, episode), group in candidates.groupby(["method", "episode"]):
        valid_gain = group[group.coverage_gain_status == "VERIFIED_AGAINST_RUN_AGGREGATE"].uncovered_coverage_gain
        rho = spearmanr(group.fixed_local_id, group.candidate_radius).statistic if group.candidate_radius.nunique() > 1 else np.nan
        rows.append({
            "method": method, "episode": episode, "labelled_budget": int(group.labelled_budget.median()),
            "n_candidates": len(group), "n_seeds": group.seed.nunique(),
            "local_id_mean": group.fixed_local_id.mean(), "local_id_sem": group.groupby("seed").fixed_local_id.mean().sem(),
            "radius_mean": group.candidate_radius.mean(), "radius_sem": group.groupby("seed").candidate_radius.mean().sem(),
            "radius_min": group.candidate_radius.min(), "radius_max": group.candidate_radius.max(),
            "radius_coefficient_of_variation": group.candidate_radius.std(ddof=0) / group.candidate_radius.mean(),
            "spearman_local_id_radius": rho,
            "coverage_gain_mean": valid_gain.mean() if len(valid_gain) else np.nan,
            "coverage_gain_status": "VERIFIED_AGAINST_RUN_AGGREGATE" if len(valid_gain) else "UNVERIFIABLE",
            "high_id_quartile_fraction": float(np.mean(group.id_quartile == 4)),
            "small_radius_quartile_fraction": float(np.mean(group.radius_quartile == 1)),
        })
    return pd.DataFrame(rows)


def make_figures(candidates: pd.DataFrame):
    FIG.mkdir(parents=True, exist_ok=True)
    lid = candidates[candidates.method == "LIDCover"]
    figure, axis = plt.subplots(figsize=(6, 4.5))
    axis.scatter(lid.fixed_local_id, lid.candidate_radius, s=8, alpha=0.25)
    axis.set_xlabel("Fixed local ID")
    axis.set_ylabel("Candidate radius")
    axis.set_yscale("symlog", linthresh=0.1)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIG / "lidcover_local_id_vs_radius.pdf", bbox_inches="tight")
    plt.close(figure)

    verified = lid[lid.coverage_gain_status == "VERIFIED_AGAINST_RUN_AGGREGATE"]
    pivot = verified.groupby(["id_quartile", "radius_quartile"]).uncovered_coverage_gain.mean().unstack()
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(pivot.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(pivot.columns)), [f"R{value}" for value in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), [f"ID{value}" for value in pivot.index])
    axis.set_xlabel("Radius quartile")
    axis.set_ylabel("Local-ID quartile")
    figure.colorbar(image, ax=axis, label="Mean verified uncovered gain")
    figure.tight_layout()
    figure.savefig(FIG / "lidcover_gain_by_id_radius_regime.pdf", bbox_inches="tight")
    plt.close(figure)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = canonical_runs()
    all_rows, validation_rows = [], []
    cache_hashes = {}
    for seed in range(1, 6):
        ids_path = ROOT / f"idpc_cache/resnet18/CIFAR100/seed{seed}/mle_local_k50.npy"
        knn_path = ROOT / f"idpc_cache/resnet18/CIFAR100/seed{seed}/knn_k50.npz"
        ids = np.load(ids_path).astype(np.float32)
        knn = np.load(knn_path)
        knn_idx, knn_dist = knn["idx"].astype(np.int64), knn["dist"].astype(np.float32)
        cache_hashes[seed] = {"id": file_sha256(ids_path), "knn": file_sha256(knn_path)}
        for episode in KEY_EPISODES:
            lid_rows, validation = reconstruct_lidcover(runs[("LIDCover", seed)], seed, episode, ids, knn_idx, knn_dist)
            all_rows.extend(lid_rows)
            validation_rows.append({"method": "LIDCover", "seed": seed, "episode": episode, **validation})
            prob_rows, prob_validation = posthoc_probcover(runs[("ProbCover", seed)], seed, episode, ids, knn_dist)
            all_rows.extend(prob_rows)
            validation_rows.append({"method": "ProbCover", "seed": seed, "episode": episode, **prob_validation})
    candidates = pd.DataFrame(all_rows)
    summary = summarise(candidates)
    validation = pd.DataFrame(validation_rows)
    candidates.to_csv(OUT / "selected_candidate_mechanism.csv", index=False)
    summary.to_csv(OUT / "mechanism_summary.csv", index=False)
    validation.to_csv(OUT / "coverage_gain_validation.csv", index=False)
    make_figures(candidates)
    lid = candidates[candidates.method == "LIDCover"]
    rho = spearmanr(lid.fixed_local_id, lid.candidate_radius).statistic
    verified = int((validation.query("method == 'LIDCover'").status == "VERIFIED_AGAINST_RUN_AGGREGATE").sum())
    provenance = {
        "feature_path": str(FEATURES), "feature_sha256": file_sha256(FEATURES),
        "cache_sha256": cache_hashes, "source_code": str(ROOT / "TypiClust/deep-al/pycls/al/IDProbCover.py"),
        "source_code_sha256": file_sha256(ROOT / "TypiClust/deep-al/pycls/al/IDProbCover.py"),
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (OUT / "README.md").write_text(f"""# LIDCover mechanism analysis

Canonical scope: CIFAR-100 / ResNet-18 / delta0=0.25 / alpha=1 / k_id=k_knn=50, seeds 1–5, compared with matched ProbCover delta0=0.25.

- LIDCover produces heterogeneous candidate radii; see `mechanism_summary.csv` for ranges and coefficients of variation.
- Across reconstructed selected candidates, Spearman(local ID, radius) = **{rho:.6f}**. This is the executed inverse radius rule, not a relationship with difficulty or informativeness.
- Exact LIDCover candidate gain passed saved aggregate validation in **{verified}/20** acquisition cells (E10/E25/E50/E75 × five seeds). E100 is evaluation-only and has no selected candidate.
- ProbCover per-candidate gain is **UNVERIFIABLE** because its run artifacts do not save gain aggregates or a historical code hash. No proxy value is inserted.
- Executed LID caches were computed on all stored training embeddings, while graph eligibility excluded validation vertices. The pool-only reference arrays are analysed separately in the reference-robustness package.
""")
    print(json.dumps({"candidate_rows": len(candidates), "summary_rows": len(summary), "validated_lid_cells": verified}, indent=2))


if __name__ == "__main__":
    main()
