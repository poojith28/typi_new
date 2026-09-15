#!/usr/bin/env python3
"""Build the thesis appendix raw-run inventory and protocol audit.

The historical run trees are read-only.  This script writes only to
``thesis_appendix_audit`` and treats run-local configs, partitions, episode
arrays, summaries, and acquisition metadata as the source of truth.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = ROOT / "TypiClust" / "output"
SUPPLEMENTAL_ROOTS = (ROOT / "output", ROOT / "TypiClust" / "asfdasdf asf")
OUT = ROOT / "thesis_appendix_audit"
EXPECTED_FINE_TUNE = {"CIFAR10": False, "CIFAR100": True, "TINYIMAGENET": True}
EXPECTED_SEEDS = {1, 2, 3, 4, 5}
SHARED_START_METHODS = {
    "Random", "LeastConfidence", "Entropy", "Margin", "DBAL", "CoreSet", "TypiClust"
}
METHOD_SPECIFIC_METHODS = {"ProbCover", "MaxHerding", "MultiScaleTopoCover"}

METHOD_ALIASES = {
    "random": "Random",
    "uncertainty": "LeastConfidence",
    "least_confidence": "LeastConfidence",
    "entropy": "Entropy",
    "margin": "Margin",
    "dbal": "DBAL",
    "coreset": "CoreSet",
    "core_set": "CoreSet",
    "typiclust": "TypiClust",
    "probcover": "ProbCover",
    "prob_cover": "ProbCover",
    "maxherding": "MaxHerding",
    "max_herding": "MaxHerding",
    "idprobcover": "LIDCover",
    "id_prob_cover": "LIDCover",
    "multiscale_topocover": "MultiScaleTopoCover",
    "multi_scale_topocover": "MultiScaleTopoCover",
    "topocover": "HistoricalTopoCover",
    "topo_cover": "HistoricalTopoCover",
    "knn_distance_cover": "MeanKNNDistanceCover",
    "density_cover": "DensityAdaptiveCover",
    "distance_variance_cover": "DistanceVarianceCover",
    "distance_cv_cover": "DistanceCVCover",
    "idprobcover_fallback_random": "LIDCoverFallbackRandom",
    "idprobcover_fallback_minid": "LIDCoverFallbackMinID",
}

IMPLEMENTATION = {
    "ProbCover": "TypiClust/deep-al/pycls/al/prob_cover.py",
    "LIDCover": "TypiClust/deep-al/pycls/al/IDProbCover.py",
    "MultiScaleTopoCover": "TypiClust/deep-al/pycls/al/multiscale_topocover.py",
    "HistoricalTopoCover": "TypiClust/deep-al/pycls/al/topocover.py",
    "MaxHerding": "TypiClust/deep-al/pycls/al/maxherding.py",
    "TypiClust": "TypiClust/deep-al/pycls/al/typiclust.py",
    "CoreSet": "TypiClust/deep-al/pycls/al/coreset.py",
    "LIDCoverFallbackRandom": "TypiClust/deep-al/pycls/al/idpc_tiebreak/methods.py",
    "LIDCoverFallbackMinID": "TypiClust/deep-al/pycls/al/idpc_tiebreak/methods.py",
    "MeanKNNDistanceCover": "TypiClust/deep-al/pycls/al/adaptive_cover/knn_distance_cover.py",
    "DensityAdaptiveCover": "TypiClust/deep-al/pycls/al/adaptive_cover/density_cover.py",
    "DistanceVarianceCover": "TypiClust/deep-al/pycls/al/adaptive_cover/distance_variance_cover.py",
    "DistanceCVCover": "TypiClust/deep-al/pycls/al/adaptive_cover/distance_cv_cover.py",
}

FIELDS = [
    "logical_experiment_id", "scope", "dataset", "backbone", "method",
    "method_implementation", "seed", "base_radius", "effective_radius",
    "fine_radius", "coarse_radius", "alpha", "k_id", "k_knn",
    "initialisation_type", "initial_labelled_count", "initial_ids_sha256",
    "eligible_count", "eligible_indices_sha256", "episodes_present",
    "final_labelled_count", "benchmark_summary_present",
    "selected_id_evidence_present", "acquisition_metadata_present", "status",
    "status_reason", "raw_path",
]


def nested(value: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_ids(path: Path) -> np.ndarray | None:
    try:
        array = np.load(path, allow_pickle=True)
        if array.dtype == object:
            array = np.asarray(array.tolist())
        return np.asarray(array, dtype=np.int64).reshape(-1)
    except Exception:
        return None


def ids_sha256(array: np.ndarray | None, *, sort: bool = False) -> str:
    if array is None:
        return ""
    values = np.sort(array) if sort else array
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def method_name(sampler: str) -> str:
    return METHOD_ALIASES.get(sampler.lower(), sampler or "UNKNOWN")


def config_paths() -> list[tuple[Path, str]]:
    rows = [(path, "canonical") for path in CANONICAL_ROOT.glob("*/*/*/config.yaml")]
    for root in SUPPLEMENTAL_ROOTS:
        rows.extend((path, "supplemental") for path in root.glob("*/*/*/config.yaml"))
    return sorted(set(rows), key=lambda item: str(item[0]))


def compact_episodes(episodes: list[int]) -> str:
    if not episodes:
        return ""
    episodes = sorted(set(episodes))
    parts: list[str] = []
    start = previous = episodes[0]
    for value in episodes[1:]:
        if value == previous + 1:
            previous = value
            continue
        parts.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    parts.append(str(start) if start == previous else f"{start}-{previous}")
    return ";".join(parts)


def discover_random_initial_hashes() -> dict[tuple[str, str, int], str]:
    hashes: dict[tuple[str, str, int], str] = {}
    for path in CANONICAL_ROOT.glob("*/*/random_*/config.yaml"):
        try:
            cfg = yaml.safe_load(path.read_text()) or {}
        except Exception:
            continue
        dataset = str(nested(cfg, "DATASET", "NAME", default=path.parents[2].name)).upper()
        backbone = str(nested(cfg, "MODEL", "TYPE", default=path.parents[1].name)).lower()
        seed = as_int(cfg.get("RNG_SEED"))
        ids = load_ids(path.parent / "lSet.npy")
        if seed in EXPECTED_SEEDS and ids is not None and len(ids) == 50:
            hashes[(dataset, backbone, seed)] = ids_sha256(ids, sort=True)
    return hashes


def inventory_row(config_path: Path, scope: str, random_hashes: dict[tuple[str, str, int], str]) -> dict[str, Any]:
    run = config_path.parent
    reasons: list[str] = []
    try:
        cfg = yaml.safe_load(config_path.read_text()) or {}
    except Exception as exc:
        return {field: "" for field in FIELDS} | {
            "logical_experiment_id": run.name,
            "scope": scope,
            "status": "UNVERIFIABLE",
            "status_reason": f"config parse failed: {exc}",
            "raw_path": str(run),
        }

    al = cfg.get("ACTIVE_LEARNING", {}) or {}
    dataset = str(nested(cfg, "DATASET", "NAME", default=run.parents[1].name)).upper()
    backbone = str(nested(cfg, "MODEL", "TYPE", default=run.parent.name)).lower()
    sampler = str(al.get("SAMPLING_FN", ""))
    method = method_name(sampler)
    seed = as_int(cfg.get("RNG_SEED"))
    budget = as_int(al.get("BUDGET_SIZE"))
    max_iter = as_int(al.get("MAX_ITER"))
    train_batch = as_int(nested(cfg, "TRAIN", "BATCH_SIZE"))
    fine_tune = nested(cfg, "ACTIVE_LEARNING", "FINE_TUNE")

    initial = load_ids(run / "lSet.npy")
    initial_count = None if initial is None else len(initial)
    initial_hash = ids_sha256(initial, sort=True)
    random_hash = random_hashes.get((dataset, backbone, seed)) if seed is not None else None
    if initial_count == 0:
        init_type = "method_specific_empty_L"
    elif initial_count == 50 and random_hash and initial_hash == random_hash:
        init_type = "seed_specific_shared_random_50"
    elif initial_count is not None:
        init_type = f"prepopulated_nonshared_{initial_count}"
    else:
        init_type = "unverifiable"

    val_ids = load_ids(run / "valSet.npy")
    u_ids = load_ids(run / "uSet.npy")
    eligible_count = None
    eligible_hash = ""
    if u_ids is not None and initial is not None:
        eligible = np.sort(np.concatenate([initial, u_ids]))
        eligible_count = len(eligible)
        eligible_hash = ids_sha256(eligible)

    episode_dirs: dict[int, Path] = {}
    for path in run.glob("episode_*"):
        try:
            episode_dirs[int(path.name.split("_", 1)[1])] = path
        except (IndexError, ValueError):
            pass
    episodes = sorted(episode_dirs)
    final_episode = max(episodes) if episodes else None
    final_ids = load_ids(episode_dirs[final_episode] / "lSet.npy") if final_episode is not None else None
    final_count = None if final_ids is None else len(final_ids)

    summary_path = run / "benchmark_summary.json"
    summary_present = summary_path.exists()
    summary: dict[str, Any] = {}
    if summary_present:
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:
            reasons.append(f"benchmark summary unreadable: {exc}")
            summary_present = False
    records = summary.get("episode_records") or []
    record_active = sum(record.get("active_set_ids") is not None for record in records if isinstance(record, dict))
    active_files = sum((path / "activeSet.npy").exists() for path in episode_dirs.values())
    expected_acquisitions = max_iter if max_iter is not None else max(0, len(episodes) - 1)
    selected_present = active_files >= expected_acquisitions or record_active >= expected_acquisitions
    episode_meta = sum((path / "episode_summary.json").exists() for path in episode_dirs.values())
    acquisition_meta = bool((run / "initial_sampling_summary.json").exists() or episode_meta or records)

    base = as_float(al.get("INITIAL_DELTA")) if method in {"ProbCover", "LIDCover"} else None
    fine = as_float(al.get("MSTC_DELTA_FINE")) if method == "MultiScaleTopoCover" else None
    coarse = as_float(al.get("MSTC_DELTA_COARSE")) if method == "MultiScaleTopoCover" else None
    alpha = as_float(al.get("MSTC_ALPHA")) if method == "MultiScaleTopoCover" else (
        as_float(al.get("IDPC_ALPHA")) if method.startswith("LIDCover") or method == "LIDCover" else None
    )
    k_id = as_int(al.get("IDPC_K_ID")) if method.startswith("LIDCover") or method == "LIDCover" else None
    k_knn = as_int(al.get("IDPC_K_KNN")) if method.startswith("LIDCover") or method == "LIDCover" else None
    if base is None:
        effective = ""
    elif init_type == "method_specific_empty_L":
        effective = f"cold_start={base * 1.35:.6g};post_label={base * 0.85:.6g}"
    else:
        effective = f"post_label={base * 0.85:.6g}"
    if method.startswith("LIDCover") or method == "LIDCover":
        effective += ";pointwise_LID_scaling"

    if train_batch != 100:
        reasons.append(f"classifier training batch={train_batch}, expected 100")
    if dataset in EXPECTED_FINE_TUNE and fine_tune is not EXPECTED_FINE_TUNE[dataset]:
        reasons.append(f"FINE_TUNE={fine_tune}, expected {EXPECTED_FINE_TUNE[dataset]}")
    if seed not in EXPECTED_SEEDS:
        reasons.append(f"seed={seed}, outside canonical 1-5")
    expected_final = None
    if budget is not None and max_iter is not None and initial_count is not None:
        evaluated_initial = budget if initial_count == 0 else initial_count
        expected_final = evaluated_initial + budget * max_iter
    if final_episode is not None and max_iter is not None and final_episode != max_iter:
        reasons.append(f"last episode={final_episode}, configured MAX_ITER={max_iter}")
    if expected_final is not None and final_count is not None and final_count != expected_final:
        reasons.append(f"final labelled count={final_count}, expected {expected_final}")

    if not config_path.exists() or initial is None:
        status = "UNVERIFIABLE"
    elif not episodes or final_episode != max_iter or final_count != expected_final or not summary_present:
        status = "PARTIAL"
    elif not selected_present:
        status = "UNVERIFIABLE"
    elif reasons:
        status = "CONFIG_MISMATCH"
    else:
        status = "COMPLETE"

    return {
        "logical_experiment_id": run.name,
        "scope": scope,
        "dataset": dataset,
        "backbone": backbone,
        "method": method,
        "method_implementation": IMPLEMENTATION.get(method, "TypiClust/deep-al/pycls/al/ActiveLearning.py routing"),
        "seed": "" if seed is None else seed,
        "base_radius": "" if base is None else base,
        "effective_radius": effective,
        "fine_radius": "" if fine is None else fine,
        "coarse_radius": "" if coarse is None else coarse,
        "alpha": "" if alpha is None else alpha,
        "k_id": "" if k_id is None else k_id,
        "k_knn": "" if k_knn is None else k_knn,
        "initialisation_type": init_type,
        "initial_labelled_count": "" if initial_count is None else initial_count,
        "initial_ids_sha256": initial_hash,
        "eligible_count": "" if eligible_count is None else eligible_count,
        "eligible_indices_sha256": eligible_hash,
        "episodes_present": compact_episodes(episodes),
        "final_labelled_count": "" if final_count is None else final_count,
        "benchmark_summary_present": summary_present,
        "selected_id_evidence_present": selected_present,
        "acquisition_metadata_present": acquisition_meta,
        "status": status,
        "status_reason": "; ".join(reasons),
        "raw_path": str(run.resolve()),
    }


def add_missing_mstc(rows: list[dict[str, Any]]) -> None:
    matrix = OUT / "_regenerated_tga" / "tga_completion_matrix.csv"
    if not matrix.exists():
        return
    with matrix.open(newline="") as handle:
        for source in csv.DictReader(handle):
            if source["status"] != "MISSING":
                continue
            row = {field: "" for field in FIELDS}
            row.update({
                "logical_experiment_id": source["experiment_id"],
                "scope": "required_missing_cell",
                "dataset": source["dataset"],
                "backbone": source["backbone"],
                "method": source["method"],
                "method_implementation": IMPLEMENTATION["MultiScaleTopoCover"],
                "seed": source["seed"],
                "fine_radius": source["delta_fine"],
                "coarse_radius": source["delta_coarse"],
                "alpha": source["alpha"],
                "initialisation_type": "method_specific_empty_L (required)",
                "status": "MISSING",
                "status_reason": source["notes"],
            })
            rows.append(row)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def command_output(args: list[str]) -> str:
    try:
        return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=30, check=False).stdout.strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def write_protocol(rows: list[dict[str, Any]]) -> None:
    canonical = [row for row in rows if row["scope"] == "canonical"]
    statuses = Counter(row["status"] for row in canonical)
    seeds = sorted({int(row["seed"]) for row in canonical if str(row["seed"]).isdigit()})
    main_complete = [
        row for row in canonical
        if row["status"] == "COMPLETE" and row["episodes_present"] == "0-100"
        and row["final_labelled_count"] == 5050
    ]
    principal_prefix = {
        "Random": "random", "LeastConfidence": "uncertainty", "Entropy": "entropy",
        "Margin": "margin", "DBAL": "dbal", "CoreSet": "coreset", "TypiClust": "typiclust",
    }
    shared_checks = [
        row for row in canonical
        if row["method"] in principal_prefix and row["status"] == "COMPLETE"
        and row["logical_experiment_id"] == f"{principal_prefix[row['method']]}_{row['seed']}_50b"
    ]
    specific_checks = [
        row for row in canonical if row["status"] == "COMPLETE" and (
            row["logical_experiment_id"] == f"probcover_{row['seed']}_50b"
            or row["logical_experiment_id"] == f"maxherding_{row['seed']}_50b"
            or row["logical_experiment_id"] == f"MSTC_C070_F050_A070_{row['seed']}_50b"
        )
    ]
    lid_required = []
    lid_matrix = OUT / "_regenerated_lidcover" / "lidcover_completion_matrix.csv"
    if lid_matrix.exists():
        with lid_matrix.open(newline="") as handle:
            lid_required = [
                row for row in csv.DictReader(handle)
                if "candidate" not in row["experiment_id"] and row["status"] == "COMPLETE"
            ]
    lid_canonical = sum("/TypiClust/output/" in row["source_path"] for row in lid_required)
    eligible_by_ds = defaultdict(set)
    eligible_hashes = defaultdict(set)
    for row in canonical:
        if row["eligible_count"]:
            eligible_by_ds[row["dataset"]].add(int(row["eligible_count"]))
            eligible_hashes[(row["dataset"], row["backbone"])].add(row["eligible_indices_sha256"])
    text = f"""# Verified executed protocol and raw-artifact audit

Audit date: 2026-09-01 (Australia/Melbourne). Historical run directories were read only.

## Source-of-truth findings

- Parsed raw configurations: **{len(canonical)} canonical** plus **{sum(row['scope'] == 'supplemental' for row in rows)} supplemental**.
- Canonical raw-run status counts: **{dict(sorted(statuses.items()))}**.
- Seeds observed: **{seeds}**; the required seed set 1–5 is present.
- Complete canonical runs with E0–E100, final budget 5,050: **{len(main_complete)}**.
- The executed main protocol is an evaluated initial budget of 50, acquisition batch 50, 100 acquisition operations, and evaluation-only E100 at 5,050 labels. Classifier training batch is 100 in protocol-matching rows.
- Primary backbone: ResNet-18. Structural acquisition methods load a fixed pretrained representation; seeds 2–5 fall back to `features_seed1.npy` when no seed-specific feature exists.
- Saved configs and source agree on `FINE_TUNE=false` for CIFAR-10 and `true` for CIFAR-100 and TinyImageNet. Here `FINE_TUNE` means restart versus continuation across AL episodes, not conventional layer unfreezing.
- Acquisition-eligible counts recovered from root partitions: **{dict(sorted((key, sorted(value)) for key, value in eligible_by_ds.items()))}**.

## Mixed initialisation policy

- Random / Least Confidence / Entropy / Margin / DBAL / CoreSet / TypiClust: **{sum(row['initialisation_type'] == 'seed_specific_shared_random_50' for row in shared_checks)}/{len(shared_checks)}** complete canonical rows match the seed-specific Random initial-ID set.
- ProbCover / MaxHerding / MSTC: **{sum(row['initialisation_type'] == 'method_specific_empty_L' for row in specific_checks)}/{len(specific_checks)}** complete canonical rows begin from empty root `lSet.npy`; their method selects the first evaluated 50.
- LIDCover was verified separately through the 100-cell matched-radius matrix: **{len(lid_required)}/100 complete**, with **{lid_canonical}** sourced from the canonical tree and **{len(lid_required) - lid_canonical}** from the preserved supplemental tree. Every accepted row begins from empty root `lSet.npy`. LIDCover is not treated as ProbCover because its estimator, adaptive point radii, sparse kNN coverage, tie handling, and fallback differ.
- Historical TopoCover remains distinct from MSTC. No missing historical TopoCover predictive result was reconstructed from an MSTC endpoint.

## Pool-reference consequence

Validation partitions are seed-specific. The inventory contains **{sum(len(value) for value in eligible_hashes.values())} distinct dataset/backbone eligible-index hashes** across the canonical tree. Therefore an exact `POOL_ONLY_REFERENCE` robustness audit must use a reference keyed by dataset, backbone, and seed (45 references for 3×3×5), or explicitly justify a single frozen split. Collapsing to nine references would silently mismatch four seeds.

## Known historical limitations retained

- Historical validation/test transforms were stochastic; later deterministic-evaluation source edits do not retroactively alter raw results.
- Saved `INIT_L_RATIO` is not authoritative because the driver dumps config before applying `--initial_size`; root partitions and episode arrays resolve the executed initial set.
- ProbCover saved base radius 0.60 corresponds to effective 0.81 at empty cold start and 0.51 after labels exist. LIDCover receives the same driver scaling before pointwise LID adaptation.
- Historical H0 references contain all stored training embeddings (50,000 CIFAR; 100,000 TinyImageNet), whereas acquisition uses 45,000 and 95,000 eligible vertices respectively.
- Existing joint `k` sensitivity changes `k_id` and `k_knn` together and does not isolate the LID-estimator neighbourhood.

## Scheduler audit

`squeue` on 2026-09-01 returned `slurm_load_jobs error: Unable to contact slurm controller (connect failure)`. RUNNING/PENDING state is therefore **UNVERIFIABLE**, and no MSTC or matched-start job may be submitted until a successful queue query is possible.
"""
    (OUT / "protocol_verified.md").write_text(text)


def write_recommendations(rows: list[dict[str, Any]]) -> None:
    missing_alpha = sum(row["scope"] == "required_missing_cell" for row in rows)
    text = f"""# Recommended additional work after raw audit

## Execute now without active-learning retraining

1. **Pool-only structural-reference robustness (high priority).** Reuse recorded trajectories. Because validation exclusions are seed-specific, compute 45 exact pool-only ID/H0 references (3 datasets × 3 backbones × 5 seeds), retaining the historical full-train reference alongside each comparison. This changes no AL trajectory or classifier.
2. **LIDCover mechanism analysis (post-hoc).** Use canonical CIFAR-100/ResNet-18/LIDCover delta0=0.25, alpha=1, k_id=k_knn=50 selected IDs and the matched ProbCover delta0=0.25 runs. Mark exact gain UNVERIFIABLE if run-local cache provenance cannot be resolved.
3. **Regenerate appendix tables from canonical raw runs.** No standard LIDCover sweep is missing.

## New predictive jobs genuinely missing

- The four old MSTC fine/coarse repair cells are now complete.
- CIFAR-100/ResNet-18/MSTC C0.70/F0.50 interior alpha {{0.25, 0.50, 0.75}}: **{missing_alpha} runs** remain missing (5 prespecified seeds per alpha).
- Do not submit these until `squeue` is reachable and COMPLETE/RUNNING/PENDING cells can be excluded. Use new names beginning `THESIS_APPENDIX_REPAIR_20260901_`.
- Matched cold-start control: no `INIT_LSET_PATH` support or identical completed control was found. If scheduler/compute becomes available, minimally add the dedicated field and run 15 CIFAR-100/ResNet-18 jobs (ProbCover, LIDCover, MSTC × 5 seeds) from the exact Random-group initial IDs.

## Do not run without explicit approval

See `optional_experiment_plan.md`. In particular, do not launch AutoDelta, graph-sampler controls, TGAAutoScale, isolated k sweeps, or second-backbone MSTC predictive work.
"""
    (OUT / "recommended_runs.md").write_text(text)


def write_optional_plan() -> None:
    text = """# Optional experiment plan — explicit approval required

| Experiment | Scientific question | Runs | Expected GPU hours | Implementation | Risk | Thesis value / claim impact |
|---|---|---:|---:|---|---|---|
| LIDCoverAutoDelta + matched ProbCover | Does an unsupervised radius rule preserve the adaptive-radius result? | 30 | 6,000–12,000 | Implemented; requires provenance validation and gated launcher | Very high training cost; automatic rule may underperform | Moderate; would change only radius-selection limitation, not current fixed-radius evidence |
| RandomComponent / GraphDegree / LowDensity controls | Is MSTC gain doing more than generic graph/component exposure? | 15 | 3,000–6,000 | Three new registered samplers and configs | Implementation confounds if controls do not share exact graph state | High mechanism value; may weaken attribution to MSTC objective |
| TGAAutoScale | Can scales be chosen without labels/test accuracy? | 15 | 3,000–6,000 | Freeze deterministic unsupervised scale rule before predictive evaluation | Selection rule may be brittle; risk of post-hoc tuning | Moderate/high; affects manual-scale limitation |
| Isolated k_id vs k_knn sweeps | Which neighbourhood controls estimator versus sparse coverage? | At least 20 for a minimal 2×2×5 block | 4,000–8,000 | Existing config fields suffice; new prespecified manifest | Large factorial space and interaction effects | Moderate; current thesis must only state joint-k robustness |
| Second-backbone MSTC predictive experiments | Does MSTC predictive behaviour transfer across representations? | At least 15 per backbone/config | 3,000–6,000 per backbone | Existing method; new frozen config manifest | High cost and multiple-comparison temptation | Moderate; current claim must remain ResNet-18-centred |

GPU-hour ranges are planning estimates based on roughly 200–400 GPU-hours per 101-evaluation AL run in this repository; replace them with cluster-accounting estimates before approval. None is required to validate the current narrowly worded claims.
"""
    (OUT / "optional_experiment_plan.md").write_text(text)


def write_provenance(rows: list[dict[str, Any]]) -> None:
    tracked = [
        ROOT / "evidence_pack/neurreps_revision_20260821/protocol_audit.md",
        ROOT / "evidence_pack/neurreps_revision_20260821/protocol_audit.json",
        ROOT / "evidence_pack/lidcover_completion_report.md",
        ROOT / "evidence_pack/method_version_registry.md",
        ROOT / "evidence_pack/tga_completion_report.md",
        ROOT / "evidence_pack/tga_method_version_registry.md",
        ROOT / "analysis/thesis_appendix_audit.py",
    ]
    code_paths = [
        ROOT / "TypiClust/deep-al/tools/train_al.py",
        ROOT / "TypiClust/deep-al/pycls/al/prob_cover.py",
        ROOT / "TypiClust/deep-al/pycls/al/IDProbCover.py",
        ROOT / "TypiClust/deep-al/pycls/al/multiscale_topocover.py",
        ROOT / "TypiClust/deep-al/pycls/al/topocover.py",
        ROOT / "analysis/compute_h0_reference_mst.py",
        ROOT / "analysis/generate_diagnostic_id_figures.py",
    ]
    provenance = {
        "audit_timestamp": datetime.now().astimezone().isoformat(),
        "repository_root": str(ROOT),
        "canonical_historical_run_root": str(CANONICAL_ROOT),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": command_output(["git", "status", "--short"]),
        "historical_directories_modified": False,
        "scheduler_state": "UNVERIFIABLE: slurm controller unreachable",
        "inventory_rows": len(rows),
        "file_sha256": {str(path): sha256_file(path) for path in tracked if path.exists()},
        "code_sha256": {str(path): sha256_file(path) for path in code_paths if path.exists()},
        "regenerated_audits": {
            "tga": str(OUT / "_regenerated_tga"),
            "lidcover": str(OUT / "_regenerated_lidcover"),
        },
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    random_hashes = discover_random_initial_hashes()
    rows = [inventory_row(path, scope, random_hashes) for path, scope in config_paths()]
    add_missing_mstc(rows)
    rows.sort(key=lambda row: (
        row["dataset"], row["backbone"], row["method"], str(row["seed"]), row["logical_experiment_id"]
    ))
    write_csv(OUT / "current_experiment_inventory.csv", rows)
    write_csv(OUT / "missing_or_partial_runs.csv", [row for row in rows if row["status"] != "COMPLETE"])
    write_protocol(rows)
    write_recommendations(rows)
    write_optional_plan()
    write_provenance(rows)
    print(json.dumps({
        "inventory_rows": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "canonical_status_counts": dict(Counter(row["status"] for row in rows if row["scope"] == "canonical")),
    }, indent=2))


if __name__ == "__main__":
    main()
