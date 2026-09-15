#!/usr/bin/env python3
"""Generate immutable run matrices and pre-acquisition provenance rows."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "TypiClust/deep-al/configs"
OUT_ROOT = ROOT / "experiments/lidcover_extensions_2026/outputs"
EVIDENCE = ROOT / "evidence/lidcover_extensions_2026"
DATE_TAG = "20260903"
SEEDS = range(1, 6)
BASE_CONFIG = {
    "CIFAR10": CONFIG_ROOT / "cifar10/al/RESNET18.yaml",
    "CIFAR100": CONFIG_ROOT / "cifar100/al/RESNET18.yaml",
    "TINYIMAGENET": CONFIG_ROOT / "tinyimagenet/al/RESNET18.yaml",
}


def array_hash(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_path(dataset: str, seed: int) -> Path:
    slug = {"CIFAR10": "cifar-10", "CIFAR100": "cifar-100", "TINYIMAGENET": "tiny-imagenet"}[dataset]
    root = ROOT / "results/results/resnet18" / slug / "pretext"
    requested = root / f"features_seed{seed}.npy"
    return requested if requested.is_file() else root / "features_seed1.npy"


def candidate_indices(dataset: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    size = {"CIFAR10": 50000, "CIFAR100": 50000, "TINYIMAGENET": 100000}[dataset]
    indices = list(range(size))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    split = int(0.9 * size)
    return np.asarray(indices[:split], dtype=np.int64), np.asarray(indices[split:], dtype=np.int64)


FIELDS = [
    "family", "array_index", "experiment_id", "dataset", "backbone", "method", "seed",
    "acquisition_batch_size", "max_iter", "acquisition_calls", "first_evaluated_budget",
    "final_evaluated_budget", "configured_delta0", "alpha", "k_id", "k_knn",
    "cold_start", "base_config", "output_root", "output_dir", "essential",
]


def row(family, array_index, experiment_id, dataset, method, seed, batch, max_iter, delta, essential):
    family_root = OUT_ROOT / family
    return {
        "family": family, "array_index": array_index, "experiment_id": experiment_id,
        "dataset": dataset, "backbone": "resnet18", "method": method, "seed": seed,
        "acquisition_batch_size": batch, "max_iter": max_iter,
        "acquisition_calls": max_iter + 1, "first_evaluated_budget": batch,
        "final_evaluated_budget": batch * (max_iter + 1), "configured_delta0": delta,
        "alpha": 1.0 if "lidcover" in method else "", "k_id": 50 if "lidcover" in method else "",
        "k_knn": 50 if ("lidcover" in method or "sparse" in method) else "",
        "cold_start": "empty_labelled_set", "base_config": str(BASE_CONFIG[dataset]),
        "output_root": str(family_root),
        "output_dir": str(family_root / dataset / "resnet18" / experiment_id),
        "essential": essential,
    }


def matrices():
    auto = []
    for dataset in ("CIFAR10", "CIFAR100", "TINYIMAGENET"):
        for method in ("probcover_auto_radius_v2", "lidcover_auto_radius_v2"):
            for seed in SEEDS:
                experiment = f"arv2_{method}_{dataset.lower()}_resnet18_s{seed}_b50_{DATE_TAG}"
                auto.append(row("auto_radius", len(auto), experiment, dataset, method, seed, 50, 100, "AUTO", "yes"))
    batch_rows = []
    for batch in (50, 100, 500):
        max_iter = 5000 // batch - 1
        for method in ("probcover_batch_size_v2", "lidcover_batch_size_v2"):
            for seed in SEEDS:
                experiment = f"bsv2_{method}_cifar100_resnet18_b{batch}_s{seed}_to5000_{DATE_TAG}"
                batch_rows.append(row("batch_size", len(batch_rows), experiment, "CIFAR100", method, seed, batch, max_iter, 0.25, "yes"))
    tf = []
    for method in (
        "lidcover_tf_minlid_minlid", "lidcover_tf_minlid_random",
        "lidcover_tf_random_minlid", "lidcover_tf_random_random",
    ):
        for seed in SEEDS:
            experiment = f"tfv1_{method}_cifar100_resnet18_s{seed}_b50_{DATE_TAG}"
            tf.append(row("tie_fallback", len(tf), experiment, "CIFAR100", method, seed, 50, 100, 0.25, "yes"))
    sparse = []
    for seed in SEEDS:
        method = "sparsefixedcover_v1"
        experiment = f"sfcv1_{method}_cifar100_resnet18_s{seed}_b50_{DATE_TAG}"
        sparse.append(row("sparse_fixed_optional", len(sparse), experiment, "CIFAR100", method, seed, 50, 100, 0.25, "no"))
    return {"auto_radius": auto, "batch_size": batch_rows, "tie_fallback": tf, "sparse_fixed_optional": sparse}


def write_csv(path: Path, rows: list[dict], fields=FIELDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    tables = matrices()
    write_csv(EVIDENCE / "auto_radius_manifest.csv", tables["auto_radius"])
    write_csv(EVIDENCE / "batch_size_manifest.csv", tables["batch_size"])
    write_csv(EVIDENCE / "tie_fallback_manifest.csv", tables["tie_fallback"])
    write_csv(EVIDENCE / "sparse_fixed_manifest.csv", tables["sparse_fixed_optional"])
    all_rows = sum(tables.values(), [])
    write_csv(EVIDENCE / "NEW_EXPERIMENT_MANIFEST.csv", all_rows)

    provenance = []
    feature_hash_cache = {}
    for item in tables["auto_radius"]:
        dataset, seed = item["dataset"], int(item["seed"])
        candidates, validation = candidate_indices(dataset, seed)
        path = feature_path(dataset, seed)
        if path not in feature_hash_cache:
            feature_hash_cache[path] = file_hash(path) if path.is_file() else "MISSING"
        provenance.append({
            "experiment_id": item["experiment_id"], "dataset": dataset, "backbone": "resnet18",
            "method": item["method"], "al_seed": seed, "representation_path": str(path),
            "representation_sha256": feature_hash_cache[path],
            "candidate_index_sha256": array_hash(candidates), "candidate_count": len(candidates),
            "held_out_validation_index_sha256": array_hash(validation), "held_out_validation_count": len(validation),
            "automatic_radius_rule": "median_exact_50th_neighbour_euclidean_l2_normalized_v2",
            "k": 50, "raw_delta_auto": "PENDING_RUN", "initial_effective_radius": "PENDING_RUN",
            "post_cold_start_effective_radius": "PENDING_RUN",
            "feature_normalisation": "rowwise_l2_float32_eps_1e-12",
            "labels_used": False, "validation_or_test_accuracy_used": False,
        })
    write_csv(EVIDENCE / "auto_radius_provenance.csv", provenance, list(provenance[0]))
    print(json.dumps({name: len(rows) for name, rows in tables.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
