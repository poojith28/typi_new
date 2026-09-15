#!/usr/bin/env python3
"""Generate the immutable 30-run v3 automatic sparse-radius matrix."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence/lidcover_extensions_2026"
OUTPUT_ROOT = ROOT / "experiments/lidcover_extensions_2026/outputs/auto_sparse_radius_v3"
MANIFEST = EVIDENCE / "auto_sparse_radius_v3_manifest.csv"
DATE_TAG = "20260904"
CONFIGS = {
    "CIFAR10": ROOT / "TypiClust/deep-al/configs/cifar10/al/RESNET18.yaml",
    "CIFAR100": ROOT / "TypiClust/deep-al/configs/cifar100/al/RESNET18.yaml",
    "TINYIMAGENET": ROOT / "TypiClust/deep-al/configs/tinyimagenet/al/RESNET18.yaml",
}
FIELDS = [
    "family", "array_index", "experiment_id", "dataset", "backbone", "method", "seed",
    "acquisition_batch_size", "max_iter", "acquisition_calls", "first_evaluated_budget",
    "final_evaluated_budget", "configured_delta0", "alpha", "k_id", "k_knn",
    "cold_start", "base_config", "output_root", "output_dir", "essential",
]


def main():
    rows = []
    for dataset in ("CIFAR10", "CIFAR100", "TINYIMAGENET"):
        for method in ("probcover_auto_sparse_radius_v3", "lidcover_auto_sparse_radius_v3"):
            for seed in range(1, 6):
                exp_id = (
                    f"asrv3_{method}_{dataset.lower()}_resnet18_"
                    f"s{seed}_b50_{DATE_TAG}"
                )
                rows.append({
                    "family": "auto_sparse_radius_v3",
                    "array_index": len(rows),
                    "experiment_id": exp_id,
                    "dataset": dataset,
                    "backbone": "resnet18",
                    "method": method,
                    "seed": seed,
                    "acquisition_batch_size": 50,
                    "max_iter": 100,
                    "acquisition_calls": 101,
                    "first_evaluated_budget": 50,
                    "final_evaluated_budget": 5050,
                    "configured_delta0": "AUTO_SPARSE_V3",
                    "alpha": 1.0 if method.startswith("lidcover") else "",
                    "k_id": 50 if method.startswith("lidcover") else "",
                    "k_knn": 50 if method.startswith("lidcover") else "",
                    "cold_start": "empty_labelled_set",
                    "base_config": str(CONFIGS[dataset]),
                    "output_root": str(OUTPUT_ROOT),
                    "output_dir": str(OUTPUT_ROOT / dataset / "resnet18" / exp_id),
                    "essential": "yes",
                })
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} unique runs to {MANIFEST}")


if __name__ == "__main__":
    main()
