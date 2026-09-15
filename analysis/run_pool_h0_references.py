#!/usr/bin/env python3
"""Run the exact pool-only H0 reference matrix with resumable new outputs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DATASET = {
    "CIFAR10": "cifar-10", "CIFAR100": "cifar-100", "TINYIMAGENET": "tiny-imagenet"
}


def task(dataset: str, backbone: str, seed: int, threads: int, python: str) -> dict:
    reference = ROOT / "outputs/thesis_appendix/pool_reference_robustness/reference"
    prefix = reference / "h0" / f"{dataset}_{backbone}_seed{seed}_pool_only_h0_reference"
    meta = Path(str(prefix) + "_meta.json")
    if meta.exists():
        return {"dataset": dataset, "backbone": backbone, "seed": seed, "status": "SKIPPED_COMPLETE"}
    features = ROOT / "results/results" / backbone / FEATURE_DATASET[dataset] / "pretext/features_seed1.npy"
    eligible = reference / "eligible_indices" / f"{dataset}_{backbone}_seed{seed}_eligible.npy"
    log = reference / "logs" / f"h0_{dataset}_{backbone}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python, str(ROOT / "analysis/compute_pool_h0_reference_mst.py"),
        "--features", str(features), "--eligible-indices", str(eligible),
        "--out", str(prefix), "--dataset", dataset, "--backbone", backbone, "--seed", str(seed),
    ]
    environment = os.environ.copy()
    environment["OPENBLAS_NUM_THREADS"] = str(threads)
    environment["OMP_NUM_THREADS"] = str(threads)
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    return {
        "dataset": dataset, "backbone": backbone, "seed": seed,
        "status": "COMPLETE" if result.returncode == 0 and meta.exists() else "FAILED",
        "returncode": result.returncode, "log": str(log),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=16)
    parser.add_argument("--python", default=str(ROOT / ".conda/envs/gudhi-py39/bin/python"))
    parser.add_argument("--backbones", nargs="+", default=["resnet18", "resnet50", "alexnet"])
    args = parser.parse_args()
    jobs = [
        (dataset, backbone, seed)
        for backbone in args.backbones for dataset in FEATURE_DATASET for seed in range(1, 6)
    ]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(task, dataset, backbone, seed, args.threads_per_worker, args.python): (dataset, backbone, seed)
            for dataset, backbone, seed in jobs
        }
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(json.dumps(row), flush=True)
    failed = [row for row in results if row["status"] == "FAILED"]
    summary = ROOT / "outputs/thesis_appendix/pool_reference_robustness/reference/h0_runner_summary.json"
    summary.write_text(json.dumps(sorted(results, key=lambda row: (row["dataset"], row["backbone"], row["seed"])), indent=2))
    if failed:
        raise SystemExit(f"{len(failed)} H0 reference jobs failed")


if __name__ == "__main__":
    main()
