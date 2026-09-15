#!/usr/bin/env python3
"""Run exact full-train and seed-specific pool-only local-ID references."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DATASET = {"CIFAR10": "cifar-10", "CIFAR100": "cifar-100", "TINYIMAGENET": "tiny-imagenet"}


def run_task(dataset: str, backbone: str, seed: int | None, threads: int) -> dict:
    reference = ROOT / "outputs/thesis_appendix/pool_reference_robustness/reference"
    kind = "full" if seed is None else "pool"
    stem = f"{dataset}_{backbone}_full_train_local_id_k50" if seed is None else f"{dataset}_{backbone}_seed{seed}_pool_only_local_id_k50"
    out = reference / "id" / kind / f"{stem}.npy"
    if out.exists() and out.with_suffix(".json").exists():
        return {"dataset": dataset, "backbone": backbone, "seed": seed, "status": "SKIPPED_COMPLETE"}
    features = ROOT / "results/results" / backbone / FEATURE_DATASET[dataset] / "pretext/features_seed1.npy"
    command = [
        str(ROOT / ".venv_diag/bin/python"), str(ROOT / "analysis/compute_pool_id_reference.py"),
        "--features", str(features), "--out", str(out), "--dataset", dataset,
        "--backbone", backbone, "--k", "50", "--threads", str(threads),
    ]
    if seed is not None:
        eligible = reference / "eligible_indices" / f"{dataset}_{backbone}_seed{seed}_eligible.npy"
        command += ["--eligible-indices", str(eligible), "--seed", str(seed)]
    log = reference / "logs" / f"id_{kind}_{dataset}_{backbone}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(threads)
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    return {
        "dataset": dataset, "backbone": backbone, "seed": seed,
        "status": "COMPLETE" if result.returncode == 0 and out.exists() else "FAILED",
        "returncode": result.returncode, "log": str(log),
    }


def run_stage(jobs, workers: int, threads: int) -> list[dict]:
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_task, *job, threads): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=16)
    args = parser.parse_args()
    pairs = [(dataset, backbone) for dataset in FEATURE_DATASET for backbone in ("resnet18", "resnet50", "alexnet")]
    rows = run_stage([(dataset, backbone, None) for dataset, backbone in pairs], args.workers, args.threads_per_worker)
    if any(row["status"] == "FAILED" for row in rows):
        raise SystemExit("full-reference ID stage failed")
    rows += run_stage(
        [(dataset, backbone, seed) for dataset, backbone in pairs for seed in range(1, 6)],
        args.workers, args.threads_per_worker,
    )
    summary = ROOT / "outputs/thesis_appendix/pool_reference_robustness/reference/id_runner_summary.json"
    summary.write_text(json.dumps(rows, indent=2))
    if any(row["status"] == "FAILED" for row in rows):
        raise SystemExit("pool-reference ID stage failed")


if __name__ == "__main__":
    main()
