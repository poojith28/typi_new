#!/usr/bin/env python3
"""Fail-closed, read-only validation for extension manifests and source hashes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence/lidcover_extensions_2026"
MANIFEST = EVIDENCE / "NEW_EXPERIMENT_MANIFEST.csv"
SNAPSHOT = EVIDENCE / "historical_source_hashes_before.json"
HISTORICAL = [
    "TypiClust/deep-al/pycls/al/prob_cover.py",
    "TypiClust/deep-al/pycls/al/IDProbCover.py",
    "TypiClust/deep-al/pycls/al/IDprocover.py",
    "TypiClust/deep-al/pycls/al/ActiveLearning.py",
    "TypiClust/deep-al/tools/train_al.py",
    "TypiClust/deep-al/pycls/core/config.py",
    "TypiClust/deep-al/pycls/datasets/data.py",
    "TypiClust/deep-al/pycls/datasets/utils/features.py",
]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["all", "auto_radius", "batch_size", "tie_fallback", "sparse_fixed_optional"], default="all")
    parser.add_argument("--check-squeue", action="store_true")
    args = parser.parse_args()
    rows = list(csv.DictReader(MANIFEST.open(newline="")))
    if args.family != "all":
        rows = [row for row in rows if row["family"] == args.family]
    errors, warnings = [], []
    ids = [row["experiment_id"] for row in rows]
    outputs = [row["output_dir"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate experiment IDs")
    if len(outputs) != len(set(outputs)):
        errors.append("duplicate output directories")
    for item in rows:
        if not Path(item["base_config"]).is_file():
            errors.append(f"missing base config: {item['base_config']}")
        if Path(item["output_dir"]).exists():
            errors.append(f"output already exists (never reused): {item['output_dir']}")
        if int(item["seed"]) not in range(1, 6):
            errors.append(f"invalid seed: {item['experiment_id']}")
        if item["cold_start"] != "empty_labelled_set":
            errors.append(f"non-empty cold start: {item['experiment_id']}")
        budget = int(item["acquisition_batch_size"])
        max_iter = int(item["max_iter"])
        if int(item["final_evaluated_budget"]) != budget * (max_iter + 1):
            errors.append(f"budget/episode mismatch: {item['experiment_id']}")
        if item["family"] == "batch_size" and int(item["final_evaluated_budget"]) != 5000:
            errors.append(f"batch study does not end exactly at 5000: {item['experiment_id']}")

    before = json.loads(SNAPSHOT.read_text())
    after = {name: sha(ROOT / name) for name in HISTORICAL}
    for name, digest in after.items():
        if before.get(name) != digest:
            errors.append(f"historical source hash changed: {name}")

    central_text = "\n".join((ROOT / name).read_text(errors="replace") for name in HISTORICAL[:4])
    for method in sorted(set(row["method"] for row in rows)):
        if method in central_text:
            errors.append(f"new method unexpectedly registered in historical source: {method}")

    if args.check_squeue:
        try:
            queue = subprocess.check_output(
                ["squeue", "-h", "-u", "s219110279", "-o", "%j %T %k"],
                text=True, timeout=15,
            )
            marker = f"lidcover_extensions_2026:{args.family}"
            if args.family == "all":
                markers = [
                    "lidcover_extensions_2026:auto_radius", "lidcover_extensions_2026:batch_size",
                    "lidcover_extensions_2026:tie_fallback", "lidcover_extensions_2026:sparse_fixed_optional",
                ]
            else:
                markers = [marker]
            for queued_marker in markers:
                if queued_marker in queue:
                    errors.append(f"matching PENDING/RUNNING scheduler family: {queued_marker}")
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            errors.append(f"could not validate scheduler queue: {exc}")
    else:
        warnings.append("scheduler queue not checked; use --check-squeue on the login node before sbatch")

    result = {
        "family": args.family, "runs_checked": len(rows), "errors": errors, "warnings": warnings,
        "historical_hashes_after": after, "status": "PASS" if not errors else "FAIL",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if not errors else 2)


if __name__ == "__main__":
    main()
