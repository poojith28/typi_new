#!/usr/bin/env python3
"""Fail-closed prelaunch checks for the isolated v3 experiment family."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "evidence/lidcover_extensions_2026/auto_sparse_radius_v3_manifest.csv"
HASH_LEDGER = ROOT / "evidence/lidcover_extensions_2026/historical_source_hashes_before.json"
METHODS = {"probcover_auto_sparse_radius_v3", "lidcover_auto_sparse_radius_v3"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    rows = list(csv.DictReader(MANIFEST.open(newline="")))
    errors = []
    if len(rows) != 30:
        errors.append(f"expected 30 rows, found {len(rows)}")
    if [int(row["array_index"]) for row in rows] != list(range(30)):
        errors.append("array indices are not exactly 0..29")
    for field in ("experiment_id", "output_dir"):
        values = [row[field] for row in rows]
        if len(values) != len(set(values)):
            errors.append(f"duplicate {field}")
    observed_cells = {(r["dataset"], r["method"], int(r["seed"])) for r in rows}
    expected_cells = {
        (dataset, method, seed)
        for dataset in ("CIFAR10", "CIFAR100", "TINYIMAGENET")
        for method in METHODS for seed in range(1, 6)
    }
    if observed_cells != expected_cells:
        errors.append("manifest does not contain the exact 3x2x5 matched matrix")
    expected_root = ROOT / "experiments/lidcover_extensions_2026/outputs/auto_sparse_radius_v3"
    for row in rows:
        if row["family"] != "auto_sparse_radius_v3" or row["method"] not in METHODS:
            errors.append(f"invalid family/method: {row['experiment_id']}")
        if int(row["acquisition_batch_size"]) != 50 or int(row["max_iter"]) != 100:
            errors.append(f"invalid acquisition protocol: {row['experiment_id']}")
        if int(row["final_evaluated_budget"]) != 5050 or row["cold_start"] != "empty_labelled_set":
            errors.append(f"invalid budget/cold start: {row['experiment_id']}")
        if row["configured_delta0"] != "AUTO_SPARSE_V3":
            errors.append(f"numeric or invalid configured radius: {row['experiment_id']}")
        if Path(row["output_root"]) != expected_root:
            errors.append(f"wrong output root: {row['experiment_id']}")
        if Path(row["output_dir"]).exists():
            errors.append(f"output already exists: {row['output_dir']}")
        if not Path(row["base_config"]).is_file():
            errors.append(f"missing config: {row['base_config']}")

    before = json.loads(HASH_LEDGER.read_text())
    after = {name: sha(ROOT / name) for name in before}
    for name, digest in after.items():
        if before[name] != digest:
            errors.append(f"historical source hash changed: {name}")

    historical_text = "\n".join((ROOT / name).read_text(errors="replace") for name in before)
    for method in METHODS:
        if method in historical_text:
            errors.append(f"v3 method leaked into historical source: {method}")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "runs_checked": len(rows),
        "errors": errors,
        "historical_hashes_after": after,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if not errors else 2)


if __name__ == "__main__":
    main()
