#!/usr/bin/env python3
"""Run one backbone through the retained ID pipeline in isolated staging.

This wrapper deliberately leaves ``outputs/diagnostic_id`` untouched.  It also
enforces the experiment's episode convention: episode 100 is evaluation-only,
so its newly-acquired batch is empty even if a legacy JSON record contains a
stale ``active_set_ids`` field.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path("/vast/s219110279")
SOURCE = ROOT / "analysis" / "generate_diagnostic_id_figures.py"
STAGING = ROOT / "outputs" / "diagnostic_id_cross_backbone_staging" / "by_backbone"
FIGURES = ROOT / "figures" / "diagnostic_id_cross_backbone_staging" / "by_backbone"
EXPECTED_ROWS = {"CIFAR10": 50_000, "CIFAR100": 50_000, "TINYIMAGENET": 100_000}
BACKBONES = ("alexnet", "resnet50")


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=BACKBONES)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("retained_diagnostic_id", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE}")
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)

    output_dir = STAGING / args.backbone
    figure_dir = FIGURES / args.backbone
    cache_dir = output_dir / "id_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pipeline.OUTPUTS_DIR = output_dir
    pipeline.FIGURES_DIR = figure_dir
    pipeline.ID_CACHE_ROOT = cache_dir

    # Record the exact fixed representations and source implementation.  On a
    # resumed job, refuse to combine caches with changed inputs.
    features = {}
    for dataset in pipeline.DATASETS:
        path = pipeline.feature_path(dataset, args.backbone)
        if not path.is_file():
            raise FileNotFoundError(path)
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[0] != EXPECTED_ROWS[dataset]:
            raise ValueError(f"Unexpected feature shape for {dataset}: {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite features: {path}")
        features[dataset] = {
            "path": str(path),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "sha256": sha256(path),
        }

    manifest = {
        "backbone": args.backbone,
        "pipeline_path": str(SOURCE),
        "pipeline_sha256": sha256(SOURCE),
        "episode_100_batch_policy": "empty_evaluation_only",
        "features": features,
    }
    manifest_path = output_dir / "provenance_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old != manifest:
            raise RuntimeError(
                f"Provenance changed for existing staging cache: {manifest_path}. "
                "Do not mix this run with the existing cache."
            )
    else:
        atomic_json(manifest_path, manifest)

    # Preserve the verified episode/budget convention independently of legacy
    # summary JSON quirks.
    original_load_run_rounds = pipeline.load_run_rounds

    def load_run_rounds_with_final_evaluation(*load_args, **load_kwargs):
        rows, error = original_load_run_rounds(*load_args, **load_kwargs)
        if rows is not None:
            for row in rows:
                if int(row["round"]) == 100:
                    row["batch"] = np.empty(0, dtype=np.int64)
        return rows, error

    pipeline.load_run_rounds = load_run_rounds_with_final_evaluation

    # Cache writes are atomic: an interrupted Slurm task cannot leave a file
    # that looks complete but contains a truncated NumPy payload.
    original_save = np.save
    original_savez_compressed = np.savez_compressed

    def atomic_save(path, array, *save_args, **save_kwargs):
        destination = Path(path)
        tmp = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        with tmp.open("wb") as handle:
            original_save(handle, array, *save_args, **save_kwargs)
        os.replace(tmp, destination)

    def atomic_savez_compressed(path, *save_args, **save_kwargs):
        destination = Path(path)
        tmp = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        with tmp.open("wb") as handle:
            original_savez_compressed(handle, *save_args, **save_kwargs)
        os.replace(tmp, destination)

    pipeline.np.save = atomic_save
    pipeline.np.savez_compressed = atomic_savez_compressed

    pipeline.run_for_backbones(
        [args.backbone],
        skip_figures=args.skip_figures,
        refresh_cache=False,
    )


if __name__ == "__main__":
    main()
