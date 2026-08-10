#!/usr/bin/env python3
"""Build a manifest of unfinished TypiClust AL runs.

The manifest is consumed by run_missing_15.sbatch.  Each row is one
train_al.py command (one seed), and completion is defined by the presence of
benchmark_summary.json in the expected output directory.
"""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path


BASE = Path("/vast/s219110279/typiclust_runs/Typiclust_runs")
OUTPUT_ROOT = Path("/vast/s219110279/TypiClust/output")

FOLDERS = [
    "cifar10_full_50b",
    "cifar10_full_50b_alexnet",
    "cifar100_full_50b",
    "cifar100_full_50b_alexnet",
    "cifar100_ablation_50b",
    "cifar100_ablation_50b_alexnet",
    "cifar100_budget_ablation",
    "cifar100_budget_ablation_alexnet",
    "tinyimagenet_full_50b",
    "tinyimagenet_full_50b_alexnet",
]

DATASET_TAG = {
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "tinyimagenet": "TINYIMAGENET",
}

BACKBONE_TAG = {
    "RESNET50": "resnet50",
    "ALEXNET": "alexnet",
    "RESNET18": "resnet18",
}

COMMAND_RE = re.compile(
    r"python\s+train_al\.py\s+\\\s*(.*?)(?=\npython\s+train_al\.py\s+\\|\Z)",
    re.S,
)
ARG_RE = re.compile(r"--([^=\s]+)(?:=|\s+)(\"[^\"]*\"|'[^']*'|\S+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(BASE / "missing_runs.tsv"),
        help="Manifest path to write",
    )
    return parser.parse_args()


def command_blocks(script_path: Path) -> list[str]:
    text = script_path.read_text(errors="replace") + "\n"
    blocks = []
    for match in COMMAND_RE.finditer(text):
        block = match.group(1).replace("\\\n", " ")
        blocks.append(" ".join(block.split()))
    return blocks


def command_args(block: str) -> dict[str, str]:
    return {key: value.strip("\"'") for key, value in ARG_RE.findall(block)}


def resolve_dataset(cfg: str) -> str | None:
    for slug, tag in DATASET_TAG.items():
        if f"/{slug}/" in cfg:
            return tag
    return None


def resolve_backbone(cfg: str) -> str | None:
    for name, tag in BACKBONE_TAG.items():
        if f"/{name}.yaml" in cfg:
            return tag
    return None


def summary_path(dataset: str, backbone: str, exp_name: str) -> Path:
    return OUTPUT_ROOT / dataset / backbone / exp_name / "benchmark_summary.json"


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    rows: list[tuple[str, str, str, str, str, str]] = []
    totals: dict[str, list[int]] = {}

    for folder in FOLDERS:
        folder_path = BASE / folder
        if not folder_path.exists():
            continue
        total = done = missing = 0

        for script_path in sorted(folder_path.glob("*.sh")):
            if script_path.name.startswith("submit"):
                continue

            for block in command_blocks(script_path):
                args_dict = command_args(block)
                cfg = args_dict.get("cfg")
                exp_name = args_dict.get("exp-name")
                if not cfg or not exp_name:
                    continue

                dataset = resolve_dataset(cfg)
                backbone = resolve_backbone(cfg)
                if not dataset or not backbone:
                    continue

                total += 1
                if summary_path(dataset, backbone, exp_name).exists():
                    done += 1
                    continue

                missing += 1
                cmd = "python train_al.py " + block
                rows.append(
                    (
                        folder,
                        script_path.name,
                        dataset,
                        backbone,
                        exp_name,
                        shlex.join(shlex.split(cmd)),
                    )
                )

        totals[folder] = [total, done, missing]

    with out_path.open("w") as handle:
        handle.write("folder\tscript\tdataset\tbackbone\texp_name\tcommand\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")

    print(f"Wrote {len(rows)} missing runs to {out_path}")
    print("folder\ttotal\tdone\tmissing")
    for folder, (total, done, missing) in totals.items():
        print(f"{folder}\t{total}\t{done}\t{missing}")


if __name__ == "__main__":
    main()
