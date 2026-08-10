from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/scratch/s219110279")
OUTPUT_ROOT = ROOT / "TypiClust/output/CIFAR100/resnet18"
LOG_ROOT = ROOT / "Typiclust_runs/rerun_unfinished"
REPORT_PATH = ROOT / "Typiclust_runs/budget500_validity_report.md"


def latest_complete_episode(run_dir: Path) -> tuple[int | None, float | None]:
    latest_idx: int | None = None
    latest_acc: float | None = None
    for ep_dir in sorted(run_dir.glob("episode_*")):
        if not ep_dir.is_dir():
            continue
        try:
            idx = int(ep_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        summary = ep_dir / "episode_summary.json"
        if not summary.exists():
            continue
        try:
            data = json.loads(summary.read_text())
        except json.JSONDecodeError:
            continue
        acc = data.get("test_acc")
        if acc is None:
            acc = data.get("test_accuracy")
        if latest_idx is None or idx > latest_idx:
            latest_idx = idx
            latest_acc = float(acc) if acc is not None else None
    return latest_idx, latest_acc


def log_text(stem: str) -> str:
    chunks = []
    prefixes = ("", "cifar100_")
    for prefix in prefixes:
        for suffix in (".err", ".out"):
            path = LOG_ROOT / f"{prefix}{stem}{suffix}"
            if path.exists():
                chunks.append(path.read_text(errors="replace"))
    return "\n".join(chunks)


def main() -> None:
    rows = []
    for run_dir in sorted(OUTPUT_ROOT.glob("*_500b")):
        stem = run_dir.name
        latest_idx, latest_acc = latest_complete_episode(run_dir)
        benchmark_summary = run_dir / "benchmark_summary.json"
        text = log_text(stem)
        exhausted = "BudgetSet cannot exceed length of unlabelled set" in text
        rows.append(
            {
                "run": stem,
                "benchmark_summary": benchmark_summary.exists(),
                "latest_complete_episode": latest_idx,
                "latest_test_acc": latest_acc,
                "pool_exhaustion_assertion": exhausted,
            }
        )

    lines = [
        "# CIFAR-100 Budget-500 Validity Report",
        "",
        "| Run | benchmark_summary.json | Latest complete episode | Latest test acc | Pool exhaustion assertion |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        acc = "" if row["latest_test_acc"] is None else f"{row['latest_test_acc']:.4f}"
        lines.append(
            "| {run} | {summary} | {episode} | {acc} | {assertion} |".format(
                run=row["run"],
                summary="yes" if row["benchmark_summary"] else "no",
                episode="" if row["latest_complete_episode"] is None else row["latest_complete_episode"],
                acc=acc,
                assertion="yes" if row["pool_exhaustion_assertion"] else "no",
            )
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(REPORT_PATH)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
