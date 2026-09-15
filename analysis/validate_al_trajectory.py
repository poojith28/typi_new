#!/usr/bin/env python3
"""Validate one completed active-learning trajectory without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_ids(path: Path) -> np.ndarray:
    array = np.load(path, allow_pickle=True)
    if array.dtype == object:
        array = np.asarray(array.tolist())
    return np.asarray(array, dtype=np.int64).reshape(-1)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--pool-size", type=int, required=True)
    parser.add_argument("--initial-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--last-episode", type=int, default=100)
    parser.add_argument("--validation-size", type=int, default=5000)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "benchmark_summary.json"
    require(run_dir.is_dir(), f"missing run directory: {run_dir}")
    require(summary_path.is_file(), f"missing summary: {summary_path}")

    summary = json.loads(summary_path.read_text())
    records = summary.get("episode_records") or []
    by_episode = {int(record["episode"]): record for record in records}
    expected_episodes = list(range(args.last_episode + 1))
    require(
        sorted(by_episode) == expected_episodes,
        f"episodes are not exactly 0-{args.last_episode}: {sorted(by_episode)}",
    )

    previous_lset: np.ndarray | None = None
    previous_active: np.ndarray | None = None

    for episode in expected_episodes:
        record = by_episode[episode]
        episode_dir = run_dir / f"episode_{episode}"
        lset_path = episode_dir / "lSet.npy"
        uset_path = episode_dir / "uSet.npy"
        active_path = episode_dir / "activeSet.npy"
        require(lset_path.is_file(), f"E{episode}: missing {lset_path}")
        require(uset_path.is_file(), f"E{episode}: missing {uset_path}")

        lset = load_ids(lset_path)
        uset = load_ids(uset_path)
        expected_labeled = args.initial_size + args.batch_size * episode
        expected_unlabeled = (
            args.pool_size - args.validation_size - expected_labeled
        )

        require(len(lset) == expected_labeled, f"E{episode}: lSet size {len(lset)} != {expected_labeled}")
        require(len(np.unique(lset)) == len(lset), f"E{episode}: duplicate lSet indices")
        require(len(uset) == expected_unlabeled, f"E{episode}: uSet size {len(uset)} != {expected_unlabeled}")
        require(len(np.unique(uset)) == len(uset), f"E{episode}: duplicate uSet indices")
        require(not len(np.intersect1d(lset, uset)), f"E{episode}: lSet/uSet overlap")
        for name, values in (("lSet", lset), ("uSet", uset)):
            require(len(values) > 0, f"E{episode}: empty {name}")
            require(
                int(values.min()) >= 0 and int(values.max()) < args.pool_size,
                f"E{episode}: {name} index range {values.min()}..{values.max()} outside pool",
            )

        require(
            int(record.get("labeled_count_before_sampling", -1)) == expected_labeled,
            f"E{episode}: summary labeled_count_before_sampling mismatch",
        )
        expected_after = (
            expected_labeled + args.batch_size
            if episode < args.last_episode
            else expected_labeled
        )
        require(
            int(record.get("labeled_count_after_sampling", -1)) == expected_after,
            f"E{episode}: summary labeled_count_after_sampling mismatch",
        )
        require(np.isfinite(float(record["test_accuracy"])), f"E{episode}: invalid test accuracy")

        if previous_lset is not None:
            require(
                len(np.intersect1d(previous_lset, lset)) == len(previous_lset),
                f"E{episode - 1}->E{episode}: cumulative lSet is not nested",
            )
            delta = np.setdiff1d(lset, previous_lset)
            require(previous_active is not None, f"E{episode - 1}: missing prior active batch")
            require(
                np.array_equal(np.sort(delta), np.sort(previous_active)),
                f"E{episode - 1}: activeSet does not equal next-lSet delta",
            )

        active_field = record.get("active_set_ids")
        if episode < args.last_episode:
            require(active_path.is_file(), f"E{episode}: missing activeSet.npy")
            require(active_field is not None, f"E{episode}: summary active_set_ids is missing")
            active = load_ids(active_path)
            field = np.asarray(active_field, dtype=np.int64).reshape(-1)
            require(len(active) == args.batch_size, f"E{episode}: activeSet size {len(active)}")
            require(len(np.unique(active)) == len(active), f"E{episode}: duplicate activeSet indices")
            require(
                int(active.min()) >= 0 and int(active.max()) < args.pool_size,
                f"E{episode}: activeSet outside pool bounds",
            )
            require(not len(np.intersect1d(lset, active)), f"E{episode}: activeSet overlaps current lSet")
            require(
                np.array_equal(np.sort(active), np.sort(field)),
                f"E{episode}: summary active_set_ids disagrees with activeSet.npy",
            )
            previous_active = active
        else:
            require(not active_path.exists(), f"E{episode}: evaluation-only episode has activeSet.npy")
            require(active_field is None, f"E{episode}: evaluation-only episode has active_set_ids")
            previous_active = None

        previous_lset = lset

    print(
        f"VALID: {run_dir} episodes=0-{args.last_episode} "
        f"final_labeled={len(previous_lset)} pool_size={args.pool_size}"
    )


if __name__ == "__main__":
    main()
