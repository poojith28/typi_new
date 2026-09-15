"""Validation helpers for an explicitly provided active-learning start set."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def sha256_file(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_provided_initial_lset(path, u_set, val_set, train_size, expected_count):
    """Load an ordered initial set and validate it against the seeded split.

    The returned unlabeled set retains its original order, which preserves the
    acquisition implementations' deterministic tie behaviour.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("Provided initial-labelled-set file does not exist: {}".format(source))

    raw = np.load(str(source), allow_pickle=True)
    try:
        initial = np.asarray(raw, dtype=np.int64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Provided initial set must contain integer indices") from exc

    if len(initial) != int(expected_count):
        raise ValueError(
            "Provided initial set has {} indices; expected exactly {}".format(
                len(initial), int(expected_count)
            )
        )
    if len(np.unique(initial)) != len(initial):
        raise ValueError("Provided initial set contains duplicate indices")
    if np.any(initial < 0) or np.any(initial >= int(train_size)):
        raise ValueError("Provided initial set contains an out-of-range training index")

    u_set = np.asarray(u_set, dtype=np.int64).reshape(-1)
    val_set = np.asarray(val_set, dtype=np.int64).reshape(-1)
    if np.intersect1d(initial, val_set).size:
        raise ValueError("Provided initial set overlaps the seeded validation set")
    if not np.all(np.isin(initial, u_set)):
        raise ValueError("Provided initial set is not a subset of the seeded acquisition pool")

    remaining = u_set[~np.isin(u_set, initial)]
    metadata = {
        "source_path": str(source),
        "source_file_sha256": sha256_file(source),
        "ordered_int64_sha256": hashlib.sha256(initial.tobytes(order="C")).hexdigest(),
        "count": int(len(initial)),
    }
    return initial, remaining, metadata
