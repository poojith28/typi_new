"""
Shared helpers for the analysis scripts in /vast/s219110279/analysis.

Conventions used here (matched by every script in this folder):

  * A *feature matrix* is a 2D float array, shape [N, D], saved as `.npy`.
  * An *indices* array maps the N rows back to original dataset indices
    (used by the AL pipeline, where lSet/uSet hold dataset-level indices).
  * A *labels* array (optional) has shape [N] with integer class ids.

Output layout for every script:
  <out_prefix>.npy        raw 2D arrays (e.g. tSNE/UMAP coords, persistence pairs)
  <out_prefix>.csv        flat CSV with the same content + label info if known
  <out_prefix>_meta.json  metadata (params, runtime, library versions)

The scripts can be run from any working directory; feature paths can be
absolute or relative to the project root / deep-al root, just like the
TypiClust feature loader in pycls/datasets/utils/features.py.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_THIS_DIR = Path(__file__).resolve().parent
VAST_ROOT = _THIS_DIR.parent
TYPI_CLUST_ROOT = VAST_ROOT / "TypiClust"
DEEP_AL_ROOT = TYPI_CLUST_ROOT / "deep-al"
PROJECT_ROOT = TYPI_CLUST_ROOT
DATA_ROOT = TYPI_CLUST_ROOT / "data"
SCAN_DATASETS_ROOT = TYPI_CLUST_ROOT / "scan" / "datasets"

# SCAN layout: scan/datasets/<name>/<cifar-10-batches-py|cifar-100-python>/...
SCAN_CIFAR_DIRS = {
    "CIFAR10": SCAN_DATASETS_ROOT / "cifar-10",
    "CIFAR100": SCAN_DATASETS_ROOT / "cifar-100",
}
CIFAR_BATCH_FOLDERS = {
    "CIFAR10": "cifar-10-batches-py",
    "CIFAR100": "cifar-100-python",
}


CIFAR10_CLASSES: Tuple[str, ...] = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def candidate_paths(raw_path: str) -> List[Path]:
    """Return ordered candidate locations for `raw_path`.

    Tries (in order): absolute, relative to cwd, relative to deep-al root,
    relative to project root, with the leading `../../` stripped if present.
    """
    raw = Path(raw_path)
    candidates: List[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        candidates.append(_THIS_DIR / raw)
        candidates.append(VAST_ROOT / raw)
        candidates.append(DEEP_AL_ROOT / raw)
        candidates.append(PROJECT_ROOT / raw)
        s = str(raw)
        if s.startswith("../../"):
            candidates.append(PROJECT_ROOT / s[6:])
            candidates.append(VAST_ROOT / s[6:])

    out: List[Path] = []
    seen = set()
    for p in candidates:
        rp = p.resolve(strict=False)
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def resolve_path(raw_path: str, must_exist: bool = True) -> Path:
    """Resolve a possibly-relative path against several roots."""
    for c in candidate_paths(raw_path):
        if c.exists():
            return c
    if must_exist:
        tried = "\n  ".join(str(c) for c in candidate_paths(raw_path))
        raise FileNotFoundError(f"Could not find '{raw_path}'. Tried:\n  {tried}")
    return candidate_paths(raw_path)[0]


# ---------------------------------------------------------------------------
# Feature / index / label loaders
# ---------------------------------------------------------------------------

def load_features(
    features_path: str | os.PathLike,
    normalize: bool = False,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Load a [N, D] feature matrix from `.npy`. Optionally L2-normalize rows."""
    fpath = resolve_path(str(features_path))
    X = np.load(fpath, allow_pickle=True)
    if X.dtype == object:
        X = np.asarray(X.tolist())
    if X.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix from {fpath}, got shape {X.shape}")
    X = X.astype(dtype, copy=False)
    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms.astype(dtype)
    return X


def load_index_array(path: str | os.PathLike) -> np.ndarray:
    """Load a 1D integer index array, even if it was saved as object dtype."""
    p = resolve_path(str(path))
    arr = np.load(p, allow_pickle=True)
    if arr.dtype == object:
        arr = np.asarray(arr.tolist())
    return arr.astype(np.int64).ravel()


def maybe_load_labels(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    arr = np.load(resolve_path(str(path)), allow_pickle=True)
    if arr.dtype == object:
        arr = np.asarray(arr.tolist())
    return arr.astype(np.int64).ravel()


# ---------------------------------------------------------------------------
# CIFAR raw images
# ---------------------------------------------------------------------------

def _load_cifar_pickles(folder: Path, batch_names: Sequence[str], label_key: str):
    """Read CIFAR10/100 batch pickle files directly (no torchvision checksum)."""
    import pickle

    data_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    for name in batch_names:
        with open(folder / name, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
        data_chunks.append(np.asarray(entry["data"], dtype=np.uint8))
        label_chunks.append(np.asarray(entry[label_key], dtype=np.int64))
    images = np.concatenate(data_chunks, axis=0).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.concatenate(label_chunks, axis=0)
    return images, labels


def _resolve_cifar_data_root(dataset: str, data_root: Optional[str] = None) -> Path:
    """Pick the directory that *contains* cifar-10-batches-py / cifar-100-python.

    Search order when ``data_root`` is not given explicitly:
      1. SCAN path:  ``TypiClust/scan/datasets/cifar-10``  (or cifar-100)
      2. Legacy path: ``TypiClust/data``  (flat layout used by torchvision)
    """
    ds = dataset.upper()
    if ds not in CIFAR_BATCH_FOLDERS:
        raise ValueError(f"Unsupported dataset {dataset!r}; expected CIFAR10 or CIFAR100")

    batch_folder = CIFAR_BATCH_FOLDERS[ds]
    if data_root is not None:
        root = Path(data_root)
        if not (root / batch_folder).exists():
            raise FileNotFoundError(
                f"Expected {batch_folder}/ under {root}, but it was not found."
            )
        return root

    candidates = [SCAN_CIFAR_DIRS[ds], DATA_ROOT]
    for root in candidates:
        if (root / batch_folder).exists():
            return root

    tried = ", ".join(str(r / batch_folder) for r in candidates)
    raise FileNotFoundError(
        f"Could not find raw {ds} pickles. Expected one of: {tried}"
    )


def load_cifar_raw(
    dataset: str,
    train: bool = True,
    data_root: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load raw CIFAR10/CIFAR100 images as [N, 32, 32, 3] uint8 + labels + class names.

    Default search order for raw pixels (same convention as SCAN ``MyPath``):
      * ``TypiClust/scan/datasets/cifar-10/cifar-10-batches-py/``
      * ``TypiClust/scan/datasets/cifar-100/cifar-100-python/``
      * fallback: ``TypiClust/data/cifar-10-batches-py/`` etc.

    Pass ``data_root`` explicitly to override (should be the parent of the
    batch folder, e.g. ``.../scan/datasets/cifar-10``).
    """
    import pickle

    ds = dataset.upper()
    root = _resolve_cifar_data_root(ds, data_root=data_root)
    folder = root / CIFAR_BATCH_FOLDERS[ds]

    if ds == "CIFAR10":
        if train:
            batches = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            batches = ["test_batch"]
        images, labels = _load_cifar_pickles(folder, batches, label_key="labels")
        with open(folder / "batches.meta", "rb") as f:
            meta = pickle.load(f, encoding="latin1")
        classes = [c.decode() if isinstance(c, bytes) else c for c in meta["label_names"]]
    elif ds == "CIFAR100":
        batches = ["train"] if train else ["test"]
        images, labels = _load_cifar_pickles(folder, batches, label_key="fine_labels")
        with open(folder / "meta", "rb") as f:
            meta = pickle.load(f, encoding="latin1")
        classes = [c.decode() if isinstance(c, bytes) else c for c in meta["fine_label_names"]]
    else:
        raise ValueError(f"Unsupported dataset {dataset!r}; expected CIFAR10 or CIFAR100")

    return images, labels, classes


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

def subsample_indices(
    n: int,
    max_points: Optional[int],
    seed: int = 0,
    stratify_labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return sorted indices for an optional subsample of size <= max_points."""
    if max_points is None or n <= max_points:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(int(seed))

    if stratify_labels is not None:
        labels = np.asarray(stratify_labels)
        if labels.shape[0] != n:
            raise ValueError("stratify_labels size mismatch")
        classes = np.unique(labels)
        per_class = max(1, max_points // len(classes))
        chosen: List[int] = []
        for c in classes:
            pool = np.where(labels == c)[0]
            take = min(per_class, len(pool))
            chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
        # If we under-sampled, top up with a random draw from the remainder.
        if len(chosen) < max_points:
            remaining = np.setdiff1d(np.arange(n), np.array(chosen), assume_unique=True)
            need = max_points - len(chosen)
            if remaining.size > 0:
                chosen.extend(rng.choice(remaining, size=min(need, remaining.size), replace=False).tolist())
        return np.sort(np.array(chosen[:max_points], dtype=np.int64))

    chosen = rng.choice(n, size=int(max_points), replace=False)
    return np.sort(chosen.astype(np.int64))


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

@dataclass
class RunMeta:
    """Metadata block written next to every output."""

    script: str
    params: Dict[str, Any]
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)
    runtime_sec: float = 0.0
    library_versions: Dict[str, str] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def collect_library_versions(modules: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in modules:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            out[name] = "MISSING"
    out["python"] = sys.version.split()[0]
    return out


def out_prefix_paths(prefix: str) -> Dict[str, Path]:
    """Resolve <prefix>.npy / <prefix>.csv / <prefix>_meta.json and ensure parent dir exists."""
    p = Path(prefix).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return {
        "prefix": p,
        "npy": p.with_suffix(".npy"),
        "csv": p.with_suffix(".csv"),
        "meta": p.with_name(p.name + "_meta.json"),
    }


def save_meta(meta: RunMeta, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(meta.to_dict(), f, indent=2, default=str)


# ---------------------------------------------------------------------------
# AL episode discovery
# ---------------------------------------------------------------------------

def find_episode_dirs(experiment_dir: str | os.PathLike) -> List[Path]:
    """Return episode_* subdirectories of `experiment_dir`, sorted by integer suffix."""
    base = Path(experiment_dir).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(base)

    eps: List[Tuple[int, Path]] = []
    for child in base.iterdir():
        if child.is_dir() and child.name.startswith("episode_"):
            try:
                idx = int(child.name.split("_", 1)[1])
            except ValueError:
                continue
            eps.append((idx, child))
    eps.sort(key=lambda t: t[0])
    return [p for _, p in eps]


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

class Timer:
    """Tiny context manager that records wall time."""

    def __init__(self) -> None:
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self.start
