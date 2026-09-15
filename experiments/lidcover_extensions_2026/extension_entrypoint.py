#!/usr/bin/env python3
"""Runtime-only registration of new experiment IDs against frozen AL sources."""

from __future__ import annotations

import sys
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEEP_AL = ROOT / "TypiClust" / "deep-al"
TOOLS = DEEP_AL / "tools"
for item in (str(DEEP_AL), str(TOOLS), str(ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

import numpy as np
import train_al
from pycls.al.ActiveLearning import ActiveLearning
from pycls.core.config import cfg

from experiments.lidcover_extensions_2026.auto_radius_v2 import (
    resolve_auto_delta as resolve_auto_delta_v2,
    write_run_provenance as write_run_provenance_v2,
)
from experiments.lidcover_extensions_2026.auto_sparse_radius_v3 import (
    resolve_auto_delta as resolve_auto_delta_v3,
    write_run_provenance as write_run_provenance_v3,
)
from experiments.lidcover_extensions_2026.samplers import (
    LIDCoverTieFallbackFactorial,
    SparseFixedCover,
)


V2_AUTO_METHODS = {"probcover_auto_radius_v2", "lidcover_auto_radius_v2"}
V3_AUTO_METHODS = {"probcover_auto_sparse_radius_v3", "lidcover_auto_sparse_radius_v3"}
AUTO_METHODS = V2_AUTO_METHODS | V3_AUTO_METHODS
REFERENCE_PROBCOVER_METHODS = {"probcover_batch_size_v2"}
REFERENCE_LIDCOVER_METHODS = {"lidcover_batch_size_v2"}
FACTORIAL = {
    "lidcover_tf_minlid_minlid": ("min_lid", "min_lid"),
    "lidcover_tf_minlid_random": ("min_lid", "random"),
    "lidcover_tf_random_minlid": ("random", "min_lid"),
    "lidcover_tf_random_random": ("random", "random"),
}
SPARSE_METHODS = {"sparsefixedcover_v1"}
EXTENSION_METHODS = (
    AUTO_METHODS | REFERENCE_PROBCOVER_METHODS | REFERENCE_LIDCOVER_METHODS
    | set(FACTORIAL) | SPARSE_METHODS
)


def _idpc_kwargs(config):
    return {
        "cfg": config,
        "delta0": config.ACTIVE_LEARNING.INITIAL_DELTA,
        "alpha": float(getattr(config.ACTIVE_LEARNING, "IDPC_ALPHA", 1.0)),
        "mode": str(getattr(config.ACTIVE_LEARNING, "IDPC_MODE", "high_id_more_centers")),
        "cache_root": str(getattr(config.ACTIVE_LEARNING, "IDPC_CACHE_ROOT", "")),
        "k_id": int(getattr(config.ACTIVE_LEARNING, "IDPC_K_ID", 50)),
        "k_knn": int(getattr(config.ACTIVE_LEARNING, "IDPC_K_KNN", 50)),
        "l2_normalize_features": bool(getattr(config.ACTIVE_LEARNING, "IDPC_L2_NORMALIZE_FEATURES", True)),
        "prefer_faiss": bool(getattr(config.ACTIVE_LEARNING, "IDPC_PREFER_FAISS", True)),
        "faiss_gpu": bool(getattr(config.ACTIVE_LEARNING, "IDPC_FAISS_GPU", True)),
        "add_self_cover": bool(getattr(config.ACTIVE_LEARNING, "IDPC_ADD_SELF_COVER", True)),
    }


def _extension_sample(self, clf_model, lSet, uSet, trainDataset, supportingModels=None):
    method = str(self.cfg.ACTIVE_LEARNING.SAMPLING_FN).lower()
    if method not in EXTENSION_METHODS:
        return _ORIGINAL_SAMPLE(self, clf_model, lSet, uSet, trainDataset, supportingModels)
    budget = int(self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)
    if budget <= 0 or budget >= len(uSet):
        raise ValueError(f"invalid acquisition budget {budget} for unlabelled pool {len(uSet)}")

    common = {"lSet": lSet, "uSet": uSet, "budgetSize": budget}
    if method in {
        "probcover_auto_radius_v2", "probcover_auto_sparse_radius_v3",
        "probcover_batch_size_v2",
    }:
        from pycls.al.prob_cover import ProbCover

        sampler = ProbCover(
            self.cfg, lSet, uSet, budgetSize=budget,
            delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
        )
        active, remaining = sampler.select_samples()
        self.latest_sampling_metadata = {
            "strategy": "historical_probcover_reference",
            "selection_mode": method,
            "method_identifier": method,
            "effective_delta": float(self.cfg.ACTIVE_LEARNING.INITIAL_DELTA),
            "historical_sampler_source": "pycls/al/prob_cover.py:ProbCover",
            "selected_count": int(len(active)),
        }
    elif method in {
        "lidcover_auto_radius_v2", "lidcover_auto_sparse_radius_v3",
        "lidcover_batch_size_v2",
    }:
        from pycls.al.IDProbCover import IDProbCover

        sampler = IDProbCover(**common, **_idpc_kwargs(self.cfg))
        active, remaining = sampler.select_samples()
        self.latest_sampling_metadata = dict(getattr(sampler, "selection_metadata", {}))
        self.latest_sampling_metadata.update({
            "selection_mode": method,
            "method_identifier": method,
            "historical_sampler_source": "pycls/al/IDProbCover.py:IDProbCover",
        })
    elif method in FACTORIAL:
        tie_rule, fallback_rule = FACTORIAL[method]
        sampler = LIDCoverTieFallbackFactorial(
            **common, **_idpc_kwargs(self.cfg), method=method,
            positive_tie_rule=tie_rule, fallback_rule=fallback_rule,
        )
        active, remaining = sampler.select_samples()
        self.latest_sampling_metadata = dict(sampler.selection_metadata)
    else:
        sampler = SparseFixedCover(
            cfg=self.cfg, lSet=lSet, uSet=uSet, budgetSize=budget,
            delta0=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
            cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, "IDPC_CACHE_ROOT", "")),
            k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, "IDPC_K_KNN", 50)),
            l2_normalize_features=True, prefer_faiss=True, faiss_gpu=True,
            add_self_cover=True,
        )
        active, remaining = sampler.select_samples()
        self.latest_sampling_metadata = dict(sampler.selection_metadata)

    if method in AUTO_METHODS:
        base = float(getattr(self.cfg.ACTIVE_LEARNING, "AUTO_DELTA_BASE", 0.0))
        self.latest_sampling_metadata.update({
            "auto_radius_active": True,
            "auto_radius_rule": str(getattr(self.cfg.ACTIVE_LEARNING, "AUTO_DELTA_RULE", "")),
            "auto_radius_base": base,
            "auto_radius_k": 50,
            "auto_radius_driver_scale": (
                float(self.cfg.ACTIVE_LEARNING.INITIAL_DELTA) / base if base > 0 else 0.0
            ),
            "labels_used_for_auto_radius": False,
            "validation_or_test_accuracy_used_for_auto_radius": False,
        })
    return np.asarray(active, dtype=np.int64), np.asarray(remaining, dtype=np.int64)


_ORIGINAL_SAMPLE = ActiveLearning.sample_from_uSet


def _configure(args):
    cfg.merge_from_file(args.cfg_file)
    cfg.EXP_NAME = args.exp_name
    cfg.OUT_DIR = str(Path(args.extension_output_root).resolve())
    cfg.ACTIVE_LEARNING.SAMPLING_FN = args.al.lower()
    cfg.ACTIVE_LEARNING.BUDGET_SIZE = args.budget
    cfg.ACTIVE_LEARNING.MAX_ITER = args.extension_max_iter
    cfg.ACTIVE_LEARNING.INITIAL_DELTA = args.initial_delta
    cfg.RNG_SEED = args.seed
    cfg.MODEL.LINEAR_FROM_FEATURES = args.linear_from_features
    cfg.ACTIVE_LEARNING.A_LOGISTIC = args.a_logistic
    cfg.ACTIVE_LEARNING.K_LOGISTIC = args.k_logistic
    if args.idpc_alpha is not None:
        cfg.ACTIVE_LEARNING.IDPC_ALPHA = args.idpc_alpha
    if args.idpc_mode is not None:
        cfg.ACTIVE_LEARNING.IDPC_MODE = args.idpc_mode
    if args.idpc_k_id is not None:
        cfg.ACTIVE_LEARNING.IDPC_K_ID = args.idpc_k_id
    if args.idpc_k_knn is not None:
        cfg.ACTIVE_LEARNING.IDPC_K_KNN = args.idpc_k_knn
    if args.idpc_log_csv is not None:
        cfg.ACTIVE_LEARNING.IDPC_LOG_CSV = args.idpc_log_csv
    if args.idpc_cache_root is not None:
        cfg.ACTIVE_LEARNING.IDPC_CACHE_ROOT = args.idpc_cache_root
    if args.auto_delta_k is not None:
        cfg.ACTIVE_LEARNING.AUTO_DELTA_K = args.auto_delta_k
    if args.auto_delta_quantile is not None:
        cfg.ACTIVE_LEARNING.AUTO_DELTA_QUANTILE = args.auto_delta_quantile
    if args.auto_delta_cache_root is not None:
        cfg.ACTIVE_LEARNING.AUTO_DELTA_CACHE_ROOT = args.auto_delta_cache_root


def main() -> int:
    parser = train_al.argparser()
    parser.add_argument("--extension-output-root", required=True)
    parser.add_argument("--extension-max-iter", required=True, type=int)
    parser.add_argument("--extension-dry-run", action="store_true")
    args = parser.parse_args()
    if args.al.lower() not in EXTENSION_METHODS:
        parser.error(f"unsupported extension method {args.al!r}")
    if args.initial_size != 0:
        parser.error("all extension protocols require --initial_size 0")
    if args.extension_max_iter < 0:
        parser.error("--extension-max-iter must be non-negative")

    _configure(args)
    expected = Path(cfg.OUT_DIR) / str(cfg.DATASET.NAME) / str(cfg.MODEL.TYPE) / cfg.EXP_NAME
    if args.extension_dry_run:
        print(json.dumps({
            "status": "DRY_RUN_PASS", "method": cfg.ACTIVE_LEARNING.SAMPLING_FN,
            "dataset": cfg.DATASET.NAME, "backbone": cfg.MODEL.TYPE, "seed": cfg.RNG_SEED,
            "budget": cfg.ACTIVE_LEARNING.BUDGET_SIZE, "max_iter": cfg.ACTIVE_LEARNING.MAX_ITER,
            "expected_final_budget": cfg.ACTIVE_LEARNING.BUDGET_SIZE * (cfg.ACTIVE_LEARNING.MAX_ITER + 1),
            "output_dir": str(expected),
        }, sort_keys=True))
        return 0
    if expected.exists():
        raise FileExistsError(f"refusing to reuse any existing output directory: {expected}")

    train_al.ADAPTIVE_COVER_METHODS.update(EXTENSION_METHODS)
    train_al.AUTO_DELTA_METHODS.update(AUTO_METHODS)
    import pycls.al.auto_delta as legacy_auto_delta_module

    if args.al.lower() in V3_AUTO_METHODS:
        legacy_auto_delta_module.resolve_auto_delta = resolve_auto_delta_v3
        legacy_auto_delta_module.write_run_provenance = write_run_provenance_v3
    else:
        legacy_auto_delta_module.resolve_auto_delta = resolve_auto_delta_v2
        legacy_auto_delta_module.write_run_provenance = write_run_provenance_v2
    ActiveLearning.sample_from_uSet = _extension_sample
    train_al.args = args
    train_al.main(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
