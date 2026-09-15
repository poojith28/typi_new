# This file is slightly modified from a code implementation by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------

import os

import numpy as np
import torch

from .Sampling import Sampling, CoreSetMIPSampling, AdversarySampler
import pycls.utils.logging as lu

logger = lu.get_logger(__name__)


def _default_sampling_metadata(strategy_name, active_set=None):
    selected_count = 0 if active_set is None else int(len(active_set))
    return {
        'strategy': strategy_name,
        'selection_mode': strategy_name,
        'boundary_variant': 'not_applicable',
        'uncertainty_mode': 'not_applicable',
        'uncertainty_active': False,
        'coverage_fraction_before': 'not_applicable',
        'coverage_fraction_after': 'not_applicable',
        'components': 'not_applicable',
        'largest_component_fraction': 'not_applicable',
        'selected_count': selected_count,
    }


def _auto_delta_metadata(cfg, effective_delta):
    base_delta = float(getattr(cfg.ACTIVE_LEARNING, 'AUTO_DELTA_BASE', 0.0))
    return {
        'auto_delta_active': True,
        'auto_delta_rule': str(getattr(cfg.ACTIVE_LEARNING, 'AUTO_DELTA_RULE', '')),
        'auto_delta_base': base_delta,
        'auto_delta_k': int(getattr(cfg.ACTIVE_LEARNING, 'AUTO_DELTA_K', 50)),
        'auto_delta_quantile': float(getattr(cfg.ACTIVE_LEARNING, 'AUTO_DELTA_QUANTILE', 0.5)),
        'auto_delta_feature_sha256': str(getattr(cfg.ACTIVE_LEARNING, 'AUTO_DELTA_FEATURE_SHA256', '')),
        'auto_delta_indices_sha256': str(getattr(cfg.ACTIVE_LEARNING, 'AUTO_DELTA_INDICES_SHA256', '')),
        'auto_delta_provenance_path': str(getattr(cfg.ACTIVE_LEARNING, 'AUTO_DELTA_PROVENANCE_PATH', '')),
        'effective_delta': float(effective_delta),
        'auto_delta_driver_scale': float(effective_delta / base_delta) if base_delta > 0.0 else 0.0,
        'labels_used_for_auto_delta': False,
        'validation_or_test_accuracy_used_for_auto_delta': False,
    }


@torch.no_grad()
def _extract_penultimate_features(data_obj, cfg, clf_model, dataset, indexes):
    indexes = np.asarray(indexes, dtype=np.int64)
    if indexes.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    batch_size = max(1, int(cfg.TRAIN.BATCH_SIZE / max(int(cfg.NUM_GPUS), 1)))
    loader = data_obj.getSequentialDataLoader(indexes=indexes, batch_size=batch_size, data=dataset)
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

    was_training = clf_model.training
    had_penultimate_flag = hasattr(clf_model, "penultimate_active")
    was_penultimate = getattr(clf_model, "penultimate_active", False)

    clf_model = clf_model.to(device)
    if had_penultimate_flag:
        clf_model.penultimate_active = True
    clf_model.eval()

    features = []
    try:
        for x, _ in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            outputs = clf_model(x)
            if not isinstance(outputs, tuple) or len(outputs) != 2:
                raise RuntimeError(
                    "Expected penultimate-active model to return (features, logits). "
                    f"Received type={type(outputs)}."
                )
            z, _ = outputs
            features.append(z.detach().cpu().numpy())
    finally:
        if had_penultimate_flag:
            clf_model.penultimate_active = was_penultimate
        clf_model.train(was_training)

    return np.concatenate(features, axis=0).astype(np.float32, copy=False)


def _resolve_idpc_feature_inputs(data_obj, cfg, clf_model, dataset):
    feature_source = str(getattr(cfg.ACTIVE_LEARNING, "IDPC_FEATURE_SOURCE", "precomputed")).lower()
    if feature_source == "precomputed":
        return {
            "feature_source": "precomputed",
            "use_cached_geometry": True,
            "feature_metadata": {
                "feature_ema": 0.0,
                "feature_ema_state_path": "",
                "feature_ema_prev_loaded": False,
            },
        }

    if feature_source != "model":
        raise ValueError(
            f"Unsupported IDPC_FEATURE_SOURCE={feature_source}. Expected 'precomputed' or 'model'."
        )

    full_index_set = np.arange(len(dataset), dtype=np.int64)
    current_features = _extract_penultimate_features(data_obj, cfg, clf_model, dataset, full_index_set)

    ema = float(getattr(cfg.ACTIVE_LEARNING, "IDPC_FEATURE_EMA", 0.0))
    if ema < 0.0 or ema >= 1.0:
        raise ValueError(f"IDPC_FEATURE_EMA must be in [0, 1). Got {ema}.")

    state_file = str(getattr(cfg.ACTIVE_LEARNING, "IDPC_FEATURE_STATE", "idpc_model_features_ema.npy") or "")
    state_path = ""
    prev_loaded = False
    blended_features = current_features

    if state_file:
        state_path = state_file
        if not os.path.isabs(state_path):
            state_path = os.path.join(getattr(cfg, "EXP_DIR", "."), state_path)

    if ema > 0.0 and state_path and os.path.exists(state_path):
        prev_features = np.load(state_path).astype(np.float32, copy=False)
        if prev_features.shape == current_features.shape:
            blended_features = ema * prev_features + (1.0 - ema) * current_features
            prev_loaded = True
        else:
            logger.info(
                "Skipping IDProbCover feature EMA warm-start due to shape mismatch: prev=%s current=%s",
                prev_features.shape,
                current_features.shape,
            )

    if state_path:
        state_dir = os.path.dirname(state_path)
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)
        np.save(state_path, blended_features.astype(np.float32, copy=False))

    return {
        "all_features": blended_features.astype(np.float32, copy=False),
        "feature_source": "model",
        "use_cached_geometry": False,
        "feature_metadata": {
            "feature_ema": ema,
            "feature_ema_state_path": state_path,
            "feature_ema_prev_loaded": prev_loaded,
        },
    }

class ActiveLearning:
    """
    Implements standard active learning methods.
    """

    def __init__(self, dataObj, cfg):
        self.dataObj = dataObj
        self.sampler = Sampling(dataObj=dataObj,cfg=cfg)
        self.cfg = cfg
        self.latest_sampling_metadata = {}
        
    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None):
        """
        Sample from uSet using cfg.ACTIVE_LEARNING.SAMPLING_FN.

        INPUT
        ------
        clf_model: Reference of task classifier model class [Typically VGG]

        supportingModels: List of models which are used for sampling process.

        OUTPUT
        -------
        Returns activeSet, uSet
        """
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE > 0, "Expected a positive budgetSize"
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE < len(uSet), "BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {} and budgetSize: {}"\
        .format(len(uSet), self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)
        self.latest_sampling_metadata = {}

        if self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "random":

            activeSet, uSet = self.sampler.random(uSet=uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)
            self.latest_sampling_metadata = _default_sampling_metadata('random_policy', activeSet)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "uncertainty":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.uncertainty(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)
            self.latest_sampling_metadata = {
                **_default_sampling_metadata('uncertainty_policy', activeSet),
                'uncertainty_mode': 'least_confidence',
                'uncertainty_active': True,
            }
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "entropy":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.entropy(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)
            self.latest_sampling_metadata = {
                **_default_sampling_metadata('uncertainty_policy', activeSet),
                'uncertainty_mode': 'entropy',
                'uncertainty_active': True,
            }
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "margin":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.margin(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)
            self.latest_sampling_metadata = {
                **_default_sampling_metadata('uncertainty_policy', activeSet),
                'uncertainty_mode': 'margin',
                'uncertainty_active': True,
            }

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "coreset":
            waslatent = clf_model.penultimate_active
            wastrain = clf_model.training
            clf_model.penultimate_active = True
            # if self.cfg.TRAIN.DATASET == "IMAGENET":
            #     clf_model.cuda(0)
            clf_model.eval()
            coreSetSampler = CoreSetMIPSampling(cfg=self.cfg, dataObj=self.dataObj)
            activeSet, uSet = coreSetSampler.query(lSet=lSet, uSet=uSet, clf_model=clf_model, dataset=trainDataset)
            
            clf_model.penultimate_active = waslatent
            clf_model.train(wastrain)
            self.latest_sampling_metadata = _default_sampling_metadata('coreset_policy', activeSet)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.startswith("typiclust"):
            from .typiclust import TypiClust
            is_scan = self.cfg.ACTIVE_LEARNING.SAMPLING_FN.endswith('dc')
            tpc = TypiClust(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, is_scan=is_scan)
            activeSet, uSet = tpc.select_samples()
            self.latest_sampling_metadata = _default_sampling_metadata('typiclust_policy', activeSet)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["probcover_auto_delta", "prob_cover_auto_delta"]:
            from .prob_cover import ProbCover
            probcov = ProbCover(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                            delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)
            activeSet, uSet = probcov.select_samples()
            self.latest_sampling_metadata = {
                **_default_sampling_metadata('probcover_auto_delta_policy', activeSet),
                **_auto_delta_metadata(self.cfg, self.cfg.ACTIVE_LEARNING.INITIAL_DELTA),
                'selection_mode': 'prob_cover_auto_delta',
            }

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["prob_cover", 'probcover']:
            from .prob_cover import ProbCover
            probcov = ProbCover(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                            delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)
            activeSet, uSet = probcov.select_samples()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in [
            "talc",
            "trajectory_adaptive_lid_cover",
            "talc_auto_delta",
        ]:
            from .talc import TrajectoryAdaptiveLIDCover

            sampler_name = self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower()
            talc = TrajectoryAdaptiveLIDCover(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta0=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                alpha_max=float(getattr(self.cfg.ACTIVE_LEARNING, 'TALC_ALPHA_MAX', 1.0)),
                coverage_target=float(getattr(self.cfg.ACTIVE_LEARNING, 'TALC_COVERAGE_TARGET', 0.9)),
                coverage_epsilon=float(getattr(self.cfg.ACTIVE_LEARNING, 'TALC_COVERAGE_EPSILON', 0.05)),
                radius_min_factor=float(getattr(self.cfg.ACTIVE_LEARNING, 'TALC_RADIUS_MIN_FACTOR', 0.5)),
                radius_max_factor=float(getattr(self.cfg.ACTIVE_LEARNING, 'TALC_RADIUS_MAX_FACTOR', 2.0)),
                topology_weight=float(getattr(self.cfg.ACTIVE_LEARNING, 'TALC_TOPOLOGY_WEIGHT', 0.5)),
                topology_quantiles=tuple(getattr(self.cfg.ACTIVE_LEARNING, 'TALC_TOPOLOGY_QUANTILES', [0.25, 0.5, 0.75])),
                min_component_size=int(getattr(self.cfg.ACTIVE_LEARNING, 'TALC_MIN_COMPONENT_SIZE', 5)),
                use_topology=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TALC_USE_TOPOLOGY', True)),
                use_uncertainty=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TALC_USE_UNCERTAINTY', True)),
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'TALC_CACHE_ROOT', './talc_cache') or './talc_cache'),
                k_id=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_ID', 50)),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_KNN', 50)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_ADD_SELF_COVER', True)),
                clf_model=clf_model,
                train_dataset=trainDataset,
                data_obj=self.dataObj,
            )
            activeSet, uSet = talc.select_samples()
            self.latest_sampling_metadata = getattr(talc, 'selection_metadata', {})
            if sampler_name == "talc_auto_delta":
                self.latest_sampling_metadata = {
                    **self.latest_sampling_metadata,
                    **_auto_delta_metadata(self.cfg, self.cfg.ACTIVE_LEARNING.INITIAL_DELTA),
                    'strategy': 'trajectory_adaptive_lid_cover_auto_delta',
                    'selection_mode': 'talc_auto_delta',
                }

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["lidcover_auto_delta", "idprobcover_auto_delta"]:
            from .IDprocover import IDProbCover
            idpc = IDProbCover(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta0=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                alpha=float(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_ALPHA', 1.0)),
                mode=str(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_MODE', 'high_id_more_centers')),
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_CACHE_ROOT', './idprobcover_cache') or './idprobcover_cache'),
                k_id=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_ID', 50)),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_KNN', 50)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_ADD_SELF_COVER', True)),
            )
            activeSet, uSet = idpc.select_samples()
            self.latest_sampling_metadata = {
                **getattr(idpc, 'selection_metadata', {}),
                **_auto_delta_metadata(self.cfg, self.cfg.ACTIVE_LEARNING.INITIAL_DELTA),
                'strategy': 'lidcover_auto_delta_policy',
                'selection_mode': 'lidcover_auto_delta',
            }

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["id_prob_cover", "idprobcover", "idprobcover_frontier_density", "id_prob_cover_frontier_density"]:
            from .IDprocover import IDProbCover
            idpc = IDProbCover(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta0=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                alpha=float(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_ALPHA', 1.0)),
                mode=str(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_MODE', 'high_id_more_centers')),
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_CACHE_ROOT', './idprobcover_cache') or './idprobcover_cache'),
                k_id=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_ID', 50)),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_KNN', 50)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_ADD_SELF_COVER', True)),
            )
            activeSet, uSet = idpc.select_samples()
            self.latest_sampling_metadata = getattr(idpc, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["idprobcover_tiebreak_min_id", "idprobcover_minid_tiebreak"]:
            from .idpc_tiebreak import IDProbCoverMinIDTieBreak
            idpc = IDProbCoverMinIDTieBreak(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta0=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                alpha=float(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_ALPHA', 1.0)),
                mode=str(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_MODE', 'high_id_more_centers')),
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_CACHE_ROOT', './idprobcover_cache') or './idprobcover_cache'),
                k_id=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_ID', 50)),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_KNN', 50)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_ADD_SELF_COVER', True)),
            )
            activeSet, uSet = idpc.select_samples()
            self.latest_sampling_metadata = getattr(idpc, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["idprobcover_tiebreak_random", "idprobcover_random_tiebreak"]:
            from .idpc_tiebreak import IDProbCoverRandomTieBreak
            idpc = IDProbCoverRandomTieBreak(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta0=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                alpha=float(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_ALPHA', 1.0)),
                mode=str(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_MODE', 'high_id_more_centers')),
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_CACHE_ROOT', './idprobcover_cache') or './idprobcover_cache'),
                k_id=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_ID', 50)),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_KNN', 50)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_ADD_SELF_COVER', True)),
            )
            activeSet, uSet = idpc.select_samples()
            self.latest_sampling_metadata = getattr(idpc, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["idprobcover_tiebreak_first_max", "idprobcover_firstmax_tiebreak"]:
            from .idpc_tiebreak import IDProbCoverFirstMaxTieBreak
            idpc = IDProbCoverFirstMaxTieBreak(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta0=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                alpha=float(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_ALPHA', 1.0)),
                mode=str(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_MODE', 'high_id_more_centers')),
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_CACHE_ROOT', './idprobcover_cache') or './idprobcover_cache'),
                k_id=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_ID', 50)),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_KNN', 50)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_ADD_SELF_COVER', True)),
            )
            activeSet, uSet = idpc.select_samples()
            self.latest_sampling_metadata = getattr(idpc, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["idprobcover_fallback_random", "lidcover_fallback_random"]:
            from .idpc_tiebreak import IDProbCoverFallbackRandom
            idpc = IDProbCoverFallbackRandom(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta0=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                alpha=float(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_ALPHA', 1.0)),
                mode=str(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_MODE', 'high_id_more_centers')),
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_CACHE_ROOT', './idprobcover_cache') or './idprobcover_cache'),
                k_id=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_ID', 50)),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_K_KNN', 50)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_ADD_SELF_COVER', True)),
            )
            activeSet, uSet = idpc.select_samples()
            self.latest_sampling_metadata = getattr(idpc, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["knn_distance_cover", "adaptive_knn_distance_cover"]:
            from .adaptive_cover import KnnDistanceCover
            sampler = KnnDistanceCover(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta0=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                alpha=float(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_ALPHA', 1.0)),
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_CACHE_ROOT', './adaptive_cover_cache') or './adaptive_cover_cache'),
                k_signal=int(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_K_SIGNAL', 50)),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_K_KNN', 50)),
                eps=float(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_EPS', 1e-8)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_ADD_SELF_COVER', True)),
            )
            activeSet, uSet = sampler.select_samples()
            self.latest_sampling_metadata = getattr(sampler, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["density_cover", "adaptive_density_cover"]:
            from .adaptive_cover import DensityCover
            sampler = DensityCover(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta0=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                alpha=float(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_ALPHA', 1.0)),
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_CACHE_ROOT', './adaptive_cover_cache') or './adaptive_cover_cache'),
                k_signal=int(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_K_SIGNAL', 50)),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_K_KNN', 50)),
                eps=float(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_EPS', 1e-8)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_ADD_SELF_COVER', True)),
            )
            activeSet, uSet = sampler.select_samples()
            self.latest_sampling_metadata = getattr(sampler, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["distance_variance_cover", "adaptive_distance_variance_cover"]:
            from .adaptive_cover import DistanceVarianceCover
            sampler = DistanceVarianceCover(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta0=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                alpha=float(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_ALPHA', 1.0)),
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_CACHE_ROOT', './adaptive_cover_cache') or './adaptive_cover_cache'),
                k_signal=int(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_K_SIGNAL', 50)),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_K_KNN', 50)),
                eps=float(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_EPS', 1e-8)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_ADD_SELF_COVER', True)),
            )
            activeSet, uSet = sampler.select_samples()
            self.latest_sampling_metadata = getattr(sampler, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["distance_cv_cover", "adaptive_distance_cv_cover"]:
            from .adaptive_cover import DistanceCVCover
            sampler = DistanceCVCover(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta0=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                alpha=float(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_ALPHA', 1.0)),
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_CACHE_ROOT', './adaptive_cover_cache') or './adaptive_cover_cache'),
                k_signal=int(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_K_SIGNAL', 50)),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_K_KNN', 50)),
                eps=float(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_EPS', 1e-8)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'ARC_ADD_SELF_COVER', True)),
            )
            activeSet, uSet = sampler.select_samples()
            self.latest_sampling_metadata = getattr(sampler, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["topocover", "topo_cover"]:
            from .topocover import TopoCover
            sampler = TopoCover(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_CACHE_ROOT', './topocover_cache') or './topocover_cache'),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_K_KNN', 50)),
                eps=float(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_EPS', 1e-8)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_ADD_SELF_COVER', True)),
                recompute_components=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_RECOMPUTE_COMPONENTS', True)),
            )
            activeSet, uSet = sampler.select_samples()
            self.latest_sampling_metadata = getattr(sampler, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in [
            "multiscale_topocover",
            "multi_scale_topocover",
            "mstc",
            "two_scale_topocover",
        ]:
            from .multiscale_topocover import MultiScaleTopoCover
            delta_coarse = getattr(self.cfg.ACTIVE_LEARNING, 'MSTC_DELTA_COARSE', None)
            if delta_coarse is None or float(delta_coarse) <= 0:
                delta_coarse = self.cfg.ACTIVE_LEARNING.INITIAL_DELTA
            delta_fine = getattr(self.cfg.ACTIVE_LEARNING, 'MSTC_DELTA_FINE', None)
            if delta_fine is None or float(delta_fine) <= 0:
                # Default fine scale to half the coarse radius when unset.
                delta_fine = 0.5 * float(delta_coarse)
            sampler = MultiScaleTopoCover(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta_coarse=float(delta_coarse),
                delta_fine=float(delta_fine),
                alpha=float(getattr(self.cfg.ACTIVE_LEARNING, 'MSTC_ALPHA', 0.7)),
                graph_batch_size=int(getattr(self.cfg.ACTIVE_LEARNING, 'MSTC_GRAPH_BATCH_SIZE', 500)),
                device=str(getattr(self.cfg.ACTIVE_LEARNING, 'MSTC_DEVICE', 'cuda') or 'cuda'),
            )
            activeSet, uSet = sampler.select_samples()
            self.latest_sampling_metadata = getattr(sampler, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in [
            "weighted_topocover",
            "weighted_topo_cover",
            "wtopocover",
            "wtopo",
        ]:
            from .topo_variants import WeightedTopoCover
            sampler = WeightedTopoCover(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_CACHE_ROOT', './topocover_cache') or './topocover_cache'),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_K_KNN', 50)),
                eps=float(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_EPS', 1e-8)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_ADD_SELF_COVER', True)),
                recompute_components=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_RECOMPUTE_COMPONENTS', True)),
            )
            activeSet, uSet = sampler.select_samples()
            self.latest_sampling_metadata = getattr(sampler, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in [
            "frontier_topocover",
            "frontier_topo_cover",
            "ftopocover",
            "ftopo",
        ]:
            from .topo_variants import FrontierTopoCover
            sampler = FrontierTopoCover(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_CACHE_ROOT', './topocover_cache') or './topocover_cache'),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_K_KNN', 50)),
                eps=float(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_EPS', 1e-8)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_ADD_SELF_COVER', True)),
                recompute_components=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_RECOMPUTE_COMPONENTS', True)),
            )
            activeSet, uSet = sampler.select_samples()
            self.latest_sampling_metadata = getattr(sampler, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in [
            "phased_topocover",
            "phased_topo_cover",
            "ptopocover",
            "ptopo",
        ]:
            from .topo_variants import PhasedTopoCover
            sampler = PhasedTopoCover(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                switch_labeled=int(getattr(self.cfg.ACTIVE_LEARNING, 'PHASED_TOPO_SWITCH_LABELED', 1000)),
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_CACHE_ROOT', './topocover_cache') or './topocover_cache'),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_K_KNN', 50)),
                eps=float(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_EPS', 1e-8)),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_FAISS_GPU', True)),
                add_self_cover=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_ADD_SELF_COVER', True)),
                recompute_components=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_RECOMPUTE_COMPONENTS', True)),
            )
            activeSet, uSet = sampler.select_samples()
            self.latest_sampling_metadata = getattr(sampler, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in [
            "persist_probcover",
            "persistprobcover",
            "ppc",
        ]:
            from .persist_cover import PersistProbCover
            sampler = PersistProbCover(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                min_component_size=int(getattr(self.cfg.ACTIVE_LEARNING, 'PERSIST_MIN_COMPONENT_SIZE', 8)),
                graph_batch_size=int(getattr(self.cfg.ACTIVE_LEARNING, 'MSTC_GRAPH_BATCH_SIZE', 500)),
                device=str(getattr(self.cfg.ACTIVE_LEARNING, 'MSTC_DEVICE', 'cuda') or 'cuda'),
            )
            activeSet, uSet = sampler.select_samples()
            self.latest_sampling_metadata = getattr(sampler, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in [
            "w1_proxy_cover",
            "w1proxycover",
            "w1cover",
            "barcode_cover",
        ]:
            from .persist_cover import W1ProxyCover
            sampler = W1ProxyCover(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_K_KNN', 50)),
                persist_quantile=float(getattr(self.cfg.ACTIVE_LEARNING, 'W1_PERSIST_QUANTILE', 0.5)),
                barcode_lambda=float(getattr(self.cfg.ACTIVE_LEARNING, 'W1_BARCODE_LAMBDA', 1.0)),
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_CACHE_ROOT', './topocover_cache') or './topocover_cache'),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_FAISS_GPU', True)),
                eps=float(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_EPS', 1e-8)),
                graph_batch_size=int(getattr(self.cfg.ACTIVE_LEARNING, 'MSTC_GRAPH_BATCH_SIZE', 500)),
                device=str(getattr(self.cfg.ACTIVE_LEARNING, 'MSTC_DEVICE', 'cuda') or 'cuda'),
            )
            activeSet, uSet = sampler.select_samples()
            self.latest_sampling_metadata = getattr(sampler, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in [
            "merge_tree_cover",
            "mergetreecover",
            "mtc",
            "merge_tree",
        ]:
            from .merge_tree_cover import MergeTreeCover
            sampler = MergeTreeCover(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                delta0=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
                cache_root=str(getattr(self.cfg.ACTIVE_LEARNING, 'MTC_CACHE_ROOT', './merge_tree_cover_cache') or './merge_tree_cover_cache'),
                k_knn=int(getattr(self.cfg.ACTIVE_LEARNING, 'MTC_K_KNN', getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_K_KNN', 50))),
                l2_normalize_features=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_L2_NORMALIZE_FEATURES', True)),
                prefer_faiss=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_PREFER_FAISS', True)),
                faiss_gpu=bool(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_FAISS_GPU', True)),
                min_component_size=int(getattr(self.cfg.ACTIVE_LEARNING, 'MTC_MIN_COMPONENT_SIZE', 5)),
                max_merge_events=int(getattr(self.cfg.ACTIVE_LEARNING, 'MTC_MAX_MERGE_EVENTS', 2000)),
                events_per_budget=int(getattr(self.cfg.ACTIVE_LEARNING, 'MTC_EVENTS_PER_BUDGET', 25)),
                rep_k=int(getattr(self.cfg.ACTIVE_LEARNING, 'MTC_REP_K', 10)),
                radius_multiplier=float(getattr(self.cfg.ACTIVE_LEARNING, 'MTC_RADIUS_MULTIPLIER', 1.0)),
                radius_min=float(getattr(self.cfg.ACTIVE_LEARNING, 'MTC_RADIUS_MIN', 0.05)),
                radius_max=float(getattr(self.cfg.ACTIVE_LEARNING, 'MTC_RADIUS_MAX', 0.0)) or None,
                scale_exp=float(getattr(self.cfg.ACTIVE_LEARNING, 'MTC_SCALE_EXP', 1.0)),
                size_exp=float(getattr(self.cfg.ACTIVE_LEARNING, 'MTC_SIZE_EXP', 0.5)),
                balance_exp=float(getattr(self.cfg.ACTIVE_LEARNING, 'MTC_BALANCE_EXP', 1.0)),
                point_lambda=float(getattr(self.cfg.ACTIVE_LEARNING, 'MTC_POINT_LAMBDA', 0.0)),
                use_probcover_fallback=bool(getattr(self.cfg.ACTIVE_LEARNING, 'MTC_PROBCOVER_FALLBACK', False)),
                rebuild_every_pick=bool(getattr(self.cfg.ACTIVE_LEARNING, 'MTC_REBUILD_EVERY_PICK', True)),
                eps=float(getattr(self.cfg.ACTIVE_LEARNING, 'TOPOCOVER_EPS', 1e-8)),
            )
            activeSet, uSet = sampler.select_samples()
            self.latest_sampling_metadata = getattr(sampler, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["maxherding", "max_herding"]:
            from .max_herding import MaxHerding
            sampler = MaxHerding(
                cfg=self.cfg,
                lSet=lSet,
                uSet=uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
            )
            activeSet, uSet = sampler.select_samples()
            self.latest_sampling_metadata = getattr(sampler, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["geometry_auto_research", "geoar"]:
            from .geometry_auto_research import GeometryAutoResearch
            geoar = GeometryAutoResearch(
                self.cfg,
                lSet,
                uSet,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                clf_model=clf_model,
                trainDataset=trainDataset,
                dataObj=self.dataObj,
            )
            activeSet, uSet = geoar.select_samples()
            self.latest_sampling_metadata = getattr(geoar, 'selection_metadata', {})

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["dcom"]:
            from .DCoM import DCoM
            dcom = DCoM(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                        max_delta=self.cfg.ACTIVE_LEARNING.MAX_DELTA,
                        lSet_deltas=self.cfg.ACTIVE_LEARNING.DELTA_LST)
            activeSet, uSet = dcom.select_samples(clf_model, trainDataset, self.dataObj)
            self.latest_sampling_metadata = _default_sampling_metadata('dcom_policy', activeSet)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "dbal" or self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "DBAL":
            activeSet, uSet = self.sampler.dbal(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, \
                uSet=uSet, clf_model=clf_model,dataset=trainDataset)
            self.latest_sampling_metadata = _default_sampling_metadata('dbal_policy', activeSet)
            
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "bald" or self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "BALD":
            activeSet, uSet = self.sampler.bald(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, uSet=uSet, clf_model=clf_model, dataset=trainDataset)
            self.latest_sampling_metadata = _default_sampling_metadata('bald_policy', activeSet)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "ensemble_var_R":
            activeSet, uSet = self.sampler.ensemble_var_R(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, uSet=uSet, clf_models=supportingModels, dataset=trainDataset)
            self.latest_sampling_metadata = _default_sampling_metadata('ensemble_policy', activeSet)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "vaal":
            adv_sampler = AdversarySampler(cfg=self.cfg, dataObj=self.dataObj)

            # Train VAE and discriminator first
            vae, disc, uSet_loader = adv_sampler.vaal_perform_training(lSet=lSet, uSet=uSet, dataset=trainDataset)

            # Do active sampling
            activeSet, uSet = adv_sampler.sample_for_labeling(vae=vae, discriminator=disc, \
                                unlabeled_dataloader=uSet_loader, uSet=uSet)
            self.latest_sampling_metadata = _default_sampling_metadata('vaal_policy', activeSet)
        else:
            print(f"{self.cfg.ACTIVE_LEARNING.SAMPLING_FN} is either not implemented or there is some spelling mistake.")
            raise NotImplementedError

        return activeSet, uSet
        
