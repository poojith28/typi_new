import os
import sys
import json
import time
from datetime import datetime
import argparse
import numpy as np

import torch
from copy import deepcopy

# local
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath(os.path.join(_THIS_DIR, '..')))

from pycls.al.ActiveLearning import ActiveLearning
import pycls.core.builders as model_builder
from pycls.core.config import cfg, dump_cfg
import pycls.core.losses as losses
import pycls.core.optimizer as optim
from pycls.datasets.data import Data
from pycls.datasets.initial_lset import load_provided_initial_lset
import pycls.utils.checkpoint as cu
import pycls.utils.logging as lu
import pycls.utils.metrics as mu
import pycls.utils.net as nu
from pycls.utils.meters import TestMeter
from pycls.utils.meters import TrainMeter
from pycls.utils.meters import ValMeter

logger = lu.get_logger(__name__)

plot_episode_xvalues = []
plot_episode_yvalues = []

plot_epoch_xvalues = []
plot_epoch_yvalues = []

plot_it_x_values = []
plot_it_y_values = []

delta_avg_lst = []
delta_std_lst = []
ADAPTIVE_COVER_METHODS = {
    'prob_cover',
    'probcover',
    'probcover_auto_delta',
    'prob_cover_auto_delta',
    'id_prob_cover',
    'idprobcover',
    'lidcover_auto_delta',
    'idprobcover_auto_delta',
    'talc',
    'trajectory_adaptive_lid_cover',
    'talc_auto_delta',
    'idprobcover_frontier_density',
    'id_prob_cover_frontier_density',
    'idprobcover_tiebreak_min_id',
    'idprobcover_minid_tiebreak',
    'idprobcover_tiebreak_random',
    'idprobcover_random_tiebreak',
    'idprobcover_tiebreak_first_max',
    'idprobcover_firstmax_tiebreak',
    'knn_distance_cover',
    'adaptive_knn_distance_cover',
    'density_cover',
    'adaptive_density_cover',
    'distance_variance_cover',
    'adaptive_distance_variance_cover',
    'distance_cv_cover',
    'adaptive_distance_cv_cover',
}
AUTO_DELTA_METHODS = {
    'probcover_auto_delta',
    'prob_cover_auto_delta',
    'lidcover_auto_delta',
    'idprobcover_auto_delta',
    'talc_auto_delta',
}


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return value


def _mean_or_zero(values):
    if not values:
        return 0.0
    return float(np.mean(values))


def _median_or_zero(values):
    if not values:
        return 0.0
    return float(np.median(values))


def _timing_summary(episode_records, initial_sampling_record=None):
    timing_records = []
    for record in episode_records:
        timing = record.get('timing', {})
        if timing.get('has_sampling', False):
            timing_records.append(timing)

    initial_sampling_record = initial_sampling_record or {}
    initial_sampling_time = float(initial_sampling_record.get('acquisition_time_sec', 0.0) or 0.0)
    acquisition_times = [float(item.get('acquisition_time_sec', 0.0)) for item in timing_records]
    train_times = [float(item.get('train_time_sec', 0.0)) for item in timing_records]
    test_times = [float(item.get('test_time_sec', 0.0)) for item in timing_records]
    round_times = [float(item.get('round_time_sec', 0.0)) for item in timing_records]
    cumulative_acquisition = float(np.sum(acquisition_times)) if acquisition_times else 0.0
    cumulative_round = float(np.sum(round_times)) if round_times else 0.0

    return {
        'sampled_rounds': int(len(timing_records)),
        'initial_sampling_time_sec': initial_sampling_time,
        'initial_sampling_recorded': bool(initial_sampling_record),
        'acquisition_time_sec': {
            'mean': _mean_or_zero(acquisition_times),
            'median': _median_or_zero(acquisition_times),
            'cumulative': cumulative_acquisition,
            'cumulative_with_initial_sampling': cumulative_acquisition + initial_sampling_time,
        },
        'train_time_sec': {
            'mean': _mean_or_zero(train_times),
            'median': _median_or_zero(train_times),
            'cumulative': float(np.sum(train_times)) if train_times else 0.0,
        },
        'test_time_sec': {
            'mean': _mean_or_zero(test_times),
            'median': _median_or_zero(test_times),
            'cumulative': float(np.sum(test_times)) if test_times else 0.0,
        },
        'round_time_sec': {
            'mean': _mean_or_zero(round_times),
            'median': _median_or_zero(round_times),
            'cumulative': cumulative_round,
            'cumulative_with_initial_sampling': cumulative_round + initial_sampling_time,
        },
    }


def write_benchmark_summary(cfg, episode_records, initial_sampling_record=None):
    summary_path = os.path.join(cfg.EXP_DIR, 'benchmark_summary.json')
    final_record = episode_records[-1] if episode_records else {}
    test_curve = [float(record.get('test_accuracy', 0.0)) for record in episode_records]
    val_curve = [float(record.get('best_val_accuracy', 0.0)) for record in episode_records]
    summary = {
        'primary_metric_name': 'final_val_accuracy',
        'primary_metric': float(final_record.get('best_val_accuracy', 0.0)),
        'final_val_accuracy': float(final_record.get('best_val_accuracy', 0.0)),
        'final_test_accuracy': float(final_record.get('test_accuracy', 0.0)),
        'val_auc': _mean_or_zero(val_curve),
        'test_auc': _mean_or_zero(test_curve),
        'balanced_accuracy': 'not_available',
        'macro_f1': 'not_available',
        'sampling_fn': cfg.ACTIVE_LEARNING.SAMPLING_FN,
        'budget_per_round': int(cfg.ACTIVE_LEARNING.BUDGET_SIZE),
        'num_rounds_completed': int(len(episode_records)),
        'dataset': cfg.DATASET.NAME,
        'model': cfg.MODEL.TYPE,
        'seed': int(cfg.RNG_SEED),
        'exp_name': cfg.EXP_NAME,
        'exp_dir': cfg.EXP_DIR,
        'timing': _timing_summary(episode_records, initial_sampling_record=initial_sampling_record),
        'initial_sampling': initial_sampling_record or {},
        'episode_records': episode_records,
    }
    with open(summary_path, 'w') as handle:
        json.dump(summary, handle, indent=2)
    return summary_path


def get_probcover_delta(base_delta, labeled_count):
    base_delta = float(base_delta)
    if labeled_count == 0:
        phase = 'cold_start'
        scale = 1.35
    else:
        phase = 'post_label_shrink'
        scale = 0.85
    return base_delta * scale, phase, scale


def apply_probcover_delta(cfg, base_delta, labeled_count, logger_obj=None):
    effective_delta, phase, scale = get_probcover_delta(base_delta, labeled_count)
    cfg.ACTIVE_LEARNING.INITIAL_DELTA = effective_delta
    message = (
        'ProbCover adaptive delta: base={:.4f}, effective={:.4f}, scale={:.2f}, '
        'phase={}, labeled_count={}'
    ).format(base_delta, effective_delta, scale, phase, labeled_count)
    print(message)
    if logger_obj is not None:
        logger_obj.info(message)
    return effective_delta, phase, scale


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def argparser():
    parser = argparse.ArgumentParser(description='Active Learning - Image Classification')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file', required=True, type=str)
    parser.add_argument('--exp-name', help='Experiment Name', required=True, type=str)
    parser.add_argument('--al', help='AL Method', required=True, type=str)
    parser.add_argument('--budget', help='Budget Per Round', required=True, type=int)
    parser.add_argument('--initial_size', help='Size of the initial random labeled set', default=0, type=int)
    parser.add_argument('--init_lset_path', help='Ordered .npy IDs for an explicit shared initial labelled set', default=None, type=str)
    parser.add_argument('--seed', help='Random seed', default=1, type=int)
    parser.add_argument('--finetune', help='Whether to continue with existing model between rounds', type=str2bool, default=False)
    parser.add_argument('--linear_from_features', help='Whether to use a linear layer from self-supervised features', action='store_true')
    parser.add_argument('--initial_delta', help='Relevant only for ProbCover and DCoM', default=0.6, type=float)
    parser.add_argument('--idpc_alpha', help='IDProbCover radius adaptation strength', default=None, type=float)
    parser.add_argument('--idpc_mode', help='IDProbCover radius adaptation mode', default=None, type=str)
    parser.add_argument('--idpc_k_id', help='IDProbCover neighbors for local ID estimation', default=None, type=int)
    parser.add_argument('--idpc_k_knn', help='IDProbCover neighbors for coverage graph construction', default=None, type=int)
    parser.add_argument('--idpc_eps', help='IDProbCover numerical stability epsilon', default=None, type=float)
    parser.add_argument('--idpc_log_csv', help='Optional CSV path for IDProbCover diagnostics', default=None, type=str)
    parser.add_argument('--idpc_cache_root', help='Optional cache root reserved for IDProbCover artifacts', default=None, type=str)
    parser.add_argument('--auto_delta_k', help='Neighbour rank for automatic base-radius selection', default=None, type=int)
    parser.add_argument('--auto_delta_quantile', help='Quantile of k-th-neighbour distances used as the automatic base radius', default=None, type=float)
    parser.add_argument('--auto_delta_cache_root', help='Content-hashed cache root for automatic base-radius selection', default=None, type=str)
    parser.add_argument('--talc_alpha_max', help='TALC maximum local-ID radius exponent', default=None, type=float)
    parser.add_argument('--talc_coverage_target', help='TALC coverage fraction at which the structural curriculum is mature', default=None, type=float)
    parser.add_argument('--talc_coverage_epsilon', help='Allowed relative loss from the maximum greedy coverage gain', default=None, type=float)
    parser.add_argument('--talc_topology_weight', help='Late-curriculum topology weight; remaining weight is margin uncertainty', default=None, type=float)
    parser.add_argument('--talc_min_component_size', help='Minimum reference component size used by TALC topology', default=None, type=int)
    parser.add_argument('--talc_use_topology', help='Enable TALC multiscale component representation', type=str2bool, default=None)
    parser.add_argument('--talc_use_uncertainty', help='Enable TALC late-stage margin uncertainty', type=str2bool, default=None)
    parser.add_argument('--talc_cache_root', help='Cache root for TALC geometry and component partitions', default=None, type=str)
    parser.add_argument('--arc_alpha', help='Adaptive-cover radius scaling strength', default=None, type=float)
    parser.add_argument('--arc_k_signal', help='Adaptive-cover neighbors for local scaling signal estimation', default=None, type=int)
    parser.add_argument('--arc_k_knn', help='Adaptive-cover neighbors for coverage graph construction', default=None, type=int)
    parser.add_argument('--arc_eps', help='Adaptive-cover numerical stability epsilon', default=None, type=float)
    parser.add_argument('--arc_cache_root', help='Optional cache root for adaptive-cover artifacts', default=None, type=str)
    parser.add_argument('--topocover_k_knn', help='TopoCover neighbors for sparse proximity graph construction', default=None, type=int)
    parser.add_argument('--topocover_eps', help='TopoCover numerical stability epsilon', default=None, type=float)
    parser.add_argument('--topocover_cache_root', help='Optional cache root for TopoCover artifacts', default=None, type=str)
    parser.add_argument('--topocover_recompute_components', help='Whether TopoCover recomputes uncovered components after each greedy pick', type=str2bool, default=None)
    parser.add_argument('--mstc_delta_coarse', help='MultiScaleTopoCover coarse radius (defaults to --initial_delta)', default=None, type=float)
    parser.add_argument('--mstc_delta_fine', help='MultiScaleTopoCover fine radius (defaults to 0.5 * coarse)', default=None, type=float)
    parser.add_argument('--mstc_alpha', help='MultiScaleTopoCover coarse/fine mix weight in [0,1]', default=None, type=float)
    parser.add_argument('--mstc_graph_batch_size', help='MultiScaleTopoCover pairwise distance batch size', default=None, type=int)
    parser.add_argument('--mstc_device', help='MultiScaleTopoCover device for distance computation (cuda/cpu)', default=None, type=str)
    parser.add_argument('--phased_topo_switch_labeled', help='PhasedTopoCover labeled-count threshold to switch from ProbCover to WeightedTopoCover', default=None, type=int)
    parser.add_argument('--persist_min_component_size', help='PersistProbCover minimum δ-component size counted as signal', default=None, type=int)
    parser.add_argument('--w1_persist_quantile', help='W1ProxyCover persistence quantile for significant MST edges', default=None, type=float)
    parser.add_argument('--w1_barcode_lambda', help='W1ProxyCover weight for barcode-edge gain vs point coverage', default=None, type=float)
    parser.add_argument('--mtc_cache_root', help='MergeTreeCover cache root', default=None, type=str)
    parser.add_argument('--mtc_k_knn', help='MergeTreeCover kNN size', default=None, type=int)
    parser.add_argument('--mtc_min_component_size', help='MergeTreeCover minimum merge component size', default=None, type=int)
    parser.add_argument('--mtc_point_lambda', help='MergeTreeCover hybrid weight for δ point coverage', default=None, type=float)
    parser.add_argument('--mtc_radius_min', help='MergeTreeCover minimum event radius', default=None, type=float)
    parser.add_argument('--mtc_radius_max', help='MergeTreeCover maximum event radius', default=None, type=float)
    parser.add_argument('--mtc_probcover_fallback', help='MergeTreeCover ProbCover fallback when events exhausted', type=str2bool, default=None)
    parser.add_argument('--k_logistic', default=50, type=int)
    parser.add_argument('--a_logistic', default=0.8, type=float)

    return parser


def is_eval_epoch(cur_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (
        (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0 or
        (cur_epoch + 1) == cfg.OPTIM.MAX_EPOCH
    )


def main(cfg):
    # Route precomputed SimCLR features to the backbone-specific embedding tree.
    os.environ['TYPI_FEATURE_BACKBONE'] = str(cfg.MODEL.TYPE)
    if not os.environ.get('TYPI_FEATURES_ROOT'):
        os.environ['TYPI_FEATURES_ROOT'] = '/vast/s219110279/results/results'

    # Setting up GPU args
    use_cuda = (cfg.NUM_GPUS > 0) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': cfg.DATA_LOADER.NUM_WORKERS, 'pin_memory': cfg.DATA_LOADER.PIN_MEMORY} if use_cuda else {}

    # Auto assign a RNG_SEED when not supplied a value
    if cfg.RNG_SEED is None:
        cfg.RNG_SEED = np.random.randint(100)

    probcover_base_delta = float(cfg.ACTIVE_LEARNING.INITIAL_DELTA)

    # Using specific GPU
    # os.environ['NVIDIA_VISIBLE_DEVICES'] = str(cfg.GPU_ID)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # print("Using GPU : {}.\n".format(cfg.GPU_ID))

    # Getting the output directory ready (default is "/output")
    cfg.OUT_DIR = os.path.join(os.path.abspath('../..'), cfg.OUT_DIR)
    if not os.path.exists(cfg.OUT_DIR):
        os.mkdir(cfg.OUT_DIR)
    # Create "DATASET/MODEL TYPE" specific directory
    dataset_out_dir = os.path.join(cfg.OUT_DIR, cfg.DATASET.NAME, cfg.MODEL.TYPE)
    if not os.path.exists(dataset_out_dir):
        os.makedirs(dataset_out_dir)
    # Creating the experiment directory inside the dataset specific directory 
    # all logs, labeled, unlabeled, validation sets are stroed here 
    # E.g., output/CIFAR10/resnet18/{timestamp or cfg.EXP_NAME based on arguments passed}
    if cfg.EXP_NAME == 'auto':
        now = datetime.now()
        exp_dir = f'{now.year}_{now.month}_{now.day}_{now.hour:02}{now.minute:02}{now.second:02}_{now.microsecond}'
    else:
        exp_dir = cfg.EXP_NAME

    exp_dir = os.path.join(dataset_out_dir, exp_dir)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
        print("Experiment Directory is {}.\n".format(exp_dir))
    else:
        print("Experiment Directory Already Exists: {}. Reusing it may lead to loss of old logs in the directory.\n".format(exp_dir))
    cfg.EXP_DIR = exp_dir

    # Save the config file in EXP_DIR
    dump_cfg(cfg)

    # Setup Logger
    lu.setup_logging(cfg)

    # Dataset preparing steps
    print("\n======== PREPARING DATA AND MODEL ========\n")
    cfg.DATASET.ROOT_DIR = os.path.join(os.path.abspath('../..'), cfg.DATASET.ROOT_DIR)
    data_obj = Data(cfg)
    train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True)
    test_data, test_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True)
    cfg.ACTIVE_LEARNING.INIT_L_RATIO = args.initial_size / train_size
    print("\nDataset {} Loaded Sucessfully.\nTotal Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))
    logger.info("Dataset {} Loaded Sucessfully. Total Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))

    lSet_path, uSet_path, valSet_path = data_obj.makeLUVSets(train_split_ratio=cfg.ACTIVE_LEARNING.INIT_L_RATIO, \
        val_split_ratio=cfg.DATASET.VAL_RATIO, data=train_data, seed_id=cfg.RNG_SEED, save_dir=cfg.EXP_DIR)

    cfg.ACTIVE_LEARNING.LSET_PATH = lSet_path
    cfg.ACTIVE_LEARNING.USET_PATH = uSet_path
    cfg.ACTIVE_LEARNING.VALSET_PATH = valSet_path

    lSet, uSet, valSet = data_obj.loadPartitions(lSetPath=cfg.ACTIVE_LEARNING.LSET_PATH, \
            uSetPath=cfg.ACTIVE_LEARNING.USET_PATH, valSetPath = cfg.ACTIVE_LEARNING.VALSET_PATH)

    provided_initial_sampling_record = {}
    if cfg.ACTIVE_LEARNING.INIT_LSET_PATH:
        if args.initial_size != 0 or len(lSet) != 0:
            raise ValueError(
                'An explicit INIT_LSET_PATH requires --initial_size=0 and an empty generated lSet.'
            )
        lSet, uSet, initial_lset_metadata = load_provided_initial_lset(
            cfg.ACTIVE_LEARNING.INIT_LSET_PATH,
            uSet,
            valSet,
            train_size,
            cfg.ACTIVE_LEARNING.BUDGET_SIZE,
        )
        cfg.ACTIVE_LEARNING.INIT_LSET_PATH = initial_lset_metadata['source_path']
        cfg.ACTIVE_LEARNING.INIT_LSET_SHA256 = initial_lset_metadata['source_file_sha256']
        cfg.ACTIVE_LEARNING.INIT_LSET_ORDERED_SHA256 = initial_lset_metadata['ordered_int64_sha256']
        cfg.ACTIVE_LEARNING.INIT_LSET_COUNT = initial_lset_metadata['count']
        cfg.ACTIVE_LEARNING.LSET_PATH = data_obj.saveSet(lSet, 'lSet', cfg.EXP_DIR)
        cfg.ACTIVE_LEARNING.USET_PATH = data_obj.saveSet(uSet, 'uSet', cfg.EXP_DIR)
        dump_cfg(cfg)
        with open(os.path.join(cfg.EXP_DIR, 'initial_lset_provenance.json'), 'w') as handle:
            json.dump(initial_lset_metadata, handle, indent=2)
        provided_initial_sampling_record = {
            'stage': 'provided_initial_lset',
            'seed': int(cfg.RNG_SEED),
            'sampling_fn': 'provided_initial_lset',
            'labeled_count_before_sampling': 0,
            'unlabeled_count_before_sampling': int(len(uSet) + len(lSet)),
            'labeled_count_after_sampling': int(len(lSet)),
            'unlabeled_count_after_sampling': int(len(uSet)),
            'active_set_size': int(len(lSet)),
            'active_set_ids': [int(idx) for idx in lSet.tolist()],
            'acquisition_time_sec': 0.0,
            'timing': {
                'acquisition_time_sec': 0.0,
                'has_sampling': False,
                'is_provided_initial_lset': True,
            },
            'sampling_metadata': {
                'initialisation_type': 'seed_specific_shared_random_50',
                'source_path': initial_lset_metadata['source_path'],
                'source_file_sha256': initial_lset_metadata['source_file_sha256'],
                'ordered_int64_sha256': initial_lset_metadata['ordered_int64_sha256'],
            },
        }
        print(
            'Loaded provided initial labelled set: count={} sha256={} source={}'.format(
                len(lSet), initial_lset_metadata['source_file_sha256'],
                initial_lset_metadata['source_path']
            )
        )

    if cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in AUTO_DELTA_METHODS:
        if len(lSet) != 0:
            raise ValueError(
                'Automatic-delta evidence requires the canonical empty cold start; '
                f'found initial labeled size {len(lSet)}.'
            )
        from pycls.al.auto_delta import resolve_auto_delta, write_run_provenance

        candidate_indices = np.concatenate([lSet, uSet]).astype(np.int64, copy=False)
        probcover_base_delta, auto_delta_metadata = resolve_auto_delta(cfg, candidate_indices)
        cfg.ACTIVE_LEARNING.INITIAL_DELTA = float(probcover_base_delta)
        cfg.ACTIVE_LEARNING.AUTO_DELTA_RULE = str(auto_delta_metadata['rule'])
        cfg.ACTIVE_LEARNING.AUTO_DELTA_BASE = float(probcover_base_delta)
        cfg.ACTIVE_LEARNING.AUTO_DELTA_FEATURE_SHA256 = str(auto_delta_metadata['feature_sha256'])
        cfg.ACTIVE_LEARNING.AUTO_DELTA_INDICES_SHA256 = str(auto_delta_metadata['candidate_indices_sha256'])
        provenance_path = os.path.join(cfg.EXP_DIR, 'auto_delta_provenance.json')
        cfg.ACTIVE_LEARNING.AUTO_DELTA_PROVENANCE_PATH = provenance_path
        auto_delta_metadata = {
            **auto_delta_metadata,
            'run_provenance_path': provenance_path,
            'driver_cold_start_scale': 1.35,
            'driver_post_label_scale': 0.85,
        }
        write_run_provenance(cfg.EXP_DIR, auto_delta_metadata)
        dump_cfg(cfg)
        print(
            'Automatic base radius resolved: rule={} delta_auto={:.8f} k={} quantile={} '
            'feature_sha256={}'.format(
                auto_delta_metadata['rule'],
                probcover_base_delta,
                auto_delta_metadata['k'],
                auto_delta_metadata['quantile'],
                auto_delta_metadata['feature_sha256'],
            )
        )
    model = model_builder.build_model(cfg).cuda()

    initial_sampling_record = provided_initial_sampling_record

    if len(lSet) == 0:
        if cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ['dcom']:
            print('Labeled Set is Empty - Create and save the first delta values list')
            lSet_deltas = [str(cfg.ACTIVE_LEARNING.INITIAL_DELTA)] * cfg.ACTIVE_LEARNING.BUDGET_SIZE
            cfg.ACTIVE_LEARNING.DELTA_LST = lSet_deltas
            delta_avg_lst.append(cfg.ACTIVE_LEARNING.INITIAL_DELTA)

        print('Labeled Set is Empty - Sampling an Initial Pool')
        if cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ADAPTIVE_COVER_METHODS:
            apply_probcover_delta(cfg, probcover_base_delta, len(lSet))
        al_obj = ActiveLearning(data_obj, cfg)
        initial_acquisition_start = time.time()
        activeSet, new_uSet = al_obj.sample_from_uSet(model, lSet, uSet, train_data)
        initial_acquisition_time_sec = time.time() - initial_acquisition_start
        initial_sampling_metadata = getattr(al_obj, 'latest_sampling_metadata', {})
        initial_sampling_record = {
            'stage': 'initial_pool_sampling',
            'seed': int(cfg.RNG_SEED),
            'sampling_fn': cfg.ACTIVE_LEARNING.SAMPLING_FN,
            'labeled_count_before_sampling': int(len(lSet)),
            'unlabeled_count_before_sampling': int(len(uSet)),
            'labeled_count_after_sampling': int(len(lSet) + len(activeSet)),
            'unlabeled_count_after_sampling': int(len(new_uSet)),
            'active_set_size': int(len(activeSet)),
            'active_set_ids': [
                int(idx) for idx in np.asarray(activeSet, dtype=np.int64).tolist()
            ],
            'acquisition_time_sec': float(initial_acquisition_time_sec),
            'timing': {
                'acquisition_time_sec': float(initial_acquisition_time_sec),
                'has_sampling': True,
                'is_initial_pool_sampling': True,
            },
            'sampling_metadata': {
                key: _safe_float(value) if not isinstance(value, (dict, list)) else value
                for key, value in initial_sampling_metadata.items()
            },
        }
        with open(os.path.join(cfg.EXP_DIR, 'initial_sampling_summary.json'), 'w') as handle:
            json.dump(initial_sampling_record, handle, indent=2)
        print(f'Initial Pool is {activeSet}')
        # Preserve the complete cold-start trajectory, including the ordered
        # pre-acquisition pool required to reproduce deterministic tie breaks.
        initial_dir = os.path.join(cfg.EXP_DIR, 'initial_pool')
        os.makedirs(initial_dir, exist_ok=True)
        data_obj.saveSets(lSet, uSet, activeSet, initial_dir)
        data_obj.saveSet(new_uSet, 'new_uSet', initial_dir)
        # Add activeSet to lSet, save new_uSet as uSet and update dataloader for the next episode
        lSet = np.append(lSet, activeSet).astype(np.int64, copy=False)
        uSet = np.asarray(new_uSet, dtype=np.int64)

    print("Data Partitioning Complete. \nLabeled Set: {}, Unlabeled Set: {}, Validation Set: {}\n".format(len(lSet), len(uSet), len(valSet)))
    logger.info("Labeled Set: {}, Unlabeled Set: {}, Validation Set: {}\n".format(len(lSet), len(uSet), len(valSet)))

    # Preparing dataloaders for initial training
    lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    test_loader = data_obj.getTestLoader(data=test_data, test_batch_size=cfg.TRAIN.BATCH_SIZE, seed_id=cfg.RNG_SEED)

    # Initialize the model.  
    model = model_builder.build_model(cfg)
    print("model: {}\n".format(cfg.MODEL.TYPE))
    logger.info("model: {}\n".format(cfg.MODEL.TYPE))

    # Construct the optimizer
    optimizer = optim.construct_optimizer(cfg, model)
    opt_init_state = deepcopy(optimizer.state_dict())
    model_init_state = deepcopy(model.state_dict().copy())

    print("optimizer: {}\n".format(optimizer))
    logger.info("optimizer: {}\n".format(optimizer))

    print("AL Query Method: {}\nMax AL Episodes: {}\n".format(cfg.ACTIVE_LEARNING.SAMPLING_FN, cfg.ACTIVE_LEARNING.MAX_ITER))
    logger.info("AL Query Method: {}\nMax AL Episodes: {}\n".format(cfg.ACTIVE_LEARNING.SAMPLING_FN, cfg.ACTIVE_LEARNING.MAX_ITER))
    episode_records = []

    for cur_episode in range(0, cfg.ACTIVE_LEARNING.MAX_ITER+1):

        print("======== EPISODE {} BEGINS ========\n".format(cur_episode))
        logger.info("======== EPISODE {} BEGINS ========\n".format(cur_episode))

        # Creating output directory for the episode
        episode_dir = os.path.join(cfg.EXP_DIR, f'episode_{cur_episode}')
        if not os.path.exists(episode_dir):
            os.mkdir(episode_dir)
        cfg.EPISODE_DIR = episode_dir

        # Train model
        print("======== TRAINING ========")
        logger.info("======== TRAINING ========")
        train_start = time.time()
        best_val_acc, best_val_epoch, checkpoint_file = train_model(lSet_loader, valSet_loader, model, optimizer, cfg)
        train_time_sec = time.time() - train_start

        print("Best Validation Accuracy: {}\nBest Epoch: {}\n".format(round(best_val_acc, 4), best_val_epoch))
        logger.info("EPISODE {} Best Validation Accuracy: {}\tBest Epoch: {}\n".format(cur_episode, round(best_val_acc, 4), best_val_epoch))

        # Test best model checkpoint
        print("======== TESTING ========\n")
        logger.info("======== TESTING ========\n")
        test_start = time.time()
        test_acc = test_model(test_loader, checkpoint_file, cfg, cur_episode)
        test_time_sec = time.time() - test_start
        print("Test Accuracy: {}.\n".format(round(test_acc, 4)))
        logger.info("EPISODE {} Test Accuracy {}.\n".format(cur_episode, test_acc))
        episode_record = {
            'episode': int(cur_episode),
            'seed': int(cfg.RNG_SEED),
            'exp_name': cfg.EXP_NAME,
            'best_val_accuracy': float(best_val_acc),
            'best_val_epoch': int(best_val_epoch),
            'test_accuracy': float(test_acc),
            'labeled_count_before_sampling': int(len(lSet)),
            'unlabeled_count_before_sampling': int(len(uSet)),
            'sampling_fn': cfg.ACTIVE_LEARNING.SAMPLING_FN,
            'train_time_sec': float(train_time_sec),
            'test_time_sec': float(test_time_sec),
        }

        # No need to perform active sampling in the last episode iteration
        if cur_episode == cfg.ACTIVE_LEARNING.MAX_ITER:
            # Save current lSet, uSet in the final episode directory
            data_obj.saveSet(lSet, 'lSet', cfg.EPISODE_DIR)
            data_obj.saveSet(uSet, 'uSet', cfg.EPISODE_DIR)
            episode_record.update({
                'labeled_count_after_sampling': int(len(lSet)),
                'unlabeled_count_after_sampling': int(len(uSet)),
                'active_set_size': 0,
                'acquisition_time_sec': 0.0,
                'round_time_sec': float(train_time_sec + test_time_sec),
                'timing': {
                    'train_time_sec': float(train_time_sec),
                    'test_time_sec': float(test_time_sec),
                    'acquisition_time_sec': 0.0,
                    'round_time_sec': float(train_time_sec + test_time_sec),
                    'has_sampling': False,
                },
                'sampling_metadata': {'selection_mode': 'final_evaluation_only'},
            })
            episode_records.append(episode_record)
            break

        # DCoM's delta-s updating
        if cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["dcom"]:
            print("======== Update the deltas dynamically ========\n")
            from pycls.al.DCoM import DCoM
            al_algo = DCoM(cfg, lSet, uSet, budgetSize=cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                                    max_delta=cfg.ACTIVE_LEARNING.MAX_DELTA,
                                    lSet_deltas=cfg.ACTIVE_LEARNING.DELTA_LST)

            lSet_labels = np.take(train_data.targets, np.asarray(lSet, dtype=np.int64))
            all_images_idx = np.array(list(lSet) + list(uSet))
            images_loader = data_obj.getSequentialDataLoader(indexes=all_images_idx,
                                                    batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
            all_labels = np.take(train_data.targets, np.asarray(all_images_idx, dtype=np.int64))

            images_pseudo_labels = get_label_from_model(images_loader, checkpoint_file, cfg)
            cfg.ACTIVE_LEARNING.DELTA_LST[
            -1 * cfg.ACTIVE_LEARNING.BUDGET_SIZE:] = al_algo.new_centroids_deltas(lSet_labels,
                                                                          all_labels=all_labels,
                                                                          pseudo_labels=images_pseudo_labels,
                                                                          budget=cfg.ACTIVE_LEARNING.BUDGET_SIZE)

            delta_lst_float = [np.float(delta) for delta in cfg.ACTIVE_LEARNING.DELTA_LST]
            delta_avg_lst.append(np.average(delta_lst_float))
            delta_std_lst.append(np.std(delta_lst_float))

        # Active Sample 
        print("======== ACTIVE SAMPLING ========\n")
        logger.info("======== ACTIVE SAMPLING ========\n")
        if cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ADAPTIVE_COVER_METHODS:
            apply_probcover_delta(cfg, probcover_base_delta, len(lSet), logger_obj=logger)
        al_obj = ActiveLearning(data_obj, cfg)
        clf_model = model_builder.build_model(cfg)
        clf_model = cu.load_checkpoint(checkpoint_file, clf_model)
        acquisition_start = time.time()
        activeSet, new_uSet = al_obj.sample_from_uSet(clf_model, lSet, uSet, train_data)
        acquisition_time_sec = time.time() - acquisition_start
        sampling_metadata = getattr(al_obj, 'latest_sampling_metadata', {})
        if cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ADAPTIVE_COVER_METHODS:
            print(
                'Coverage selection metadata: active={} (mode={}, variant={}, delta={:.4f}, phase={}, bald_mean={:.4f}, bald_max={:.4f}, variance_mean={:.4f}, variance_max={:.4f}, dropout_iters={}, cover_anchors={}, diversity_lambda={:.2f}, selected_id_mean={:.4f}, selected_signal_mean={:.4f}, selected_radius_mean={:.4f}, k_id={}, k_knn={}, alpha={:.2f})'.format(
                    sampling_metadata.get('boundary_scores_active', False),
                    sampling_metadata.get('selection_mode', 'unknown'),
                    sampling_metadata.get('boundary_variant', 'unknown'),
                    sampling_metadata.get('effective_delta', 0.0),
                    sampling_metadata.get('delta_phase', 'unknown'),
                    sampling_metadata.get('epistemic_bald_mean', 0.0),
                    sampling_metadata.get('epistemic_bald_max', 0.0),
                    sampling_metadata.get('predictive_variance_mean', 0.0),
                    sampling_metadata.get('predictive_variance_max', 0.0),
                    sampling_metadata.get('dropout_iterations', 0),
                    sampling_metadata.get('cover_anchor_count', 0),
                    sampling_metadata.get('diversity_lambda', 0.0),
                    sampling_metadata.get('selected_id_mean', 0.0),
                    sampling_metadata.get('selected_signal_mean', 0.0),
                    sampling_metadata.get('selected_radius_mean', 0.0),
                    sampling_metadata.get('k_id', sampling_metadata.get('k_signal', 0)),
                    sampling_metadata.get('k_knn', 0),
                    sampling_metadata.get('alpha', 0.0),
                )
            )
            logger.info(
                'Coverage selection metadata: active={} (mode={}, variant={}, delta={:.4f}, phase={}, bald_mean={:.4f}, bald_max={:.4f}, variance_mean={:.4f}, variance_max={:.4f}, dropout_iters={}, cover_anchors={}, diversity_lambda={:.2f}, selected_id_mean={:.4f}, selected_signal_mean={:.4f}, selected_radius_mean={:.4f}, k_id={}, k_knn={}, alpha={:.2f})'.format(
                    sampling_metadata.get('boundary_scores_active', False),
                    sampling_metadata.get('selection_mode', 'unknown'),
                    sampling_metadata.get('boundary_variant', 'unknown'),
                    sampling_metadata.get('effective_delta', 0.0),
                    sampling_metadata.get('delta_phase', 'unknown'),
                    sampling_metadata.get('epistemic_bald_mean', 0.0),
                    sampling_metadata.get('epistemic_bald_max', 0.0),
                    sampling_metadata.get('predictive_variance_mean', 0.0),
                    sampling_metadata.get('predictive_variance_max', 0.0),
                    sampling_metadata.get('dropout_iterations', 0),
                    sampling_metadata.get('cover_anchor_count', 0),
                    sampling_metadata.get('diversity_lambda', 0.0),
                    sampling_metadata.get('selected_id_mean', 0.0),
                    sampling_metadata.get('selected_signal_mean', 0.0),
                    sampling_metadata.get('selected_radius_mean', 0.0),
                    sampling_metadata.get('k_id', sampling_metadata.get('k_signal', 0)),
                    sampling_metadata.get('k_knn', 0),
                    sampling_metadata.get('alpha', 0.0),
                )
            )

        if cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ['geometry_auto_research', 'geoar']:
            print(
                'GeometryAutoResearch metadata: graph_mode={} radius={} uncertainty_mode={} uncertainty_active={} components={} selected_id_mean={:.4f} selected_coverage_gain_mean={:.4f} selected_uncertainty_mean={:.4f} selected_score_mean={:.4f}'.format(
                    sampling_metadata.get('graph_mode', 'unknown'),
                    sampling_metadata.get('radius', None),
                    sampling_metadata.get('uncertainty_mode', 'unknown'),
                    sampling_metadata.get('uncertainty_active', False),
                    sampling_metadata.get('components', 0),
                    sampling_metadata.get('selected_id_mean', 0.0),
                    sampling_metadata.get('selected_coverage_gain_mean', 0.0),
                    sampling_metadata.get('selected_uncertainty_mean', 0.0),
                    sampling_metadata.get('selected_score_mean', 0.0),
                )
            )
            logger.info(
                'GeometryAutoResearch metadata: graph_mode={} radius={} uncertainty_mode={} uncertainty_active={} components={} selected_id_mean={:.4f} selected_coverage_gain_mean={:.4f} selected_uncertainty_mean={:.4f} selected_score_mean={:.4f}'.format(
                    sampling_metadata.get('graph_mode', 'unknown'),
                    sampling_metadata.get('radius', None),
                    sampling_metadata.get('uncertainty_mode', 'unknown'),
                    sampling_metadata.get('uncertainty_active', False),
                    sampling_metadata.get('components', 0),
                    sampling_metadata.get('selected_id_mean', 0.0),
                    sampling_metadata.get('selected_coverage_gain_mean', 0.0),
                    sampling_metadata.get('selected_uncertainty_mean', 0.0),
                    sampling_metadata.get('selected_score_mean', 0.0),
                )
            )

        # Save current lSet, new_uSet and activeSet in the episode directory
        data_obj.saveSets(lSet, uSet, activeSet, cfg.EPISODE_DIR)

        # Add activeSet to lSet, save new_uSet as uSet and update dataloader for the next episode
        lSet = np.append(lSet, activeSet).astype(np.int64, copy=False)
        uSet = np.asarray(new_uSet, dtype=np.int64)
        episode_record.update({
            'labeled_count_after_sampling': int(len(lSet)),
            'unlabeled_count_after_sampling': int(len(uSet)),
            'active_set_size': int(len(activeSet)),
            'acquisition_time_sec': float(acquisition_time_sec),
            'round_time_sec': float(train_time_sec + test_time_sec + acquisition_time_sec),
            'active_set_ids': [int(idx) for idx in np.asarray(activeSet, dtype=np.int64).tolist()],
            'timing': {
                'train_time_sec': float(train_time_sec),
                'test_time_sec': float(test_time_sec),
                'acquisition_time_sec': float(acquisition_time_sec),
                'round_time_sec': float(train_time_sec + test_time_sec + acquisition_time_sec),
                'has_sampling': True,
            },
            'sampling_metadata': {
                key: _safe_float(value) if not isinstance(value, (dict, list)) else value
                for key, value in sampling_metadata.items()
            },
        })
        episode_records.append(episode_record)
        with open(os.path.join(cfg.EPISODE_DIR, 'episode_summary.json'), 'w') as handle:
            json.dump(episode_record, handle, indent=2)

        lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
        valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
        uSet_loader = data_obj.getSequentialDataLoader(indexes=uSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)

        print("Active Sampling Complete. After Episode {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(cur_episode, len(lSet), len(uSet), len(activeSet)))
        logger.info("Active Sampling Complete. After Episode {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(cur_episode, len(lSet), len(uSet), len(activeSet)))
        print("================================\n\n")
        logger.info("================================\n\n")

        # add avg delta to cfg.ACTIVE_LEARNING.DELTA_LST towards the next active sampling
        if cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ['dcom']:
            delta_lst_float = [np.float(delta) for delta in cfg.ACTIVE_LEARNING.DELTA_LST]
            next_initial_deltas = [str(round(np.average(delta_lst_float), 2))] * cfg.ACTIVE_LEARNING.BUDGET_SIZE
            cfg.ACTIVE_LEARNING.DELTA_LST.extend(next_initial_deltas)
            print("Current delta list: ", cfg.ACTIVE_LEARNING.DELTA_LST)
            print("Current delta avg list: ", delta_avg_lst)
            print("Current delta std list: ", delta_std_lst)
        print('Current accuracy values: ', plot_episode_yvalues)

        if not cfg.ACTIVE_LEARNING.FINE_TUNE:
            # start model from scratch
            print('Starting model from scratch - ignoring existing weights.')
            model = model_builder.build_model(cfg)
            # Construct the optimizer
            optimizer = optim.construct_optimizer(cfg, model)
            print(model.load_state_dict(model_init_state))
            print(optimizer.load_state_dict(opt_init_state))

        os.remove(checkpoint_file)

    write_benchmark_summary(cfg, episode_records, initial_sampling_record=initial_sampling_record)



def train_model(train_loader, val_loader, model, optimizer, cfg):
    global plot_episode_xvalues
    global plot_episode_yvalues

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    start_epoch = 0
    loss_fun = losses.get_loss_fun()

    # Create meters
    train_meter = TrainMeter(len(train_loader))
    val_meter = ValMeter(len(val_loader))

    # Perform the training loop
    # print("Len(train_loader):{}".format(len(train_loader)))
    logger.info('Start epoch: {}'.format(start_epoch + 1))
    val_set_acc = 0.

    temp_best_val_acc = 0.
    temp_best_val_epoch = 0

    # Best checkpoint model and optimizer states
    best_model_state = None
    best_opt_state = None

    val_acc_epochs_x = []
    val_acc_epochs_y = []

    clf_train_iterations = cfg.OPTIM.MAX_EPOCH * int(len(train_loader)/cfg.TRAIN.BATCH_SIZE)
    clf_change_lr_iter = clf_train_iterations // 25
    clf_iter_count = 0

    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):

        # Train for one epoch
        train_loss, clf_iter_count = train_epoch(train_loader, model, loss_fun, optimizer, train_meter, \
                                        cur_epoch, cfg, clf_iter_count, clf_change_lr_iter, clf_train_iterations)

        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            nu.compute_precise_bn_stats(model, train_loader)


        # Model evaluation
        if is_eval_epoch(cur_epoch):
            # Original code[PYCLS] passes on testLoader but we want to compute on val Set
            val_loader.dataset.no_aug = True
            val_set_err = test_epoch(val_loader, model, val_meter, cur_epoch)
            val_set_acc = 100. - val_set_err
            val_loader.dataset.no_aug = False
            if temp_best_val_acc < val_set_acc:
                temp_best_val_acc = val_set_acc
                temp_best_val_epoch = cur_epoch + 1

                # Save best model and optimizer state for checkpointing
                model.eval()

                best_model_state = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
                best_opt_state = optimizer.state_dict()

                model.train()

            # Since we start from 0 epoch
            val_acc_epochs_x.append(cur_epoch+1)
            val_acc_epochs_y.append(val_set_acc)

        plot_epoch_xvalues.append(cur_epoch+1)
        plot_epoch_yvalues.append(train_loss)

        # save_plot_values([plot_epoch_xvalues, plot_epoch_yvalues, plot_it_x_values, plot_it_y_values, val_acc_epochs_x, val_acc_epochs_y],\
        #     ["plot_epoch_xvalues", "plot_epoch_yvalues", "plot_it_x_values", "plot_it_y_values","val_acc_epochs_x","val_acc_epochs_y"], out_dir=cfg.EPISODE_DIR, isDebug=False)
        logger.info("Successfully logged numpy arrays!!")

        # Plot arrays
        # plot_arrays(x_vals=plot_epoch_xvalues, y_vals=plot_epoch_yvalues, \
        # x_name="Epochs", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)
        #
        # plot_arrays(x_vals=val_acc_epochs_x, y_vals=val_acc_epochs_y, \
        # x_name="Epochs", y_name="Validation Accuracy", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)

        # save_plot_values([plot_epoch_xvalues, plot_epoch_yvalues, plot_it_x_values, plot_it_y_values, val_acc_epochs_x, val_acc_epochs_y], \
        #         ["plot_epoch_xvalues", "plot_epoch_yvalues", "plot_it_x_values", "plot_it_y_values","val_acc_epochs_x","val_acc_epochs_y"], out_dir=cfg.EPISODE_DIR)

        print('Training Epoch: {}/{}\tTrain Loss: {}\tVal Accuracy: {}'.format(cur_epoch+1, cfg.OPTIM.MAX_EPOCH, round(train_loss, 4), round(val_set_acc, 4)))

    # Save the best model checkpoint (Episode level)
    if best_model_state is None:
        logger.warning(
            "No validation improvement this episode (best_val_acc=%s); "
            "falling back to final model weights for checkpoint.",
            temp_best_val_acc,
        )
        best_model_state = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
        best_opt_state = optimizer.state_dict()
    checkpoint_file = cu.save_checkpoint(info="vlBest_acc_"+str(int(temp_best_val_acc)), \
        model_state=best_model_state, optimizer_state=best_opt_state, epoch=temp_best_val_epoch, cfg=cfg)

    print('\nWrote Best Model Checkpoint to: {}\n'.format(checkpoint_file.split('/')[-1]))
    logger.info('Wrote Best Model Checkpoint to: {}\n'.format(checkpoint_file))

    # plot_arrays(x_vals=plot_epoch_xvalues, y_vals=plot_epoch_yvalues, \
    #     x_name="Epochs", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)
    #
    # plot_arrays(x_vals=plot_it_x_values, y_vals=plot_it_y_values, \
    #     x_name="Iterations", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)
    #
    # plot_arrays(x_vals=val_acc_epochs_x, y_vals=val_acc_epochs_y, \
    #     x_name="Epochs", y_name="Validation Accuracy", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)

    plot_epoch_xvalues = []
    plot_epoch_yvalues = []
    plot_it_x_values = []
    plot_it_y_values = []

    best_val_acc = temp_best_val_acc
    best_val_epoch = temp_best_val_epoch

    return best_val_acc, best_val_epoch, checkpoint_file


def test_model(test_loader, checkpoint_file, cfg, cur_episode):

    global plot_episode_xvalues
    global plot_episode_yvalues

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    test_meter = TestMeter(len(test_loader))

    model = model_builder.build_model(cfg)
    model = cu.load_checkpoint(checkpoint_file, model)

    test_err = test_epoch(test_loader, model, test_meter, cur_episode)
    test_acc = 100. - test_err

    plot_episode_xvalues.append(cur_episode)
    plot_episode_yvalues.append(test_acc)

    # plot_arrays(x_vals=plot_episode_xvalues, y_vals=plot_episode_yvalues, \
    #     x_name="Episodes", y_name="Test Accuracy", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EXP_DIR)
    #
    # save_plot_values([plot_episode_xvalues, plot_episode_yvalues], \
    #     ["plot_episode_xvalues", "plot_episode_yvalues"], out_dir=cfg.EXP_DIR)

    return test_acc


def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch, cfg, clf_iter_count, clf_change_lr_iter, clf_max_iter):
    """Performs one epoch of training."""
    global plot_episode_xvalues
    global plot_episode_yvalues

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    # Shuffle the data
    #loader.shuffle(train_loader, cur_epoch)
    if cfg.NUM_GPUS>1:  train_loader.sampler.set_epoch(cur_epoch)

    # Update the learning rate
    # Currently we only support LR schedules for only 'SGD' optimizer
    lr = optim.get_epoch_lr(cfg, cur_epoch)
    if cfg.OPTIM.TYPE == "sgd":
        optim.set_lr(optimizer, lr)

    if torch.cuda.is_available():
        model.cuda()

    # Enable training mode
    model.train()
    train_meter.iter_tic() #This basically notes the start time in timer class defined in utils/timer.py

    len_train_loader = len(train_loader)
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        #ensuring that inputs are floatTensor as model weights are
        inputs = inputs.type(torch.cuda.FloatTensor)
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Perform the forward pass
        preds = model(inputs)
        # Compute the loss
        loss = loss_fun(preds, labels)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update the parametersSWA
        optimizer.step()
        # Compute the errors
        top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
        # Combine the stats across the GPUs
        # if cfg.NUM_GPUS > 1:
        #     #Average error and losses across GPUs
        #     #Also this this calls wait method on reductions so we are ensured
        #     #to obtain synchronized results
        #     loss, top1_err = du.scaled_all_reduce(
        #         [loss, top1_err]
        #     )
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err = loss.item(), top1_err.item()
        # #Only master process writes the logs which are used for plotting
        # if du.is_master_proc():
        if cur_iter != 0 and cur_iter%19 == 0:
            #because cur_epoch starts with 0
            plot_it_x_values.append((cur_epoch)*len_train_loader + cur_iter)
            plot_it_y_values.append(loss)
            # save_plot_values([plot_it_x_values, plot_it_y_values],["plot_it_x_values", "plot_it_y_values"], out_dir=cfg.EPISODE_DIR, isDebug=False)
            # print(plot_it_x_values)
            # print(plot_it_y_values)
            #Plot loss graphs
            # plot_arrays(x_vals=plot_it_x_values, y_vals=plot_it_y_values, x_name="Iterations", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR,)
            print('Training Epoch: {}/{}\tIter: {}/{}'.format(cur_epoch+1, cfg.OPTIM.MAX_EPOCH, cur_iter, len(train_loader)))

        #Compute the difference in time now from start time initialized just before this for loop.
        train_meter.iter_toc()
        train_meter.update_stats(top1_err=top1_err, loss=loss, \
            lr=lr, mb_size=inputs.size(0) * cfg.NUM_GPUS)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    return loss, clf_iter_count


def get_label_from_model(images_loader, checkpoint_file, cfg, model=None):
    """
    returns the labels of the images according to the checkpoint file model
    """
    get_label_meter = TestMeter(len(images_loader))
    if model is None:
        model = model_builder.build_model(cfg)
        model = cu.load_checkpoint(checkpoint_file, model)

    pred = get_label_epoch(images_loader, model, get_label_meter)
    return pred

@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""

    global plot_episode_xvalues
    global plot_episode_yvalues

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    if torch.cuda.is_available():
        model.cuda()

    # Enable eval mode
    model.eval()
    test_meter.iter_tic()

    misclassifications = 0.
    totalSamples = 0.

    for cur_iter, (inputs, labels) in enumerate(test_loader):
        with torch.no_grad():
            # Transfer the data to the current GPU device
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            inputs = inputs.type(torch.cuda.FloatTensor)
            # Compute the predictions
            preds = model(inputs)
            # Compute the errors
            top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
            # Combine the errors across the GPUs
            # if cfg.NUM_GPUS > 1:
            #     top1_err = du.scaled_all_reduce([top1_err])
            #     #as above returns a list
            #     top1_err = top1_err[0]
            # Copy the errors from GPU to CPU (sync point)
            top1_err = top1_err.item()
            # Multiply by Number of GPU's as top1_err is scaled by 1/Num_GPUs
            misclassifications += top1_err * inputs.size(0) * cfg.NUM_GPUS
            totalSamples += inputs.size(0)*cfg.NUM_GPUS
            test_meter.iter_toc()
            # Update and log stats
            test_meter.update_stats(
                top1_err=top1_err, mb_size=inputs.size(0) * cfg.NUM_GPUS
            )
            test_meter.log_iter_stats(cur_epoch, cur_iter)
            test_meter.iter_tic()
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()

    return misclassifications/totalSamples

@torch.no_grad()
def get_label_epoch(images_loader, model, get_label_meter):
    """get labels according to the model."""
    if torch.cuda.is_available():
        model.cuda()

    # Enable eval mode
    model.eval()
    get_label_meter.iter_tic()

    all_preds = []
    for cur_iter, (inputs, _) in enumerate(images_loader):
        with torch.no_grad():
            # Transfer the data to the current GPU device
            inputs = inputs.cuda().type(torch.cuda.FloatTensor)
            # Compute the predictions
            preds = model(inputs)
            all_preds += preds

    final_preds = [torch.argmax(p).item() for p in all_preds]
    model.train()

    return final_preds


if __name__ == "__main__":
    args = argparser().parse_args()
    cfg.merge_from_file(args.cfg_file)
    cfg.EXP_NAME = args.exp_name
    cfg.ACTIVE_LEARNING.SAMPLING_FN = args.al
    cfg.ACTIVE_LEARNING.BUDGET_SIZE = args.budget
    if args.init_lset_path is not None:
        cfg.ACTIVE_LEARNING.INIT_LSET_PATH = args.init_lset_path
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
    if args.idpc_eps is not None:
        cfg.ACTIVE_LEARNING.IDPC_EPS = args.idpc_eps
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
    if args.talc_alpha_max is not None:
        cfg.ACTIVE_LEARNING.TALC_ALPHA_MAX = args.talc_alpha_max
    if args.talc_coverage_target is not None:
        cfg.ACTIVE_LEARNING.TALC_COVERAGE_TARGET = args.talc_coverage_target
    if args.talc_coverage_epsilon is not None:
        cfg.ACTIVE_LEARNING.TALC_COVERAGE_EPSILON = args.talc_coverage_epsilon
    if args.talc_topology_weight is not None:
        cfg.ACTIVE_LEARNING.TALC_TOPOLOGY_WEIGHT = args.talc_topology_weight
    if args.talc_min_component_size is not None:
        cfg.ACTIVE_LEARNING.TALC_MIN_COMPONENT_SIZE = args.talc_min_component_size
    if args.talc_use_topology is not None:
        cfg.ACTIVE_LEARNING.TALC_USE_TOPOLOGY = args.talc_use_topology
    if args.talc_use_uncertainty is not None:
        cfg.ACTIVE_LEARNING.TALC_USE_UNCERTAINTY = args.talc_use_uncertainty
    if args.talc_cache_root is not None:
        cfg.ACTIVE_LEARNING.TALC_CACHE_ROOT = args.talc_cache_root
    if args.arc_alpha is not None:
        cfg.ACTIVE_LEARNING.ARC_ALPHA = args.arc_alpha
    if args.arc_k_signal is not None:
        cfg.ACTIVE_LEARNING.ARC_K_SIGNAL = args.arc_k_signal
    if args.arc_k_knn is not None:
        cfg.ACTIVE_LEARNING.ARC_K_KNN = args.arc_k_knn
    if args.arc_eps is not None:
        cfg.ACTIVE_LEARNING.ARC_EPS = args.arc_eps
    if args.arc_cache_root is not None:
        cfg.ACTIVE_LEARNING.ARC_CACHE_ROOT = args.arc_cache_root
    if args.topocover_k_knn is not None:
        cfg.ACTIVE_LEARNING.TOPOCOVER_K_KNN = args.topocover_k_knn
    if args.topocover_eps is not None:
        cfg.ACTIVE_LEARNING.TOPOCOVER_EPS = args.topocover_eps
    if args.topocover_cache_root is not None:
        cfg.ACTIVE_LEARNING.TOPOCOVER_CACHE_ROOT = args.topocover_cache_root
    if args.topocover_recompute_components is not None:
        cfg.ACTIVE_LEARNING.TOPOCOVER_RECOMPUTE_COMPONENTS = args.topocover_recompute_components
    if args.mstc_delta_coarse is not None:
        cfg.ACTIVE_LEARNING.MSTC_DELTA_COARSE = args.mstc_delta_coarse
    if args.mstc_delta_fine is not None:
        cfg.ACTIVE_LEARNING.MSTC_DELTA_FINE = args.mstc_delta_fine
    if args.mstc_alpha is not None:
        cfg.ACTIVE_LEARNING.MSTC_ALPHA = args.mstc_alpha
    if args.mstc_graph_batch_size is not None:
        cfg.ACTIVE_LEARNING.MSTC_GRAPH_BATCH_SIZE = args.mstc_graph_batch_size
    if args.mstc_device is not None:
        cfg.ACTIVE_LEARNING.MSTC_DEVICE = args.mstc_device
    if args.phased_topo_switch_labeled is not None:
        cfg.ACTIVE_LEARNING.PHASED_TOPO_SWITCH_LABELED = args.phased_topo_switch_labeled
    if args.persist_min_component_size is not None:
        cfg.ACTIVE_LEARNING.PERSIST_MIN_COMPONENT_SIZE = args.persist_min_component_size
    if args.w1_persist_quantile is not None:
        cfg.ACTIVE_LEARNING.W1_PERSIST_QUANTILE = args.w1_persist_quantile
    if args.w1_barcode_lambda is not None:
        cfg.ACTIVE_LEARNING.W1_BARCODE_LAMBDA = args.w1_barcode_lambda
    if args.mtc_cache_root is not None:
        cfg.ACTIVE_LEARNING.MTC_CACHE_ROOT = args.mtc_cache_root
    if args.mtc_k_knn is not None:
        cfg.ACTIVE_LEARNING.MTC_K_KNN = args.mtc_k_knn
    if args.mtc_min_component_size is not None:
        cfg.ACTIVE_LEARNING.MTC_MIN_COMPONENT_SIZE = args.mtc_min_component_size
    if args.mtc_point_lambda is not None:
        cfg.ACTIVE_LEARNING.MTC_POINT_LAMBDA = args.mtc_point_lambda
    if args.mtc_radius_min is not None:
        cfg.ACTIVE_LEARNING.MTC_RADIUS_MIN = args.mtc_radius_min
    if args.mtc_radius_max is not None:
        cfg.ACTIVE_LEARNING.MTC_RADIUS_MAX = args.mtc_radius_max
    if args.mtc_probcover_fallback is not None:
        cfg.ACTIVE_LEARNING.MTC_PROBCOVER_FALLBACK = args.mtc_probcover_fallback
    main(cfg)
