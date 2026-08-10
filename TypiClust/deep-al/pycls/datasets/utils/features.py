import os

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEEP_AL_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_DEEP_AL_ROOT, '..'))

# Dataset folder names under <embed_root>/<backbone>/<dataset_dir>/pretext/
_DATASET_DIRS = {
    'CIFAR10': 'cifar-10',
    'CIFAR100': 'cifar-100',
    'TINYIMAGENET': 'tiny-imagenet',
}

# Legacy scan layout (resnet18 SimCLR features only, no backbone subfolder).
_LEGACY_SCAN_FEATURES = {
    'train': {
        'CIFAR10': '../../scan/results/cifar-10/pretext/features_seed1.npy',
        'CIFAR100': '../../scan/results/cifar-100/pretext/features_seed1.npy',
        'TINYIMAGENET': '../../scan/results/tiny-imagenet/pretext/features_seed1.npy',
        'IMAGENET50': '../../dino/runs/trainfeat.pth',
        'IMAGENET100': '../../dino/runs/trainfeat.pth',
        'IMAGENET200': '../../dino/runs/trainfeat.pth',
    },
    'test': {
        'CIFAR10': '../../scan/results/cifar-10/pretext/test_features_seed1.npy',
        'CIFAR100': '../../scan/results/cifar-100/pretext/test_features_seed1.npy',
        'TINYIMAGENET': '../../scan/results/tiny-imagenet/pretext/test_features_seed1.npy',
        'IMAGENET50': '../../dino/runs/testfeat.pth',
        'IMAGENET100': '../../dino/runs/testfeat.pth',
        'IMAGENET200': '../../dino/runs/testfeat.pth',
    },
}


def feature_embeddings_root():
    """Root directory containing <backbone>/<dataset>/pretext/ feature trees."""
    return os.environ.get(
        'TYPI_FEATURES_ROOT',
        '/vast/s219110279/results/results',
    )


def feature_backbone(backbone=None):
    if backbone is None:
        backbone = os.environ.get('TYPI_FEATURE_BACKBONE', 'resnet18')
    return backbone


def _backbone_feature_path(ds_name, seed, train, backbone):
    ds_dir = _DATASET_DIRS.get(ds_name)
    if ds_dir is None:
        return None
    split = 'features' if train else 'test_features'
    fname = f'{split}_seed{seed}.npy'
    return os.path.join(
        feature_embeddings_root(),
        backbone,
        ds_dir,
        'pretext',
        fname,
    )


def _candidate_feature_paths(raw_path, backbone_path=None):
    candidates = []

    if backbone_path:
        candidates.append(backbone_path)

    if os.path.isabs(raw_path):
        candidates.append(raw_path)
    else:
        candidates.append(os.path.abspath(os.path.join(_THIS_DIR, raw_path)))
        candidates.append(os.path.abspath(os.path.join(_DEEP_AL_ROOT, raw_path)))
        if raw_path.startswith('../../'):
            candidates.append(os.path.abspath(os.path.join(_PROJECT_ROOT, raw_path[6:])))

    normalized = []
    seen = set()
    for path in candidates:
        norm = os.path.normpath(path)
        if norm not in seen:
            normalized.append(norm)
            seen.add(norm)
    return normalized


def _feature_seeds_to_try(requested_seed):
    """Embeddings on disk are usually features_seed1.npy; AL seed can be 1-5."""
    override = os.environ.get('TYPI_FEATURE_SEED')
    seeds = []
    for s in (requested_seed, int(override) if override else None, 1):
        if s is None:
            continue
        s = int(s)
        if s not in seeds:
            seeds.append(s)
    return seeds


def load_features(ds_name, seed=1, train=True, normalized=True, backbone=None):
    """Load pretrained SimCLR features for a dataset and backbone."""
    split = 'train' if train else 'test'
    backbone = feature_backbone(backbone)

    raw_path = _LEGACY_SCAN_FEATURES[split].get(ds_name)
    if raw_path is None:
        raise KeyError(f'Unknown dataset for features: {ds_name}')

    fname = None
    candidate_paths = []
    for try_seed in _feature_seeds_to_try(seed):
        backbone_path = _backbone_feature_path(ds_name, try_seed, train, backbone)
        legacy = raw_path.format(seed=try_seed)
        candidate_paths = _candidate_feature_paths(legacy, backbone_path=backbone_path)
        fname = next((path for path in candidate_paths if os.path.exists(path)), None)
        if fname is not None:
            break

    if fname is None:
        raise FileNotFoundError(
            'Could not find pretrained features for dataset={} split={} seed={} backbone={}.'
            ' Tried: {}'.format(ds_name, split, seed, backbone, candidate_paths)
        )
    if fname.endswith('.npy'):
        features = np.load(fname)
    elif fname.endswith('.pth'):
        features = torch.load(fname)
    else:
        raise Exception('Unsupported filetype')
    if normalized:
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
    return features
