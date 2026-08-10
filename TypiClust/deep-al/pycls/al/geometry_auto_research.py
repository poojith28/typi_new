import csv
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pycls.datasets.utils as ds_utils

try:
    import faiss
except ImportError:  # pragma: no cover - optional acceleration
    faiss = None


def _safe_normalize(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    min_val = float(values.min())
    max_val = float(values.max())
    if max_val > min_val:
        return (values - min_val) / (max_val - min_val)
    return np.zeros_like(values, dtype=np.float32)


def _summary_stats(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
    return {
        'mean': float(values.mean()),
        'min': float(values.min()),
        'max': float(values.max()),
        'std': float(values.std()),
    }


def _parse_round_index(episode_dir):
    if not episode_dir:
        return -1
    match = re.search(r'episode_(\d+)', str(episode_dir))
    return int(match.group(1)) if match else -1


def _build_bruteforce_knn(embeddings, k, device=None, batch_size=2048):
    num_points = len(embeddings)
    effective_k = min(max(1, int(k)), max(1, num_points - 1))
    if num_points <= 1:
        return (
            np.zeros((num_points, 0), dtype=np.int64),
            np.zeros((num_points, 0), dtype=np.float32),
        )

    tensor = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    neighbors = np.empty((num_points, effective_k), dtype=np.int64)
    distances = np.empty((num_points, effective_k), dtype=np.float32)

    for start in range(0, num_points, batch_size):
        stop = min(start + batch_size, num_points)
        cur = tensor[start:stop]
        dists = torch.cdist(cur, tensor, p=2)
        row_idx = torch.arange(stop - start, device=tensor.device)
        dists[row_idx, start + row_idx] = float('inf')
        topk_dists, topk_idx = torch.topk(dists, k=effective_k, largest=False, dim=1)
        neighbors[start:stop] = topk_idx.cpu().numpy().astype(np.int64, copy=False)
        distances[start:stop] = topk_dists.cpu().numpy().astype(np.float32, copy=False)

    return neighbors, distances


def build_knn_graph(embeddings, k, graph_mode='knn', radius=None):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    num_points = len(embeddings)
    if num_points == 0:
        return {
            'neighbors': np.zeros((0, 0), dtype=np.int64),
            'distances': np.zeros((0, 0), dtype=np.float32),
            'coverage_neighbors': [],
            'radius': None,
            'graph_mode': graph_mode,
        }

    effective_k = min(max(1, int(k)), max(1, num_points - 1))
    if faiss is not None:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        if torch.cuda.is_available():
            try:
                index = faiss.index_cpu_to_all_gpus(index)
            except Exception:
                pass
        index.add(embeddings)
        distances_sq, neighbors = index.search(embeddings, effective_k + 1)
        neighbors = neighbors[:, 1:]
        distances = np.sqrt(np.maximum(distances_sq[:, 1:], 0.0)).astype(np.float32, copy=False)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        neighbors, distances = _build_bruteforce_knn(embeddings, effective_k, device=device)

    if radius is None and graph_mode == 'radius' and distances.shape[1] > 0:
        radius = float(np.median(distances[:, -1]))

    coverage_neighbors = []
    for idx in range(num_points):
        if graph_mode == 'radius' and radius is not None and distances.shape[1] > 0:
            nbrs = neighbors[idx][distances[idx] <= float(radius)]
        else:
            nbrs = neighbors[idx]
        closed_nbrs = np.concatenate(([idx], np.asarray(nbrs, dtype=np.int64)))
        coverage_neighbors.append(np.unique(closed_nbrs).astype(np.int64, copy=False))

    return {
        'neighbors': neighbors.astype(np.int64, copy=False),
        'distances': distances.astype(np.float32, copy=False),
        'coverage_neighbors': coverage_neighbors,
        'radius': radius,
        'graph_mode': graph_mode,
    }


def compute_local_id(neighbor_distances, k):
    neighbor_distances = np.asarray(neighbor_distances, dtype=np.float32)
    if neighbor_distances.ndim != 2 or neighbor_distances.shape[1] == 0:
        return np.zeros(neighbor_distances.shape[0], dtype=np.float32)

    effective_k = min(max(2, int(k)), neighbor_distances.shape[1])
    dists = np.clip(neighbor_distances[:, :effective_k], a_min=1e-12, a_max=None)
    kth = dists[:, effective_k - 1:effective_k]
    logs = np.log(np.clip(kth / dists[:, :effective_k - 1], a_min=1.0, a_max=None))
    mean_logs = np.mean(logs, axis=1)
    ids = 1.0 / np.clip(mean_logs, a_min=1e-12, a_max=None)
    ids = np.nan_to_num(ids, nan=0.0, posinf=0.0, neginf=0.0)
    return ids.astype(np.float32, copy=False)


def compute_component_labels(coverage_neighbors):
    num_points = len(coverage_neighbors)
    parents = np.arange(num_points, dtype=np.int64)

    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parents[root_b] = root_a

    for idx, nbrs in enumerate(coverage_neighbors):
        for nbr in np.asarray(nbrs, dtype=np.int64):
            union(idx, int(nbr))

    labels = np.array([find(i) for i in range(num_points)], dtype=np.int64)
    _, compact = np.unique(labels, return_inverse=True)
    return compact.astype(np.int64, copy=False)


def compute_coverage_gain(candidate_indices, covered_mask, coverage_neighbors, component_labels=None, graph_mode='knn'):
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    covered_mask = np.asarray(covered_mask, dtype=bool)
    gains = np.zeros(len(candidate_indices), dtype=np.float32)
    selected_coverage = []

    if component_labels is not None:
        component_labels = np.asarray(component_labels, dtype=np.int64)

    for pos, idx in enumerate(candidate_indices):
        if graph_mode == 'component' and component_labels is not None:
            cover_idx = np.flatnonzero(component_labels == component_labels[idx]).astype(np.int64, copy=False)
        elif graph_mode == 'hybrid' and component_labels is not None:
            component_cover = np.flatnonzero(component_labels == component_labels[idx]).astype(np.int64, copy=False)
            cover_idx = np.unique(np.concatenate([coverage_neighbors[idx], component_cover])).astype(np.int64, copy=False)
        else:
            cover_idx = np.asarray(coverage_neighbors[idx], dtype=np.int64)
        selected_coverage.append(cover_idx)
        gains[pos] = float(np.count_nonzero(~covered_mask[cover_idx]))

    return gains, selected_coverage


def compute_uncertainty_scores(probabilities=None, logits=None, mode='entropy'):
    if probabilities is None and logits is None:
        raise ValueError('Expected probabilities or logits for uncertainty computation.')

    if probabilities is None:
        logits = np.asarray(logits, dtype=np.float32)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)

    probabilities = np.asarray(probabilities, dtype=np.float32)
    probabilities = probabilities / np.clip(probabilities.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)

    mode = str(mode).lower()
    if mode in ['least_confidence', 'least-confidence', 'lc']:
        scores = 1.0 - probabilities.max(axis=1)
    elif mode == 'margin':
        sorted_probs = np.sort(probabilities, axis=1)
        scores = 1.0 - (sorted_probs[:, -1] - sorted_probs[:, -2])
    elif mode == 'entropy':
        scores = -(probabilities * np.log(np.clip(probabilities, a_min=1e-12, a_max=None))).sum(axis=1)
    else:
        raise ValueError('Unsupported uncertainty mode: {}'.format(mode))
    return scores.astype(np.float32, copy=False)


def greedy_geometry_batch(
    uncertainty_scores,
    local_id_scores,
    coverage_neighbors,
    budget,
    alpha,
    beta,
    gamma,
    graph_mode='knn',
    component_labels=None,
    initial_covered_mask=None,
    recompute_greedy=True,
    candidate_indices=None,
):
    num_points = len(uncertainty_scores)
    if candidate_indices is None:
        candidate_indices = np.arange(num_points, dtype=np.int64)
    else:
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    selected = []
    selected_details = []
    covered_mask = np.zeros(num_points, dtype=bool) if initial_covered_mask is None else np.asarray(initial_covered_mask, dtype=bool).copy()

    normalized_uncertainty = _safe_normalize(uncertainty_scores)
    normalized_id = _safe_normalize(local_id_scores)

    cached_gains = None
    cached_cover_sets = None

    for step in range(min(int(budget), num_points)):
        remaining = np.setdiff1d(candidate_indices, np.asarray(selected, dtype=np.int64), assume_unique=False)
        if len(remaining) == 0:
            break

        if recompute_greedy or cached_gains is None:
            raw_coverage, cover_sets = compute_coverage_gain(
                remaining,
                covered_mask,
                coverage_neighbors,
                component_labels=component_labels,
                graph_mode=graph_mode,
            )
            cached_gains = raw_coverage
            cached_cover_sets = cover_sets
        else:
            raw_coverage = cached_gains
            cover_sets = cached_cover_sets

        normalized_coverage = _safe_normalize(raw_coverage)
        scores = (
            float(alpha) * normalized_uncertainty[remaining] +
            float(beta) * normalized_coverage +
            float(gamma) * normalized_id[remaining]
        )
        best_pos = int(np.argmax(scores))
        best_idx = int(remaining[best_pos])
        cover_idx = np.asarray(cover_sets[best_pos], dtype=np.int64)

        selected.append(best_idx)
        selected_details.append({
            'step': int(step),
            'candidate_index': best_idx,
            'score': float(scores[best_pos]),
            'uncertainty': float(uncertainty_scores[best_idx]),
            'uncertainty_norm': float(normalized_uncertainty[best_idx]),
            'coverage_gain': float(raw_coverage[best_pos]),
            'coverage_gain_norm': float(normalized_coverage[best_pos]),
            'local_id': float(local_id_scores[best_idx]),
            'local_id_norm': float(normalized_id[best_idx]),
            'covered_size_after': int(np.count_nonzero(np.logical_or(covered_mask, np.isin(np.arange(num_points), cover_idx)))),
        })
        covered_mask[cover_idx] = True

        if not recompute_greedy:
            cached_gains = None
            cached_cover_sets = None

    return (
        np.asarray(selected, dtype=np.int64),
        covered_mask.astype(bool, copy=False),
        selected_details,
    )


def save_round_diagnostics(csv_path, round_info):
    if not csv_path:
        return

    dir_name = os.path.dirname(csv_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    fieldnames = [
        'episode',
        'round',
        'selected_ids',
        'selected_id_mean',
        'selected_id_min',
        'selected_id_max',
        'selected_id_std',
        'selected_coverage_gain_mean',
        'selected_coverage_gain_min',
        'selected_coverage_gain_max',
        'selected_coverage_gain_std',
        'selected_uncertainty_mean',
        'selected_uncertainty_min',
        'selected_uncertainty_max',
        'selected_uncertainty_std',
        'selected_score_mean',
        'selected_score_min',
        'selected_score_max',
        'selected_score_std',
        'graph_mode',
        'uncertainty_mode',
        'id_mode',
        'alpha',
        'beta',
        'gamma',
        'k_id',
        'k_knn',
        'radius',
        'recompute_greedy',
    ]

    row = {key: round_info.get(key, '') for key in fieldnames}
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


class GeometryAutoResearch:
    def __init__(self, cfg, lSet, uSet, budgetSize, clf_model=None, trainDataset=None, dataObj=None):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.all_features = ds_utils.load_features(self.ds_name, self.seed)
        self.lSet = np.asarray(lSet, dtype=np.int64)
        self.uSet = np.asarray(uSet, dtype=np.int64)
        self.budgetSize = int(budgetSize)
        self.clf_model = clf_model
        self.trainDataset = trainDataset
        self.dataObj = dataObj
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(np.int64, copy=False)
        self.rel_features = self.all_features[self.relevant_indices].astype(np.float32, copy=False)
        self.selection_metadata = {}

    def _collect_probabilities(self):
        if self.clf_model is None or self.trainDataset is None or self.dataObj is None or len(self.uSet) == 0 or len(self.lSet) == 0:
            return None

        old_mode = self.clf_model.training
        self.clf_model.eval()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.clf_model = self.clf_model.to(device)
        loader = self.dataObj.getSequentialDataLoader(
            indexes=self.uSet,
            batch_size=max(1, int(self.cfg.TRAIN.BATCH_SIZE / max(1, self.cfg.NUM_GPUS))),
            data=self.trainDataset,
        )
        loader.dataset.no_aug = True

        preds = []
        try:
            for x, _ in loader:
                with torch.no_grad():
                    x = x.to(device).float()
                    preds.append(F.softmax(self.clf_model(x), dim=1).cpu().numpy())
        finally:
            loader.dataset.no_aug = False
            self.clf_model.train(old_mode)

        if not preds:
            return None
        return np.concatenate(preds, axis=0).astype(np.float32, copy=False)

    def _initial_covered_mask(self, coverage_neighbors, component_labels, graph_mode):
        covered_mask = np.zeros(len(self.relevant_indices), dtype=bool)
        if len(self.lSet) == 0:
            return covered_mask

        labeled_rel = np.arange(len(self.lSet), dtype=np.int64)
        if graph_mode == 'component' and component_labels is not None:
            labeled_components = np.unique(component_labels[labeled_rel])
            covered_mask[np.isin(component_labels, labeled_components)] = True
        elif graph_mode == 'hybrid' and component_labels is not None:
            labeled_components = np.unique(component_labels[labeled_rel])
            covered_mask[np.isin(component_labels, labeled_components)] = True
            for idx in labeled_rel:
                covered_mask[np.asarray(coverage_neighbors[idx], dtype=np.int64)] = True
        else:
            for idx in labeled_rel:
                covered_mask[np.asarray(coverage_neighbors[idx], dtype=np.int64)] = True
        return covered_mask

    def _id_preference_scores(self, local_ids):
        mode = str(self.cfg.ACTIVE_LEARNING.GEOAR_MODE).lower()
        normalized = _safe_normalize(local_ids)
        if mode == 'high_id_more_weight':
            return normalized
        if mode == 'low_id_more_weight':
            return 1.0 - normalized
        return np.zeros_like(normalized, dtype=np.float32)

    def select_samples(self):
        if len(self.uSet) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

        graph_mode = str(self.cfg.ACTIVE_LEARNING.GEOAR_GRAPH_MODE).lower()
        uncertainty_mode = str(getattr(self.cfg.ACTIVE_LEARNING, 'GEOAR_UNCERTAINTY', 'entropy')).lower()
        radius = float(self.cfg.ACTIVE_LEARNING.GEOAR_RADIUS)
        if radius <= 0.0:
            radius = None

        graph = build_knn_graph(
            self.rel_features,
            k=self.cfg.ACTIVE_LEARNING.GEOAR_K_KNN,
            graph_mode=graph_mode,
            radius=radius,
        )
        local_ids = compute_local_id(graph['distances'], self.cfg.ACTIVE_LEARNING.GEOAR_K_ID)
        component_labels = compute_component_labels(graph['coverage_neighbors'])
        id_scores = self._id_preference_scores(local_ids)

        probabilities = self._collect_probabilities()
        if probabilities is None:
            uncertainty_scores = np.zeros(len(self.uSet), dtype=np.float32)
            uncertainty_active = False
        else:
            uncertainty_scores = compute_uncertainty_scores(probabilities=probabilities, mode=uncertainty_mode)
            uncertainty_active = True

        unlabeled_rel = np.arange(len(self.lSet), len(self.relevant_indices), dtype=np.int64)
        full_uncertainty = np.zeros(len(self.relevant_indices), dtype=np.float32)
        full_uncertainty[unlabeled_rel] = uncertainty_scores
        initial_covered_mask = self._initial_covered_mask(graph['coverage_neighbors'], component_labels, graph_mode)
        coverage_fraction_before = float(initial_covered_mask.mean()) if len(initial_covered_mask) else 0.0

        selected_rel, _, selected_details = greedy_geometry_batch(
            uncertainty_scores=full_uncertainty,
            local_id_scores=id_scores,
            coverage_neighbors=graph['coverage_neighbors'],
            budget=self.budgetSize,
            alpha=self.cfg.ACTIVE_LEARNING.GEOAR_ALPHA,
            beta=self.cfg.ACTIVE_LEARNING.GEOAR_BETA,
            gamma=self.cfg.ACTIVE_LEARNING.GEOAR_GAMMA,
            graph_mode=graph_mode,
            component_labels=component_labels,
            initial_covered_mask=initial_covered_mask,
            recompute_greedy=bool(self.cfg.ACTIVE_LEARNING.GEOAR_RECOMPUTE_GREEDY),
            candidate_indices=unlabeled_rel,
        )

        activeSet = self.relevant_indices[selected_rel]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))), dtype=np.int64)

        selected_local_ids = local_ids[selected_rel]
        selected_cov = np.array([item['coverage_gain'] for item in selected_details], dtype=np.float32)
        selected_unc = np.array([item['uncertainty'] for item in selected_details], dtype=np.float32)
        selected_scores = np.array([item['score'] for item in selected_details], dtype=np.float32)
        component_counts = np.bincount(component_labels) if len(component_labels) else np.zeros(0, dtype=np.int64)
        selected_component_labels = component_labels[selected_rel] if len(selected_rel) else np.zeros(0, dtype=np.int64)
        selected_component_hist = {
            str(int(label)): int(np.sum(selected_component_labels == label))
            for label in np.unique(selected_component_labels)
        }

        round_idx = _parse_round_index(getattr(self.cfg, 'EPISODE_DIR', ''))
        csv_path = str(getattr(self.cfg.ACTIVE_LEARNING, 'GEOAR_LOG_CSV', '') or '')
        if csv_path:
            if not os.path.isabs(csv_path):
                csv_path = os.path.join(getattr(self.cfg, 'EXP_DIR', '.'), csv_path)
            save_round_diagnostics(csv_path, {
                'episode': round_idx,
                'round': round_idx,
                'selected_ids': json.dumps(activeSet.tolist()),
                'selected_id_mean': _summary_stats(selected_local_ids)['mean'],
                'selected_id_min': _summary_stats(selected_local_ids)['min'],
                'selected_id_max': _summary_stats(selected_local_ids)['max'],
                'selected_id_std': _summary_stats(selected_local_ids)['std'],
                'selected_coverage_gain_mean': _summary_stats(selected_cov)['mean'],
                'selected_coverage_gain_min': _summary_stats(selected_cov)['min'],
                'selected_coverage_gain_max': _summary_stats(selected_cov)['max'],
                'selected_coverage_gain_std': _summary_stats(selected_cov)['std'],
                'selected_uncertainty_mean': _summary_stats(selected_unc)['mean'],
                'selected_uncertainty_min': _summary_stats(selected_unc)['min'],
                'selected_uncertainty_max': _summary_stats(selected_unc)['max'],
                'selected_uncertainty_std': _summary_stats(selected_unc)['std'],
                'selected_score_mean': _summary_stats(selected_scores)['mean'],
                'selected_score_min': _summary_stats(selected_scores)['min'],
                'selected_score_max': _summary_stats(selected_scores)['max'],
                'selected_score_std': _summary_stats(selected_scores)['std'],
                'graph_mode': graph_mode,
                'uncertainty_mode': uncertainty_mode,
                'id_mode': self.cfg.ACTIVE_LEARNING.GEOAR_MODE,
                'alpha': float(self.cfg.ACTIVE_LEARNING.GEOAR_ALPHA),
                'beta': float(self.cfg.ACTIVE_LEARNING.GEOAR_BETA),
                'gamma': float(self.cfg.ACTIVE_LEARNING.GEOAR_GAMMA),
                'k_id': int(self.cfg.ACTIVE_LEARNING.GEOAR_K_ID),
                'k_knn': int(self.cfg.ACTIVE_LEARNING.GEOAR_K_KNN),
                'radius': '' if graph['radius'] is None else float(graph['radius']),
                'recompute_greedy': bool(self.cfg.ACTIVE_LEARNING.GEOAR_RECOMPUTE_GREEDY),
            })

        self.selection_metadata = {
            'strategy': 'geometry_policy',
            'round': int(round_idx),
            'selected_count': int(len(activeSet)),
            'graph_mode': graph_mode,
            'radius': None if graph['radius'] is None else float(graph['radius']),
            'uncertainty_mode': uncertainty_mode,
            'uncertainty_active': bool(uncertainty_active),
            'local_id_mean': float(local_ids[unlabeled_rel].mean()) if len(unlabeled_rel) else 0.0,
            'local_id_max': float(local_ids[unlabeled_rel].max()) if len(unlabeled_rel) else 0.0,
            'pool_local_id_stats': _summary_stats(local_ids[unlabeled_rel]) if len(unlabeled_rel) else _summary_stats(np.zeros(0, dtype=np.float32)),
            'selected_id_mean': float(selected_local_ids.mean()) if len(selected_local_ids) else 0.0,
            'selected_coverage_gain_mean': float(selected_cov.mean()) if len(selected_cov) else 0.0,
            'selected_uncertainty_mean': float(selected_unc.mean()) if len(selected_unc) else 0.0,
            'selected_score_mean': float(selected_scores.mean()) if len(selected_scores) else 0.0,
            'components': int(np.unique(component_labels).size) if len(component_labels) else 0,
            'largest_component_fraction': float(component_counts.max() / max(len(component_labels), 1)) if len(component_counts) else 0.0,
            'component_size_stats': _summary_stats(component_counts.astype(np.float32)) if len(component_counts) else _summary_stats(np.zeros(0, dtype=np.float32)),
            'selected_component_histogram': selected_component_hist,
            'coverage_fraction_before': coverage_fraction_before,
            'coverage_fraction_after': min(1.0, coverage_fraction_before + float(selected_cov.sum()) / max(len(self.relevant_indices), 1)),
            'initial_covered_fraction': coverage_fraction_before,
        }
        print('Finished GeometryAutoResearch selection of {} samples.'.format(len(activeSet)))
        print('Active set is {}'.format(activeSet))
        return activeSet, remainSet


def smoke_test_geometry_auto_research():
    rng = np.random.RandomState(0)
    embeddings = rng.randn(24, 8).astype(np.float32)
    probs = rng.rand(24, 4).astype(np.float32)
    probs = probs / probs.sum(axis=1, keepdims=True)
    graph = build_knn_graph(embeddings, k=5)
    local_ids = compute_local_id(graph['distances'], k=4)
    component_labels = compute_component_labels(graph['coverage_neighbors'])
    uncertainty = compute_uncertainty_scores(probabilities=probs, mode='entropy')
    selected, covered_mask, details = greedy_geometry_batch(
        uncertainty_scores=uncertainty,
        local_id_scores=local_ids,
        coverage_neighbors=graph['coverage_neighbors'],
        component_labels=component_labels,
        budget=4,
        alpha=1.0,
        beta=1.0,
        gamma=0.5,
        graph_mode='knn',
        recompute_greedy=True,
    )
    return {
        'selected': selected.tolist(),
        'covered_fraction': float(np.mean(covered_mask)),
        'steps': len(details),
    }


if __name__ == '__main__':
    print(smoke_test_geometry_auto_research())
