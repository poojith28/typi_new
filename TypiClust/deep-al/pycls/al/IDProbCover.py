#!/usr/bin/env python3

import csv
import os

import numpy as np
import torch

import pycls.datasets.utils as ds_utils


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


def _write_idpc_diagnostics(csv_path, round_info):
    if not csv_path:
        return

    dir_name = os.path.dirname(csv_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    fieldnames = [
        'selected_size',
        'selected_id_mean',
        'selected_id_min',
        'selected_id_max',
        'selected_id_std',
        'selected_radius_mean',
        'selected_radius_min',
        'selected_radius_max',
        'selected_radius_std',
        'selected_gain_mean',
        'selected_gain_min',
        'selected_gain_max',
        'selected_gain_std',
        'positive_gain_count',
        'zero_gain_fallback_count',
        'zero_gain_fallback_fraction',
        'first_zero_gain_selection_index',
        'first_zero_gain_selection_number',
        'coverage_fraction_before',
        'coverage_fraction_after',
        'k_id',
        'k_knn',
        'base_delta',
        'alpha',
        'median_id',
    ]

    row = {key: round_info.get(key, '') for key in fieldnames}
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def _default_cache_paths(cache_root, dataset, seed, k_id, k_knn):
    base = os.path.join(cache_root, dataset, f"seed{seed}")
    ids_path = os.path.join(base, f"mle_local_k{k_id}.npy")
    knn_path = os.path.join(base, f"knn_k{k_knn}.npz")
    return ids_path, knn_path


def compute_or_load_ids_mle(x, ids_cache_path, k_id=50):
    _safe_mkdir(os.path.dirname(ids_cache_path))
    if os.path.exists(ids_cache_path):
        ids = np.load(ids_cache_path).astype(np.float32)
        if ids.shape[0] != x.shape[0]:
            raise RuntimeError(
                f"[ID cache] shape mismatch: cached {ids.shape} vs X {x.shape} at {ids_cache_path}"
            )
        print(f"[ID cache] loaded: {ids_cache_path}")
        return ids, {
            'cache_hit': True,
            'cache_path': ids_cache_path,
            'estimator': 'mle',
        }

    try:
        from skdim.id import MLE
    except Exception as exc:
        raise RuntimeError(
            "scikit-dimension (skdim) is required to compute IDs inside IDProbCover.\n"
            "Install: pip install scikit-dimension\n"
            f"Original error: {exc}"
        ) from exc

    print(f"[ID cache] computing MLE local IDs: X={x.shape}, k_id={k_id}")
    mle = MLE()
    ids = mle.fit_transform_pw(x.astype(np.float32), n_neighbors=int(k_id))
    ids = np.asarray(ids, dtype=np.float32)
    median_id = np.nanmedian(ids)
    ids = np.nan_to_num(ids, nan=median_id, posinf=median_id, neginf=median_id).astype(np.float32)
    np.save(ids_cache_path, ids)
    print(f"[ID cache] saved: {ids_cache_path}")
    return ids, {
        'cache_hit': False,
        'cache_path': ids_cache_path,
        'estimator': 'mle',
    }


def _knn_faiss(x, k, use_gpu=True):
    import faiss  # type: ignore

    x = np.asarray(x, dtype=np.float32)
    _, dim = x.shape
    index = faiss.IndexFlatL2(dim)
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception:
            pass
    index.add(x)
    d2, idx = index.search(x, k + 1)
    idx = idx[:, 1:].astype(np.int32)
    dist = np.sqrt(np.maximum(d2[:, 1:], 0.0)).astype(np.float32)
    return idx, dist


def _knn_sklearn(x, k):
    from sklearn.neighbors import NearestNeighbors

    x = np.asarray(x, dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="auto")
    nn.fit(x)
    dist, idx = nn.kneighbors(x, return_distance=True)
    idx = idx[:, 1:].astype(np.int32)
    dist = dist[:, 1:].astype(np.float32)
    return idx, dist


def compute_or_load_knn(x, knn_cache_path, k_knn=50, prefer_faiss=True, faiss_gpu=True):
    _safe_mkdir(os.path.dirname(knn_cache_path))
    if os.path.exists(knn_cache_path):
        cached = np.load(knn_cache_path)
        idx = cached["idx"].astype(np.int32)
        dist = cached["dist"].astype(np.float32)
        if idx.shape[0] != x.shape[0]:
            raise RuntimeError(
                f"[kNN cache] N mismatch: cached idx {idx.shape} vs X {x.shape} at {knn_cache_path}"
            )
        if idx.shape[1] != k_knn:
            raise RuntimeError(
                f"[kNN cache] k mismatch: cached k={idx.shape[1]} but requested k_knn={k_knn}. "
                "Delete cache or request the cached k."
            )
        print(f"[kNN cache] loaded: {knn_cache_path}")
        return idx, dist, {
            'cache_hit': True,
            'cache_path': knn_cache_path,
            'backend': 'cache',
        }

    print(
        f"[kNN cache] computing kNN: X={x.shape}, k_knn={k_knn} "
        f"(prefer_faiss={prefer_faiss}, faiss_gpu={faiss_gpu})"
    )
    idx = dist = None
    backend = None
    if prefer_faiss:
        try:
            idx, dist = _knn_faiss(x, k=int(k_knn), use_gpu=faiss_gpu)
            backend = 'faiss_gpu' if faiss_gpu else 'faiss_cpu'
        except Exception as exc:
            print(f"[kNN cache] FAISS failed, falling back to sklearn. Reason: {exc}")
    if idx is None or dist is None:
        idx, dist = _knn_sklearn(x, k=int(k_knn))
        backend = 'sklearn'

    np.savez_compressed(knn_cache_path, idx=idx, dist=dist)
    print(f"[kNN cache] saved: {knn_cache_path}")
    return idx, dist, {
        'cache_hit': False,
        'cache_path': knn_cache_path,
        'backend': backend,
    }


class IDProbCover:
    def __init__(
        self,
        cfg,
        lSet,
        uSet,
        budgetSize,
        delta0,
        alpha=1.0,
        mode="high_id_more_centers",
        cache_root="./idprobcover_cache",
        k_id=50,
        k_knn=50,
        l2_normalize_features=True,
        prefer_faiss=True,
        faiss_gpu=True,
        add_self_cover=True,
    ):
        self.cfg = cfg
        self.ds_name = self.cfg["DATASET"]["NAME"]
        self.seed = self.cfg["RNG_SEED"]

        self.lSet = np.asarray(lSet).astype(np.int64)
        self.uSet = np.asarray(uSet).astype(np.int64)
        self.budgetSize = int(budgetSize)
        self.delta0 = float(delta0)
        self.alpha = float(alpha)
        self.mode = mode
        self.cache_root = cache_root
        self.k_id = int(k_id)
        self.k_knn = int(k_knn)
        self.l2_normalize_features = bool(l2_normalize_features)
        self.prefer_faiss = bool(prefer_faiss)
        self.faiss_gpu = bool(faiss_gpu)
        self.add_self_cover = bool(add_self_cover)

        self.selection_metadata = {
            'strategy': 'id_probcover_policy',
            'selection_mode': 'id_prob_cover',
            'effective_delta': float(self.delta0),
            'delta_phase': 'id_adaptive',
            'k_id': int(self.k_id),
            'k_knn': int(self.k_knn),
            'alpha': float(self.alpha),
            'mode': self.mode,
            'cache_root': self.cache_root,
        }

        self.all_features = ds_utils.load_features(self.ds_name, self.seed).astype(np.float32)
        if self.l2_normalize_features:
            self.all_features /= (np.linalg.norm(self.all_features, axis=1, keepdims=True) + 1e-12)

        ids_path, knn_path = _default_cache_paths(
            self.cache_root,
            self.ds_name,
            self.seed,
            self.k_id,
            self.k_knn,
        )
        self.ids_all, id_cache_meta = compute_or_load_ids_mle(self.all_features, ids_path, k_id=self.k_id)
        self.knn_idx_all, self.knn_dist_all, knn_cache_meta = compute_or_load_knn(
            self.all_features,
            knn_path,
            k_knn=self.k_knn,
            prefer_faiss=self.prefer_faiss,
            faiss_gpu=self.faiss_gpu,
        )
        self.selection_metadata.update({
            'id_cache_hit': bool(id_cache_meta['cache_hit']),
            'id_cache_path': id_cache_meta['cache_path'],
            'id_estimator': id_cache_meta['estimator'],
            'knn_cache_hit': bool(knn_cache_meta['cache_hit']),
            'knn_cache_path': knn_cache_meta['cache_path'],
            'knn_backend': knn_cache_meta['backend'],
        })

        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(np.int64)
        self.Nr = len(self.relevant_indices)
        self.L = len(self.lSet)
        self.g2r = -np.ones(self.all_features.shape[0], dtype=np.int32)
        self.g2r[self.relevant_indices] = np.arange(self.Nr, dtype=np.int32)
        self.rel_ids = self.ids_all[self.relevant_indices].astype(np.float32)
        self.delta_per_x = self._compute_delta_per_x(self.rel_ids)
        self.out_neighbors, self.in_sources = self._build_rel_adjacency()

    def _compute_delta_per_x(self, ids):
        median_id = np.median(ids) + 1e-12
        ratio = (ids + 1e-12) / median_id
        if self.mode == "high_id_more_centers":
            scale = ratio ** (-self.alpha)
        elif self.mode == "low_id_more_centers":
            scale = ratio ** self.alpha
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return (self.delta0 * scale).astype(np.float32)

    def _build_rel_adjacency(self):
        out_neighbors = [None] * self.Nr
        in_sources = [[] for _ in range(self.Nr)]

        for rel_x, global_x in enumerate(self.relevant_indices):
            neigh_g = self.knn_idx_all[global_x]
            neigh_d = self.knn_dist_all[global_x]

            neigh_r = self.g2r[neigh_g]
            mask_rel = neigh_r >= 0
            neigh_r = neigh_r[mask_rel]
            neigh_d = neigh_d[mask_rel]

            thr = self.delta_per_x[rel_x]
            mask_thr = neigh_d < thr
            neigh_r = neigh_r[mask_thr].astype(np.int32)

            if self.add_self_cover and (neigh_r.size == 0 or not np.any(neigh_r == rel_x)):
                neigh_r = np.concatenate([np.array([rel_x], dtype=np.int32), neigh_r], axis=0)

            out_neighbors[rel_x] = neigh_r
            for rel_y in neigh_r:
                in_sources[int(rel_y)].append(rel_x)

        in_sources = [np.asarray(srcs, dtype=np.int32) for srcs in in_sources]
        return out_neighbors, in_sources

    def select_samples(self):
        covered = np.zeros(self.Nr, dtype=bool)
        for rel_x in range(self.L):
            covered[self.out_neighbors[rel_x]] = True

        current_degree = np.zeros(self.Nr, dtype=np.int32)
        for rel_x in range(self.Nr):
            current_degree[rel_x] = int(np.sum(~covered[self.out_neighbors[rel_x]]))

        selected_rel = []
        selected_mask = np.zeros(self.Nr, dtype=bool)
        selected_gains = []
        coverage_before = covered.copy()

        for it in range(self.budgetSize):
            cand_deg = current_degree.copy()
            cand_deg[:self.L] = -1
            cand_deg[selected_mask] = -1
            best = int(cand_deg.max())
            if best < 0:
                break

            cands = np.where(cand_deg == best)[0]
            rel_x = int(cands[np.argmin(self.rel_ids[cands])])

            if best <= 0:
                pool = np.where((~selected_mask) & (np.arange(self.Nr) >= self.L))[0]
                if pool.size == 0:
                    break
                rel_x = int(pool[np.argmin(self.rel_ids[pool])])

            selected_rel.append(rel_x)
            selected_mask[rel_x] = True
            selected_gains.append(best)

            newly = self.out_neighbors[rel_x]
            newly = newly[~covered[newly]]

            pick_id = float(self.rel_ids[rel_x])
            pick_global = int(self.relevant_indices[rel_x])
            if newly.size == 0:
                current_degree[rel_x] = 0
                cov_ratio = float(np.mean(covered))
                print(
                    f"[IDProbCover] it={it:03d} pick_rel={rel_x} ID={pick_id:.3f} "
                    f"pick_global={pick_global} best_gain={best} covered={cov_ratio:.3f} "
                    f"maxdeg={int(current_degree.max())}"
                )
                continue

            covered[newly] = True
            for rel_y in newly:
                srcs = self.in_sources[int(rel_y)]
                if srcs.size:
                    current_degree[srcs] -= 1
            current_degree[rel_x] = 0

            cov_ratio = float(np.mean(covered))
            print(
                f"[IDProbCover] it={it:03d} pick_rel={rel_x} ID={pick_id:.3f} "
                f"pick_global={pick_global} best_gain={best} covered={cov_ratio:.3f} "
                f"maxdeg={int(current_degree.max())}"
            )

        selected_rel = np.asarray(selected_rel, dtype=np.int32)
        activeSet = self.relevant_indices[selected_rel]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))), dtype=np.int64)

        selected_ids = self.rel_ids[selected_rel] if len(selected_rel) else np.zeros(0, dtype=np.float32)
        selected_radii = self.delta_per_x[selected_rel] if len(selected_rel) else np.zeros(0, dtype=np.float32)
        id_stats = _summary_stats(selected_ids)
        radius_stats = _summary_stats(selected_radii)
        gain_stats = _summary_stats(selected_gains)
        median_id = float(np.median(self.rel_ids)) if len(self.rel_ids) else 0.0

        self.selection_metadata.update({
            'coverage_fraction_before': float(coverage_before.mean()) if len(coverage_before) else 0.0,
            'coverage_fraction_after': float(covered.mean()) if len(covered) else 0.0,
            'selected_count': int(len(activeSet)),
            'selected_id_mean': id_stats['mean'],
            'selected_id_min': id_stats['min'],
            'selected_id_max': id_stats['max'],
            'selected_id_std': id_stats['std'],
            'selected_radius_mean': radius_stats['mean'],
            'selected_radius_min': radius_stats['min'],
            'selected_radius_max': radius_stats['max'],
            'selected_radius_std': radius_stats['std'],
            'selected_coverage_gain_mean': gain_stats['mean'],
            'selected_coverage_gain_min': gain_stats['min'],
            'selected_coverage_gain_max': gain_stats['max'],
            'selected_coverage_gain_std': gain_stats['std'],
            'median_local_id': median_id,
        })

        csv_path = str(getattr(self.cfg.ACTIVE_LEARNING, 'IDPC_LOG_CSV', '') or '')
        if csv_path:
            if not os.path.isabs(csv_path):
                csv_path = os.path.join(getattr(self.cfg, 'EXP_DIR', '.'), csv_path)
            _write_idpc_diagnostics(csv_path, {
                'selected_size': int(len(activeSet)),
                'selected_id_mean': id_stats['mean'],
                'selected_id_min': id_stats['min'],
                'selected_id_max': id_stats['max'],
                'selected_id_std': id_stats['std'],
                'selected_radius_mean': radius_stats['mean'],
                'selected_radius_min': radius_stats['min'],
                'selected_radius_max': radius_stats['max'],
                'selected_radius_std': radius_stats['std'],
                'selected_gain_mean': gain_stats['mean'],
                'selected_gain_min': gain_stats['min'],
                'selected_gain_max': gain_stats['max'],
                'selected_gain_std': gain_stats['std'],
                'coverage_fraction_before': self.selection_metadata['coverage_fraction_before'],
                'coverage_fraction_after': self.selection_metadata['coverage_fraction_after'],
                'k_id': int(self.k_id),
                'k_knn': int(self.k_knn),
                'base_delta': float(self.delta0),
                'alpha': float(self.alpha),
                'median_id': median_id,
            })

        return activeSet, remainSet
