import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from pycls.al.auto_delta import (
    AUTO_DELTA_RULE,
    resolve_auto_delta,
    select_auto_delta_from_knn_distances,
    write_run_provenance,
)


class AutoDeltaTests(unittest.TestCase):
    def test_selects_median_kth_distance(self):
        distances = np.array(
            [[0.1, 0.2], [0.2, 0.4], [0.3, 0.6], [0.4, 0.8]],
            dtype=np.float32,
        )
        self.assertAlmostEqual(select_auto_delta_from_knn_distances(distances), 0.5)

    def test_rejects_nonfinite_distances(self):
        with self.assertRaises(ValueError):
            select_auto_delta_from_knn_distances(np.array([[0.1, np.nan]], dtype=np.float32))

    def test_resolver_is_label_free_and_cacheable(self):
        features = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            cfg = SimpleNamespace(
                DATASET=SimpleNamespace(NAME="CIFAR10"),
                MODEL=SimpleNamespace(TYPE="resnet18"),
                RNG_SEED=1,
                ACTIVE_LEARNING=SimpleNamespace(
                    AUTO_DELTA_K=1,
                    AUTO_DELTA_QUANTILE=0.5,
                    AUTO_DELTA_CACHE_ROOT=directory,
                    AUTO_DELTA_PREFER_FAISS=True,
                    AUTO_DELTA_FAISS_GPU=False,
                    IDPC_CACHE_ROOT="",
                ),
            )
            with mock.patch(
                "pycls.al.auto_delta.ds_utils.load_features",
                return_value=features,
            ):
                delta, metadata = resolve_auto_delta(cfg, np.arange(len(features)))
                delta_cached, metadata_cached = resolve_auto_delta(cfg, np.arange(len(features)))

            self.assertGreater(delta, 0.0)
            self.assertEqual(delta, delta_cached)
            self.assertEqual(metadata["rule"], AUTO_DELTA_RULE)
            self.assertFalse(metadata["labels_used"])
            self.assertFalse(metadata["validation_or_test_accuracy_used"])
            self.assertTrue(metadata_cached["auto_delta_cache_hit"])

            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            provenance = write_run_provenance(str(run_dir), metadata_cached)
            saved = json.loads(Path(provenance).read_text())
            self.assertEqual(saved["delta_auto"], delta)


if __name__ == "__main__":
    unittest.main()
