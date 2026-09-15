import hashlib

import numpy as np
import pytest

from pycls.datasets.initial_lset import load_provided_initial_lset


def test_provided_initial_lset_preserves_order_and_removes_only_selected(tmp_path):
    source = tmp_path / "initial.npy"
    initial = np.asarray([7, 2, 5], dtype=np.int64)
    np.save(str(source), initial)
    u_set = np.asarray([0, 2, 3, 5, 7, 8], dtype=np.int64)
    val_set = np.asarray([1, 4, 6, 9], dtype=np.int64)

    loaded, remaining, metadata = load_provided_initial_lset(
        source, u_set, val_set, train_size=10, expected_count=3
    )

    assert loaded.tolist() == [7, 2, 5]
    assert remaining.tolist() == [0, 3, 8]
    assert metadata["source_file_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert metadata["count"] == 3


@pytest.mark.parametrize("initial", ([2, 2, 3], [2, 4, 5], [2, 5, 12]))
def test_provided_initial_lset_rejects_invalid_partition(tmp_path, initial):
    source = tmp_path / "initial.npy"
    np.save(str(source), np.asarray(initial))
    with pytest.raises(ValueError):
        load_provided_initial_lset(
            source,
            np.asarray([0, 2, 3, 5, 7, 8]),
            np.asarray([1, 4, 6, 9]),
            train_size=10,
            expected_count=3,
        )
