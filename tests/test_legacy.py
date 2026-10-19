import numpy as np
import pytest
from pathlib import Path

from recursive_group_selection import RecursiveGroupSelector


def test_read_version_one_native_selection(tmp_path):
    pytest.importorskip("catboost")
    # This fixture was generated with the real 0.1.0 library before uninstalling it.
    directory = Path(__file__).parent / 'fixtures/legacy_v1'
    with np.load(directory / 'expected_predictions.npz') as saved:
        X_test, expected, selected_ids = saved['X_test'], saved['predictions'], saved['selected_ids']
    restored = RecursiveGroupSelector.load(directory)
    np.testing.assert_array_equal(restored.predict(X_test), expected)
    assert restored.selected_group_ids_ == tuple(selected_ids)
    assert restored.stop_reason_ == "min_groups_reached"
    assert len(restored.history_) == 1
