"""Explicit group grouping and input-column validation."""
from numbers import Integral

import numpy as np


class FeatureGroups:
    def __init__(self, feature_groups, n_features):
        if not feature_groups:
            raise ValueError("feature_groups must contain at least one group.")
        groups = {}
        for group, columns in feature_groups.items():
            if isinstance(group, bool) or not isinstance(group, Integral) or group < 1:
                raise ValueError("Group IDs must be positive integers.")
            values = tuple(columns)
            if not values or any(isinstance(c, bool) or not isinstance(c, Integral) for c in values):
                raise ValueError("Every group needs nonempty integer column indices.")
            groups[int(group)] = tuple(sorted(int(c) for c in values))
        flattened = [c for cols in groups.values() for c in cols]
        if len(flattened) != n_features or sorted(flattened) != list(range(n_features)):
            raise ValueError("feature_groups must partition every input column exactly once.")
        self._groups = dict(sorted(groups.items(), key=lambda item: item[1][0]))
        self.group_ids = tuple(self._groups)
        self.n_features = n_features

    def columns(self, group_ids):
        return tuple(sorted(c for group in group_ids for c in self._groups[group]))

    def relative(self, group_ids):
        columns = self.columns(group_ids)
        positions = {column: index for index, column in enumerate(columns)}
        return {group: tuple(positions[c] for c in self._groups[group]) for group in group_ids}

    def as_dict(self):
        return {group: list(columns) for group, columns in self._groups.items()}


def numeric_matrix(X, *, name, n_features=None, feature_names=None):
    if feature_names is not None and hasattr(X, "columns"):
        if list(X.columns) != list(feature_names):
            raise ValueError(f"{name} columns do not match the fitted feature names/order.")
    array = np.asarray(X)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{name} must be a nonempty 2D feature matrix.")
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values; preprocess before selection.")
    if n_features is not None and array.shape[1] != n_features:
        raise ValueError(f"{name} must have {n_features} original input columns.")
    return array


def labels(y, *, name, n_samples):
    array = np.asarray(y)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or len(array) != n_samples:
        raise ValueError(f"{name} must contain one label per input row.")
    if array.dtype.kind in "iuf" and not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite labels.")
    return array
