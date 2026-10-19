"""Class-stratified paired recording/group-cluster bootstrap for fixed-model F1."""
from numbers import Integral

import numpy as np

from .results import BootstrapComparison


def _vector(value, name, n=None):
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or not len(array) or (n is not None and len(array) != n):
        raise ValueError(f"{name} must contain one value per row in a nonempty sample.")
    if array.dtype.kind in "biuf" and not np.isfinite(array).all():
        raise ValueError(f"{name} must not contain missing or infinite values.")
    if array.dtype.kind == "O" and any(
        value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value))
        for value in array
    ):
        raise ValueError(f"{name} must not contain missing values.")
    return array


def _unique(value, name, **kwargs):
    try:
        return np.unique(value, **kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must use mutually comparable scalar labels/IDs.") from exc


def _macro_f1(counts):
    scores = []
    for predicted_slot, tp_slot in ((1, 2), (3, 4)):
        denominator = counts[..., 0, :] + counts[..., predicted_slot, :]
        values = np.divide(2 * counts[..., tp_slot, :], denominator,
                           out=np.zeros_like(denominator), where=denominator != 0)
        scores.append(values.mean(axis=-1))
    return np.stack(scores, axis=-1)


class PairedClusterBootstrap:
    """Resample complete clusters with shared weights for two prediction vectors.

    Strata, if supplied, must be constant within each cluster. With strata=y_true,
    this is the class-stratified design used by the Z24 notebook. Cluster and
    stratum order are sorted; batch_size participates in seed reproducibility.
    The sole supported metric in this release is fixed-class F1 macro.
    """

    def __init__(self, n_resamples=50_000, confidence_level=0.90,
                 random_state=42, batch_size=1_000):
        for name, value, minimum in (("n_resamples", n_resamples, 2), ("batch_size", batch_size, 1)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if isinstance(confidence_level, bool) or not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1.")
        if random_state is not None and (isinstance(random_state, bool) or not isinstance(random_state, Integral) or random_state < 0):
            raise ValueError("random_state must be a nonnegative integer or None.")
        self.n_resamples = int(n_resamples)
        self.confidence_level = float(confidence_level)
        self.random_state = int(random_state) if random_state is not None else None
        self.batch_size = int(batch_size)

    def compare(self, y_true, reference_predictions, candidate_predictions,
                cluster_ids, strata=None, classes=None):
        y = _vector(y_true, "y_true")
        predictions = [_vector(reference_predictions, "reference_predictions", len(y)),
                       _vector(candidate_predictions, "candidate_predictions", len(y))]
        cluster_ids = _vector(cluster_ids, "cluster_ids", len(y))
        if classes is None:
            class_values = _unique(np.concatenate([y, *predictions]), "classes")
        else:
            class_values = _vector(classes, "classes")
            if len(_unique(class_values, "classes")) != len(class_values):
                raise ValueError("classes must not contain duplicates.")
        if len(class_values) < 2 or any(not np.isin(value, class_values).all() for value in [y, *predictions]):
            raise ValueError("At least two fixed classes are required; labels and predictions must belong to classes.")
        clusters, codes = _unique(cluster_ids, "cluster_ids", return_inverse=True)
        if len(clusters) < 2:
            raise ValueError("At least two recording/group clusters are required.")
        if strata is None:
            stratum_codes = np.zeros(len(y), dtype=int)
        else:
            _, stratum_codes = _unique(_vector(strata, "strata", len(y)), "strata", return_inverse=True)
        cluster_strata = np.empty(len(clusters), dtype=int)
        for index in range(len(clusters)):
            values = np.unique(stratum_codes[codes == index])
            if len(values) != 1:
                raise ValueError("Each cluster must belong to exactly one stratum; split labels are not valid cluster strata.")
            cluster_strata[index] = values[0]
        strata_indices = [np.flatnonzero(cluster_strata == value) for value in np.unique(cluster_strata)]
        if any(len(indices) < 2 for indices in strata_indices):
            raise ValueError("Each stratum needs at least two independent clusters.")

        lookup = {value: index for index, value in enumerate(class_values)}
        true_codes = np.array([lookup[value] for value in y])
        # Per cluster: true support, reference support/TP, candidate support/TP.
        counts = np.zeros((len(clusters), 5, len(class_values)), dtype=float)
        np.add.at(counts[:, 0, :], (codes, true_codes), 1)
        for predicted, support_slot, tp_slot in zip(predictions, (1, 3), (2, 4)):
            pred_codes = np.array([lookup[value] for value in predicted])
            np.add.at(counts[:, support_slot, :], (codes, pred_codes), 1)
            correct = pred_codes == true_codes
            np.add.at(counts[:, tp_slot, :], (codes[correct], true_codes[correct]), 1)
        observed = _macro_f1(counts.sum(axis=0))
        rng = np.random.default_rng(self.random_state)
        differences = np.empty(self.n_resamples)
        flat_counts = counts.reshape(len(clusters), -1)
        for start in range(0, self.n_resamples, self.batch_size):
            stop = min(start + self.batch_size, self.n_resamples)
            weights = np.zeros((stop - start, len(clusters)), dtype=int)
            for indices in strata_indices:
                weights[:, indices] = rng.multinomial(
                    len(indices), np.full(len(indices), 1 / len(indices)), size=len(weights),
                )
            batch_counts = (weights @ flat_counts).reshape(-1, 5, len(class_values))
            scores = _macro_f1(batch_counts)
            differences[start:stop] = scores[:, 1] - scores[:, 0]
        lower = float(np.quantile(differences, 1 - self.confidence_level, method="linear"))
        return BootstrapComparison(
            float(observed[0]), float(observed[1]), differences, lower,
            self.confidence_level, self.n_resamples, self.random_state, self.batch_size,
            len(y), len(clusters), tuple(len(indices) for indices in strata_indices),
            tuple(class_values.tolist()), strata is not None,
        )
