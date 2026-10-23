"""Evaluate fixed classifiers once and retain predictions for downstream work."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, log_loss,
                             precision_recall_fscore_support)

from ..groups import labels
from ..reporting.export import save_reports


def _frozen_array(value):
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ClassificationSplit:
    name: str
    y_true: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray | None
    classes: tuple
    n_features: int
    inference_seconds: float
    metrics: dict
    per_class: object
    text_report: str


class ClassificationEvaluation:
    def __init__(self, splits):
        import pandas as pd
        self.splits = dict(splits)
        self.metrics = pd.DataFrame([split.metrics for split in self.splits.values()])

    def __getitem__(self, split):
        return self.splits[split]

    def style(self):
        return self.metrics.style.format({"Samples": "{:.0f}", "Accuracy": "{:.2%}",
            "Precision macro": "{:.2%}", "Recall macro": "{:.2%}", "F1 macro": "{:.2%}",
            "Log loss": "{:.4f}"}, na_rep="Unavailable")

    def print_report(self):
        for name, split in self.splits.items():
            print(f"\n{'=' * 30}\n{name.upper()}\n{'=' * 30}")
            print(f"Accuracy: {split.metrics['Accuracy']:.4f}\n\nClassification Report:\n")
            print(split.text_report)

    def save(self, directory, *, prefix="evaluation", overwrite=False):
        from ..reporting.export import safe_name
        safe_name(prefix)
        tables = {f"{prefix}_metrics.csv": self.metrics}
        for index, split in enumerate(self.splits.values()):
            # Split names are user display text, never interpolated into file paths.
            tables[f"{prefix}_split_{index}_class_metrics.csv"] = split.per_class
        return save_reports(directory, tables=tables, prefix=prefix, overwrite=overwrite)


def evaluate_classifier(*, model, splits, adapter=None, include_probabilities=True):
    """Evaluate supplied rows in their existing order; never fit a model.

    Probability columns follow adapter.classes(model), including unsorted classes.
    Macro metrics include all model classes with zero_division=0. Timing includes
    only label prediction/normalization, not probabilities, metrics or reporting.
    An adapter without predict_proba works with include_probabilities=False.
    """
    import pandas as pd
    if not splits:
        raise ValueError("At least one named evaluation split is required.")
    classes = np.asarray(adapter.classes(model) if adapter is not None else model.classes_).reshape(-1)
    if len(classes) < 2 or len(np.unique(classes)) != len(classes):
        raise ValueError("A classifier with at least two unique classes is required.")
    prepared = {}
    for name, pair in splits.items():
        if not isinstance(name, str) or not name or not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("splits must map nonempty names to (X, y) pairs.")
        X, y = pair
        if not hasattr(X, "shape") or len(X.shape) != 2 or not X.shape[0] or not X.shape[1]:
            raise ValueError("Evaluation inputs must be nonempty feature matrices.")
        y = labels(y, name=name, n_samples=len(X))
        if not np.isin(y, classes).all():
            raise ValueError("Evaluation contains labels unknown to the model.")
        prepared[name] = (X, y)
    probability_predictor = getattr(adapter if adapter is not None else model, "predict_proba", None)
    if include_probabilities and not callable(probability_predictor):
        raise ValueError("Probability evaluation needs predict_proba; disable include_probabilities otherwise.")
    evaluated = {}
    for name, (X, y) in prepared.items():
        tick = perf_counter()
        predicted = np.asarray(adapter.predict(model, X) if adapter is not None else model.predict(X)).reshape(-1)
        elapsed = perf_counter() - tick
        if len(predicted) != len(y) or not np.isin(predicted, classes).all():
            raise ValueError("Predictions must contain one known class per evaluation row.")
        precision, recall, f1, support = precision_recall_fscore_support(
            y, predicted, labels=classes, average=None, zero_division=0)
        probabilities, loss = None, np.nan
        if include_probabilities:
            probabilities = np.asarray(probability_predictor(model, X) if adapter is not None
                                       else probability_predictor(X), dtype=float)
            if (probabilities.shape != (len(y), len(classes)) or not np.isfinite(probabilities).all()
                    or np.any(probabilities < 0) or np.any(probabilities > 1)
                    or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-7, rtol=0)):
                raise ValueError("Probabilities must be finite normalized rows in model class order.")
            order = np.argsort(classes)
            loss = float(log_loss(y, probabilities[:, order], labels=classes[order]))
        metrics = {"Split": name, "Samples": len(y), "Accuracy": float(accuracy_score(y, predicted)),
                   "Precision macro": float(precision.mean()), "Recall macro": float(recall.mean()),
                   "F1 macro": float(f1.mean()), "Log loss": loss}
        per_class = pd.DataFrame({"Class": classes, "Samples": support,
                                  "Precision": precision, "Recall": recall, "F1": f1})
        evaluated[name] = ClassificationSplit(name, _frozen_array(y), _frozen_array(predicted),
            _frozen_array(probabilities) if probabilities is not None else None, tuple(classes.tolist()),
            X.shape[1], elapsed, metrics, per_class,
            classification_report(y, predicted, labels=classes, zero_division=0))
    return ClassificationEvaluation(evaluated)
