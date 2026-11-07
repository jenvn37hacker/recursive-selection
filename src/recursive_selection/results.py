"""Serializable records, independent of any boosting library."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from numbers import Integral


def validate_search_options(retry, shap, gate, random_state):
    for name, value in [('retry', retry), ('shap', shap), ('gate', gate)]:
        if not isinstance(value, bool):
            raise TypeError(f'{name} must be a bool.')
    if retry and not gate:
        raise ValueError('retry=True requires gate=True.')
    if isinstance(random_state, bool) or not isinstance(random_state, Integral) or random_state < 0:
        raise ValueError('random_state must be a nonnegative integer.')


@dataclass(frozen=True)
class MacroScores:
    precision_macro: float
    recall_macro: float
    f1_macro: float

    @classmethod
    def from_predictions(cls, y, predictions, classes):
        predicted = np.asarray(predictions).reshape(-1)
        if len(predicted) != len(y) or not np.isin(predicted, classes).all():
            raise ValueError(
                "Predictions must match validation rows and known classes."
            )
        precision, recall, f1, _ = precision_recall_fscore_support(
            y,
            predicted,
            labels=classes,
            average=None,
            zero_division=0,
        )
        return cls(float(precision.mean()), float(recall.mean()), float(f1.mean()))

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class QualityDecision:
    passed: bool
    worst_margin: float
    failed_metrics: tuple[str, ...]


@dataclass(frozen=True)
class TrialRecord:
    """An evaluated removal. With gate=False, accepted means visited, not quality-pass."""
    trial_id: int
    configuration_id: int
    group_id: int
    groups_before: tuple[int, ...]
    proposed_groups: tuple[int, ...]
    importance: float
    metrics: MacroScores
    change_from_full: MacroScores
    accepted: bool
    worst_margin: float
    failed_metrics: tuple[str, ...]
    train_seconds: float
    validation_predict_seconds: float
    removed_group_ids: tuple[int, ...] = ()
    metric_kind: str = "mean"
    f1_reference_before: float | None = None
    f1_reference_after: float | None = None

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class AcceptedState:
    configuration_id: int
    trial_id: int
    group_ids: tuple[int, ...]
    metrics: MacroScores


@dataclass(frozen=True)
class ShapSnapshot:
    configuration_id: int
    trial_id: int
    group_ids: tuple[int, ...]
    # Immutable pairs preserve original input-group order.
    importance: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class SelectionResult:
    estimator: Any
    selected_group_ids: tuple[int, ...]
    selected_columns: tuple[int, ...]
    selected_feature_names: tuple[str, ...] | None
    baseline_metrics: MacroScores
    effective_floors: MacroScores
    final_metrics: MacroScores
    history: tuple[TrialRecord, ...]
    accepted_history: tuple[AcceptedState, ...]
    shap_history: tuple[ShapSnapshot, ...]
    timings: dict[str, float]
    stop_reason: str
    cv_evaluation: "CVEvaluationTrace | FixedCVEvaluationTrace | None" = None
    speed_options: Any = None
    gate_metrics: tuple[str, ...] = ("precision_macro", "recall_macro", "f1_macro")
    recording_split: Any = None
    retry: bool = True
    shap: bool = True
    gate: bool = True
    random_state: int = 42
    selected_trial_id: int | None = None
    validation_f1_reference: str | None = None
    validation_truncate_decimals: int | None = None

    @property
    def selected_state(self):
        if self.selected_trial_id is not None and (
            isinstance(self.selected_trial_id, bool) or not isinstance(self.selected_trial_id, Integral)
            or self.selected_trial_id < 0
        ):
            raise ValueError('Selected trial ID must be a nonnegative integer.')
        if not self.gate and self.selected_trial_id is None:
            raise ValueError('No-gate selection requires an explicit selected trial ID.')
        trial = self.accepted_history[-1].trial_id if self.selected_trial_id is None else self.selected_trial_id
        for state in self.accepted_history:
            if state.trial_id == trial:
                return state
        raise ValueError('Selected trial does not identify a visited/accepted state.')

    @property
    def ranking_source(self):
        return 'SHAP' if self.shap else 'Random ranking'

    @property
    def search_options(self):
        return dict(retry=self.retry, shap=self.shap, gate=self.gate, random_state=self.random_state)


@dataclass(frozen=True)
class CVFold:
    """Train-local row indices; stopping rows never overlap scored held-out rows."""

    train_rows: tuple[int, ...]
    stopping_rows: tuple[int, ...]
    heldout_rows: tuple[int, ...]


@dataclass(frozen=True)
class CVScoreSummary:
    fold_scores: tuple[MacroScores, ...]
    mean: MacroScores
    standard_deviation: MacroScores
    standard_error: MacroScores
    training_seconds: float = 0.0
    prediction_seconds: float = 0.0


@dataclass(frozen=True)
class CVPrunedEvaluation:
    """Observed folds and rigorous all-fold upper bounds; no full CV mean."""

    fold_scores: tuple[MacroScores, ...]
    total_folds: int
    upper_bounds: MacroScores
    training_seconds: float = 0.0
    prediction_seconds: float = 0.0


@dataclass(frozen=True)
class CVEvaluationTrace:
    folds: tuple[CVFold, ...]
    grouped: bool
    baseline: CVScoreSummary
    trials: tuple[CVScoreSummary | CVPrunedEvaluation, ...]
    final: CVScoreSummary
    validation_baseline: MacroScores
    validation_trials: tuple[MacroScores | None, ...]
    validation_final: MacroScores
    labels: tuple = ()
    group_ids: tuple | None = None


@dataclass(frozen=True)
class ValidationFold:
    """Development-local indices; Validation selects checkpoints and scores F1."""

    train_rows: tuple[int, ...]
    validation_rows: tuple[int, ...]


@dataclass(frozen=True)
class FixedCVEvaluationTrace:
    """External two-part folds, complete CV scores, and accepted refit budgets.

    best_rounds[0] describes Full; subsequent entries correspond to every trial.
    refit_rounds corresponds to accepted_history (visited states if ungated), including Full.
    final describes the selected state, which can precede the terminal state.
    """

    folds: tuple[ValidationFold, ...]
    grouped: bool
    baseline: CVScoreSummary
    trials: tuple[CVScoreSummary, ...]
    final: CVScoreSummary
    best_rounds: tuple[tuple[int, ...], ...]
    refit_rounds: tuple[int, ...]
    labels: tuple
    group_ids: tuple | None = None
    reference: str = 'full'
    fold_kind: str = 'partition'
    checkpoint_metric: str = 'configured'
    truncate_decimals: int | None = None
