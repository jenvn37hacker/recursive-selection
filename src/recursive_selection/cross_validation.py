"""Fixed Train-only CV membership and fresh-model scoring for automatic gates."""

from time import perf_counter

import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
    train_test_split,
)

from .results import CVFold, CVScoreSummary, MacroScores


def summarize_scores(scores, *, training_seconds=0.0, prediction_seconds=0.0):
    scores = tuple(scores)
    values = np.asarray(
        [list(score.as_dict().values()) for score in scores], dtype=float
    )
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 3:
        raise ValueError("CV needs at least two folds of all three macro metrics.")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("CV scores must be finite and in [0, 1].")
    sd = values.std(axis=0, ddof=1)
    return CVScoreSummary(
        scores,
        MacroScores(*values.mean(axis=0)),
        MacroScores(*sd),
        MacroScores(*(sd / np.sqrt(len(scores)))),
        training_seconds,
        prediction_seconds,
    )


def _has_classes(y, rows, classes):
    return np.array_equal(np.unique(y[rows]), classes)


def validate_folds(folds, y, groups=None):
    """Validate coverage/class support and, if supplied, recording disjointness."""
    classes = np.unique(y)
    heldout_count = np.zeros(len(y), dtype=int)
    for fold in folds:
        parts = [
            np.asarray(rows, dtype=int)
            for rows in (fold.train_rows, fold.stopping_rows, fold.heldout_rows)
        ]
        if any(len(part) == 0 or len(np.unique(part)) != len(part) for part in parts):
            raise ValueError(
                "CV fit, stopping and held-out rows must be nonempty and unique."
            )
        combined = np.concatenate(parts)
        if not np.array_equal(np.sort(combined), np.arange(len(y))):
            raise ValueError(
                "Each CV fold must partition Train into fit/stopping/held-out rows exactly once."
            )
        if any(not _has_classes(y, part, classes) for part in parts):
            raise ValueError(
                "Each CV fit/stopping/held-out partition must contain every training class."
            )
        if groups is not None:
            sets = [set(groups[part].tolist()) for part in parts]
            if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
                raise ValueError(
                    "CV groups overlap between fit, stopping or held-out rows."
                )
        heldout_count[parts[2]] += 1
    if not np.array_equal(heldout_count, np.ones(len(y), dtype=int)):
        raise ValueError("Every Train row must be scored in exactly one CV fold.")


def make_folds(X, y, policy, groups=None):
    if min(np.unique(y, return_counts=True)[1]) < policy.n_splits:
        raise ValueError("Every class needs at least n_splits training samples for CV.")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.ndim != 1 or len(groups) != len(y):
            raise ValueError(
                "cv_groups must contain one recording/group ID per Train row."
            )
        if any(
            value is None
            or (isinstance(value, (float, np.floating)) and not np.isfinite(value))
            for value in groups
        ):
            raise ValueError("cv_groups must not contain missing or non-finite IDs.")
        if any(
            len(np.unique(groups[y == label])) < policy.n_splits
            for label in np.unique(y)
        ):
            raise ValueError("Each class needs at least n_splits distinct CV groups.")
        splitter = StratifiedGroupKFold(
            n_splits=policy.n_splits, shuffle=True, random_state=policy.random_state
        )
        partitions = splitter.split(X, y, groups=groups)
    else:
        partitions = StratifiedKFold(
            n_splits=policy.n_splits, shuffle=True, random_state=policy.random_state
        ).split(X, y)
    folds = []
    classes = np.unique(y)
    for number, (fit_pool, heldout) in enumerate(partitions):
        if groups is None:
            stopping_count = max(len(classes), int(np.ceil(len(fit_pool) * 0.25)))
            if stopping_count > len(fit_pool) - len(classes):
                raise ValueError(
                    "Not enough Train rows for class-stratified fit/stopping/held-out CV."
                )
            train, stopping = train_test_split(
                fit_pool,
                test_size=stopping_count,
                random_state=policy.random_state + number,
                stratify=y[fit_pool],
            )
        else:
            # Prefer roughly 25% stopping groups. For smaller datasets reduce the
            # inner split count, never fall back to leaking recording groups.
            found = None
            max_inner = min(
                4,
                min(
                    len(np.unique(groups[fit_pool][y[fit_pool] == label]))
                    for label in classes
                ),
            )
            for inner_count in range(max_inner, 1, -1):
                inner = StratifiedGroupKFold(
                    n_splits=inner_count,
                    shuffle=True,
                    random_state=policy.random_state + number,
                )
                for fit_rows, stop_rows in inner.split(
                    X[fit_pool], y[fit_pool], groups[fit_pool]
                ):
                    proposed = (fit_pool[fit_rows], fit_pool[stop_rows])
                    if all(_has_classes(y, rows, classes) for rows in proposed):
                        found = proposed
                        break
                if found is not None:
                    break
            if found is None:
                raise ValueError(
                    "Cannot make class-complete grouped stopping folds; supply more recordings."
                )
            train, stopping = found
        folds.append(
            CVFold(
                tuple(map(int, train)),
                tuple(map(int, stopping)),
                tuple(map(int, heldout)),
            )
        )
    folds = tuple(folds)
    validate_folds(folds, y, groups)
    return folds, groups


def evaluate_folds(
    adapter,
    X,
    y,
    folds,
    classes,
    *,
    check_model,
    forbidden_models=(),
    emit=None,
    trial_id=0,
):
    scores, models = [], []
    train_seconds = prediction_seconds = 0.0
    for number, fold in enumerate(folds, 1):
        train, stopping, heldout = (
            np.asarray(rows, dtype=int)
            for rows in (fold.train_rows, fold.stopping_rows, fold.heldout_rows)
        )
        model = adapter.new_model()
        if any(model is previous for previous in (*models, *forbidden_models)):
            raise ValueError("CV backend must return a fresh model for every fold.")
        models.append(model)
        if emit:
            emit(
                "cv_fold_started",
                trial_id=trial_id,
                fold=number,
                folds=len(folds),
                n_features=X.shape[1],
            )
        tick = perf_counter()
        adapter.fit(model, X[train], y[train], (X[stopping], y[stopping]))
        train_seconds += perf_counter() - tick
        check_model(adapter, model, X.shape[1], classes)
        tick = perf_counter()
        score = MacroScores.from_predictions(
            y[heldout], adapter.predict(model, X[heldout]), classes
        )
        prediction_seconds += perf_counter() - tick
        scores.append(score)
        if emit:
            emit(
                "cv_fold_completed",
                trial_id=trial_id,
                fold=number,
                folds=len(folds),
                n_features=X.shape[1],
                metrics=score.as_dict(),
            )
    return summarize_scores(
        scores, training_seconds=train_seconds, prediction_seconds=prediction_seconds
    )


def validate_cv_trace(result, policy):
    """Validate recorded CV evidence without training, prediction or SHAP."""
    trace = result.cv_evaluation
    if (
        trace is None
        or len(trace.folds) != policy.n_splits
        or len(trace.trials) != len(result.history)
    ):
        raise ValueError(
            "CV trace is missing or differs from selection history/fold count."
        )
    if trace.grouped != (trace.group_ids is not None):
        raise ValueError("CV grouping metadata is inconsistent.")
    validate_folds(
        trace.folds,
        np.asarray(trace.labels),
        np.asarray(trace.group_ids) if trace.grouped else None,
    )

    def same(a, b):
        return np.allclose(
            list(a.as_dict().values()), list(b.as_dict().values()), atol=1e-12, rtol=0
        )

    for summary in (trace.baseline, *trace.trials, trace.final):
        if len(summary.fold_scores) != policy.n_splits:
            raise ValueError("CV summary fold count differs from membership.")
        expected = summarize_scores(summary.fold_scores)
        if any(
            not same(getattr(summary, name), getattr(expected, name))
            for name in ("mean", "standard_deviation", "standard_error")
        ):
            raise ValueError("CV mean/SD/SE differs from recorded fold scores.")
        if any(
            not np.isfinite(value) or value < 0
            for value in (summary.training_seconds, summary.prediction_seconds)
        ):
            raise ValueError("Invalid CV timing.")
    if not same(trace.baseline.mean, result.baseline_metrics):
        raise ValueError("CV reference mean differs from selection baseline.")
    if not same(policy.thresholds(trace.baseline), result.effective_floors):
        raise ValueError(
            "CV thresholds differ from the fixed full-reference mean minus SE."
        )
    selected = trace.baseline
    validation_selected = trace.validation_baseline
    if len(trace.validation_trials) != len(result.history):
        raise ValueError(
            "Validation diagnostic history length differs from CV history."
        )
    for trial, summary, validation in zip(
        result.history, trace.trials, trace.validation_trials
    ):
        if not same(trial.metrics, summary.mean):
            raise ValueError("Trial gate metrics differ from its CV mean.")
        if trial.accepted:
            selected, validation_selected = summary, validation
    if selected != trace.final or not same(result.final_metrics, trace.final.mean):
        raise ValueError(
            "Final CV result differs from the last accepted configuration."
        )
    if not same(validation_selected, trace.validation_final):
        raise ValueError(
            "Final Validation diagnostics differ from the last accepted configuration."
        )
    if (
        int(result.timings.get("cv_fit_count", -1))
        != (1 + len(result.history)) * policy.n_splits
    ):
        raise ValueError("Recorded CV fit count differs from history.")
