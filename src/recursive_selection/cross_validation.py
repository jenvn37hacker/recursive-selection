"""Fixed Train-only CV membership and fresh-model scoring for automatic gates."""

from time import perf_counter
from copy import copy, deepcopy
from concurrent.futures import ThreadPoolExecutor
from math import fsum

from threadpoolctl import threadpool_limits

import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
    train_test_split,
)

from .results import CVFold, CVScoreSummary, CVPrunedEvaluation, MacroScores


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


def cv_upper_bounds(scores, total_folds):
    """Outward-rounded maximum possible all-fold mean for bounded [0, 1] scores."""
    scores = tuple(scores)
    if not 0 < len(scores) <= total_folds:
        raise ValueError("Invalid observed/total fold counts.")
    for score in scores:
        if any(not np.isfinite(v) or not 0 <= v <= 1 for v in score.as_dict().values()):
            raise ValueError("CV scores must be finite and in [0, 1].")
    remaining = total_folds - len(scores)
    return MacroScores(
        *(
            min(
                1.0,
                np.nextafter(
                    np.nextafter(
                        fsum([*(getattr(score, key) for score in scores), remaining])
                        / total_folds,
                        np.inf,
                    ),
                    np.inf,
                ),
            )
            for key in scores[0].as_dict()
        )
    )


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
    speed_options=None,
    floors=None,
    tolerance=1e-12,
):
    scores, models = [], []
    train_seconds = prediction_seconds = 0.0
    jobs, threads = (
        speed_options.resources(len(folds), adapter) if speed_options else (1, None)
    )
    if speed_options is not None and getattr(adapter, "backend_name", "") == "catboost":
        # Concurrent CatBoost fits must not redirect the notebook's global log
        # streams. Keep the deployment adapter unchanged; worker fits are quiet.
        worker_adapter = copy(adapter)
        worker_adapter.params = deepcopy(adapter.params)
        worker_adapter.fit_params = deepcopy(adapter.fit_params)
        for options in (worker_adapter.params, worker_adapter.fit_params):
            for key in ("verbose", "verbose_eval", "logging_level", "silent"):
                options.pop(key, None)
        worker_adapter.params["verbose"] = False
        worker_adapter.fit_params["verbose"] = False
        adapter = worker_adapter
    can_prune = (
        speed_options is not None and speed_options.early_reject and floors is not None
    )

    def run_fold(model, fold):
        train, stopping, heldout = (
            np.asarray(rows, dtype=int)
            for rows in (fold.train_rows, fold.stopping_rows, fold.heldout_rows)
        )
        tick = perf_counter()
        adapter.fit(model, X[train], y[train], (X[stopping], y[stopping]))
        training = perf_counter() - tick
        check_model(adapter, model, X.shape[1], classes)
        tick = perf_counter()
        score = MacroScores.from_predictions(
            y[heldout], adapter.predict(model, X[heldout]), classes
        )
        return score, training, perf_counter() - tick

    # Worker fits never invoke notebook/recorder APIs. Events are emitted on the
    # caller thread, with deterministic fold order, after each bounded wave.
    with (
        threadpool_limits(limits=threads),
        ThreadPoolExecutor(max_workers=jobs) as executor,
    ):
        position = 0
        while position < len(folds):
            size = 1 if can_prune and position == 0 else jobs
            wave = []
            for index in range(position, min(position + size, len(folds))):
                model = adapter.new_model()
                if any(model is previous for previous in (*models, *forbidden_models)):
                    raise ValueError(
                        "CV backend must return a fresh model for every fold."
                    )
                models.append(model)
                if threads is not None and hasattr(model, "set_params"):
                    backend = getattr(adapter, "backend_name", "")
                    if backend in ("catboost", "xgboost", "lightgbm"):
                        parameter_name = (
                            "thread_count" if backend == "catboost" else "n_jobs"
                        )
                        thread_settings = {parameter_name: threads}
                        model.set_params(**thread_settings)
                if emit:
                    emit(
                        "cv_fold_started",
                        trial_id=trial_id,
                        fold=index + 1,
                        folds=len(folds),
                        n_features=X.shape[1],
                    )
                wave.append((index, executor.submit(run_fold, model, folds[index])))
            for index, future in wave:
                score, training, predicting = future.result()
                train_seconds += training
                prediction_seconds += predicting
                scores.append(score)
                if emit:
                    emit(
                        "cv_fold_completed",
                        trial_id=trial_id,
                        fold=index + 1,
                        folds=len(folds),
                        n_features=X.shape[1],
                        metrics=score.as_dict(),
                    )
            position += len(wave)
            if can_prune and position < len(folds):
                bounds = cv_upper_bounds(scores, len(folds))
                if any(
                    getattr(bounds, key) < floor - tolerance
                    for key, floor in floors.as_dict().items()
                ):
                    if emit:
                        emit(
                            "cv_pruned",
                            trial_id=trial_id,
                            completed_folds=position,
                            folds=len(folds),
                            upper_bounds=bounds.as_dict(),
                            n_features=X.shape[1],
                        )
                    return CVPrunedEvaluation(
                        tuple(scores),
                        len(folds),
                        bounds,
                        train_seconds,
                        prediction_seconds,
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

    if not isinstance(trace.baseline, CVScoreSummary) or not isinstance(
        trace.final, CVScoreSummary
    ):
        raise ValueError("Reference and accepted CV summaries must contain all folds.")
    for summary in (trace.baseline, *trace.trials, trace.final):
        if isinstance(summary, CVPrunedEvaluation):
            if result.speed_options is None or not result.speed_options.early_reject:
                raise ValueError("Pruned CV requires enabled early rejection.")
            if (
                summary.total_folds != policy.n_splits
                or not 0 < len(summary.fold_scores) < policy.n_splits
            ):
                raise ValueError("Invalid pruned CV fold counts.")
            if not same(
                summary.upper_bounds,
                cv_upper_bounds(summary.fold_scores, policy.n_splits),
            ):
                raise ValueError("CV upper bounds differ from observed fold scores.")
            if policy.evaluate(summary.upper_bounds, result.effective_floors).passed:
                raise ValueError("Pruned CV cannot certify rejection.")
        else:
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
        pruned = isinstance(summary, CVPrunedEvaluation)
        expected_scores = summary.upper_bounds if pruned else summary.mean
        if not same(trial.metrics, expected_scores) or trial.metric_kind != (
            "cv_upper_bound" if pruned else "mean"
        ):
            raise ValueError(
                "Trial gate metrics differ from its CV mean or upper bound."
            )
        if pruned and trial.accepted:
            raise ValueError("A pruned candidate cannot be accepted.")
        if result.speed_options is not None:
            if trial.accepted != (validation is not None):
                raise ValueError(
                    "CV-first fits/Validation diagnostics must exist only for accepted trials."
                )
            if not trial.accepted and (
                trial.train_seconds != 0 or trial.validation_predict_seconds != 0
            ):
                raise ValueError(
                    "A rejected CV-first candidate must not have a deployment fit."
                )
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
    if int(result.timings.get("cv_fit_count", -1)) != len(
        trace.baseline.fold_scores
    ) + sum(len(item.fold_scores) for item in trace.trials):
        raise ValueError("Recorded CV fit count differs from history.")

    if result.speed_options is not None:
        accepted_count = sum(trial.accepted for trial in result.history)
        if result.timings.get("trial_model_fit_count") != accepted_count:
            raise ValueError(
                "CV-first deployment fit count differs from accepted history."
            )
        expected_pruned = sum(
            isinstance(item, CVPrunedEvaluation) for item in trace.trials
        )
        if result.timings.get("cv_pruned_trials") != expected_pruned:
            raise ValueError("Pruned trial count differs from CV evidence.")
