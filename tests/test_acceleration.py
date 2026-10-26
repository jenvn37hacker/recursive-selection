"""Bounds, batch backoff/retry, CV-first fits, concurrency and native replay."""

from dataclasses import replace
from threading import Lock, get_ident
from time import sleep

import numpy as np
import pytest

from recursive_selection import (
    CVReferenceQualityPolicy,
    RecursiveFeatureSelector,
    SelectionSpeedOptions,
)
from recursive_selection.acceleration import RemovalPlanner
from recursive_selection.cross_validation import (
    cv_upper_bounds,
    evaluate_folds,
    make_folds,
)
from recursive_selection.evaluation import audit_selection
from recursive_selection.reporting import SelectionReport
from recursive_selection.results import CVPrunedEvaluation, MacroScores
from test_cv_policy import CVOracle, scripted_data
from test_selection import TieRanker


def run_fast(fraction=0.0, grouped=False, jobs=2, early=True):
    X, y = scripted_data(grouped)
    adapter = CVOracle()
    selector = RecursiveFeatureSelector(
        model_adapter=adapter,
        ranker=TieRanker(),
        quality_policy=CVReferenceQualityPolicy(),
        speed_options=SelectionSpeedOptions(
            removal_fraction=fraction, cv_jobs=jobs, cpu_budget=2, early_reject=early
        ),
    )
    selector.fit(
        X,
        y,
        eval_set=(X, 1 - y),
        cv_groups=np.repeat(np.arange(20), 3) if grouped else None,
    )
    audit_selection(
        selector.result_,
        feature_groups=selector.feature_groups_.as_dict(),
        quality_policy=selector.quality_policy,
    )
    return selector, adapter


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("jobs", [1, 2])
def test_cv_first_and_exact_pruning_preserve_singleton_path(grouped, jobs):
    fast, adapter = run_fast(grouped=grouped, jobs=jobs)
    complete, slow_adapter = run_fast(grouped=grouped, jobs=jobs, early=False)
    assert fast.selected_columns_ == complete.selected_columns_ == (0, 1)
    assert [(t.group_id, t.accepted) for t in fast.history_] == [
        (4, False),
        (3, True),
        (4, True),
        (2, False),
        (1, False),
    ]
    assert fast.timings_["trial_model_fit_count"] == 2
    assert fast.timings_["cv_fit_count"] == 12
    assert complete.timings_["cv_fit_count"] == 18
    assert len(adapter.fit_calls) == 15  # 1 full + 2 accepted + 12 CV
    assert (
        len(slow_adapter.fit_calls) == 21
    )  # CV-first still avoids rejected deployment fits
    assert fast.timings_["cv_pruned_trials"] == 3
    for trial, cv, validation in zip(
        fast.history_, fast.cv_evaluation_.trials, fast.cv_evaluation_.validation_trials
    ):
        if trial.accepted:
            assert trial.metric_kind == "mean" and len(cv.fold_scores) == 3
            assert validation is not None
        else:
            assert isinstance(cv, CVPrunedEvaluation)
            assert trial.metric_kind == "cv_upper_bound"
            assert validation is None and trial.train_seconds == 0
            assert not hasattr(cv, "mean")


def test_batch_success_reduces_trials_and_preserves_minimum(tmp_path):
    selector, adapter = run_fast(fraction=0.5)
    assert [t.removed_group_ids for t in selector.history_] == [(4, 3), (2,), (1,)]
    assert selector.selected_columns_ == (0, 1)
    assert len(selector.shap_history_) == 2  # one SHAP update for a batch of two
    assert len(adapter.fit_calls) == 10
    report = SelectionReport(
        selector.result_, feature_groups=selector.feature_groups_.as_dict()
    )
    assert report.trials.iloc[0]["Removed IDs"] == [4, 3]
    assert report.trials.iloc[1]["Score source"] == "Train CV rejection upper bound"
    assert report.trials["Completed folds"].tolist() == [3, 1, 1]
    assert report.selection_costs.set_index("Stage").loc["Trial training", "Calls"] == 1
    assert len(report.cv_scores) == 8
    assert report.metadata()["trial_models_trained"] == 1
    report.save(tmp_path / "report")


def test_failed_batches_shrink_without_marking_their_members_failed():
    selector, _ = run_fast(fraction=1.0)
    assert [t.removed_group_ids for t in selector.history_] == [
        (4, 3, 2),
        (4,),
        (3,),
        (4, 2),
        (4,),
        (2,),
        (1,),
    ]
    assert selector.selected_columns_ == (0, 1)
    report = SelectionReport(
        selector.result_, feature_groups=selector.feature_groups_.as_dict()
    )
    assert report.retry_group_ids == (4,)
    assert len(report.retry_evidence) == 2
    for trial in selector.history_:
        assert len(trial.proposed_groups) >= selector.min_features
    planner = RemovalPlanner(2, 1.0)
    assert planner.choose((1, 2, 3, 4), {g: g for g in range(1, 5)}) == (1, 2)
    planner.observe((1, 2), False)
    assert planner.failed == set()
    assert planner.choose((1, 2, 3, 4), {g: g for g in range(1, 5)}) == (1,)


def test_bounds_are_safe_for_every_metric_and_not_just_one_bad_fold():
    bounds = cv_upper_bounds([MacroScores(0.70, 0.95, 0.99)], 3)
    assert bounds.precision_macro == pytest.approx(0.90)
    assert bounds.recall_macro > 0.94
    assert (
        not CVReferenceQualityPolicy()
        .evaluate(bounds, MacroScores(0.94, 0.94, 0.94))
        .passed
    )
    recoverable = cv_upper_bounds([MacroScores(0.90, 0.90, 0.90)], 3)
    assert (
        CVReferenceQualityPolicy()
        .evaluate(recoverable, MacroScores(0.94, 0.94, 0.94))
        .passed
    )
    rng = np.random.default_rng(6)
    for _ in range(100):
        scores = [MacroScores(*row) for row in rng.random((5, 3))]
        for completed in range(1, 5):
            bound = cv_upper_bounds(scores[:completed], 5)
            actual = np.mean(
                [list(score.as_dict().values()) for score in scores], axis=0
            )
            assert np.all(actual <= list(bound.as_dict().values()))
    # Equality at a zero-tolerance threshold must never be pruned due to rounding.
    policy = CVReferenceQualityPolicy(tolerance=0)
    assert policy.evaluate(
        cv_upper_bounds([MacroScores(0.7, 0.7, 0.7)], 3), MacroScores(0.9, 0.9, 0.9)
    ).passed


def test_pruned_evidence_and_batch_order_tampering_are_detected():
    selector, _ = run_fast(fraction=1)
    trace = selector.cv_evaluation_
    bad = replace(trace.trials[0], upper_bounds=MacroScores(1, 1, 1))
    kwargs = dict(
        feature_groups=selector.feature_groups_.as_dict(),
        quality_policy=selector.quality_policy,
    )
    with pytest.raises(ValueError, match="upper bounds"):
        audit_selection(
            replace(
                selector.result_,
                cv_evaluation=replace(trace, trials=(bad, *trace.trials[1:])),
            ),
            **kwargs,
        )
    trial = replace(selector.history_[0], removed_group_ids=(1, 2, 3))
    with pytest.raises(ValueError, match="batch backoff"):
        audit_selection(
            replace(selector.result_, history=(trial, *selector.history_[1:])), **kwargs
        )


def test_parallel_folds_are_bounded_and_recorder_callbacks_stay_on_caller():
    class ConcurrentOracle(CVOracle):
        def __init__(self):
            super().__init__()
            self.lock = Lock()
            self.active = self.maximum = 0

        def fit(self, *args):
            with self.lock:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
            try:
                sleep(0.03)
                super().fit(*args)
            finally:
                with self.lock:
                    self.active -= 1

    adapter = ConcurrentOracle()
    options = SelectionSpeedOptions(cpu_budget=2, cv_jobs=2)
    if options.resources(3, adapter)[0] < 2:
        pytest.skip("Only one available CPU")
    X, y = scripted_data()
    folds, _ = make_folds(X, y, CVReferenceQualityPolicy())
    callback_threads = []
    summary = evaluate_folds(
        adapter,
        X,
        y,
        folds,
        np.unique(y),
        check_model=RecursiveFeatureSelector._check_model,
        speed_options=options,
        emit=lambda *a, **kw: callback_threads.append(get_ident()),
    )
    assert adapter.maximum == 2 and len(summary.fold_scores) == 3
    assert set(callback_threads) == {get_ident()}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cv_jobs": 0},
        {"cv_jobs": True},
        {"cpu_budget": -1},
        {"removal_fraction": -1},
        {"removal_fraction": float("nan")},
        {"early_reject": 1},
    ],
)
def test_invalid_speed_options(kwargs):
    with pytest.raises(ValueError):
        SelectionSpeedOptions(**kwargs)


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["catboost", "xgboost", "lightgbm"])
def test_native_accelerated_bundle_restores_options_and_audit(tmp_path, backend):
    pytest.importorskip(backend)
    from sklearn.datasets import make_classification
    from recursive_selection import CatBoostAdapter, XGBoostAdapter, LightGBMAdapter

    if backend == "catboost":
        adapter = CatBoostAdapter(
            params=dict(
                iterations=6,
                depth=2,
                thread_count=1,
                verbose=False,
                random_seed=42,
                allow_writing_files=False,
            )
        )
    elif backend == "xgboost":
        adapter = XGBoostAdapter(
            params=dict(n_estimators=6, max_depth=2, n_jobs=1, random_state=42)
        )
    else:
        adapter = LightGBMAdapter(
            params=dict(
                n_estimators=6,
                max_depth=2,
                num_leaves=4,
                min_child_samples=2,
                n_jobs=1,
                random_state=42,
                verbosity=-1,
            )
        )
    X, y = make_classification(
        n_samples=120, n_features=6, n_informative=4, n_redundant=0, random_state=4
    )
    selector = RecursiveFeatureSelector(
        model_adapter=adapter,
        quality_policy=CVReferenceQualityPolicy(),
        speed_options=SelectionSpeedOptions(cpu_budget=2, removal_fraction=0.5),
        min_features=2,
    )
    selector.fit(X[:90], y[:90], eval_set=(X[90:], y[90:]))
    path = selector.save(tmp_path / backend)
    loaded = RecursiveFeatureSelector.load(path)
    assert loaded.speed_options == selector.speed_options
    assert loaded.cv_evaluation_ == selector.cv_evaluation_
    assert loaded.history_records() == selector.history_records()
    assert loaded.result_.speed_options == selector.speed_options
    np.testing.assert_array_equal(loaded.predict(X), selector.predict(X))
    audit_selection(
        loaded.result_,
        feature_groups=loaded.feature_groups_.as_dict(),
        quality_policy=loaded.quality_policy,
        min_groups=2,
    )


def test_accepted_cv_candidate_deployment_errors_propagate():
    class BrokenDeployment(CVOracle):
        def fit(self, model, X, y, eval_set):
            if len(X) == 60 and X.shape[1] < 4:
                raise RuntimeError("deployment fit failed")
            super().fit(model, X, y, eval_set)

    X, y = scripted_data()
    selector = RecursiveFeatureSelector(
        model_adapter=BrokenDeployment(),
        ranker=TieRanker(),
        quality_policy=CVReferenceQualityPolicy(),
        speed_options=SelectionSpeedOptions(),
    )
    with pytest.raises(RuntimeError, match="deployment fit failed"):
        selector.fit(X, y, eval_set=(X, y))
    with pytest.raises(RuntimeError, match="fit"):
        selector.predict(X)


def test_group_selection_acceleration_respects_whole_groups_and_minimum():
    from recursive_selection import RecursiveGroupSelector

    X, y = scripted_data()
    selector = RecursiveGroupSelector(
        model_adapter=CVOracle(),
        ranker=TieRanker(),
        quality_policy=CVReferenceQualityPolicy(),
        speed_options=SelectionSpeedOptions(removal_fraction=0.5),
        min_groups=1,
    )
    selector.fit(X, y, eval_set=(X, y), feature_groups={1: [0, 1], 2: [2], 3: [3]})
    assert selector.selected_group_ids_ == (1,)
    assert selector.selected_columns_ == (0, 1)
    audit_selection(
        selector.result_,
        feature_groups=selector.feature_groups_.as_dict(),
        quality_policy=selector.quality_policy,
        min_groups=1,
    )
