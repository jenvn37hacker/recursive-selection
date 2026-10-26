"""Automatic full-reference gate, fixed folds, retries, leakage and native bundles."""

from dataclasses import replace
import json

import numpy as np
import pytest

from recursive_selection import (
    CVReferenceQualityPolicy,
    MacroQualityPolicy,
    RecursiveFeatureSelector,
    CatBoostAdapter,
)
from recursive_selection.cross_validation import summarize_scores, validate_cv_trace
from recursive_selection.results import MacroScores
from test_selection import ScriptedAdapter, TieRanker


def scripted_data(grouped=False):
    y = np.repeat(np.arange(20) % 2, 3) if grouped else np.arange(60) % 2
    X = np.tile(np.arange(1, 5, dtype=float), (len(y), 1)) + y[:, None] * 0.01
    return X, y


class CVOracle(ScriptedAdapter):
    def predict(self, model, X):
        y = (X[:, 0] - np.floor(X[:, 0]) > 0.005).astype(int)
        return (
            y
            if model.groups in {(1, 2, 3, 4), (1, 2, 4), (1, 2)}
            else np.zeros(len(X), dtype=int)
        )


def selected(grouped=False, recorder=None):
    X, y = scripted_data(grouped)
    adapter = CVOracle()
    selector = RecursiveFeatureSelector(
        model_adapter=adapter,
        ranker=TieRanker(),
        quality_policy=CVReferenceQualityPolicy(),
        recorder=recorder,
    )
    groups = np.repeat(np.arange(20), 3) if grouped else None
    # Deliberately poor outer Validation predictions must not become the CV gate.
    selector.fit(X, y, eval_set=(X, 1 - y), cv_groups=groups)
    return selector, adapter, X, y, groups


def test_thresholds_are_full_mean_minus_sample_se_not_fixed_percentages():
    summary = summarize_scores([MacroScores(v, v, v) for v in [0.90, 0.92, 0.94]])
    policy = CVReferenceQualityPolicy()
    assert summary.standard_deviation.f1_macro == pytest.approx(0.02)
    assert summary.standard_error.f1_macro == pytest.approx(0.02 / np.sqrt(3))
    assert policy.thresholds(summary).f1_macro == pytest.approx(
        0.92 - 0.02 / np.sqrt(3)
    )
    with pytest.raises(TypeError, match="CVScoreSummary"):
        policy.thresholds(summary.mean)
    assert CVReferenceQualityPolicy(se_multiplier=0).thresholds(summary) == summary.mean


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_splits": True},
        {"n_splits": 1},
        {"n_splits": 2.5},
        {"random_state": -1},
        {"se_multiplier": -1},
        {"se_multiplier": float("nan")},
        {"tolerance": float("inf")},
    ],
)
def test_invalid_cv_policy_settings(kwargs):
    with pytest.raises(ValueError):
        CVReferenceQualityPolicy(**kwargs)


@pytest.mark.parametrize("grouped", [False, True])
def test_fixed_folds_cv_only_gate_fresh_fits_and_retries(grouped):
    selector, adapter, X, y, groups = selected(grouped)
    assert selector.baseline_metrics_.f1_macro == 1
    assert selector.cv_evaluation_.validation_baseline.f1_macro == 0
    assert selector.selected_columns_ == (0, 1)
    assert [r.group_id for r in selector.history_] == [4, 3, 4, 2, 1]
    assert [r.accepted for r in selector.history_] == [False, True, True, False, False]
    assert selector.timings_["cv_fit_count"] == 18
    assert len(adapter.models) == 24
    assert len({id(m) for m in adapter.models}) == 24
    trace = selector.cv_evaluation_
    assert trace.grouped == grouped
    for fold in trace.folds:
        fit, stop, held = map(
            set, (fold.train_rows, fold.stopping_rows, fold.heldout_rows)
        )
        assert not fit & stop and not fit & held and not stop & held
        assert fit | stop | held == set(range(len(X)))
        if grouped:
            assert not set(groups[list(fit)]) & set(groups[list(held)])
            assert not set(groups[list(stop)]) & set(groups[list(held)])
    validate_cv_trace(selector.result_, selector.quality_policy)
    original = trace.folds
    selector.fit(X, y, eval_set=(X, 1 - y), cv_groups=groups)
    assert selector.cv_evaluation_.folds == original


def test_cv_audit_and_report_preserve_scope_and_detect_tampering(tmp_path):
    from recursive_selection.evaluation import audit_selection
    from recursive_selection.reporting import SelectionReport

    selector, *_ = selected()
    kwargs = dict(
        feature_groups=selector.feature_groups_.as_dict(),
        quality_policy=selector.quality_policy,
    )
    audit_selection(selector.result_, **kwargs)
    report = SelectionReport(
        selector.result_, feature_groups=kwargs["feature_groups"], unit_name="feature"
    )
    assert set(report.trials["Score source"]) == {"Train CV mean"}
    assert len(report.cv_scores) == 18
    report.save(tmp_path / "reports")
    assert (tmp_path / "reports/selection_cv_membership.csv").exists()
    assert report.metadata()["score_source"] == "Train CV mean"
    bad = replace(
        selector.cv_evaluation_.baseline, standard_error=MacroScores(0.1, 0.1, 0.1)
    )
    tampered = replace(
        selector.result_, cv_evaluation=replace(selector.cv_evaluation_, baseline=bad)
    )
    with pytest.raises(ValueError, match="SE differs"):
        audit_selection(tampered, **kwargs)


@pytest.mark.parametrize("groups", [np.zeros(60), np.arange(10), np.full(60, np.nan)])
def test_invalid_recording_membership_does_not_silently_fall_back(groups):
    X, y = scripted_data()
    with pytest.raises(ValueError):
        RecursiveFeatureSelector(
            model_adapter=CVOracle(),
            ranker=TieRanker(),
            quality_policy=CVReferenceQualityPolicy(),
        ).fit(X, y, eval_set=(X, y), cv_groups=groups)
    with pytest.raises(ValueError, match="requires CVReference"):
        RecursiveFeatureSelector(model_adapter=CVOracle()).fit(
            X, y, eval_set=(X, y), cv_groups=np.arange(60)
        )


@pytest.mark.integration
def test_native_cv_bundle_roundtrip_and_reference_model_reuse(tmp_path):
    pytest.importorskip("catboost")
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=120,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        random_state=4,
    )
    adapter = CatBoostAdapter(
        params=dict(
            iterations=8,
            depth=2,
            thread_count=1,
            verbose=False,
            random_seed=42,
            allow_writing_files=False,
        )
    )
    full = adapter.new_model()
    adapter.fit(full, X[:90], y[:90], (X[90:], y[90:]))
    selector = RecursiveFeatureSelector(
        model_adapter=adapter, quality_policy=CVReferenceQualityPolicy(), min_features=3
    )
    selector.fit(X[:90], y[:90], eval_set=(X[90:], y[90:]), full_model=full)
    assert selector.timings_["full_training_seconds"] == 0
    path = selector.save(tmp_path / "native")
    data = json.loads((path / "selection.json").read_text())
    assert data["format_version"] == 4 and data["policy_type"] == "cv_reference"
    loaded = RecursiveFeatureSelector.load(path)
    assert isinstance(loaded.quality_policy, CVReferenceQualityPolicy)
    assert loaded.cv_evaluation_ == selector.cv_evaluation_
    np.testing.assert_array_equal(loaded.predict(X[90:]), selector.predict(X[90:]))
    loaded.fit(X[:90], y[:90], eval_set=(X[90:], y[90:]))


def test_cv_constraints_cannot_be_applied_to_outer_validation_metrics():
    from recursive_selection.evaluation import evaluate_classifier, compare_evaluations

    selector, adapter, X, y, _ = selected()
    full = evaluate_classifier(
        model=selector.full_model_,
        adapter=adapter,
        splits={"Validation": (X, y)},
        include_probabilities=False,
    )
    candidate = evaluate_classifier(
        model=selector.estimator_,
        adapter=adapter,
        splits={"Validation": (selector.transform(X), y)},
        include_probabilities=False,
    )
    with pytest.raises(ValueError, match="Train CV"):
        compare_evaluations(reference=full, candidate=candidate).validation_constraints(
            policy=selector.quality_policy, baseline=selector.baseline_metrics_
        )


def test_cv_no_removal_still_calibrates_and_audits():
    from recursive_selection.evaluation import audit_selection

    X, y = scripted_data()
    selector = RecursiveFeatureSelector(
        model_adapter=CVOracle(),
        ranker=TieRanker(),
        quality_policy=CVReferenceQualityPolicy(),
        min_features=4,
    ).fit(X, y, eval_set=(X, 1 - y))
    assert not selector.history_ and selector.timings_["cv_fit_count"] == 3
    assert selector.cv_evaluation_.final == selector.cv_evaluation_.baseline
    audit_selection(
        selector.result_,
        feature_groups=selector.feature_groups_.as_dict(),
        quality_policy=selector.quality_policy,
        min_groups=4,
    )
