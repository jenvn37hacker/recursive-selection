"""Fixed Validation F1 reference, one-fit trials, batch retries and native replay."""

from dataclasses import replace
import json

import numpy as np
import pytest

from recursive_selection import (
    BatchRemovalOptions,
    F1ReferenceQualityPolicy,
    RecursiveFeatureSelector,
    CatBoostAdapter,
    XGBoostAdapter,
    LightGBMAdapter,
)
from recursive_selection.evaluation import audit_selection
from recursive_selection.reporting import SelectionReport
from recursive_selection.results import MacroScores
from test_selection import ScriptedAdapter, TieRanker


def test_only_f1_controls_gate_and_reference_is_fixed():
    policy = F1ReferenceQualityPolicy()
    reference = MacroScores(0.99, 0.99, 0.8)
    floors = policy.thresholds(reference)
    assert policy.evaluate(MacroScores(0.1, 0.2, 0.8), floors).passed
    assert policy.evaluate(MacroScores(0.9, 0.9, 0.9), floors).passed
    # A lower F1 than the previous accepted .9 is still valid against original .8.
    assert policy.evaluate(MacroScores(0.1, 0.1, 0.81), floors).passed
    assert policy.evaluate(MacroScores(0.1, 0.1, 0.8 - 5e-13), floors).passed
    failed = policy.evaluate(MacroScores(1, 1, 0.79), floors)
    assert not failed.passed and failed.failed_metrics == ("f1_macro",)
    assert failed.worst_margin == pytest.approx(-0.01)
    for value in [float("nan"), float("inf"), -1, 1.01]:
        with pytest.raises(ValueError):
            policy.thresholds(MacroScores(0.8, 0.8, value))
    with pytest.raises(ValueError):
        F1ReferenceQualityPolicy(tolerance=-1)


@pytest.mark.parametrize("fraction", [0, 0.1, 0.5, 1])
def test_one_full_train_fit_per_trial_reuse_and_exact_retry(fraction, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Validation F1 selection must never train CV models")

    monkeypatch.setattr("recursive_selection.selector.evaluate_folds", forbidden)
    X = np.tile(np.arange(1, 5, dtype=float), (12, 1))
    y = np.arange(12) % 2
    adapter, ranker = ScriptedAdapter(), TieRanker()
    full = adapter.new_model()
    adapter.fit(full, X, y, (X, y))
    adapter.fit_calls.clear()
    selector = RecursiveFeatureSelector(
        model_adapter=adapter,
        ranker=ranker,
        quality_policy=F1ReferenceQualityPolicy(),
        speed_options=BatchRemovalOptions(fraction),
    ).fit(X, y, eval_set=(X, y), full_model=full)
    assert selector.full_model_ is full
    assert selector.timings_["full_training_seconds"] == 0
    assert selector.cv_evaluation_ is None and selector.timings_["cv_fit_count"] == 0
    assert (
        len(adapter.fit_calls)
        == len(selector.history_)
        == selector.timings_["trial_model_fit_count"]
    )
    assert len(ranker.states) == 1 + sum(t.accepted for t in selector.history_)
    assert selector.selected_group_ids_ == (1, 2)
    if fraction == 1:
        assert [t.removed_group_ids for t in selector.history_] == [
            (4, 3, 2),
            (4,),
            (3,),
            (4, 2),
            (4,),
            (2,),
            (1,),
        ]
    audit_selection(
        selector.result_,
        feature_groups=selector.feature_groups_.as_dict(),
        quality_policy=selector.quality_policy,
    )
    report = SelectionReport(
        selector.result_, feature_groups=selector.feature_groups_.as_dict()
    )
    assert report.gate_constraints.Metric.tolist() == ["F1 macro"]
    assert report.metadata()["gate_metrics"] == ["f1_macro"]
    assert "Required Precision macro" not in report.retry_evidence
    with pytest.raises(ValueError, match="gate metrics"):
        audit_selection(
            replace(selector.result_, gate_metrics=("precision_macro",)),
            feature_groups=selector.feature_groups_.as_dict(),
            quality_policy=selector.quality_policy,
        )


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["catboost", "xgboost", "lightgbm"])
@pytest.mark.parametrize("batch", [False, True])
def test_native_f1_roundtrip_and_report(tmp_path, backend, batch):
    pytest.importorskip(backend)
    from sklearn.datasets import make_classification

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
        quality_policy=F1ReferenceQualityPolicy(),
        speed_options=BatchRemovalOptions(0.5) if batch else None,
        min_features=2,
    )
    selector.fit(X[:90], y[:90], eval_set=(X[90:], y[90:]))
    from recursive_selection.evaluation import evaluate_classifier, compare_evaluations

    full_eval = evaluate_classifier(
        model=selector.full_model_,
        adapter=adapter,
        splits={"Validation": (X[90:], y[90:])},
    )
    selected_eval = evaluate_classifier(
        model=selector.estimator_,
        adapter=adapter,
        splits={"Validation": (selector.transform(X[90:]), y[90:])},
    )
    constraints = compare_evaluations(
        reference=full_eval, candidate=selected_eval
    ).validation_constraints(
        policy=selector.quality_policy, baseline=selector.baseline_metrics_
    )
    assert constraints.Metric.tolist() == ["F1 macro"] and constraints.Passed.all()
    if backend == "catboost" and batch:
        from recursive_selection.plotting import plot_accepted_scores
        import matplotlib.pyplot as plt

        fig, ax = plot_accepted_scores(selector.result_)
        assert len([line for line in ax.lines if line.get_linestyle() == "--"]) == 1
        plt.close(fig)
    path = selector.save(tmp_path / backend)
    loaded = RecursiveFeatureSelector.load(path)
    assert loaded.quality_policy == selector.quality_policy
    assert loaded.speed_options == selector.speed_options
    assert loaded.result_.gate_metrics == ("f1_macro",)
    assert loaded.history_records() == selector.history_records()
    np.testing.assert_array_equal(loaded.predict(X), selector.predict(X))
    audit_selection(
        loaded.result_,
        feature_groups=loaded.feature_groups_.as_dict(),
        quality_policy=loaded.quality_policy,
        min_groups=2,
    )
    report = SelectionReport(
        loaded.result_, feature_groups=loaded.feature_groups_.as_dict()
    )
    report.save(tmp_path / "report")
    assert (tmp_path / "report/selection_gate_constraints.csv").is_file()
    data = json.loads((path / "selection.json").read_text())
    data["gate_metrics"] = ["precision_macro", "f1_macro"]
    (path / "selection.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="gate metrics"):
        RecursiveFeatureSelector.load(path)


@pytest.mark.parametrize("value", [-0.1, 1.1, True, float("nan")])
def test_invalid_batch_fraction(value):
    with pytest.raises(ValueError):
        BatchRemovalOptions(value)


def test_minimum_one_and_shap_only_after_acceptance():
    class AlwaysPass(ScriptedAdapter):
        def predict(self, model, X):
            return np.arange(len(X)) % 2

    X = np.tile(np.arange(1, 5, dtype=float), (12, 1))
    y = np.arange(12) % 2
    adapter = AlwaysPass()
    selector = RecursiveFeatureSelector(
        model_adapter=adapter,
        ranker=TieRanker(),
        quality_policy=F1ReferenceQualityPolicy(),
        speed_options=BatchRemovalOptions(1),
    ).fit(X, y, eval_set=(X, y))
    assert selector.stop_reason_ == "min_groups_reached"
    assert len(selector.selected_columns_) == 1 and len(selector.history_) == 1
    assert len(selector.shap_history_) == 2
    assert len(adapter.fit_calls) == 2  # baseline plus one candidate
    audit_selection(
        selector.result_,
        feature_groups=selector.feature_groups_.as_dict(),
        quality_policy=selector.quality_policy,
    )
