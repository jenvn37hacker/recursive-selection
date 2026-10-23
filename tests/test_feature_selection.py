"""Individual-column API shares the group engine and its downstream contracts."""
import json

import numpy as np
import pytest

from recursive_selection import (
    CatBoostAdapter, MacroQualityPolicy, RecursiveFeatureSelector, RecursiveGroupSelector,
)
from test_backends import sample_data
from test_selection import ScriptedAdapter, TieRanker, make_selector


def feature_selector(**kwargs):
    return RecursiveFeatureSelector(
        model_adapter=ScriptedAdapter(), ranker=TieRanker(),
        quality_policy=MacroQualityPolicy(0.8, 0.8, 0.8, 1), **kwargs,
    )


def data():
    return np.tile(np.arange(1, 5, dtype=float), (12, 1)), np.arange(12) % 2


def test_feature_search_matches_singleton_group_search_including_retry():
    X, y = data()
    names = ["a", "b", "c", "d"]
    features = feature_selector().fit(X, y, eval_set=(X, y), feature_names=names)
    groups = make_selector().fit(X, y, eval_set=(X, y),
        feature_groups={c + 1: [c] for c in range(4)}, feature_names=names)
    assert features.selected_columns_ == groups.selected_columns_ == (0, 1)
    assert features.selected_feature_names_ == ("a", "b")
    assert [t.group_id for t in features.history_] == [4, 3, 4, 2, 1]
    assert [t.accepted for t in features.history_] == [False, True, True, False, False]
    for name in ("accepted_history_", "shap_history_", "baseline_metrics_", "final_metrics_", "stop_reason_"):
        assert getattr(features, name) == getattr(groups, name)
    assert features.model_adapter.fit_calls == groups.model_adapter.fit_calls
    np.testing.assert_array_equal(features.predict(X), groups.predict(X))
    np.testing.assert_array_equal(features.transform(X), X[:, :2])


def test_min_features_and_full_model_reuse_with_repeat_fit():
    X, y = data()
    selector = feature_selector(min_features=3)
    adapter = selector.model_adapter
    full = adapter.new_model()
    adapter.fit(full, X, y, (X, y))
    adapter.fit_calls.clear()
    assert selector.fit(X, y, eval_set=(X, y), full_model=full) is selector
    assert selector.min_features == selector.min_groups == 3
    assert selector.selected_columns_ == (0, 1, 3)
    assert len(adapter.fit_calls) == 2
    assert selector.timings_["full_training_seconds"] == 0
    selector.fit(X, y, eval_set=(X, y))
    assert len(selector.history_) == 2
    assert selector.stop_reason_ == "min_groups_reached"
    with pytest.raises(ValueError, match="nonempty 2D"):
        selector.fit(X[0], y, eval_set=(X, y))
    with pytest.raises(RuntimeError, match="fit"):
        selector.predict(X)


@pytest.mark.parametrize("minimum", [True, 0, -1, 1.5])
def test_invalid_feature_minimum_rejected(minimum):
    with pytest.raises(ValueError, match="min_features"):
        feature_selector(min_features=minimum)


def test_excessive_minimum_and_no_removal_boundary():
    X, y = data()
    with pytest.raises(ValueError, match="min_features exceeds"):
        feature_selector(min_features=5).fit(X, y, eval_set=(X, y))
    selector = feature_selector(min_features=4).fit(X, y, eval_set=(X, y))
    assert selector.selected_columns_ == (0, 1, 2, 3)
    assert not selector.history_
    assert len(selector.shap_history_) == 1


def test_feature_api_infers_names_and_checks_original_dataframe_order():
    pd = pytest.importorskip("pandas")
    X, y = data()
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    selector = feature_selector().fit(frame, y, eval_set=(frame, y))
    assert selector.selected_feature_names_ == ("a", "b")
    np.testing.assert_array_equal(selector.predict(frame), y)
    with pytest.raises(ValueError, match="names/order"):
        selector.predict(frame[["b", "a", "c", "d"]])


def test_feature_results_support_reports_charts_and_audit(tmp_path):
    pytest.importorskip("pandas")
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from recursive_selection.evaluation import audit_selection
    from recursive_selection.plotting import plot_importance_evolution, plot_retry_history
    from recursive_selection.reporting import SelectionReport
    X, y = data()
    selector = feature_selector().fit(X, y, eval_set=(X, y))
    groups = selector.feature_groups_.as_dict()
    labels = {1: "S1_mean", 2: "S1_std", 3: "S2_mean", 4: "S2_std"}
    report = SelectionReport(selector.result_, feature_groups=groups, unit_name="feature", group_labels=labels)
    assert report.trials.columns.is_unique
    assert report.retry_group_ids == (4,)
    assert report.trials["Feature tried"].tolist() == ["S2_std", "S2_mean", "S2_std", "S1_std", "S1_mean"]
    assert set(report.initial_ranking["feature"]) == set(labels.values())
    assert report.trials["Features before"].tolist() == [4, 4, 3, 2, 2]
    report.save(tmp_path / "report")
    report.style("trials").to_html()
    audit_selection(selector.result_, feature_groups=groups, quality_policy=selector.quality_policy,
                    min_groups=selector.min_features, unit_name="feature")
    for plot in (plot_importance_evolution, plot_retry_history):
        fig, axes = plot(selector.result_, unit_name="feature")
        fig.canvas.draw()
        plt.close(fig)


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["feature", "group"])
def test_native_roundtrip_restores_mode_and_old_group_metadata(mode, tmp_path):
    pytest.importorskip("catboost")
    X, y, X_val, y_val = sample_data()
    selector_type = RecursiveFeatureSelector if mode == "feature" else RecursiveGroupSelector
    minimum = {"min_features": 4} if mode == "feature" else {"min_groups": 2}
    selector = selector_type(
        model_adapter=CatBoostAdapter(params=dict(iterations=8, depth=2, random_seed=42,
            thread_count=1, verbose=False, allow_writing_files=False)),
        quality_policy=MacroQualityPolicy(0, 0, 0, 1), **minimum,
    )
    fit_kwargs = {} if mode == "feature" else {"feature_groups": {1: [0, 1], 2: [2, 3], 3: [4, 5]}}
    selector.fit(X, y, eval_set=(X_val, y_val), feature_names=[f"f{i}" for i in range(6)], **fit_kwargs)
    saved = selector.save(tmp_path / mode)
    loaded = selector_type.load(saved)
    assert type(loaded) is selector_type
    assert loaded.selected_columns_ == selector.selected_columns_
    assert loaded.selected_feature_names_ == selector.selected_feature_names_
    assert loaded.history_records() == selector.history_records()
    np.testing.assert_array_equal(loaded.predict(X_val), selector.predict(X_val))
    if mode == "feature":
        assert loaded.min_features == 4
        loaded.fit(X, y, eval_set=(X_val, y_val))
        assert len(loaded.selected_columns_) == 4
    wrong_type = RecursiveGroupSelector if mode == "feature" else RecursiveFeatureSelector
    with pytest.raises(ValueError, match="Saved selection uses"):
        wrong_type.load(saved)
    metadata_path = saved / "selection.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if mode == "group":
        # 0.4.0 metadata did not specify a mode and must remain loadable.
        metadata["format_version"] = 2
        metadata.pop("selection_mode")
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        legacy = RecursiveGroupSelector.load(saved)
        np.testing.assert_array_equal(legacy.predict(X_val), selector.predict(X_val))
    else:
        metadata["feature_groups"] = {"1": [0, 1], "3": [2], "4": [3], "5": [4], "6": [5]}
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with pytest.raises(ValueError, match="singleton"):
            RecursiveFeatureSelector.load(saved)
