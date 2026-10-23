import copy

import numpy as np
import pytest

from recursive_selection import MacroQualityPolicy, RecursiveGroupSelector
from recursive_selection.groups import FeatureGroups
from recursive_selection.importance import MeanAbsoluteShapRanker
from recursive_selection.results import MacroScores


class FakeModel:
    pass


class ScriptedAdapter:
    """Deterministic validation oracle: rank ties force rejection then retry."""
    def __init__(self):
        self.fit_calls = []
        self.models = []

    def new_model(self):
        model = FakeModel()
        self.models.append(model)
        return model

    def fit(self, model, X, y, eval_set):
        model.groups = tuple(int(x) for x in X[0])
        model.classes_ = np.unique(y)
        model.n_features_in_ = X.shape[1]
        self.fit_calls.append(model.groups)

    def predict(self, model, X):
        if model.groups in {(1, 2, 3, 4), (1, 2, 4), (1, 2)}:
            return np.arange(len(X)) % 2
        return np.zeros(len(X), dtype=int)

    def classes(self, model):
        return model.classes_

    def feature_count(self, model):
        return model.n_features_in_

    def validate_full_model(self, model):
        assert isinstance(model, FakeModel)


class TieRanker:
    def __init__(self):
        self.states = []

    def rank(self, adapter, model, X, y, groups):
        self.states.append(tuple(groups))
        return {group: 0.0 for group in groups}


@pytest.fixture
def inputs():
    X = np.tile(np.arange(1, 5, dtype=float), (12, 1))
    y = np.arange(12) % 2
    return X, y, {group: [group - 1] for group in range(1, 5)}


def make_selector(adapter=None, **kwargs):
    return RecursiveGroupSelector(
        model_adapter=adapter if adapter is not None else ScriptedAdapter(),
        ranker=kwargs.pop("ranker", TieRanker()),
        quality_policy=kwargs.pop("quality_policy", MacroQualityPolicy(0.8, 0.8, 0.8, 1.0)),
        **kwargs,
    )


def test_retry_state_ties_fresh_models_and_final_exhaustion(inputs):
    X, y, groups = inputs
    original = X.copy()
    adapter, ranker = ScriptedAdapter(), TieRanker()
    selector = make_selector(adapter, ranker=ranker).fit(X, y, eval_set=(X, y), feature_groups=groups)
    assert selector.selected_group_ids_ == (1, 2)
    assert [r.group_id for r in selector.history_] == [4, 3, 4, 2, 1]
    assert [r.accepted for r in selector.history_] == [False, True, True, False, False]
    assert [r.configuration_id for r in selector.history_] == [0, 0, 1, 2, 2]
    assert selector.history_[0].groups_before == selector.history_[1].groups_before
    assert selector.history_[2].groups_before == (1, 2, 4)
    assert ranker.states == [(1, 2, 3, 4), (1, 2, 4), (1, 2)]
    assert len({id(model) for model in adapter.models}) == 6
    assert selector.stop_reason_ == "all_candidates_rejected"
    assert selector.rejected_group_ids_ == (1, 2)
    assert len(selector.shap_history_) == 1 + 4 - len(selector.selected_group_ids_)
    np.testing.assert_array_equal(selector.transform(X), X[:, :2])
    np.testing.assert_array_equal(selector.predict(X), y)
    np.testing.assert_array_equal(X, original)
    assert selector.history_records()[0]["metrics"]["f1_macro"] < 0.8


def test_reuse_full_model_and_repeat_fit_resets_state(inputs):
    X, y, groups = inputs
    adapter = ScriptedAdapter()
    full = adapter.new_model()
    adapter.fit(full, X, y, (X, y))
    adapter.fit_calls.clear()
    selector = make_selector(adapter)
    selector.fit(X, y, eval_set=(X, y), feature_groups=groups, full_model=full)
    assert len(adapter.fit_calls) == 5
    assert full.groups == (1, 2, 3, 4)
    assert selector.timings_["full_training_seconds"] == 0
    first = [(r.group_id, r.accepted) for r in selector.history_]
    selector.fit(X, y, eval_set=(X, y), feature_groups=groups)
    assert [(r.group_id, r.accepted) for r in selector.history_] == first


def test_thresholds_are_computed_once_and_fixed_to_full(inputs):
    class TrackingPolicy:
        def __init__(self):
            self.policy = MacroQualityPolicy(0.8, 0.8, 0.8, 0.2)
            self.calls = []

        def thresholds(self, baseline):
            self.calls.append(baseline)
            return self.policy.thresholds(baseline)

        def evaluate(self, scores, floors):
            return self.policy.evaluate(scores, floors)

    X, y, groups = inputs
    policy = TrackingPolicy()
    selector = make_selector(quality_policy=policy).fit(X, y, eval_set=(X, y), feature_groups=groups)
    assert policy.calls == [selector.baseline_metrics_]
    assert selector.effective_floors_ == MacroScores(0.8, 0.8, 0.8)


def test_minimum_group_count_stops_before_more_trials(inputs):
    X, y, groups = inputs
    selector = make_selector(min_groups=3).fit(X, y, eval_set=(X, y), feature_groups=groups)
    assert selector.selected_group_ids_ == (1, 2, 4)
    assert selector.stop_reason_ == "min_groups_reached"
    assert len(selector.history_) == 2


def test_infeasible_baseline_stops_before_ranking(inputs):
    class BadAdapter(ScriptedAdapter):
        def predict(self, model, X):
            return np.zeros(len(X), dtype=int)

    X, y, groups = inputs
    ranker = TieRanker()
    selector = make_selector(BadAdapter(), ranker=ranker)
    with pytest.raises(ValueError, match="Full model"):
        selector.fit(X, y, eval_set=(X, y), feature_groups=groups)
    assert ranker.states == []
    with pytest.raises(RuntimeError):
        selector.predict(X)


def test_training_errors_are_not_counted_as_rejections(inputs):
    class BrokenAdapter(ScriptedAdapter):
        def fit(self, model, X, y, eval_set):
            if X.shape[1] < 4:
                raise RuntimeError("backend exploded")
            super().fit(model, X, y, eval_set)

    X, y, groups = inputs
    selector = make_selector(BrokenAdapter())
    with pytest.raises(RuntimeError, match="backend exploded"):
        selector.fit(X, y, eval_set=(X, y), feature_groups=groups)
    with pytest.raises(RuntimeError):
        selector.predict(X)


def test_recorder_cannot_mutate_selection_state(inputs):
    class MutatingRecorder:
        def __init__(self):
            self.events = []

        def record(self, event):
            self.events.append(copy.deepcopy(event))
            if "ranking" in event:
                event["ranking"].clear()

    X, y, groups = inputs
    recorder = MutatingRecorder()
    selector = make_selector(recorder=recorder).fit(X, y, eval_set=(X, y), feature_groups=groups)
    assert selector.selected_group_ids_ == (1, 2)
    assert [e["event"] for e in recorder.events].count("trial_completed") == 5
    assert recorder.events[-1]["event"] == "run_finished"


def test_explicit_groups_preserve_original_column_order():
    groups = FeatureGroups({7: [3, 0], 2: [2, 1]}, 4)
    assert groups.group_ids == (7, 2)
    assert groups.columns((7,)) == (0, 3)
    assert groups.relative((7,)) == {7: (0, 1)}
    assert groups.columns((2, 7)) == (0, 1, 2, 3)


@pytest.mark.parametrize("groups", [{1: [0], 2: [0, 1, 2, 3]}, {1: [0, 1]}, {1: []}, {0: [0, 1, 2, 3]}])
def test_invalid_groups_are_rejected(groups):
    with pytest.raises(ValueError):
        FeatureGroups(groups, 4)


def test_validation_and_input_schema_checks(inputs):
    X, y, groups = inputs
    selector = make_selector()
    with pytest.raises(ValueError, match="original input columns"):
        selector.fit(X, y, eval_set=(X[:, :2], y), feature_groups=groups)
    with pytest.raises(ValueError, match="unique strings"):
        selector.fit(X, y, eval_set=(X, y), feature_groups=groups, feature_names=["x"] * 4)
    selector.fit(X, y, eval_set=(X, y), feature_groups=groups)
    with pytest.raises(ValueError, match="original input columns"):
        selector.predict(X[:, :2])


def test_dataframe_column_order_is_checked(inputs):
    pd = pytest.importorskip("pandas")
    X, y, groups = inputs
    frame = pd.DataFrame(X, columns=list("abcd"))
    selector = make_selector().fit(frame, y, eval_set=(frame, y), feature_groups=groups)
    with pytest.raises(ValueError, match="columns"):
        selector.predict(frame[list("bacd")])


def test_macro_policy_uses_percentage_points_and_tolerance():
    policy = MacroQualityPolicy()
    floors = policy.thresholds(MacroScores(0.95, 0.93, 0.94))
    np.testing.assert_allclose(list(floors.as_dict().values()), [0.91, 0.89, 0.90])
    assert policy.evaluate(MacroScores(0.91 - 5e-13, 0.89, 0.90), floors).passed
    assert not policy.evaluate(MacroScores(0.91 - 2e-12, 0.89, 0.90), floors).passed
    with pytest.raises(ValueError):
        MacroQualityPolicy(max_drop_from_full=4)


def test_ranker_aggregates_absolute_values_and_removes_no_additional_columns():
    class Provider:
        def feature_shap(self, model, X, y):
            return np.array([[[1.0, -3.0, 10.0]], [[-1.0, 3.0, -10.0]]])

    scores = MeanAbsoluteShapRanker().rank(Provider(), None, np.ones((2, 3)), [0, 1], {1: (0, 1), 2: (2,)})
    assert scores == {1: 2.0, 2: 10.0}


def test_bad_ranker_and_reused_trial_model_are_rejected(inputs):
    class BadRanker(TieRanker):
        def rank(self, adapter, model, X, y, groups):
            return {group: float("nan") for group in groups}

    X, y, groups = inputs
    with pytest.raises(ValueError, match="finite nonnegative"):
        make_selector(ranker=BadRanker()).fit(X, y, eval_set=(X, y), feature_groups=groups)

    class ReusedAdapter(ScriptedAdapter):
        def new_model(self):
            if not self.models:
                self.models.append(FakeModel())
            return self.models[0]

    with pytest.raises(ValueError, match="fresh model"):
        make_selector(ReusedAdapter()).fit(X, y, eval_set=(X, y), feature_groups=groups)
