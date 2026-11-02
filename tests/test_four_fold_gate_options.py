"""Four-fold resource preset, real scheduling and full/best gate provenance."""

from threading import Barrier, Lock, get_ident

import numpy as np
import pytest

from recursive_selection import (BatchRemovalOptions, CatBoostAdapter,
    CVF1ReferenceQualityPolicy, RecursiveFeatureSelector)
from recursive_selection.evaluation import audit_selection
from recursive_selection.results import MacroScores
from test_fixed_cv import CVAdapter
from test_selection import ScriptedAdapter, TieRanker


def four_splits():
    X = np.tile(np.arange(1, 5, dtype=float), (24, 1))
    y = np.arange(24) % 2
    folds = [(np.flatnonzero(np.arange(24)//6 != f), np.arange(6*f, 6*(f+1)))
             for f in range(4)]
    return X, y, folds


def test_four_fold_preset_caps_cpu_threads_without_mutating_adapter(monkeypatch):
    from recursive_selection import acceleration
    adapter = CatBoostAdapter(params=dict(thread_count=2))
    options = BatchRemovalOptions.four_fold(removal_fraction=.05)
    assert options.to_config() == dict(removal_fraction=.05, cv_jobs=4, cpu_budget=4)
    monkeypatch.setattr(acceleration, 'cpu_count', lambda: 6)
    assert options.resources(4, adapter) == (4, 1)
    monkeypatch.setattr(acceleration, 'cpu_count', lambda: 2)
    assert options.resources(4, adapter) == (2, 1)
    assert adapter.params['thread_count'] == 2
    # Existing options/defaults and GPU serialization keep their old behavior.
    assert BatchRemovalOptions().cv_jobs == 1
    adapter.params['task_type'] = 'GPU'
    assert options.resources(4, adapter)[0] == 1


def test_four_folds_actually_overlap_in_baseline_and_trial(monkeypatch):
    from recursive_selection import acceleration
    monkeypatch.setattr(acceleration, 'cpu_count', lambda: 6)

    class ConcurrentAdapter(CVAdapter):
        def __init__(self):
            super().__init__()
            self.barrier = Barrier(4, timeout=15)
            self.lock = Lock()
            self.activity = dict(active=0, maximum=0, worker_ids=set())

        def fit(self, model, X, y, eval_set):
            with self.lock:
                self.activity['active'] += 1
                self.activity['maximum'] = max(self.activity['maximum'], self.activity['active'])
                self.activity['worker_ids'].add(get_ident())
            try:
                self.barrier.wait()
                ScriptedAdapter.fit(self, model, X, y, eval_set)
                model.rounds = 3
            finally:
                with self.lock:
                    self.activity['active'] -= 1

    class Recorder:
        def __init__(self):
            self.events, self.thread_ids = [], []

        def record(self, event):
            self.events.append(event)
            self.thread_ids.append(get_ident())

    X, y, folds = four_splits()
    adapter, recorder = ConcurrentAdapter(), Recorder()
    selector = RecursiveFeatureSelector(model_adapter=adapter, ranker=TieRanker(),
        retry=False, gate=False, min_features=3, recorder=recorder,
        quality_policy=CVF1ReferenceQualityPolicy(reference='best'),
        speed_options=BatchRemovalOptions.four_fold())
    selector.fit_cv(X, y, cv_splits=folds)
    assert adapter.activity['maximum'] == 4
    assert len(adapter.activity['worker_ids']) >= 4
    assert set(recorder.thread_ids) == {get_ident()}
    evaluations = [e for e in recorder.events if e['event'] == 'cv_evaluation_started']
    assert [e['trial_id'] for e in evaluations] == [0, 1]
    assert all(e['cv_workers'] == 4 and e['cv_threads_per_model'] == 1 for e in evaluations)
    assert all(e['gate_reference'] is None for e in recorder.events)
    assert selector.timings_['cv_fit_count'] == 8
    assert selector.gate_reference_ is selector.initial_f1_gate_ is selector.final_f1_gate_ is None
    audit_selection(selector.result_, feature_groups=selector.feature_groups_.as_dict(),
                    quality_policy=selector.quality_policy, min_groups=3)


@pytest.mark.parametrize('reference', ['full', 'best'])
@pytest.mark.parametrize('retry', [False, True])
def test_gate_reference_advances_only_after_accepted_improvement(reference, retry):
    class ImprovedAdapter(CVAdapter):
        def predict(self, model, matrix):
            if model.groups not in {(1, 2, 3, 4), (1, 2, 4), (1, 2)}:
                return np.zeros(len(matrix), dtype=int)
            prediction = np.arange(len(matrix)) % 2
            if model.groups != (1, 2, 4):
                prediction[0] = 1
            return prediction

    X, y, folds = four_splits()
    policy = CVF1ReferenceQualityPolicy(reference=reference, truncate_decimals=2)
    selector = RecursiveFeatureSelector(model_adapter=ImprovedAdapter(), ranker=TieRanker(),
        retry=retry, quality_policy=policy)
    selector.fit_cv(X, y, cv_splits=folds)
    assert selector.gate_reference_ == reference
    assert selector.initial_f1_gate_ == policy.gate_value(selector.baseline_metrics_.f1_macro)
    expected = 1. if reference == 'best' else selector.initial_f1_gate_
    assert selector.final_f1_gate_ == expected
    gate = selector.initial_f1_gate_
    for trial in selector.history_:
        assert trial.f1_reference_before == gate
        if reference == 'best' and trial.accepted:
            gate = max(gate, policy.gate_value(trial.metrics.f1_macro))
        assert trial.f1_reference_after == gate
    assert gate == selector.final_f1_gate_
    audit_selection(selector.result_, feature_groups=selector.feature_groups_.as_dict(), quality_policy=policy)


def test_best_gate_keeps_truncation_and_rejected_scores_do_not_advance_it():
    full = CVF1ReferenceQualityPolicy(reference='full', truncate_decimals=2)
    best = CVF1ReferenceQualityPolicy(reference='best', truncate_decimals=2)
    floors = MacroScores(0., 0., .68)
    improved, lower, same_bucket = (MacroScores(0., 0., f) for f in (.705, .695, .701))
    assert full.advance(floors, improved).f1_macro == .68
    raised = best.advance(floors, improved)
    assert raised.f1_macro == .70
    assert full.evaluate(lower, floors).passed
    assert not best.evaluate(lower, raised).passed
    assert best.evaluate(same_bucket, raised).passed
    assert best.advance(raised, same_bucket).f1_macro == .70


@pytest.mark.integration
@pytest.mark.parametrize('reference', ['full', 'best'])
def test_native_four_fold_matches_two_workers_and_reloads_gate(tmp_path, reference):
    pytest.importorskip('catboost')
    from sklearn.datasets import make_classification
    from sklearn.model_selection import StratifiedKFold
    X, y = make_classification(n_samples=72, n_features=4, n_informative=3,
                              n_redundant=0, n_classes=3, random_state=17)
    folds = list(StratifiedKFold(4, shuffle=True, random_state=42).split(X, y))
    selectors = []
    for options in (BatchRemovalOptions(.5, cv_jobs=2, cpu_budget=4),
                    BatchRemovalOptions.four_fold(removal_fraction=.5)):
        adapter = CatBoostAdapter(params=dict(iterations=8, depth=2, thread_count=2,
            random_seed=42, verbose=False, allow_writing_files=False),
            early_stopping_rounds=2, early_stopping_metric='accuracy')
        selector = RecursiveFeatureSelector(model_adapter=adapter, retry=False,
            quality_policy=CVF1ReferenceQualityPolicy(reference=reference, truncate_decimals=2),
            speed_options=options, min_features=2)
        selector.fit_cv(X, y, cv_splits=folds)
        audit_selection(selector.result_, feature_groups=selector.feature_groups_.as_dict(),
                        quality_policy=selector.quality_policy, min_groups=2)
        selectors.append(selector)
    a, b = selectors
    assert a.cv_evaluation_.best_rounds == b.cv_evaluation_.best_rounds
    assert a.cv_evaluation_.baseline.mean == b.cv_evaluation_.baseline.mean
    assert [s.mean for s in a.cv_evaluation_.trials] == [s.mean for s in b.cv_evaluation_.trials]
    assert a.selected_columns_ == b.selected_columns_
    assert [(t.removed_group_ids, t.accepted) for t in a.history_] == [
        (t.removed_group_ids, t.accepted) for t in b.history_]
    b.save(tmp_path/'bundle')
    restored = RecursiveFeatureSelector.load(tmp_path/'bundle')
    assert restored.gate_reference_ == b.gate_reference_ == reference
    assert restored.initial_f1_gate_ == b.initial_f1_gate_
    assert restored.final_f1_gate_ == b.final_f1_gate_
    assert restored.speed_options == b.speed_options
    np.testing.assert_array_equal(restored.predict(X), b.predict(X))
