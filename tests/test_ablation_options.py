"""Ablation contracts, selected-versus-terminal state and safe random ranking."""
from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from recursive_selection import (BatchRemovalOptions, CVF1ReferenceQualityPolicy,
    F1ReferenceQualityPolicy, RecursiveFeatureSelector, RecursiveGroupSelector)
from recursive_selection.evaluation import audit_selection
from recursive_selection.reporting import SelectionReport
from test_fixed_cv import CVAdapter
from test_selection import ScriptedAdapter, TieRanker


def data():
    X = np.tile(np.arange(1, 5, dtype=float), (24, 1))
    y = np.arange(24) % 2
    folds = [(np.flatnonzero(np.arange(24)//6 != f), np.arange(6*f, 6*(f+1))) for f in range(4)]
    return X, y, folds


@pytest.mark.parametrize('retry,shap,gate', list(product([False, True], repeat=3)))
@pytest.mark.parametrize('mode', ['feature', 'group'])
@pytest.mark.parametrize('cv', [False, True])
def test_all_combinations_and_audit(retry, shap, gate, mode, cv):
    adapter = CVAdapter() if cv else ScriptedAdapter()
    cls = RecursiveFeatureSelector if mode == 'feature' else RecursiveGroupSelector
    options = dict(retry=retry, shap=shap, gate=gate,
        ranker=TieRanker() if shap else None, speed_options=BatchRemovalOptions(.5),
        quality_policy=CVF1ReferenceQualityPolicy(reference='full') if cv else F1ReferenceQualityPolicy())
    if retry and not gate:
        with pytest.raises(ValueError, match='retry=True requires gate=True'):
            cls(model_adapter=adapter, **options)
        assert not adapter.fit_calls
        return
    X, y, folds = data()
    selector = cls(model_adapter=adapter, **options)
    kwargs = {'feature_groups': {i+1: [i] for i in range(4)}} if mode == 'group' else {}
    if cv: selector.fit_cv(X, y, cv_splits=folds, **kwargs)
    else: selector.fit(X, y, eval_set=(X, y), **kwargs)
    audit_selection(selector.result_, feature_groups=selector.feature_groups_.as_dict(),
                    quality_policy=selector.quality_policy)
    report = SelectionReport(selector.result_, feature_groups=selector.feature_groups_.as_dict())
    assert report.metadata()['search_options']['gate'] == gate
    assert report.metadata()['shap_calls'] == (len(selector.shap_history_) if shap else 0)
    assert (('SHAP share' if shap else 'Random ranking share') in report.initial_ranking)
    if not gate:
        assert report.gate_constraints.empty
        assert all(t.accepted for t in selector.history_)
        assert report.trials.Decision.eq('Visited (gate disabled)').all()
        assert len(selector.terminal_group_ids_) == 1
        assert selector.selected_trial_id_ == 0  # Full is first among equal maxima.
        assert selector.selected_columns_ == (0, 1, 2, 3)
        np.testing.assert_array_equal(selector.predict(X), y)
        if cv: assert selector.refit_rounds_ == 6
    if not retry:
        failed = set()
        for t in selector.history_:
            removed = t.removed_group_ids or (t.group_id,)
            assert not failed.intersection(removed)
            if not t.accepted and len(removed) == 1: failed.update(removed)


def test_no_retry_keeps_failed_singleton_after_other_acceptance():
    X, y, folds = data()
    selectors = []
    for retry in [True, False]:
        s = RecursiveFeatureSelector(model_adapter=CVAdapter(), ranker=TieRanker(), retry=retry,
            quality_policy=CVF1ReferenceQualityPolicy(reference='full'))
        s.fit_cv(X, y, cv_splits=folds)
        selectors.append(s)
    assert selectors[0].selected_group_ids_ == (1, 2)
    assert selectors[1].selected_group_ids_ == (1, 2, 4)
    assert [t.group_id for t in selectors[1].history_] == [4, 3, 2, 1]


def test_random_ranker_is_seeded_reset_and_never_calls_shap():
    class NoShap(CVAdapter):
        def feature_shap(self, *args):raise AssertionError('SHAP must not be called')
    X, y, folds = data()
    s = RecursiveFeatureSelector(model_adapter=NoShap(), shap=False, retry=False, gate=False,
        random_state=18, quality_policy=CVF1ReferenceQualityPolicy())
    s.fit_cv(X, y, cv_splits=folds)
    snapshots = s.shap_history_
    s.fit_cv(X, y, cv_splits=folds)
    assert s.shap_history_ == snapshots
    other = RecursiveFeatureSelector(model_adapter=NoShap(), shap=False, retry=False, gate=False,
        random_state=19, quality_policy=CVF1ReferenceQualityPolicy()).fit_cv(X, y, cv_splits=folds)
    assert other.initial_ranking_ != s.initial_ranking_
    with pytest.raises(ValueError, match='omit ranker'):
        RecursiveFeatureSelector(model_adapter=NoShap(), shap=False, ranker=TieRanker())
    with pytest.raises(ValueError, match='random ranking'):
        audit_selection(replace(s.result_, random_state=99), feature_groups=s.feature_groups_.as_dict(), quality_policy=s.quality_policy)


@pytest.mark.parametrize('cv', [False, True])
def test_no_gate_selects_intermediate_raw_best_not_terminal_or_truncated_tie(cv):
    class Intermediate(CVAdapter):
        def predict(self, model, X):
            pred = np.arange(len(X)) % 2
            if len(model.groups) != 3:pred[0] = 1
            return pred
    X, y, folds = data()
    s = RecursiveFeatureSelector(model_adapter=Intermediate(), ranker=TieRanker(), retry=False, gate=False,
        quality_policy=CVF1ReferenceQualityPolicy(truncate_decimals=0) if cv else F1ReferenceQualityPolicy())
    if cv: s.fit_cv(X, y, cv_splits=folds)
    else: s.fit(X, y, eval_set=(X, y))
    assert s.selected_group_ids_ == (1, 2, 3)
    assert s.estimator_.groups == s.selected_group_ids_
    assert s.selected_trial_id_ == 1 and s.last_trial_id_ == 3
    assert s.terminal_group_ids_ == (1,)
    assert s.final_metrics_.f1_macro == 1
    assert tuple(s.final_ranking_) == s.selected_group_ids_
    if cv: assert s.refit_rounds_ == 9 and s.estimator_.rounds == 9
    np.testing.assert_array_equal(s.predict(X), y)
    audit_selection(s.result_, feature_groups=s.feature_groups_.as_dict(),quality_policy=s.quality_policy)
    with pytest.raises(ValueError, match='selected trial|Selected trial'):
        audit_selection(replace(s.result_, selected_trial_id=3), feature_groups=s.feature_groups_.as_dict(),quality_policy=s.quality_policy)


@pytest.mark.parametrize('key,value', [('retry',1),('shap',None),('gate','false'),('random_state',True),('random_state',-1)])
def test_invalid_options_fail_before_fit(key, value):
    with pytest.raises((TypeError, ValueError)):
        RecursiveFeatureSelector(model_adapter=CVAdapter(), **{key:value})


@pytest.mark.parametrize('shap', [False, True])
def test_train_only_cv_no_gate_completes_folds_without_pruning(shap):
    from recursive_selection import CVReferenceQualityPolicy, SelectionSpeedOptions
    from test_cv_policy import CVOracle, scripted_data
    X, y = scripted_data()
    s = RecursiveFeatureSelector(model_adapter=CVOracle(), retry=False, gate=False, shap=shap,
        ranker=TieRanker() if shap else None, quality_policy=CVReferenceQualityPolicy(),
        speed_options=SelectionSpeedOptions(cv_jobs=2, cpu_budget=2, early_reject=True))
    s.fit(X, y, eval_set=(X, 1-y))
    assert s.timings_['cv_pruned_trials'] == 0
    assert all(len(v.fold_scores)==3 for v in s.cv_evaluation_.trials)
    assert s.selected_trial_id_ == 0
    audit_selection(s.result_, feature_groups=s.feature_groups_.as_dict(),quality_policy=s.quality_policy)


def test_recording_cv_no_gate_preserves_test_isolation():
    from recursive_selection import SetupRecordingSplit
    from test_recording_split import setup_metadata
    metadata = setup_metadata()
    X = np.tile(np.arange(1,5,dtype=float),(len(metadata),1))
    s = RecursiveFeatureSelector(model_adapter=CVAdapter(), retry=False, gate=False,
        shap=False, quality_policy=CVF1ReferenceQualityPolicy())
    s.fit_recording_cv(X,metadata.condition_id.to_numpy(),metadata=metadata,split_policy=SetupRecordingSplit.z24())
    assert s.split_ is not None
    assert not set(s.test_indices_) & set(s.development_indices_)
    assert s.cv_evaluation_.labels == tuple(metadata.condition_id.iloc[list(s.development_indices_)])
    audit_selection(s.result_,feature_groups=s.feature_groups_.as_dict(),quality_policy=s.quality_policy)


VALID = [(r,s,g) for r,s,g in product([False,True],repeat=3) if not (r and not g)]


@pytest.mark.integration
@pytest.mark.parametrize('backend', ['catboost','lightgbm','xgboost','sklearn_gbc'])
@pytest.mark.parametrize('retry,shap,gate',VALID)
def test_native_ablation_smoke_and_bundle(tmp_path,backend,retry,shap,gate):
    import json
    from sklearn.datasets import make_classification
    from sklearn.model_selection import StratifiedKFold
    from recursive_selection import CatBoostAdapter,LightGBMAdapter,XGBoostAdapter,SklearnGBCAdapter
    if backend != 'sklearn_gbc':pytest.importorskip(backend)
    else:pytest.importorskip('shap')
    X,y=make_classification(n_samples=72,n_features=4,n_informative=3,n_redundant=0,n_classes=3,random_state=17)
    if backend=='catboost':
        adapter=CatBoostAdapter(params=dict(iterations=8,depth=2,random_seed=42,thread_count=1,verbose=False,allow_writing_files=False),early_stopping_rounds=2)
    elif backend=='lightgbm':
        adapter=LightGBMAdapter(params=dict(n_estimators=8,max_depth=2,num_leaves=4,min_child_samples=2,n_jobs=1,verbosity=-1,random_state=42),early_stopping_rounds=2)
    elif backend=='xgboost':
        adapter=XGBoostAdapter(params=dict(n_estimators=8,max_depth=2,n_jobs=1,random_state=42),early_stopping_rounds=2)
    else:
        adapter=SklearnGBCAdapter(params=dict(n_estimators=8,max_depth=2,random_state=42),early_stopping_rounds=2)
    s=RecursiveFeatureSelector(model_adapter=adapter,retry=retry,shap=shap,gate=gate,random_state=17,
        quality_policy=CVF1ReferenceQualityPolicy(reference='full',truncate_decimals=2),
        min_features=2,speed_options=BatchRemovalOptions(.5))
    s.fit_cv(X,y,cv_splits=list(StratifiedKFold(4,shuffle=True,random_state=42).split(X,y)))
    audit_selection(s.result_,feature_groups=s.feature_groups_.as_dict(),quality_policy=s.quality_policy,min_groups=2)
    report=SelectionReport(s.result_,feature_groups=s.feature_groups_.as_dict())
    report.save(tmp_path/'reports')
    s.save(tmp_path/'model')
    saved=json.loads((tmp_path/'model/selection.json').read_text())
    assert saved['format_version']==11
    loaded=RecursiveFeatureSelector.load(tmp_path/'model',trusted=(backend=='sklearn_gbc'))
    assert loaded.result_.search_options == s.result_.search_options
    assert loaded.selected_trial_id_ == s.selected_trial_id_
    assert loaded.refit_rounds_ == s.refit_rounds_
    model = loaded.estimator_
    actual_rounds = (model.tree_count_ if backend=='catboost' else
                     model.current_iteration() if backend=='lightgbm' else
                     model.get_booster().num_boosted_rounds() if backend=='xgboost' else model.n_estimators_)
    assert actual_rounds == s.refit_rounds_
    np.testing.assert_array_equal(loaded.predict(X),s.predict(X))
    np.testing.assert_array_equal(loaded.transform(X),s.transform(X))
    if not shap:
        assert s.timings_['initial_shap_seconds']==s.timings_['shap_update_seconds']==0
        assert list((tmp_path/'reports').glob('*random_ranking_evolution.csv'))
        assert not list((tmp_path/'reports').glob('*shap_evolution.csv'))
    saved['search_options'].update(retry=True,gate=False)
    (tmp_path/'model/selection.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='retry=True requires gate=True'):
        RecursiveFeatureSelector.load(tmp_path/'model',trusted=True)


@pytest.mark.integration
@pytest.mark.parametrize('train_cv', [False,True])
def test_native_legacy_modes_random_no_gate_bundle(tmp_path,train_cv):
    from sklearn.datasets import make_classification
    from recursive_selection import CatBoostAdapter,CVReferenceQualityPolicy,SelectionSpeedOptions
    X,y=make_classification(n_samples=90,n_features=4,n_informative=3,n_redundant=0,random_state=17)
    adapter=CatBoostAdapter(params=dict(iterations=8,depth=2,random_seed=42,thread_count=1,verbose=False,allow_writing_files=False),early_stopping_rounds=2)
    s=RecursiveFeatureSelector(model_adapter=adapter,retry=False,gate=False,shap=False,min_features=2,
        quality_policy=CVReferenceQualityPolicy() if train_cv else F1ReferenceQualityPolicy(),
        speed_options=SelectionSpeedOptions(cv_jobs=2,cpu_budget=2) if train_cv else BatchRemovalOptions(.5))
    s.fit(X[:60],y[:60],eval_set=(X[60:],y[60:]))
    s.save(tmp_path/'bundle')
    loaded=RecursiveFeatureSelector.load(tmp_path/'bundle')
    assert loaded.result_.search_options == s.result_.search_options
    assert loaded.selected_trial_id_ == s.selected_trial_id_
    np.testing.assert_array_equal(loaded.predict(X),s.predict(X))
    audit_selection(loaded.result_,feature_groups=loaded.feature_groups_.as_dict(),quality_policy=loaded.quality_policy,min_groups=2)


def test_random_charts_and_ungated_recorder(tmp_path,capsys):
    from recursive_selection.plotting import plot_accepted_scores,plot_importance_evolution
    from recursive_selection.integrations.notebook import NotebookSelectionRecorder
    from matplotlib import pyplot as plt
    X,y,folds=data()
    s=RecursiveFeatureSelector(model_adapter=CVAdapter(),retry=False,gate=False,shap=False,
        quality_policy=CVF1ReferenceQualityPolicy()).fit_cv(X,y,cv_splits=folds)
    fig,ax=plot_importance_evolution(s.result_)
    assert fig.axes[-1].get_ylabel()=='Random priority'
    fig.savefig(tmp_path/'random.png');plt.close(fig)
    fig,ax=plot_accepted_scores(s.result_)
    assert 'Visited' in ax.get_title()
    assert not any('minimum' in line.get_label() for line in ax.lines)
    assert any('Selected subset' in coll.get_label() for coll in ax.collections)
    plt.close(fig)
    recorder=NotebookSelectionRecorder.__new__(NotebookSelectionRecorder)
    recorder.verbose=True;recorder.unit_name='feature'
    recorder.record(dict(event='trial_completed',gate_enabled=False,metric_kind='mean',
        metrics=dict(precision_macro=.5,recall_macro=.5,f1_macro=.5),worst_margin=0,accepted=True))
    assert 'VISITED (gate disabled)' in capsys.readouterr().out
