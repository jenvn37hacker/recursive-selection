"""Single-fold selection must preserve Train-only gates and separate refits."""

from dataclasses import replace
import json

import numpy as np
import pytest

from recursive_selection import (BatchRemovalOptions, CVF1ReferenceQualityPolicy,
    F1ReferenceQualityPolicy, HoldoutSelection, RecursiveFeatureSelector, RecursiveGroupSelector)
from recursive_selection.evaluation import audit_selection
from recursive_selection.reporting import SelectionReport
from recursive_selection.results import MacroScores
from test_selection import ScriptedAdapter


class HoldoutAdapter(ScriptedAdapter):
    def __init__(self):
        super().__init__()
        self.fit_rows, self.shap_rows, self.refit_calls = [], [], []

    def fit(self, model, X, y, eval_set):
        super().fit(model, X, y, eval_set)
        self.fit_rows.append((X.copy(), eval_set[0].copy()))

    def feature_shap(self, model, X, y):
        self.shap_rows.append(X.copy())
        return np.zeros((len(X), 1, X.shape[1]))

    def best_rounds(self, model):
        return getattr(model, 'rounds', model.n_features_in_ + 2)

    def refit(self, X, y, rounds):
        self.refit_calls.append((X.copy(), y.copy(), rounds))
        model = self.new_model()
        model.groups = tuple(int(x) for x in X[0])
        model.n_features_in_, model.classes_, model.rounds = X.shape[1], np.unique(y), rounds
        return model


def data():
    return np.tile(np.arange(1, 5, dtype=float), (20, 1)), np.arange(20) % 2


def selector(adapter=None, **kwargs):
    return RecursiveFeatureSelector(model_adapter=adapter or HoldoutAdapter(),
        quality_policy=kwargs.pop('quality_policy', F1ReferenceQualityPolicy(truncate_decimals=2)),
        speed_options=BatchRemovalOptions(.5), **kwargs)


@pytest.mark.parametrize('retry,gate,shap', [(True,True,True), (False,True,True),
                                          (False,False,True), (False,True,False)])
def test_train_only_selection_and_own_selected_budget(retry, gate, shap):
    X, y = data()
    original = X.copy()
    adapter = HoldoutAdapter()
    selected = selector(adapter, retry=retry, gate=gate, shap=shap)
    run = HoldoutSelection(selected).fit(X[:12], y[:12], eval_set=(X[12:], y[12:]),
        train_ids=np.arange(12), validation_ids=np.arange(12,20), test_ids=[20,21],
        train_groups=['D1/r1']*6+['D2/r1']*6,
        validation_groups=['D1/r2']*4+['D2/r2']*4,
        test_groups=['D1/r3','D2/r3'], fold_id=1, test_used_for_fold_choice=True)
    assert len(adapter.fit_rows) == 1 + len(selected.history_)
    assert all(len(train)==12 and len(val)==8 for train,val in adapter.fit_rows)
    assert all(len(train)==12 for train in adapter.shap_rows)
    assert len(adapter.refit_calls) == 2
    assert run.full_refit_rounds_ == 6
    assert run.refit_rounds_ == len(selected.selected_columns_) + 2
    assert [c[2] for c in adapter.refit_calls] == [run.full_refit_rounds_,run.refit_rounds_]
    assert all(len(Xdev)==20 for Xdev,_,_ in adapter.refit_calls)
    assert selected.estimator_ is not run.estimator_
    assert selected.full_model_ is not run.full_model_
    assert run.provenance_['test_used_for_fold_choice'] is True
    assert run.provenance_['test_used_for_selection'] is False
    assert selected.cv_evaluation_ is None
    audit_selection(selected.result_, feature_groups=selected.feature_groups_.as_dict(),
                    quality_policy=selected.quality_policy)
    np.testing.assert_array_equal(run.predict(X, model='full'), y)
    np.testing.assert_array_equal(X, original)


def test_reuse_train_checkpoint_without_retraining_full():
    X, y = data()
    adapter = HoldoutAdapter()
    full = adapter.new_model()
    adapter.fit(full, X[:12], y[:12], (X[12:], y[12:]))
    adapter.fit_rows.clear()
    run = HoldoutSelection(selector(adapter)).fit(
        X[:12], y[:12], eval_set=(X[12:], y[12:]), full_model=full)
    assert run.selector.full_model_ is full
    assert len(adapter.fit_rows) == len(run.selector.history_)
    assert run.selector.timings_['full_training_seconds'] == 0


@pytest.mark.parametrize('kwargs,match', [
    ({'train_ids':np.arange(12), 'validation_ids':np.arange(8)}, 'overlap'),
    ({'train_ids':[0]*12, 'validation_ids':np.arange(12,20)}, 'unique'),
    ({'train_ids':np.arange(12)}, 'both'),
    ({'train_groups':['same']*12, 'validation_groups':['same']*8}, 'overlap'),
    ({'train_ids':np.arange(12), 'validation_ids':np.arange(12,20), 'test_ids':[19]}, 'overlap'),
    ({'test_used_for_fold_choice':1}, 'bool'),
])
def test_invalid_memberships_fail_before_fitting(kwargs, match):
    X, y = data()
    adapter = HoldoutAdapter()
    with pytest.raises((ValueError,TypeError), match=match):
        HoldoutSelection(selector(adapter)).fit(X[:12], y[:12], eval_set=(X[12:],y[12:]), **kwargs)
    assert not adapter.fit_rows and not adapter.shap_rows and not adapter.refit_calls


def test_does_not_accept_cv_policy_or_test_values():
    X, y = data()
    with pytest.raises(ValueError, match='single-Validation'):
        HoldoutSelection(selector(quality_policy=CVF1ReferenceQualityPolicy()))
    run = HoldoutSelection(selector())
    with pytest.raises(TypeError, match='X_test'):
        run.fit(X[:12], y[:12], eval_set=(X[12:],y[12:]), X_test=X)
    with pytest.raises(RuntimeError):
        run.predict(X)
    with pytest.raises(ValueError, match='requires gate'):
        selector(retry=True, gate=False)


@pytest.mark.parametrize('reference', ['full','best'])
def test_truncation_boundary_and_reference_policy(reference):
    policy = F1ReferenceQualityPolicy(reference=reference, truncate_decimals=2)
    scores = lambda f: MacroScores(.5,.5,f)
    floors = policy.thresholds(scores(.689989))
    assert floors.f1_macro == .68
    assert policy.evaluate(scores(.680001), floors).passed
    assert not policy.evaluate(scores(.679999), floors).passed
    assert policy.gate_value(.58) == .58
    raised = policy.advance(floors,scores(.705))
    assert raised.f1_macro == (.70 if reference=='best' else .68)
    assert policy.evaluate(scores(.69), raised).passed == (reference=='full')


@pytest.mark.parametrize('reference', ['full','best'])
def test_single_validation_report_gate_history_and_tamper(reference, tmp_path):
    X, y = data()
    selected = selector(quality_policy=F1ReferenceQualityPolicy(reference=reference,truncate_decimals=2))
    selected.fit(X[:12], y[:12], eval_set=(X[12:],y[12:]))
    report = SelectionReport(selected.result_, feature_groups=selected.feature_groups_.as_dict())
    assert report.metadata()['validation_f1_reference'] == reference
    report.save(tmp_path/'reports')
    assert (tmp_path/'reports/selection_validation_f1_reference_history.csv').is_file()
    if reference=='best':
        import matplotlib.pyplot as plt
        from recursive_selection.plotting import plot_accepted_scores
        fig,ax=plot_accepted_scores(selected.result_)
        assert any(line.get_label()=='Best Validation F1 gate so far' for line in ax.lines)
        plt.close(fig)
    damaged = replace(selected.result_, history=(replace(selected.history_[0], f1_reference_before=.1),
                                                *selected.history_[1:]))
    with pytest.raises(ValueError,match='reference transition'):
        audit_selection(damaged, feature_groups=selected.feature_groups_.as_dict(),
                        quality_policy=selected.quality_policy)


def test_group_selection_uses_same_holdout_workflow():
    X,y = data()
    grouped = RecursiveGroupSelector(model_adapter=HoldoutAdapter(),
        quality_policy=F1ReferenceQualityPolicy(), min_groups=1)
    run=HoldoutSelection(grouped).fit(X[:12],y[:12],eval_set=(X[12:],y[12:]),
                                    feature_groups={1:[0,1],2:[2,3]})
    assert run.selector._selection_mode == 'group'
    assert run.refit_rounds_ == len(run.selected_columns_)+2


@pytest.mark.parametrize('reference,gate,expected', [('full',True,2),('best',True,3),('full',False,3)])
def test_best_gate_advances_and_ungated_refit_uses_best_not_terminal(reference, gate, expected):
    class RisingThenDropping(HoldoutAdapter):
        def predict(self, model, X):
            predicted=np.arange(len(X))%2
            if model.n_features_in_ != 3:
                predicted[0]=1-predicted[0]
            return predicted
    X,y=data()
    selected=RecursiveFeatureSelector(model_adapter=RisingThenDropping(),min_features=2,
        quality_policy=F1ReferenceQualityPolicy(reference=reference,truncate_decimals=2),
        retry=False,gate=gate)
    run=HoldoutSelection(selected).fit(X[:12],y[:12],eval_set=(X[12:],y[12:]))
    assert len(run.selected_columns_)==expected
    assert run.refit_rounds_==expected+2
    if gate:
        assert selected.final_f1_gate_ == (1. if reference=='best' else selected.initial_f1_gate_)
    audit_selection(selected.result_,feature_groups=selected.feature_groups_.as_dict(),
                    quality_policy=selected.quality_policy,min_groups=2)


@pytest.mark.integration
@pytest.mark.parametrize('backend', ['catboost','xgboost','lightgbm','sklearn_gbc'])
@pytest.mark.parametrize('policy_options', [{}, {'reference':'full','truncate_decimals':2},
                                        {'reference':'best','truncate_decimals':2}])
def test_real_backends_roundtrip(tmp_path, backend, policy_options):
    pytest.importorskip('sklearn' if backend=='sklearn_gbc' else backend)
    from sklearn.datasets import make_classification
    from recursive_selection import CatBoostAdapter, XGBoostAdapter, LightGBMAdapter, SklearnGBCAdapter
    X,y=make_classification(n_samples=72,n_features=5,n_informative=3,n_redundant=0,random_state=13)
    adapters={
        'catboost':lambda:CatBoostAdapter(params=dict(iterations=8,depth=2,thread_count=1,random_seed=42,
            verbose=False,allow_writing_files=False,loss_function='Logloss')),
        'xgboost':lambda:XGBoostAdapter(params=dict(n_estimators=8,max_depth=2,n_jobs=1,random_state=42)),
        'lightgbm':lambda:LightGBMAdapter(params=dict(n_estimators=8,max_depth=2,num_leaves=4,
            min_child_samples=2,n_jobs=1,verbosity=-1,random_state=42)),
        'sklearn_gbc':lambda:SklearnGBCAdapter(params=dict(n_estimators=8,max_depth=2,random_state=42)),
    }
    if backend=='sklearn_gbc':
        pytest.importorskip('shap')
    adapter=adapters[backend]()
    selected=RecursiveFeatureSelector(model_adapter=adapter,
        quality_policy=F1ReferenceQualityPolicy(**policy_options),
        speed_options=BatchRemovalOptions(.5),min_features=3,retry=False)
    run=HoldoutSelection(selected).fit(X[:40],y[:40],eval_set=(X[40:60],y[40:60]),fold_id=2,
        train_ids=np.arange(40),validation_ids=np.arange(40,60),test_ids=np.arange(60,72))
    assert adapter.early_stopping_metric=='accuracy' and adapter.early_stopping_rounds==5
    assert run.full_refit_rounds_==adapter.best_rounds(selected.full_model_)
    assert run.refit_rounds_==adapter.best_rounds(selected.estimator_)
    path=run.save(tmp_path/'bundle')
    assert json.loads((path/'selection/selection.json').read_text())['format_version']==(12 if policy_options else 11)
    if not policy_options:
        assert json.loads((path/'selection/selection.json').read_text())['policy']=={'tolerance':1e-12}
    if backend=='sklearn_gbc':
        with pytest.raises(ValueError,match='trusted'):
            HoldoutSelection.load(path)
    restored=HoldoutSelection.load(path,trusted=backend=='sklearn_gbc')
    for model in ('full','selected'):
        np.testing.assert_array_equal(restored.predict(X[60:],model=model),run.predict(X[60:],model=model))
        np.testing.assert_allclose(restored.predict_proba(X[60:],model=model),run.predict_proba(X[60:],model=model),atol=0,rtol=0)
    assert restored.provenance_==run.provenance_
    assert restored.selector.gate_reference_==policy_options.get('reference','full')
    audit_selection(restored.selector.result_, feature_groups=restored.selector.feature_groups_.as_dict(),
                    quality_policy=restored.selector.quality_policy,min_groups=3)
    with pytest.raises(FileExistsError):
        run.save(path)
    model_path=path/'development_selected'/adapter.model_filename
    with model_path.open('ab') as stream:
        stream.write(b'corrupted')
    with pytest.raises(ValueError,match='hash mismatch'):
        HoldoutSelection.load(path,trusted=True)
