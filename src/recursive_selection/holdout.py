"""One baseline-selected Train/Validation split, then separate Development refits.

The wrapped selector keeps its Train-only checkpoint and Validation evidence.
This object predicts with a fresh Development model. Test values are never
accepted by fit; optional Test IDs are checked only for split overlap.
"""

from numbers import Integral
from pathlib import Path
from time import perf_counter
import json

import numpy as np

from .groups import numeric_matrix, labels
from .policies import F1ReferenceQualityPolicy, CVF1ReferenceQualityPolicy


def _round_budget(adapter, model):
    rounds = adapter.best_rounds(model)
    if isinstance(rounds, bool) or not isinstance(rounds, Integral) or rounds < 1:
        raise ValueError('Checkpoint best rounds must be positive one-based integers.')
    return int(rounds)


def _identities(train, validation, test, n_train, n_validation, *, name, unique):
    if train is None and validation is None and test is None:
        return None
    if train is None or validation is None:
        raise ValueError(f'Supply both Train and Validation {name}.')
    parts = []
    for values, count in ((train, n_train), (validation, n_validation), (test, None)):
        if values is None:
            parts.append(None)
            continue
        values = np.asarray(values)
        if values.ndim != 1 or not len(values) or (count is not None and len(values) != count):
            raise ValueError(f'{name} must contain one identity per corresponding row.')
        items = values.tolist()
        if any(v is None or not isinstance(v, (str, int, float)) or isinstance(v, bool)
               or (isinstance(v, float) and not np.isfinite(v)) for v in items):
            raise ValueError(f'{name} must be finite scalar strings or numbers.')
        if unique and len(set(items)) != len(items):
            raise ValueError(f'{name} must be unique within each split.')
        parts.append(items)
    for i, left in enumerate(parts):
        for right in parts[i+1:]:
            if left is not None and right is not None and set(left) & set(right):
                raise ValueError(f'{name} overlap between Train, Validation or Test.')
    return dict(zip(('Train', 'Validation', 'Test'), parts))


class HoldoutSelection:
    """Run Proposed on one fold with the caller's baseline winner parameters.

    ``selector`` is a RecursiveFeatureSelector or RecursiveGroupSelector with a
    single-Validation F1ReferenceQualityPolicy. Existing fit/fit_cv APIs and
    their estimators are unchanged. To evaluate four independent outer splits,
    create four independent selectors/wrappers with their own winner adapters.
    Preprocessing and baseline hyperparameter selection belong to the caller.
    """

    def __init__(self, selector):
        from .selector import RecursiveFeatureSelector, RecursiveGroupSelector
        if not isinstance(selector, (RecursiveFeatureSelector, RecursiveGroupSelector)):
            raise TypeError('selector must be a recursive feature or group selector.')
        if (not isinstance(selector.quality_policy, F1ReferenceQualityPolicy)
                or isinstance(selector.quality_policy, CVF1ReferenceQualityPolicy)):
            raise ValueError('HoldoutSelection requires a single-Validation F1ReferenceQualityPolicy.')
        self.selector = selector
        self._is_fitted = False

    def fit(self, X_train, y_train, *, eval_set, full_model=None,
            feature_names=None, feature_groups=None, fold_id=None,
            train_ids=None, validation_ids=None, test_ids=None,
            train_groups=None, validation_groups=None, test_groups=None,
            test_used_for_fold_choice=False):
        """Select on Train/Validation, freeze columns, then refit on their union.

        A reused full_model MUST be the Train-only baseline checkpoint with the
        same parameters/stopping protocol, never its Development refit. Model
        training provenance cannot be inferred from estimator parameters alone.
        Recording IDs must be globally unique (e.g. condition/recording keys).
        IDs/groups are optional; when supplied, overlap is rejected before fit.
        Test IDs/groups contain metadata only; no Test X/y is accepted or scored.
        """
        self._is_fitted = False
        if not all(callable(getattr(self.selector.model_adapter, name, None))
                   for name in ('best_rounds', 'refit')):
            raise TypeError('The adapter must implement best_rounds() and refit().')
        if not isinstance(test_used_for_fold_choice, bool):
            raise TypeError('test_used_for_fold_choice must be a bool.')
        if fold_id is not None and (isinstance(fold_id, bool) or not isinstance(fold_id, (str, Integral))):
            raise ValueError('fold_id must be an integer, string or None.')
        if feature_names is None and hasattr(X_train, 'columns'):
            feature_names = list(X_train.columns)
        names = tuple(feature_names) if feature_names is not None else None
        train = numeric_matrix(X_train, name='X_train', feature_names=names)
        y_train = labels(y_train, name='y_train', n_samples=len(train))
        if not isinstance(eval_set, (tuple, list)) or len(eval_set) != 2:
            raise ValueError('eval_set must be (X_validation, y_validation).')
        validation = numeric_matrix(eval_set[0], name='X_validation',
            n_features=train.shape[1], feature_names=names)
        y_validation = labels(eval_set[1], name='y_validation', n_samples=len(validation))
        if not np.array_equal(np.unique(y_train), np.unique(y_validation)):
            raise ValueError('Train and Validation must contain the same classes.')
        ids = _identities(train_ids, validation_ids, test_ids, len(train), len(validation),
                          name='sample IDs', unique=True)
        groups = _identities(train_groups, validation_groups, test_groups, len(train), len(validation),
                             name='recording groups', unique=False)
        if ids is not None and groups is not None:
            for split in ('Train', 'Validation', 'Test'):
                if ids[split] is not None and groups[split] is not None and len(ids[split]) != len(groups[split]):
                    raise ValueError('Sample IDs and recording groups have different lengths.')
        options = dict(eval_set=(validation, y_validation), full_model=full_model, feature_names=names)
        if self.selector._selection_mode == 'group':
            options['feature_groups'] = feature_groups
        elif feature_groups is not None:
            raise ValueError('Individual-feature selection does not accept feature_groups.')
        tick = perf_counter()
        self.selector.fit(train, y_train, **options)
        selection_seconds = perf_counter() - tick
        self.full_refit_rounds_ = _round_budget(self.selector.adapter_, self.selector.full_model_)
        self.refit_rounds_ = _round_budget(self.selector.adapter_, self.selector.estimator_)
        # Freeze the selected schema before either Development fit. SHAP was
        # computed only by selector.fit on Train, never on this concatenation.
        self.selected_columns_ = tuple(self.selector.selected_columns_)
        development = np.concatenate((train, validation), axis=0)
        y_development = np.concatenate((y_train, y_validation))
        tick = perf_counter()
        self.full_model_ = self.selector.adapter_.refit(development, y_development, self.full_refit_rounds_)
        self.estimator_ = self.selector.adapter_.refit(
            development[:, self.selected_columns_], y_development, self.refit_rounds_)
        self.timings_ = dict(selection_seconds=selection_seconds, development_refit_seconds=perf_counter()-tick)
        checkpoints = (self.selector.full_model_, self.selector.estimator_)
        if self.full_model_ is self.estimator_ or any(
                model is checkpoint for model in (self.full_model_, self.estimator_) for checkpoint in checkpoints):
            raise ValueError('Development refits must be fresh models, separate from checkpoints.')
        for model, width in ((self.full_model_, train.shape[1]), (self.estimator_, len(self.selected_columns_))):
            self.selector._check_model(self.selector.adapter_, model, width, self.selector.classes_)
        self.provenance_ = dict(protocol='single_validation_then_development_refit',
            fold_id=int(fold_id) if isinstance(fold_id, Integral) else fold_id,
            train_rows=len(train), validation_rows=len(validation), development_rows=len(development),
            sample_ids=ids, recording_groups=groups,
            test_used_for_fold_choice=test_used_for_fold_choice, test_used_for_selection=False,
            test_used_for_feature_selection=False,
            independent_test_estimate=False if test_used_for_fold_choice else None,
            reused_full_checkpoint=full_model is not None,
            refit_rule='own checkpoint best rounds; no cross-fold median',
            ranking_pool='Train', gate_pool='Validation', test_evaluated=False)
        self._is_fitted = True
        return self

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError('Fit or load HoldoutSelection before prediction/save.')

    def _prediction_input(self, X, model):
        self._check_fitted()
        if model not in ('full', 'selected'):
            raise ValueError("model must be 'full' or 'selected'.")
        X = numeric_matrix(X, name='X', n_features=self.selector.n_features_in_,
                           feature_names=self.selector.feature_names_in_)
        return (self.full_model_, X) if model == 'full' else (self.estimator_, X[:, self.selected_columns_])

    def predict(self, X, *, model='selected'):
        """Predict using the frozen Full or Selected Development refit."""
        estimator, values = self._prediction_input(X, model)
        predicted = np.asarray(self.selector.adapter_.predict(estimator, values)).reshape(-1)
        if len(predicted) != len(values) or not np.isin(predicted, self.selector.classes_).all():
            raise ValueError('Backend returned invalid prediction rows or classes.')
        return predicted

    def predict_proba(self, X, *, model='selected'):
        estimator, values = self._prediction_input(X, model)
        probability = np.asarray(self.selector.adapter_.predict_proba(estimator, values))
        if (probability.shape != (len(values), len(self.selector.classes_))
                or not np.isfinite(probability).all() or (probability < 0).any()
                or not np.allclose(probability.sum(axis=1), 1.)):
            raise ValueError('Backend returned invalid class probabilities.')
        return probability

    def save(self, directory):
        """Save checkpoint evidence and separate Full/Selected Development models."""
        from .persistence import _encode, _hash
        self._check_fitted()
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError('Choose an empty/new HoldoutSelection directory.')
        self.selector.save(directory/'selection')
        files = {}
        for role, model in (('train_full', self.selector.full_model_),
                            ('development_full', self.full_model_), ('development_selected', self.estimator_)):
            destination = directory/role/self.selector.adapter_.model_filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.selector.adapter_.save_model(model, destination)
            files[role] = _hash(destination)
        for path in (directory/'selection').iterdir():
            if path.is_file():
                files['selection/'+path.name] = _hash(path)
        metadata = dict(format_version=1, protocol='single_validation_then_development_refit',
            full_refit_rounds=self.full_refit_rounds_, selected_refit_rounds=self.refit_rounds_,
            selected_columns=self.selected_columns_, provenance=self.provenance_, timings=self.timings_, files=files)
        (directory/'holdout.json').write_text(json.dumps(_encode(metadata), indent=2, allow_nan=False)+'\n')
        return directory

    @classmethod
    def load(cls, directory, *, trusted=False):
        """Restore fixed models; no fitting, ranking or Test evaluation occurs."""
        from .persistence import _decode, _hash, load_selector
        directory = Path(directory)
        metadata = _decode(json.loads((directory/'holdout.json').read_text()))
        if metadata.get('format_version') != 1 or metadata.get('protocol') != 'single_validation_then_development_refit':
            raise ValueError('Unsupported HoldoutSelection metadata.')
        # Hash the selection evidence before loading its model (including joblib).
        selection_meta = json.loads((directory/'selection/selection.json').read_text())
        from .backends import adapter_from_specification
        adapter = adapter_from_specification(_decode(selection_meta['backend']))
        expected = {'train_full', 'development_full', 'development_selected',
                    'selection/selection.json', 'selection/'+adapter.model_filename}
        if set(metadata.get('files', {})) != expected:
            raise ValueError('HoldoutSelection file manifest differs from the fixed schema.')
        for role, checksum in metadata['files'].items():
            path = directory/role if role.startswith('selection/') else directory/role/adapter.model_filename
            if _hash(path) != checksum:
                raise ValueError('HoldoutSelection file hash mismatch: '+role)
        selector = load_selector(directory/'selection', trusted=trusted)
        instance = cls(selector)
        for name in ('full_refit_rounds', 'selected_refit_rounds'):
            value = metadata[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError('Invalid HoldoutSelection refit budget.')
        instance.full_refit_rounds_ = metadata['full_refit_rounds']
        instance.refit_rounds_ = metadata['selected_refit_rounds']
        instance.selected_columns_ = tuple(metadata['selected_columns'])
        if instance.selected_columns_ != selector.selected_columns_:
            raise ValueError('HoldoutSelection columns differ from checkpoint selection.')
        for role, attribute, width in (
            ('train_full', 'full_model_', selector.n_features_in_),
            ('development_full', None, selector.n_features_in_),
            ('development_selected', None, len(instance.selected_columns_))):
            model = adapter.load_model(directory/role/adapter.model_filename, selector.classes_)
            selector._check_model(adapter, model, width, selector.classes_)
            if attribute:
                setattr(selector, attribute, model)
            else:
                setattr(instance, 'full_model_' if role == 'development_full' else 'estimator_', model)
        if (_round_budget(adapter, selector.full_model_) != instance.full_refit_rounds_
                or _round_budget(adapter, selector.estimator_) != instance.refit_rounds_):
            raise ValueError('HoldoutSelection refit budgets differ from their checkpoints.')
        instance.provenance_, instance.timings_ = metadata['provenance'], metadata['timings']
        instance._is_fitted = True
        return instance
