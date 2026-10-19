"""Sequential, validation-constrained elimination of complete feature groups."""
from copy import deepcopy
from numbers import Integral
from time import perf_counter

import numpy as np

from .groups import FeatureGroups, labels, numeric_matrix
from .importance import MeanAbsoluteShapRanker
from .policies import MacroQualityPolicy
from .results import AcceptedState, MacroScores, SelectionResult, ShapSnapshot, TrialRecord


class RecursiveGroupSelector:
    """Select feature groups with fresh refits and configuration-specific retries.

    Supply a model_adapter explicitly. Test data must not be passed as eval_set.
    """

    def __init__(self, *, model_adapter, ranker=None,
                 quality_policy=None, min_groups=1, recorder=None):
        if model_adapter is None:
            raise ValueError("model_adapter is required.")
        if isinstance(min_groups, bool) or not isinstance(min_groups, Integral) or min_groups < 1:
            raise ValueError("min_groups must be a positive integer.")
        self.model_adapter = model_adapter
        self.ranker = ranker if ranker is not None else MeanAbsoluteShapRanker()
        self.quality_policy = quality_policy if quality_policy is not None else MacroQualityPolicy()
        self.min_groups = int(min_groups)
        self.recorder = recorder
        self._is_fitted = False

    def _emit(self, kind, **payload):
        if self.recorder is not None:
            self.recorder.record(deepcopy({"event": kind, **payload}))

    def _rank(self, adapter, model, X, y, feature_groups, group_ids):
        scores = dict(self.ranker.rank(adapter, model, X, y, feature_groups.relative(group_ids)))
        if set(scores) != set(group_ids):
            raise ValueError("Ranker must return exactly the retained group IDs.")
        scores = {group: float(scores[group]) for group in group_ids}
        if any(not np.isfinite(value) or value < 0 for value in scores.values()):
            raise ValueError("Ranker must return finite nonnegative group importance.")
        return scores

    @staticmethod
    def _check_model(adapter, model, n_features, expected_classes=None):
        if adapter.feature_count(model) != n_features:
            raise ValueError("Fitted model feature count differs from the candidate schema.")
        classes = np.asarray(adapter.classes(model)).reshape(-1)
        if len(classes) < 2 or len(np.unique(classes)) != len(classes):
            raise ValueError("A classification model with at least two distinct classes is required.")
        if expected_classes is not None and not np.array_equal(classes, expected_classes):
            raise ValueError("Candidate model classes/order differ from the full model.")
        return classes

    def fit(self, X_train, y_train, *, eval_set, feature_groups, feature_names=None, full_model=None):
        """Fit the entire search. Existing estimators/inputs are never retrained in place.

        feature_names is optional for arrays and inferred for named DataFrames.
        full_model must already be fitted with the same input columns, labels,
        model parameters and training protocol. The caller owns its provenance.
        """
        self._is_fitted = False
        started = perf_counter()
        if feature_names is None and hasattr(X_train, "columns"):
            feature_names = list(X_train.columns)
        names = tuple(feature_names) if feature_names is not None else None
        X = numeric_matrix(X_train, name="X_train", feature_names=names)
        if names is not None:
            if len(names) != X.shape[1] or any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
                raise ValueError("feature_names must be unique strings, one per original column.")
        y = labels(y_train, name="y_train", n_samples=len(X))
        if not isinstance(eval_set, (tuple, list)) or len(eval_set) != 2:
            raise ValueError("eval_set must be the pair (X_val, y_val).")
        X_val = numeric_matrix(eval_set[0], name="X_val", n_features=X.shape[1], feature_names=names)
        y_val = labels(eval_set[1], name="y_val", n_samples=len(X_val))
        groups = FeatureGroups(feature_groups, X.shape[1])
        if self.min_groups > len(groups.group_ids):
            raise ValueError("min_groups exceeds the number of supplied feature groups.")
        train_classes = np.unique(y)
        if len(train_classes) < 2 or not np.isin(y_val, train_classes).all():
            raise ValueError("Training needs at least two classes; validation labels must be known.")
        adapter = self.model_adapter
        timings = {"full_training_seconds": 0.0, "baseline_prediction_seconds": 0.0,
                   "initial_shap_seconds": 0.0, "trial_training_seconds": 0.0,
                   "validation_prediction_seconds": 0.0, "shap_update_seconds": 0.0}
        self._emit("run_started", group_ids=groups.group_ids)
        if full_model is None:
            full_model = adapter.new_model()
            tick = perf_counter()
            adapter.fit(full_model, X, y, (X_val, y_val))
            timings["full_training_seconds"] = perf_counter() - tick
        else:
            adapter.validate_full_model(full_model)
        classes = self._check_model(adapter, full_model, X.shape[1])
        if set(classes.tolist()) != set(train_classes.tolist()):
            raise ValueError("full_model class labels differ from the supplied training labels.")
        tick = perf_counter()
        baseline = MacroScores.from_predictions(y_val, adapter.predict(full_model, X_val), classes)
        timings["baseline_prediction_seconds"] = perf_counter() - tick
        floors = self.quality_policy.thresholds(baseline)
        if not self.quality_policy.evaluate(baseline, floors).passed:
            raise ValueError("Full model does not meet the fixed validation floors; selection did not start.")
        current_model = full_model
        current_groups = groups.group_ids
        current_scores = baseline
        failed = set()
        attempted = set()
        tick = perf_counter()
        ranking = self._rank(adapter, current_model, X, y, groups, current_groups)
        timings["initial_shap_seconds"] = perf_counter() - tick
        initial_ranking = dict(ranking)
        history = []
        accepted = [AcceptedState(0, 0, current_groups, baseline)]
        snapshots = [ShapSnapshot(0, 0, current_groups, tuple(ranking.items()))]
        self._emit("initialized", baseline=baseline.as_dict(), floors=floors.as_dict(), ranking=ranking)
        loop_started = perf_counter()
        configuration_id = 0
        while True:
            if len(current_groups) <= self.min_groups:
                stop_reason = "min_groups_reached"
                break
            eligible = [group for group in current_groups if group not in failed]
            if not eligible:
                stop_reason = "all_candidates_rejected"
                break
            group = min(eligible, key=lambda key: (ranking[key], -key))
            proposed = tuple(s for s in current_groups if s != group)
            columns = groups.columns(proposed)
            X_candidate = X[:, columns]
            X_val_candidate = X_val[:, columns]
            trial_id = len(history) + 1
            trial_model = adapter.new_model()
            if trial_model is current_model or trial_model is full_model:
                raise ValueError("Backend must return a fresh model for every trial.")
            self._emit("trial_started", trial_id=trial_id, configuration_id=configuration_id,
                       group_id=group, groups_before=current_groups, proposed_groups=proposed)
            tick = perf_counter()
            # Training/prediction errors propagate; they are never metric rejections.
            adapter.fit(trial_model, X_candidate, y, (X_val_candidate, y_val))
            train_seconds = perf_counter() - tick
            timings["trial_training_seconds"] += train_seconds
            self._check_model(adapter, trial_model, len(columns), classes)
            tick = perf_counter()
            predictions = adapter.predict(trial_model, X_val_candidate)
            predict_seconds = perf_counter() - tick
            timings["validation_prediction_seconds"] += predict_seconds
            scores = MacroScores.from_predictions(y_val, predictions, classes)
            decision = self.quality_policy.evaluate(scores, floors)
            changes = MacroScores(*(scores.as_dict()[name] - baseline.as_dict()[name]
                                    for name in baseline.as_dict()))
            trial = TrialRecord(trial_id, configuration_id, group, current_groups, proposed,
                                ranking[group], scores, changes, decision.passed,
                                decision.worst_margin, decision.failed_metrics,
                                train_seconds, predict_seconds)
            history.append(trial)
            attempted.add(group)
            self._emit("trial_completed", **trial.as_dict())
            if not decision.passed:
                failed.add(group)
                continue
            current_groups = proposed
            current_model = trial_model
            current_scores = scores
            failed.clear()
            configuration_id += 1
            accepted.append(AcceptedState(configuration_id, trial_id, current_groups, scores))
            tick = perf_counter()
            ranking = self._rank(adapter, current_model, X_candidate, y, groups, current_groups)
            timings["shap_update_seconds"] += perf_counter() - tick
            snapshots.append(ShapSnapshot(configuration_id, trial_id, current_groups, tuple(ranking.items())))
            self._emit("configuration_accepted", configuration_id=configuration_id,
                       trial_id=trial_id, group_ids=current_groups, ranking=ranking)
        timings["elimination_loop_seconds"] = perf_counter() - loop_started
        timings["total_fit_seconds"] = perf_counter() - started
        columns = groups.columns(current_groups)
        selected_names = tuple(names[c] for c in columns) if names is not None else None
        result = SelectionResult(current_model, current_groups, columns, selected_names, baseline,
                                 floors, current_scores, tuple(history), tuple(accepted),
                                 tuple(snapshots), dict(timings), stop_reason)
        self._emit("run_finished", group_ids=current_groups, stop_reason=stop_reason,
                   trial_count=len(history), shap_calls=len(snapshots))
        self.adapter_ = adapter
        self.full_model_ = full_model
        self.classes_ = classes.copy()
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = names
        self.feature_groups_ = groups
        self.initial_ranking_ = initial_ranking
        self.final_ranking_ = dict(ranking)
        self.attempted_group_ids_ = tuple(sorted(attempted))
        self.rejected_group_ids_ = tuple(sorted(failed))
        self._set_result(result)
        return self

    def _set_result(self, result):
        self.result_ = result
        self.estimator_ = result.estimator
        self.selected_group_ids_ = result.selected_group_ids
        self.selected_columns_ = result.selected_columns
        self.selected_feature_names_ = result.selected_feature_names
        self.baseline_metrics_ = result.baseline_metrics
        self.effective_floors_ = result.effective_floors
        self.final_metrics_ = result.final_metrics
        self.history_ = result.history
        self.accepted_history_ = result.accepted_history
        self.shap_history_ = result.shap_history
        self.timings_ = dict(result.timings)
        self.stop_reason_ = result.stop_reason
        self._is_fitted = True

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call fit() successfully or load() a saved selector first.")

    def transform(self, X):
        """Select retained columns from the complete original feature schema."""
        self._check_fitted()
        matrix = numeric_matrix(X, name="X", n_features=self.n_features_in_,
                                feature_names=self.feature_names_in_)
        return matrix[:, self.selected_columns_]

    def predict(self, X):
        """Predict from the full original input schema, selecting columns internally."""
        selected = self.transform(X)
        predictions = np.asarray(self.adapter_.predict(self.estimator_, selected)).reshape(-1)
        if len(predictions) != len(selected) or not np.isin(predictions, self.classes_).all():
            raise ValueError("Backend returned invalid prediction rows or classes.")
        return predictions

    def history_records(self):
        self._check_fitted()
        return [record.as_dict() for record in self.history_]

    def save(self, directory):
        """Save native model + explicit schema/history; does not pickle Python objects."""
        from .persistence import save_selector
        self._check_fitted()
        return save_selector(self, directory)

    @classmethod
    def load(cls, directory):
        from .persistence import load_selector
        return load_selector(directory)
