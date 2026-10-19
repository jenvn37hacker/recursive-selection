"""Small standalone example, independent of Z24 and QUGS notebook data.

Run after installing your wheel and a backend:
    python examples/classification.py --backend catboost
"""
import argparse

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from recursive_group_selection import (MacroQualityPolicy, RecursiveGroupSelector,
                                        CatBoostAdapter, XGBoostAdapter, LightGBMAdapter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["catboost", "xgboost", "lightgbm"], default="catboost")
    args = parser.parse_args()
    if args.backend == "catboost":
        adapter = CatBoostAdapter(params=dict(iterations=60, depth=3, random_seed=42,
                                  verbose=False, thread_count=1, allow_writing_files=False))
    elif args.backend == "xgboost":
        adapter = XGBoostAdapter(params=dict(n_estimators=60, max_depth=3, random_state=42, n_jobs=1))
    else:
        adapter = LightGBMAdapter(params=dict(n_estimators=60, max_depth=3, random_state=42, n_jobs=1, verbosity=-1))
    X, y = make_classification(n_samples=240, n_features=12, n_informative=7,
                                n_redundant=0, n_classes=3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    selector = RecursiveGroupSelector(
        model_adapter=adapter,
        # Loose floors for a small demonstration, not the experiment 005 policy.
        quality_policy=MacroQualityPolicy(0.3, 0.3, 0.3, max_drop_from_full=0.15),
    )
    selector.fit(X_train, y_train, eval_set=(X_val, y_val),
                 feature_groups={group + 1: list(range(3 * group, 3 * group + 3)) for group in range(4)},
                 feature_names=[f"feature_{i}" for i in range(12)])
    print("Selected groups:", selector.selected_group_ids_)
    print("Selected columns:", selector.selected_columns_)
    print("Validation macros:", selector.final_metrics_.as_dict())
    print("Trials:", len(selector.history_))
    print("SHAP calls:", len(selector.shap_history_))
    print("Stop reason:", selector.stop_reason_)


if __name__ == "__main__":
    main()
