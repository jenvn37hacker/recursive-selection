"""Restore a built-in adapter from a saved bundle; no automatic model discovery."""
from .catboost import CatBoostAdapter
from .lightgbm import LightGBMAdapter
from .xgboost import XGBoostAdapter


def adapter_from_specification(spec):
    adapters = {"catboost": CatBoostAdapter, "xgboost": XGBoostAdapter, "lightgbm": LightGBMAdapter}
    name = spec["name"]
    if name not in adapters:
        raise ValueError("Native persistence supports the three built-in backends only.")
    return adapters[name](params=spec["params"], fit_params=spec["fit_params"],
                          early_stopping_rounds=spec["early_stopping_rounds"])
