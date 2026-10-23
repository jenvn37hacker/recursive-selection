"""Public API; optional model libraries are imported only when used."""
from .backends import CatBoostAdapter, LightGBMAdapter, XGBoostAdapter
from .importance import MeanAbsoluteShapRanker
from .policies import MacroQualityPolicy
from .selector import RecursiveFeatureSelector, RecursiveGroupSelector

__version__ = "0.5.0"
__all__ = [
    "RecursiveGroupSelector", "RecursiveFeatureSelector", "MacroQualityPolicy", "MeanAbsoluteShapRanker",
    "CatBoostAdapter", "XGBoostAdapter", "LightGBMAdapter",
]
