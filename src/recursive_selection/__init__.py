"""Public API; optional model libraries are imported only when used."""

from .backends import CatBoostAdapter, LightGBMAdapter, XGBoostAdapter
from .importance import MeanAbsoluteShapRanker
from .policies import MacroQualityPolicy, CVReferenceQualityPolicy
from .selector import RecursiveFeatureSelector, RecursiveGroupSelector

__version__ = "1.1.0"
__all__ = [
    "RecursiveGroupSelector",
    "RecursiveFeatureSelector",
    "MacroQualityPolicy",
    "CVReferenceQualityPolicy",
    "MeanAbsoluteShapRanker",
    "CatBoostAdapter",
    "XGBoostAdapter",
    "LightGBMAdapter",
]
