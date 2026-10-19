"""Public API; optional model libraries are imported only when used."""
from .backends import CatBoostAdapter, LightGBMAdapter, XGBoostAdapter
from .importance import MeanAbsoluteShapRanker
from .policies import MacroQualityPolicy
from .selector import RecursiveGroupSelector

__version__ = "0.4.0"
__all__ = [
    "RecursiveGroupSelector", "MacroQualityPolicy", "MeanAbsoluteShapRanker",
    "CatBoostAdapter", "XGBoostAdapter", "LightGBMAdapter",
]
