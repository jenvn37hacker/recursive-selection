"""Public API; optional model libraries are imported only when used."""

from .acceleration import SelectionSpeedOptions, BatchRemovalOptions
from .backends import CatBoostAdapter, LightGBMAdapter, XGBoostAdapter, SklearnGBCAdapter
from .importance import MeanAbsoluteShapRanker
from .policies import (
    MacroQualityPolicy,
    CVReferenceQualityPolicy,
    F1ReferenceQualityPolicy,
    CVF1ReferenceQualityPolicy,
)
from .selector import RecursiveFeatureSelector, RecursiveGroupSelector
from .recording_split import SetupRecordingSplit, RecordingSplit
from .holdout import HoldoutSelection

__version__ = "1.10.0"
__all__ = [
    "RecursiveGroupSelector",
    "RecursiveFeatureSelector",
    "MacroQualityPolicy",
    "CVReferenceQualityPolicy",
    "SelectionSpeedOptions",
    "BatchRemovalOptions",
    "F1ReferenceQualityPolicy",
    "CVF1ReferenceQualityPolicy",
    "MeanAbsoluteShapRanker",
    "CatBoostAdapter",
    "XGBoostAdapter",
    "LightGBMAdapter",
    "SklearnGBCAdapter",
    "SetupRecordingSplit",
    "RecordingSplit",
    "HoldoutSelection",
]
