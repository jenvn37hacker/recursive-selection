"""Validation acceptance rules. They never change group or model state."""
from dataclasses import asdict, dataclass
from math import isfinite

from .results import MacroScores, QualityDecision


@dataclass(frozen=True)
class MacroQualityPolicy:
    min_precision_macro: float = 0.87
    min_recall_macro: float = 0.87
    min_f1_macro: float = 0.87
    max_drop_from_full: float = 0.04
    tolerance: float = 1e-12

    def __post_init__(self):
        for name in ("min_precision_macro", "min_recall_macro", "min_f1_macro",
                     "max_drop_from_full"):
            value = getattr(self, name)
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1.")
        if not isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative.")

    def thresholds(self, baseline):
        return MacroScores(
            max(self.min_precision_macro, baseline.precision_macro - self.max_drop_from_full),
            max(self.min_recall_macro, baseline.recall_macro - self.max_drop_from_full),
            max(self.min_f1_macro, baseline.f1_macro - self.max_drop_from_full),
        )

    def evaluate(self, scores, thresholds):
        margins = {name: score - thresholds.as_dict()[name]
                   for name, score in scores.as_dict().items()}
        if not all(isfinite(value) for value in margins.values()):
            raise ValueError("Quality scores and thresholds must be finite.")
        failed = tuple(name for name, margin in margins.items() if margin < -self.tolerance)
        return QualityDecision(not failed, min(margins.values()), failed)

    def to_config(self):
        return asdict(self)
