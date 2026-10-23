"""Approximate confidence-bound inversion; no bootstrap-tail p-value is inferred."""
from math import isfinite
from numbers import Real

from .results import NonInferiorityDecision


def evaluate_noninferiority(result, margin):
    """Reject H0: delta <= -margin only if the one-sided lower bound exceeds -margin.

    The conclusion inherits the percentile bootstrap's approximation and the
    validity of the caller's cluster/stratum definitions and evaluation protocol.
    A failed decision means insufficient evidence, not proven inferiority.
    """
    if isinstance(margin, bool) or not isinstance(margin, Real) or not isfinite(margin) or not 0 <= margin <= 1:
        raise ValueError("margin must be finite and between 0 and 1; 0.04 means four percentage points.")
    return NonInferiorityDecision(
        passed=bool(result.lower_bound > -float(margin)),
        margin=float(margin), lower_bound=result.lower_bound,
        confidence_level=result.confidence_level, alpha=1 - result.confidence_level,
    )
