"""Bootstrap summaries and explicit one-sided non-inferiority decisions."""
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NonInferiorityDecision:
    passed: bool
    margin: float
    lower_bound: float
    confidence_level: float
    alpha: float
    method: str = "one-sided percentile bootstrap confidence-bound comparison"
    null_hypothesis: str = "candidate minus reference <= -margin"
    alternative_hypothesis: str = "candidate minus reference > -margin"

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class BootstrapComparison:
    reference_score: float
    candidate_score: float
    bootstrap_differences: np.ndarray
    lower_bound: float
    confidence_level: float
    n_resamples: int
    random_state: int | None
    batch_size: int
    n_samples: int
    n_clusters: int
    clusters_per_stratum: tuple[int, ...]
    classes: tuple
    stratified: bool
    metric: str = "f1_macro"
    method: str = "paired cluster percentile bootstrap"

    def __post_init__(self):
        values = np.asarray(self.bootstrap_differences, dtype=float).copy()
        if values.shape != (self.n_resamples,) or not np.isfinite(values).all():
            raise ValueError("Bootstrap differences must be finite, one per replicate.")
        if not 0 < self.confidence_level < 1 or not np.isfinite(self.lower_bound):
            raise ValueError("A finite lower bound and confidence level in (0, 1) are required.")
        values.setflags(write=False)
        object.__setattr__(self, "bootstrap_differences", values)

    @property
    def observed_difference(self):
        return self.candidate_score - self.reference_score

    @property
    def observed_difference_pp(self):
        return 100 * self.observed_difference

    @property
    def lower_bound_pp(self):
        return 100 * self.lower_bound

    @property
    def bootstrap_differences_pp(self):
        return 100 * self.bootstrap_differences

    def noninferiority(self, margin):
        """Use an explicit margin in score units, not percentage points."""
        from .noninferiority import evaluate_noninferiority
        return evaluate_noninferiority(self, margin)

    def as_dict(self):
        """Return summary/configuration without embedding all replicates."""
        from .. import __version__
        return {
            "format_version": 1, "package_version": __version__,
            "metric": self.metric, "method": self.method,
            "difference_direction": "candidate minus reference",
            "score_unit": "0-to-1", "reported_difference_unit": "percentage points",
            "reference_score": self.reference_score, "candidate_score": self.candidate_score,
            "observed_difference_pp": self.observed_difference_pp,
            "lower_bound_pp": self.lower_bound_pp,
            "bound": "one-sided lower", "quantile_method": "linear",
            "confidence_level": self.confidence_level,
            "n_resamples": self.n_resamples, "random_state": self.random_state,
            "batch_size": self.batch_size, "n_samples": self.n_samples,
            "n_clusters": self.n_clusters, "stratified": self.stratified,
            "clusters_per_stratum": list(self.clusters_per_stratum),
            "classes": [value.item() if isinstance(value, np.generic) else value for value in self.classes],
            "scope": "Fixed predictions; no refitting or selection; approximate percentile bound, not a p-value",
        }

    def save(self, directory):
        """Export summary and replicates; overwrite these two report files only."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "bootstrap_comparison.json").write_text(
            json.dumps(self.as_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        with (directory / "bootstrap_differences.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["difference", "difference_pp"])
            writer.writerows(zip(self.bootstrap_differences, self.bootstrap_differences_pp))
        return directory
