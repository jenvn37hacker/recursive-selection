"""Explicit prediction measurement and aggregation of existing pipeline timings."""
from dataclasses import dataclass
from numbers import Integral
from time import perf_counter

import numpy as np

from ..reporting.export import save_reports


@dataclass(frozen=True)
class PredictionTiming:
    seconds: tuple[float, ...]
    warmup: int
    n_samples: int

    @property
    def median_seconds(self):
        return float(np.median(self.seconds))


def benchmark_prediction(predict, X, *, repeats=1, warmup=0):
    """Opt-in extra prediction calls; this function never fits or computes metrics."""
    if (isinstance(repeats, bool) or not isinstance(repeats, Integral) or repeats < 1
            or isinstance(warmup, bool) or not isinstance(warmup, Integral) or warmup < 0 or len(X) < 1):
        raise ValueError("Use positive repeats/sample count and nonnegative warmup.")
    for _ in range(warmup):
        predict(X)
    times = []
    for _ in range(repeats):
        tick = perf_counter()
        predict(X)
        times.append(perf_counter() - tick)
    return PredictionTiming(tuple(times), int(warmup), len(X))


class ProcessingBenchmark:
    """Aggregate recorded single-run label-inference and feature timings.

    No additional prediction, warmup or feature extraction occurs here.
    Feature time includes exactly the preparation the caller chose to measure.
    Batch averages are not individual-request latency measurements.
    """
    @classmethod
    def from_evaluations(cls, *, reference, candidate, reference_feature_times,
                         candidate_feature_times, reference_groups, candidate_groups,
                         unit_name="group", reference_label=None, candidate_label=None):
        import pandas as pd
        from ..evaluation.comparison import compare_evaluations
        from ..reporting.export import safe_name
        safe_name(unit_name)
        compare_evaluations(reference=reference, candidate=candidate)
        for count in (reference_groups, candidate_groups):
            if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
                raise ValueError("Group counts must be positive integers.")
        rows = []
        for evaluation, feature_times, count, label in (
            (reference, reference_feature_times, reference_groups, reference_label or f"Full {unit_name}s"),
            (candidate, candidate_feature_times, candidate_groups, candidate_label or f"Selected {unit_name}s")):
            if set(feature_times) != set(evaluation.splits):
                raise ValueError("Feature timings must cover exactly the evaluation splits.")
            for name, split in evaluation.splits.items():
                feature, inference = float(feature_times[name]), split.inference_seconds
                if any(not np.isfinite(t) or t < 0 for t in (feature, inference)):
                    raise ValueError("Timings must be finite and nonnegative.")
                if count > split.n_features:
                    raise ValueError("Group count cannot exceed input feature count.")
                total, samples = feature + inference, len(split.y_true)
                rows.append({"Model": label, "Split": name, "Samples": samples,
                    unit_name.title() + "s": int(count), "Features": split.n_features,
                    "Feature extraction (s)": feature, "Model inference (s)": inference,
                    "Model inference (ms)": inference * 1000, "Total processing (s)": total,
                    "Batch average (ms/sample)": total * 1000 / samples})
        obj = cls()
        obj.metrics = pd.DataFrame(rows)
        obj.method = {"repeats": 1, "warmup": 0, "inference": "label prediction only",
                      "feature_time": "caller-supplied extraction/preprocessing duration",
                      "excluded": ["probability prediction", "metrics", "reporting", "hardware collection"],
                      "average": "batch processing time divided by sample count"}
        return obj

    def save(self, directory, *, prefix="benchmark", environment=None, overwrite=False):
        from ..reporting.export import safe_name
        safe_name(prefix)
        tables = {f"{prefix}_processing_times.csv": self.metrics}
        if environment is not None:
            tables[f"{prefix}_hardware.csv"] = environment.table
        return save_reports(directory, tables=tables,
            documents={f"{prefix}_benchmark_method.json": self.method},
            prefix=prefix + "_benchmark", overwrite=overwrite)
