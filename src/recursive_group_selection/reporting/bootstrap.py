"""Report compatibility exports around an existing bootstrap comparison."""
import numpy as np

from .export import save_reports


class BootstrapReport:
    def __init__(self, result, *, reference, candidate, split="Test", margin=None,
                 scope="Fixed fitted models and existing evaluation split", recording_overlap=None):
        from .. import __version__
        actual = [reference[split].metrics["F1 macro"], candidate[split].metrics["F1 macro"]]
        if (not np.allclose([result.reference_score, result.candidate_score], actual, atol=1e-12, rtol=0)
                or result.n_samples != len(reference[split].y_true)
                or not np.array_equal(reference[split].y_true, candidate[split].y_true)):
            raise ValueError("Bootstrap scores/rows differ from the compared evaluations.")
        self.result = result
        bound_name = f"one_sided_{100 * result.confidence_level:g}_lower_pp"
        self.summary = {"method": result.method, "library_version": __version__,
            "replicates": result.n_resamples, "seed": result.random_state,
            "test_segments": result.n_samples, "recording_clusters": result.n_clusters,
            "train_test_recording_overlap": recording_overlap,
            "full_f1": result.reference_score, "selected_f1": result.candidate_score,
            "observed_change_pp": result.observed_difference_pp, bound_name: result.lower_bound_pp,
            "confidence_level": result.confidence_level, "scope": scope}
        self.decision = result.noninferiority(margin) if margin is not None else None

    def save(self, directory, *, overwrite=False):
        import pandas as pd
        documents = {"bootstrap_summary.json": self.summary}
        if self.decision is not None:
            documents["noninferiority_decision.json"] = {**self.decision.as_dict(), "scope": self.summary["scope"]}
        return save_reports(directory,
            tables={"f1_bootstrap_differences.csv": pd.DataFrame({"Delta F1 macro": self.result.bootstrap_differences})},
            documents=documents, prefix="bootstrap_report", overwrite=overwrite)
