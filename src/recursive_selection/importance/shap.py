"""Aggregate backend-normalized SHAP attributions over group groups."""
import numpy as np


class MeanAbsoluteShapRanker:
    def rank(self, adapter, model, X, y, groups):
        values = np.asarray(adapter.feature_shap(model, X, y))
        if (values.ndim != 3 or values.shape[0] != len(X)
                or values.shape[2] != X.shape[1] or values.shape[1] < 1):
            raise ValueError("Backend SHAP must have shape (samples, outputs, features).")
        if not np.isfinite(values).all():
            raise ValueError("Backend SHAP contains non-finite values.")
        absolute = np.abs(values)
        return {group: float(absolute[:, :, columns].mean())
                for group, columns in groups.items()}
