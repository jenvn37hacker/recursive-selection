"""Optional plotting; importing this module does not import Matplotlib."""
from .importance import importance_evolution_data, plot_importance_evolution
from .retries import retry_history_data, plot_retry_history
from .charts import (
    plot_group_importance, plot_accepted_scores, plot_confusion_matrix,
    plot_processing_times, plot_metric_comparison, plot_selection_mask,
    plot_bootstrap_distribution,
)

__all__ = [
    "importance_evolution_data", "plot_importance_evolution", "retry_history_data", "plot_retry_history",
    "plot_group_importance", "plot_accepted_scores", "plot_confusion_matrix",
    "plot_processing_times", "plot_metric_comparison", "plot_selection_mask", "plot_bootstrap_distribution",
]
