from dataclasses import replace
import os
import subprocess
import sys

import numpy as np
import pytest

from recursive_selection.plotting import (
    importance_evolution_data, plot_importance_evolution, retry_history_data, plot_retry_history,
    plot_group_importance, plot_accepted_scores, plot_confusion_matrix,
    plot_processing_times, plot_metric_comparison, plot_selection_mask, plot_bootstrap_distribution,
)
from test_selection import make_selector


@pytest.fixture
def selection_result():
    X = np.tile(np.arange(1, 5, dtype=float), (12, 1))
    y = np.arange(12) % 2
    return make_selector().fit(X, y, eval_set=(X, y), feature_groups={s: [s - 1] for s in range(1, 5)}).result_


def test_plotting_import_does_not_load_matplotlib():
    subprocess.run([sys.executable, "-c",
        "import sys; import recursive_selection.plotting; assert 'matplotlib' not in sys.modules"],
        check=True, env=os.environ.copy(), capture_output=True)


def test_evolution_nan_mask_shared_scale_and_labels(selection_result):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    data = importance_evolution_data(selection_result)
    assert data["trial_ids"] == (0, 2, 3)
    assert np.isnan(data["importance"][2, 1:]).all()
    assert np.isnan(data["importance"][3, 2])
    assert np.isfinite(data["importance"][:2]).all()
    fig, ax = plot_importance_evolution(selection_result, title="Example", group_labels={1: "Channel A"})
    fig.canvas.draw()
    assert ax.get_yticklabels()[0].get_text() == "Channel A"
    assert ax.images[0].norm.vmin == 0 and ax.images[0].norm.vmax == 1
    assert ax.images[0].get_array().mask.sum() == 3
    plt.close(fig)


def test_retry_chart_data_and_empty_case(selection_result):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    rows = retry_history_data(selection_result)
    assert [r["group_id"] for r in rows] == [4, 4]
    assert [r["trial_id"] for r in rows] == [1, 3]
    assert [r["accepted"] for r in rows] == [False, True]
    fig, axes = plot_retry_history(selection_result, unit_name="sensor", group_labels={4: "S4"})
    fig.canvas.draw()
    assert axes[0].get_yticklabels()[0].get_text() == "S4"
    plt.close(fig)
    empty = replace(selection_result, history=())
    fig, axes = plot_retry_history(empty)
    fig.canvas.draw()
    assert not axes[0].axison
    plt.close(fig)


def test_full_only_history_can_be_plotted(selection_result):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    result = replace(selection_result, shap_history=selection_result.shap_history[:1])
    fig, ax = plot_importance_evolution(result)
    fig.canvas.draw()
    assert ax.images[0].get_array().shape == (4, 1)
    plt.close(fig)


def test_ranking_and_accepted_score_charts_use_recorded_values(selection_result):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plot_group_importance({1: 0.1, 2: 0.3}, group_labels={1: "A", 2: "B"})
    fig.canvas.draw()
    assert [bar.get_height() for bar in ax.patches] == [0.3, 0.1]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["B", "A"]
    plt.close(fig)
    fig, ax = plot_accepted_scores(selection_result)
    fig.canvas.draw()
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [4, 3, 2])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [100, 100, 100])
    plt.close(fig)


def test_top_importance_bars_preserve_full_count_and_names():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plot_group_importance({1: .1, 2: .3, 3: .2}, top_n=2,
        group_labels={1: "S1_mean", 2: "S1_std", 3: "S2_mean"}, total_groups=10, unit_name="feature")
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["S1_std", "S2_mean"]
    assert "3/10 features retained" in ax.get_title()
    assert "Showing top 2" in ax.get_title()
    plt.close(fig)
    with pytest.raises(ValueError, match="top_n"):
        plot_group_importance({1: .1}, top_n=0)


def test_confusion_and_metric_comparison_match_supplied_data():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plot_confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], labels=[0, 1])
    fig.canvas.draw()
    np.testing.assert_array_equal(ax.images[0].get_array(), [[1, 1], [0, 2]])
    plt.close(fig)
    fig, ax = plot_metric_comparison([0.9, 0.85], [0.88, 0.82], ["Accuracy", "F1"])
    fig.canvas.draw()
    np.testing.assert_allclose([bar.get_height() for bar in ax.patches], [90, 85, 88, 82])
    plt.close(fig)
    with pytest.raises(ValueError, match="0-to-1"):
        plot_metric_comparison([90], [88], ["Accuracy"])


def test_timing_chart_preserves_millisecond_segment_height():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plot_processing_times([1.0, 0.2], [0.001, 0.0005], ["Full", "Selected"])
    fig.canvas.draw()
    np.testing.assert_allclose([bar.get_height() for bar in ax.patches], [1, 0.2, 0.001, 0.0005])
    np.testing.assert_allclose([bar.get_y() for bar in ax.patches[2:]], [1, 0.2])
    plt.close(fig)


def test_selection_mask_matches_cells_and_rejects_wrong_labels():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plot_selection_mask([[1, 1], [0, 0]], ["mean", "std"], group_labels=["A", "B"])
    fig.canvas.draw()
    np.testing.assert_array_equal(ax.images[0].get_array(), [[1, 1], [0, 0]])
    plt.close(fig)
    with pytest.raises(ValueError, match="dimensions"):
        plot_selection_mask([[1, 1]], ["mean"])


def test_bootstrap_histogram_handles_degenerate_distribution_and_confidence():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from recursive_selection.evaluation import PairedClusterBootstrap
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    result = PairedClusterBootstrap(n_resamples=30, confidence_level=0.95).compare(
        y, y, y, np.arange(len(y)), strata=y,
    )
    fig, ax = plot_bootstrap_distribution(result)
    fig.canvas.draw()
    assert ax.get_xlim()[0] < 0 < ax.get_xlim()[1]
    assert ax.lines[0].get_xdata()[0] == 0
    assert sum(patch.get_height() for patch in ax.patches) == 30
    assert "95%" in ax.texts[0].get_text()
    plt.close(fig)
