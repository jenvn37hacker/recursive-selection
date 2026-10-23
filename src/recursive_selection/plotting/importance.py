"""Visualize recorded accepted configurations without recomputing explanations."""
import numpy as np


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('Install recursive-selection[plot] to draw charts.') from exc
    return plt


def importance_evolution_data(result):
    """Return original group IDs, accepted trial IDs, counts and an NaN-padded matrix."""
    snapshots = result.shap_history
    if not snapshots:
        raise ValueError("At least the full-model importance snapshot is required.")
    ids = tuple(snapshots[0].group_ids)
    matrix = np.full((len(ids), len(snapshots)), np.nan)
    positions = {group: index for index, group in enumerate(ids)}
    for column, snapshot in enumerate(snapshots):
        values = dict(snapshot.importance)
        if set(values) != set(snapshot.group_ids) or not set(values).issubset(positions):
            raise ValueError("Importance snapshots do not match their group schemas.")
        for group, value in values.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError("Importance values must be finite and nonnegative.")
            matrix[positions[group], column] = value
    return {
        "group_ids": ids, "trial_ids": tuple(s.trial_id for s in snapshots),
        "group_counts": tuple(len(s.group_ids) for s in snapshots), "importance": matrix,
    }


def plot_importance_evolution(result, *, title=None, group_labels=None, unit_name="group",
                              value_label="Mean absolute importance", figsize=None, ax=None):
    """Return (Figure, Axes); never call show(), savefig(), fit() or a ranker.

    Values share one linear scale across configurations. Missing groups are NaN,
    not zero. The caller supplies attribution-specific labels for custom rankers.
    """
    plt = _pyplot()
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    data = importance_evolution_data(result)
    values = data["importance"]
    labels = group_labels or {}
    finite = values[np.isfinite(values)]
    vmax = float(finite.max()) if len(finite) else 0.0
    cmap = plt.get_cmap("YlGnBu").with_extremes(bad="#b0b0b0")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (max(12, 0.55 * values.shape[1]), max(5, 0.38 * len(values) + 1)))
    else:
        fig = ax.figure
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap,
                      norm=Normalize(vmin=0, vmax=vmax if vmax > 0 else 1),
                      aspect="auto", interpolation="nearest")
    removed = np.zeros_like(values, dtype=bool)
    removed[:, 1:] = np.isfinite(values[:, :-1]) & np.isnan(values[:, 1:])
    rows, columns = np.where(removed)
    ax.scatter(columns, rows, marker="x", s=65, linewidths=2, color="#991b1b", zorder=4)
    for row, column in np.argwhere(np.isnan(values) & ~removed):
        ax.text(column, row, "NaN", ha="center", va="center", fontsize=4.5, color="#374151")
    ax.set_yticks(range(len(data["group_ids"])), [labels.get(g, str(g)) for g in data["group_ids"]])
    ax.set_xticks(range(len(data["trial_ids"])), [
        f"{'Full' if trial == 0 else 'Trial ' + str(trial)}\n{count} {unit_name}s"
        for trial, count in zip(data["trial_ids"], data["group_counts"])
    ], rotation=65, ha="right")
    ax.set_ylabel(f"Original {unit_name} ID")
    ax.set_xlabel("Accepted model configuration (trial number; rejected trials omitted)")
    ax.set_title((title or "Group Importance During Recursive Elimination") +
                 "\nFull model and accepted configurations | Shared scale, no normalization")
    fig.colorbar(image, ax=ax, pad=0.02, label=value_label)
    ax.legend(handles=[
        Patch(facecolor="#b0b0b0", label=f"Removed {unit_name} (NaN: importance unavailable)"),
        Line2D([], [], color="#991b1b", marker="x", linestyle="None", markersize=8,
               markeredgewidth=2, label="Removed at this accepted trial"),
    ], loc="upper left", bbox_to_anchor=(0, -0.23), frameon=False)
    fig.tight_layout()
    return fig, ax
