"""Task-agnostic multi-panel wrapper, shared by every reconstruction task."""

from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_multi_panel(
    predictions_by_dataset: Dict[str, Dict[str, pd.DataFrame]],
    plot_comparison_fn: Callable[..., plt.Axes],
    ncols: int = 4,
    figsize_per_panel: Tuple[float, float] = (2.5, 2.5),
    **plot_kwargs,
) -> plt.Figure:
    """Arrange one subplot per dataset, each a multi-model comparison.

    A thin, general-purpose wrapper - not specific to any one task. For
    each dataset in `predictions_by_dataset`, creates one subplot and
    calls `plot_comparison_fn` on it, passing that dataset's own
    per-model predictions dict as the first argument and `ax=` set to
    that subplot. Works with *any* of this module's `*_comparison`
    functions (e.g. `plot_energy_calibration_comparison`,
    `plot_roc_curve_comparison`, ...), since they all share the same
    shape: first argument is a `Dict[str, pd.DataFrame]` of per-model
    predictions, and they all accept an `ax` keyword.

    Args:
        predictions_by_dataset: Mapping from dataset name (used as each
            subplot's title) to that dataset's own per-model predictions
            dict - the same shape any single `*_comparison` function
            expects as its own first argument.
        plot_comparison_fn: One of this module's `*_comparison`
            functions, called once per dataset.
        ncols: Number of subplot columns. Rows are computed automatically
            from the number of datasets.
        figsize_per_panel: `(width, height)` in inches for each
            individual subplot - the overall figure size scales with the
            grid.
        **plot_kwargs: Passed through unchanged to every call of
            `plot_comparison_fn` - e.g. `truth_col=...`, `pred_col=...`,
            whatever that particular function needs beyond the
            predictions dict and `ax`.

    Returns:
        The Figure containing one subplot per dataset (any leftover grid
        slots, if the dataset count doesn't fill it exactly, are hidden
        rather than left blank-but-visible).
    """
    n_datasets = len(predictions_by_dataset)
    n_rows = -(-n_datasets // ncols)  # integer ceiling division
    fig, axes = plt.subplots(
        n_rows, ncols,
        figsize=(ncols * figsize_per_panel[0], n_rows * figsize_per_panel[1]),
        constrained_layout=True
    )
    flat_axes = np.ravel(axes)  # flatten to 1D array for consistent indexing
    for i, (dataset_name, predictions) in enumerate(
        predictions_by_dataset.items()
    ):
        plot_comparison_fn(predictions, ax=flat_axes[i], **plot_kwargs)
        flat_axes[i].set_title(dataset_name, fontsize=14)
    for j in range(n_datasets, len(flat_axes)):
        flat_axes[j].set_visible(False)  # hide unused axes
    return fig
