"""Unit tests for `nubench.evaluation.plotting.multi_panel`."""

from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")  # No display available/needed for tests.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nubench.evaluation.plotting import plot_multi_panel  # noqa: E402


def _fake_comparison_plot(
    predictions: Dict[str, pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    y_offset: float = 0.0,
) -> plt.Axes:
    """A minimal stand-in with the same shape as a real `*_comparison`
    function - lets `plot_multi_panel`'s tests stay decoupled from any
    one specific task.
    """
    if ax is None:
        fig, ax = plt.subplots()
    for model_name, df in predictions.items():
        ax.plot(df["x"], df["y"] + y_offset, label=model_name)
    ax.legend()
    return ax


def _make_multi_panel_predictions(
    dataset_names, model_names
) -> Dict[str, Dict[str, pd.DataFrame]]:
    return {
        dataset_name: {
            model_name: pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]})
            for model_name in model_names
        }
        for dataset_name in dataset_names
    }


def test_plot_multi_panel_one_axes_per_dataset() -> None:
    """Each dataset should get its own subplot, titled with its name."""
    predictions_by_dataset = _make_multi_panel_predictions(
        ["arca", "orca", "gvd"], ["ModelA", "ModelB"]
    )

    fig = plot_multi_panel(
        predictions_by_dataset, _fake_comparison_plot, ncols=2
    )

    titles = [ax.get_title() for ax in fig.axes if ax.get_visible()]
    assert set(titles) == {"arca", "orca", "gvd"}


def test_plot_multi_panel_calls_plot_fn_per_dataset() -> None:
    """Each subplot should actually contain that dataset's model lines."""
    predictions_by_dataset = _make_multi_panel_predictions(
        ["arca", "orca"], ["ModelA", "ModelB"]
    )

    fig = plot_multi_panel(
        predictions_by_dataset, _fake_comparison_plot, ncols=2
    )

    for ax in fig.axes:
        if ax.get_visible() and ax.get_title() in {"arca", "orca"}:
            labels = [line.get_label() for line in ax.lines]
            assert set(labels) == {"ModelA", "ModelB"}


def test_plot_multi_panel_passes_through_kwargs() -> None:
    """Extra keyword args should reach `plot_comparison_fn` unchanged."""
    predictions_by_dataset = _make_multi_panel_predictions(
        ["arca"], ["ModelA"]
    )

    fig = plot_multi_panel(
        predictions_by_dataset,
        _fake_comparison_plot,
        ncols=1,
        y_offset=10.0,
    )

    ax = fig.axes[0]
    (line,) = ax.lines
    assert np.allclose(line.get_data()[1], [10.0, 11.0, 12.0])


def test_plot_multi_panel_hides_unused_axes() -> None:
    """Leftover grid slots, when the dataset count doesn't divide evenly,
    are hidden rather than left blank-but-visible.
    """
    predictions_by_dataset = _make_multi_panel_predictions(
        ["arca", "orca", "gvd"], ["ModelA"]
    )

    fig = plot_multi_panel(
        predictions_by_dataset, _fake_comparison_plot, ncols=2
    )

    # 3 datasets in a 2-column grid -> 2x2 = 4 slots, 1 left unused.
    assert len(fig.axes) == 4
    visible = [ax for ax in fig.axes if ax.get_visible()]
    hidden = [ax for ax in fig.axes if not ax.get_visible()]
    assert len(visible) == 3
    assert len(hidden) == 1


def test_plot_multi_panel_single_dataset() -> None:
    """A single dataset should still work - the 1x1/1xN axes-array edge
    cases that `plt.subplots` returns differently from the general case.
    """
    predictions_by_dataset = _make_multi_panel_predictions(
        ["arca"], ["ModelA"]
    )

    fig = plot_multi_panel(
        predictions_by_dataset, _fake_comparison_plot, ncols=4
    )

    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 1
    assert visible[0].get_title() == "arca"
