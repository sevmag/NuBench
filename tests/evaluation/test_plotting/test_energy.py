"""Unit tests for `nubench.evaluation.plotting.energy`."""

import matplotlib

matplotlib.use("Agg")  # No display available/needed for tests.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nubench.evaluation.plotting import (  # noqa: E402
    plot_energy_calibration,
    plot_energy_calibration_by_topology_figure,
    plot_energy_calibration_comparison,
    plot_energy_calibration_figure,
)


def _make_calibration_result() -> pd.DataFrame:
    # p50 deliberately different from bin_center everywhere, so the
    # calibration line is never confused with the "ideal" y=x diagonal.
    return pd.DataFrame(
        {
            "bin_center": [10.0, 100.0, 1000.0],
            "p16": [8.0, 90.0, 950.0],
            "p50": [12.0, 115.0, 1080.0],
            "p84": [16.0, 140.0, 1200.0],
        }
    )


def test_plot_energy_calibration_returns_axes() -> None:
    """The function should return a matplotlib Axes."""
    ax = plot_energy_calibration(_make_calibration_result())
    assert isinstance(ax, plt.Axes)


def test_plot_energy_calibration_draws_correct_median_line() -> None:
    """One plotted line should carry the bin_center/p50 data exactly."""
    result = _make_calibration_result()
    ax = plot_energy_calibration(result)
    data_lines = [
        line
        for line in ax.lines
        if np.allclose(line.get_data()[0], result["bin_center"])
        and np.allclose(line.get_data()[1], result["p50"])
    ]
    assert len(data_lines) == 1


def test_plot_energy_calibration_draws_a_band() -> None:
    """`fill_between` should add a filled region (a Collection) to the Axes."""
    ax = plot_energy_calibration(_make_calibration_result())
    assert len(ax.collections) >= 1


def test_plot_energy_calibration_uses_log_log_scale() -> None:
    """Both axes should be log - unlike the resolution plots (log-x only)."""
    ax = plot_energy_calibration(_make_calibration_result())
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"


def test_plot_energy_calibration_reuses_provided_axes() -> None:
    """Passing an existing `ax` should draw onto it, not create a new one."""
    fig, ax = plt.subplots()
    returned_ax = plot_energy_calibration(_make_calibration_result(), ax=ax)
    assert returned_ax is ax


def test_plot_energy_calibration_sets_legend_label() -> None:
    """The `label` argument should end up on the plotted line for a legend."""
    ax = plot_energy_calibration(_make_calibration_result(), label="DynEdge")
    labels = [line.get_label() for line in ax.lines]
    assert "DynEdge" in labels


def test_plot_energy_calibration_draws_ideal_line() -> None:
    """A y=x 'ideal reconstruction' reference line should also be drawn."""
    result = _make_calibration_result()
    ax = plot_energy_calibration(result)
    ideal_lines = [
        line for line in ax.lines if line.get_label() == "ideal"
    ]
    assert len(ideal_lines) == 1
    x_data, y_data = ideal_lines[0].get_data()
    assert np.allclose(x_data, y_data)


def _make_calibration_predictions_df(
    truth: np.ndarray, scale: float
) -> pd.DataFrame:
    return pd.DataFrame({"truth": truth, "pred": truth * scale})


def test_plot_energy_calibration_comparison_one_line_per_model() -> None:
    """Each model should get exactly one labelled calibration line.

    Filtering by `get_label() == model_name` naturally excludes the
    "ideal" reference line(s), which share that separate, fixed label
    rather than any model's name.
    """
    rng = np.random.default_rng(seed=60)
    truth = rng.uniform(10, 1000, size=300)
    predictions = {
        "ModelA": _make_calibration_predictions_df(truth, scale=1.1),
        "ModelB": _make_calibration_predictions_df(truth, scale=0.9),
    }

    ax = plot_energy_calibration_comparison(
        predictions, truth_col="truth", pred_col="pred"
    )

    for model_name in predictions:
        model_lines = [
            line for line in ax.lines if line.get_label() == model_name
        ]
        assert len(model_lines) == 1
    assert len(ax.collections) == 2


def test_plot_energy_calibration_comparison_dedupes_ideal_line() -> None:
    """`plot_energy_calibration` redraws the "ideal" line once per model
    (harmless on its own, since every model's own copy is identical),
    but with several models overlaid that means several duplicate
    lines. Only one should survive - as an actual Line2D removed from
    the Axes, not just filtered out of this one legend, since a multi-
    panel grid later rebuilds its own legend straight from the Axes'
    lines (see `nubench.multipanel`'s `_consolidate_legends`), bypassing
    whatever this function's own `ax.legend()` call filtered down to.
    """
    rng = np.random.default_rng(seed=61)
    truth = rng.uniform(10, 1000, size=300)
    predictions = {
        "ModelA": _make_calibration_predictions_df(truth, scale=1.1),
        "ModelB": _make_calibration_predictions_df(truth, scale=0.9),
        "ModelC": _make_calibration_predictions_df(truth, scale=1.0),
    }

    ax = plot_energy_calibration_comparison(
        predictions, truth_col="truth", pred_col="pred"
    )

    assert len([
        line for line in ax.lines if line.get_label() == "ideal"
    ]) == 1
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_labels.count("ideal") == 1


def test_plot_energy_calibration_comparison_shares_bins() -> None:
    """Every model must be binned with the same shared edges.

    Same disjoint-ranges trick as the resolution plots: with correctly
    shared bins, each model's own line must show NaNs where only the
    *other* model has data.
    """
    truth_low = np.linspace(10, 100, 200)
    truth_high = np.linspace(500, 1000, 200)
    predictions = {
        "Low": _make_calibration_predictions_df(truth_low, scale=1.0),
        "High": _make_calibration_predictions_df(truth_high, scale=1.0),
    }

    ax = plot_energy_calibration_comparison(
        predictions, truth_col="truth", pred_col="pred", n_bins=10
    )

    for model_name in predictions:
        model_line = next(
            line for line in ax.lines if line.get_label() == model_name
        )
        assert np.isnan(model_line.get_data()[1]).any()


def test_plot_energy_calibration_comparison_reuses_provided_axes() -> None:
    """Passing an existing `ax` should draw onto it, not create a new one."""
    fig, ax = plt.subplots()
    truth = np.linspace(10, 1000, 100)
    predictions = {
        "ModelA": _make_calibration_predictions_df(truth, scale=1.0)
    }

    returned_ax = plot_energy_calibration_comparison(
        predictions, truth_col="truth", pred_col="pred", ax=ax
    )

    assert returned_ax is ax


def test_plot_energy_calibration_figure_returns_fig_and_two_axes() -> None:
    """The function should return (fig, (ax_hist, ax_main))."""
    truth = np.linspace(10, 1000, 300)
    predictions = {
        "ModelA": _make_calibration_predictions_df(truth, scale=1.0)
    }

    fig, axes = plot_energy_calibration_figure(
        predictions, truth_col="truth", pred_col="pred"
    )

    assert isinstance(fig, plt.Figure)
    assert len(axes) == 2
    ax_hist, ax_main = axes
    assert isinstance(ax_hist, plt.Axes)
    assert isinstance(ax_main, plt.Axes)


def test_plot_energy_calibration_figure_top_axis_hides_yaxis() -> None:
    """The marginal histogram panel shouldn't show a y-axis at all."""
    truth = np.linspace(10, 1000, 300)
    predictions = {
        "ModelA": _make_calibration_predictions_df(truth, scale=1.0)
    }

    _, (ax_hist, _) = plot_energy_calibration_figure(
        predictions, truth_col="truth", pred_col="pred"
    )

    assert ax_hist.yaxis.get_visible() is False


def test_plot_energy_calibration_figure_top_axis_has_histograms() -> None:
    """One step histogram per model, plus one for the truth distribution."""
    truth = np.linspace(10, 1000, 300)
    predictions = {
        "ModelA": _make_calibration_predictions_df(truth, scale=1.0),
        "ModelB": _make_calibration_predictions_df(truth, scale=1.0),
    }

    _, (ax_hist, _) = plot_energy_calibration_figure(
        predictions, truth_col="truth", pred_col="pred"
    )

    # Step histograms are drawn as Polygon patches, not lines.
    assert len(ax_hist.patches) == len(predictions) + 1


def test_plot_energy_calibration_figure_bottom_axis_has_calibration() -> None:
    """The bottom panel should carry one calibration line per model."""
    truth = np.linspace(10, 1000, 300)
    predictions = {
        "ModelA": _make_calibration_predictions_df(truth, scale=1.0),
        "ModelB": _make_calibration_predictions_df(truth, scale=1.0),
    }

    _, (_, ax_main) = plot_energy_calibration_figure(
        predictions, truth_col="truth", pred_col="pred"
    )

    for model_name in predictions:
        model_lines = [
            line for line in ax_main.lines if line.get_label() == model_name
        ]
        assert len(model_lines) == 1


def test_plot_energy_calibration_figure_shares_xaxis() -> None:
    """The two panels should share the same x-axis (same view limits)."""
    truth = np.linspace(10, 1000, 300)
    predictions = {
        "ModelA": _make_calibration_predictions_df(truth, scale=1.0)
    }

    _, (ax_hist, ax_main) = plot_energy_calibration_figure(
        predictions, truth_col="truth", pred_col="pred"
    )

    assert ax_hist.get_xlim() == ax_main.get_xlim()


def test_plot_energy_calibration_figure_reuses_provided_axes() -> None:
    """Passing `ax_hist`/`ax_main` should draw onto them, not new ones."""
    fig, axes = plt.subplots(2, 1)
    truth = np.linspace(10, 1000, 300)
    predictions = {
        "ModelA": _make_calibration_predictions_df(truth, scale=1.0)
    }

    returned_fig, (returned_hist, returned_main) = (
        plot_energy_calibration_figure(
            predictions,
            truth_col="truth",
            pred_col="pred",
            ax_hist=axes[0],
            ax_main=axes[1],
        )
    )

    assert returned_fig is fig
    assert returned_hist is axes[0]
    assert returned_main is axes[1]


def _make_topology_calibration_predictions_df() -> pd.DataFrame:
    n = 100
    truth = np.tile(np.linspace(10, 1000, n), 2)
    is_track = np.array([True] * n + [False] * n)
    # Track predictions scaled 2x truth, cascade scaled 0.5x - a known,
    # easily distinguishable difference between the two columns.
    pred = np.where(is_track, truth * 2.0, truth * 0.5)
    return pd.DataFrame({"truth": truth, "pred": pred, "is_track": is_track})


def test_plot_energy_calibration_by_topology_figure_returns_2x2_axes() -> (
    None
):
    """The figure should have a 2x2 grid of Axes."""
    predictions = {"ModelA": _make_topology_calibration_predictions_df()}

    fig, axes = plot_energy_calibration_by_topology_figure(
        predictions,
        truth_col="truth",
        pred_col="pred",
        is_track_col="is_track",
    )

    assert axes.shape == (2, 2)


def test_plot_energy_calibration_by_topology_figure_columns_use_subset() -> (
    None
):
    """Each column's calibration line should reflect only its topology."""
    predictions = {"ModelA": _make_topology_calibration_predictions_df()}

    _, axes = plot_energy_calibration_by_topology_figure(
        predictions,
        truth_col="truth",
        pred_col="pred",
        is_track_col="is_track",
        n_bins=10,
    )

    track_line = next(
        line for line in axes[1, 0].lines if line.get_label() == "ModelA"
    )
    cascade_line = next(
        line for line in axes[1, 1].lines if line.get_label() == "ModelA"
    )
    track_x, track_y = track_line.get_data()
    cascade_x, cascade_y = cascade_line.get_data()
    # Track column: predictions are 2x truth -> calibration line sits
    # above y=x. Cascade column: 0.5x truth -> sits below y=x.
    assert np.all(track_y > track_x)
    assert np.all(cascade_y < cascade_x)


def test_plot_energy_calibration_by_topology_figure_shares_bins() -> None:
    """Both columns must share the same combined-range bins.

    Track events confined to a low truth range and cascade events to a
    disjoint high range - with correctly shared bins (built from the
    combined range across both topologies), each column's calibration
    line must show NaNs where only the *other* topology has data.
    """
    n = 100
    truth_low = np.linspace(10, 100, n)
    truth_high = np.linspace(500, 1000, n)
    truth = np.concatenate([truth_low, truth_high])
    is_track = np.array([True] * n + [False] * n)
    predictions = {
        "ModelA": pd.DataFrame(
            {"truth": truth, "pred": truth, "is_track": is_track}
        )
    }

    _, axes = plot_energy_calibration_by_topology_figure(
        predictions,
        truth_col="truth",
        pred_col="pred",
        is_track_col="is_track",
        n_bins=10,
    )

    track_line = next(
        line for line in axes[1, 0].lines if line.get_label() == "ModelA"
    )
    cascade_line = next(
        line for line in axes[1, 1].lines if line.get_label() == "ModelA"
    )
    assert np.isnan(track_line.get_data()[1]).any()
    assert np.isnan(cascade_line.get_data()[1]).any()
