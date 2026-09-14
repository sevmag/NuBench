"""Unit tests for `nubench.evaluation.plotting.inelasticity`."""

from typing import Dict

import matplotlib

matplotlib.use("Agg")  # No display available/needed for tests.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nubench.evaluation.plotting import (  # noqa: E402
    plot_inelasticity_distribution_by_energy_regime,
    plot_inelasticity_distribution_by_energy_regime_comparison,
    plot_inelasticity_figure,
    plot_inelasticity_resolution,
    plot_inelasticity_resolution_comparison,
)


def _make_inelasticity_result() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bin_center": [10.0, 100.0, 1000.0],
            "p16": [0.01, 0.02, 0.01],
            "p50": [0.05, 0.08, 0.04],
            "p84": [0.15, 0.20, 0.10],
        }
    )


def test_plot_inelasticity_resolution_returns_axes() -> None:
    """The function should return a matplotlib Axes."""
    ax = plot_inelasticity_resolution(_make_inelasticity_result())
    assert isinstance(ax, plt.Axes)


def test_plot_inelasticity_resolution_draws_correct_median_line() -> None:
    """The plotted line should carry the bin_center/p50 data exactly."""
    result = _make_inelasticity_result()
    ax = plot_inelasticity_resolution(result)
    assert len(ax.lines) >= 1
    x_data, y_data = ax.lines[0].get_data()
    assert np.allclose(x_data, result["bin_center"])
    assert np.allclose(y_data, result["p50"])


def test_plot_inelasticity_resolution_draws_a_band() -> None:
    """`fill_between` should add a filled region (a Collection) to the Axes."""
    ax = plot_inelasticity_resolution(_make_inelasticity_result())
    assert len(ax.collections) >= 1


def test_plot_inelasticity_resolution_uses_log_xscale() -> None:
    """True energy spans orders of magnitude, so the x-axis should be log."""
    ax = plot_inelasticity_resolution(_make_inelasticity_result())
    assert ax.get_xscale() == "log"


def test_plot_inelasticity_resolution_reuses_provided_axes() -> None:
    """Passing an existing `ax` should draw onto it, not create a new one."""
    fig, ax = plt.subplots()
    returned_ax = plot_inelasticity_resolution(
        _make_inelasticity_result(), ax=ax
    )
    assert returned_ax is ax


def test_plot_inelasticity_resolution_sets_legend_label() -> None:
    """The `label` argument should end up on the plotted line for a legend."""
    ax = plot_inelasticity_resolution(
        _make_inelasticity_result(), label="DynEdge"
    )
    labels = [line.get_label() for line in ax.lines]
    assert "DynEdge" in labels


def _make_inelasticity_predictions_df(
    energy: np.ndarray, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    n = len(energy)
    truth = rng.uniform(0, 1, size=n)
    pred = np.clip(truth + rng.normal(scale=0.1, size=n), 0.0, 1.0)
    return pd.DataFrame({"truth": truth, "pred": pred, "energy": energy})


def test_plot_inelasticity_resolution_comparison_one_line_per_model() -> None:
    """Each model in `predictions` should get its own labelled line + band."""
    energy = np.linspace(10, 1000, 300)
    predictions = {
        "ModelA": _make_inelasticity_predictions_df(energy, seed=30),
        "ModelB": _make_inelasticity_predictions_df(energy, seed=31),
    }

    ax = plot_inelasticity_resolution_comparison(
        predictions,
        truth_col="truth",
        pred_col="pred",
        energy_col="energy",
    )

    assert len(ax.lines) == 2
    assert len(ax.collections) == 2
    labels = {line.get_label() for line in ax.lines}
    assert labels == {"ModelA", "ModelB"}


def test_plot_inelasticity_resolution_comparison_shares_bins() -> None:
    """Every model must be binned with the same shared edges.

    Same disjoint-ranges trick as energy/direction/vertex: with correctly
    shared bins (built from the combined energy range), each model must
    show NaNs where only the *other* model has data. If a model's own
    `bins=None` default were used instead of the shared `bins`, neither
    model would ever show an empty/NaN bin.
    """
    energy_low = np.linspace(10, 100, 200)
    energy_high = np.linspace(500, 1000, 200)
    predictions = {
        "Low": _make_inelasticity_predictions_df(energy_low, seed=32),
        "High": _make_inelasticity_predictions_df(energy_high, seed=33),
    }

    ax = plot_inelasticity_resolution_comparison(
        predictions,
        truth_col="truth",
        pred_col="pred",
        energy_col="energy",
        n_bins=10,
    )

    y_low = ax.lines[0].get_data()[1]
    y_high = ax.lines[1].get_data()[1]

    assert np.isnan(y_low).any()
    assert np.isnan(y_high).any()


def test_plot_inelasticity_resolution_comparison_reuses_axes() -> None:
    """Passing an existing `ax` should draw onto it, not create a new one."""
    fig, ax = plt.subplots()
    energy = np.linspace(10, 1000, 100)
    predictions = {
        "ModelA": _make_inelasticity_predictions_df(energy, seed=34)
    }

    returned_ax = plot_inelasticity_resolution_comparison(
        predictions,
        truth_col="truth",
        pred_col="pred",
        energy_col="energy",
        ax=ax,
    )

    assert returned_ax is ax


def _make_inelasticity_distribution_result() -> Dict[str, pd.DataFrame]:
    bin_left = [0.0, 0.2, 0.4, 0.6, 0.8]
    return {
        "low_energy_pred": pd.DataFrame(
            {"bin_left": bin_left, "percentage": [40, 30, 15, 10, 5]}
        ),
        "high_energy_pred": pd.DataFrame(
            {"bin_left": bin_left, "percentage": [5, 10, 15, 30, 40]}
        ),
        "low_energy_truth": pd.DataFrame(
            {"bin_left": bin_left, "percentage": [35, 30, 20, 10, 5]}
        ),
        "high_energy_truth": pd.DataFrame(
            {"bin_left": bin_left, "percentage": [5, 10, 20, 30, 35]}
        ),
    }


def test_plot_inelasticity_distribution_by_energy_regime_returns_axes() -> (
    None
):
    """The function should return a matplotlib Axes."""
    ax = plot_inelasticity_distribution_by_energy_regime(
        _make_inelasticity_distribution_result()
    )
    assert isinstance(ax, plt.Axes)


def test_plot_inelasticity_distribution_by_energy_regime_low_dashed() -> (
    None
):
    """The low-energy predicted line should carry its data and be solid."""
    result = _make_inelasticity_distribution_result()
    ax = plot_inelasticity_distribution_by_energy_regime(
        result, label="DynEdge"
    )
    lines = [
        line
        for line in ax.lines
        if line.get_label() == "DynEdge (low energy)"
    ]
    assert len(lines) == 1
    assert lines[0].get_linestyle() == "-"
    x_data, y_data = lines[0].get_data()
    assert np.allclose(x_data, result["low_energy_pred"]["bin_left"])
    assert np.allclose(y_data, result["low_energy_pred"]["percentage"])


def test_plot_inelasticity_distribution_by_energy_regime_high_dashed() -> (
    None
):
    """The high-energy predicted line should carry its data and be dashed."""
    result = _make_inelasticity_distribution_result()
    ax = plot_inelasticity_distribution_by_energy_regime(
        result, label="DynEdge"
    )
    lines = [
        line
        for line in ax.lines
        if line.get_label() == "DynEdge (high energy)"
    ]
    assert len(lines) == 1
    assert lines[0].get_linestyle() == "--"
    x_data, y_data = lines[0].get_data()
    assert np.allclose(x_data, result["high_energy_pred"]["bin_left"])
    assert np.allclose(y_data, result["high_energy_pred"]["percentage"])


def test_plot_inelasticity_distribution_by_energy_regime_truth_black() -> (
    None
):
    """Truth lines are always black, regardless of the model's `color`."""
    ax = plot_inelasticity_distribution_by_energy_regime(
        _make_inelasticity_distribution_result(),
        label="DynEdge",
        color="pink",
    )
    low_truth = [
        line for line in ax.lines if line.get_label() == "truth (low energy)"
    ]
    high_truth = [
        line
        for line in ax.lines
        if line.get_label() == "truth (high energy)"
    ]
    assert len(low_truth) == 1
    assert low_truth[0].get_linestyle() == "-"
    assert low_truth[0].get_color() == "black"
    assert len(high_truth) == 1
    assert high_truth[0].get_linestyle() == "--"
    assert high_truth[0].get_color() == "black"


def test_plot_inelasticity_distribution_by_energy_regime_no_none_label() -> (
    None
):
    """Without a `label`, predicted lines shouldn't get "None (...)" text."""
    ax = plot_inelasticity_distribution_by_energy_regime(
        _make_inelasticity_distribution_result()
    )
    labels = [line.get_label() for line in ax.lines]
    assert not any("None" in str(text) for text in labels)


def test_plot_inelasticity_distribution_by_energy_regime_no_band() -> None:
    """There's no shaded band on this plot."""
    ax = plot_inelasticity_distribution_by_energy_regime(
        _make_inelasticity_distribution_result()
    )
    assert len(ax.collections) == 0


def test_plot_inelasticity_distribution_by_energy_regime_reuses_axes() -> (
    None
):
    """Passing an existing `ax` should draw onto it, not create a new one."""
    fig, ax = plt.subplots()
    returned_ax = plot_inelasticity_distribution_by_energy_regime(
        _make_inelasticity_distribution_result(), ax=ax
    )
    assert returned_ax is ax


def _make_inelasticity_distribution_predictions_df(
    seed: int, n: int = 400
) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    return pd.DataFrame(
        {
            "pred": rng.uniform(0, 1, size=n),
            "truth": rng.uniform(0, 1, size=n),
            "energy": rng.uniform(10, 10_000, size=n),
        }
    )


def test_inelasticity_distribution_comparison_lines_per_model() -> None:
    """Each model contributes a low+high energy predicted line pair."""
    predictions = {
        "ModelA": _make_inelasticity_distribution_predictions_df(seed=10),
        "ModelB": _make_inelasticity_distribution_predictions_df(seed=11),
    }

    ax = plot_inelasticity_distribution_by_energy_regime_comparison(
        predictions,
        pred_col="pred",
        truth_col="truth",
        energy_col="energy",
        n_bins=10,
    )

    labels = [line.get_label() for line in ax.lines]
    for model_name in predictions:
        assert f"{model_name} (low energy)" in labels
        assert f"{model_name} (high energy)" in labels


def test_inelasticity_distribution_comparison_shares_bins() -> None:
    """Every model must be binned with the same shared edges.

    Same disjoint-ranges trick used throughout: with correctly shared
    bins, each model's high-energy predicted line must use identical
    "bin_left" x-values.
    """
    predictions = {
        "ModelA": _make_inelasticity_distribution_predictions_df(seed=10),
        "ModelB": _make_inelasticity_distribution_predictions_df(seed=11),
    }

    ax = plot_inelasticity_distribution_by_energy_regime_comparison(
        predictions,
        pred_col="pred",
        truth_col="truth",
        energy_col="energy",
        n_bins=10,
    )

    line_a = next(
        line
        for line in ax.lines
        if line.get_label() == "ModelA (high energy)"
    )
    line_b = next(
        line
        for line in ax.lines
        if line.get_label() == "ModelB (high energy)"
    )
    assert np.allclose(line_a.get_data()[0], line_b.get_data()[0])


def test_inelasticity_distribution_comparison_reuses_axes() -> None:
    """Passing an existing `ax` should draw onto it, not create a new one."""
    fig, ax = plt.subplots()
    predictions = {
        "ModelA": _make_inelasticity_distribution_predictions_df(seed=12)
    }

    returned_ax = plot_inelasticity_distribution_by_energy_regime_comparison(
        predictions,
        pred_col="pred",
        truth_col="truth",
        energy_col="energy",
        n_bins=10,
        ax=ax,
    )

    assert returned_ax is ax


def test_plot_inelasticity_figure_returns_fig_and_two_axes() -> None:
    """The figure should have two distinct Axes, one per panel."""
    energy = np.linspace(10, 1000, 300)
    predictions = {
        "ModelA": _make_inelasticity_predictions_df(energy, seed=10)
    }

    fig, (ax_distribution, ax_resolution) = plot_inelasticity_figure(
        predictions, truth_col="truth", pred_col="pred", energy_col="energy"
    )

    assert isinstance(fig, plt.Figure)
    assert ax_resolution is not ax_distribution


def test_plot_inelasticity_figure_left_panel_is_distribution() -> None:
    """Left panel: the linear-scale value-distribution plot."""
    energy = np.linspace(10, 1000, 300)
    predictions = {
        "ModelA": _make_inelasticity_predictions_df(energy, seed=10)
    }

    _, (ax_distribution, _) = plot_inelasticity_figure(
        predictions, truth_col="truth", pred_col="pred", energy_col="energy"
    )

    assert ax_distribution.get_xscale() == "linear"
    labels = [line.get_label() for line in ax_distribution.lines]
    assert "ModelA (low energy)" in labels


def test_plot_inelasticity_figure_right_panel_is_resolution() -> None:
    """Right panel: the log-scale residual-vs-energy plot."""
    energy = np.linspace(10, 1000, 300)
    predictions = {
        "ModelA": _make_inelasticity_predictions_df(energy, seed=10)
    }

    _, (_, ax_resolution) = plot_inelasticity_figure(
        predictions, truth_col="truth", pred_col="pred", energy_col="energy"
    )

    assert ax_resolution.get_xscale() == "log"
    labels = [line.get_label() for line in ax_resolution.lines]
    assert "ModelA" in labels
