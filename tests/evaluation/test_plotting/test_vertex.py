"""Unit tests for `nubench.evaluation.plotting.vertex`."""

from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")  # No display available/needed for tests.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402

from nubench.evaluation.plotting import (  # noqa: E402
    plot_vertex_contour_by_topology,
    plot_vertex_contour_by_topology_comparison,
    plot_vertex_resolution_by_topology,
    plot_vertex_resolution_by_topology_comparison,
)


def _make_vertex_topology_result() -> Dict[str, pd.DataFrame]:
    track = pd.DataFrame(
        {
            "bin_center": [10.0, 100.0, 1000.0],
            "p16": [1.0, 2.0, 3.0],
            "p50": [5.0, 10.0, 15.0],
            "p84": [12.0, 20.0, 30.0],
        }
    )
    cascade = pd.DataFrame(
        {
            "bin_center": [10.0, 100.0, 1000.0],
            "p16": [2.0, 4.0, 6.0],
            "p50": [10.0, 20.0, 30.0],
            "p84": [24.0, 40.0, 60.0],
        }
    )
    return {"track": track, "cascade": cascade}


def test_plot_vertex_resolution_by_topology_returns_axes() -> None:
    """The function should return a matplotlib Axes."""
    ax = plot_vertex_resolution_by_topology(_make_vertex_topology_result())
    assert isinstance(ax, plt.Axes)


def test_plot_vertex_resolution_by_topology_track_is_solid() -> None:
    """The track line should carry the track data and use a solid style."""
    result = _make_vertex_topology_result()
    ax = plot_vertex_resolution_by_topology(result, label="DynEdge")
    track_lines = [
        line for line in ax.lines if line.get_label() == "DynEdge (track)"
    ]
    assert len(track_lines) == 1
    assert track_lines[0].get_linestyle() == "-"
    x_data, y_data = track_lines[0].get_data()
    assert np.allclose(x_data, result["track"]["bin_center"])
    assert np.allclose(y_data, result["track"]["p50"])


def test_plot_vertex_resolution_by_topology_cascade_is_dashed() -> None:
    """The cascade line should carry the cascade data and be dashed."""
    result = _make_vertex_topology_result()
    ax = plot_vertex_resolution_by_topology(result, label="DynEdge")
    cascade_lines = [
        line for line in ax.lines if line.get_label() == "DynEdge (cascade)"
    ]
    assert len(cascade_lines) == 1
    assert cascade_lines[0].get_linestyle() == "--"
    x_data, y_data = cascade_lines[0].get_data()
    assert np.allclose(x_data, result["cascade"]["bin_center"])
    assert np.allclose(y_data, result["cascade"]["p50"])


def test_plot_vertex_resolution_by_topology_no_none_label() -> None:
    """Without a `label`, lines shouldn't get a "None (...)" placeholder."""
    ax = plot_vertex_resolution_by_topology(_make_vertex_topology_result())
    labels = [line.get_label() for line in ax.lines]
    assert not any("None" in str(text) for text in labels)


def test_plot_vertex_resolution_by_topology_draws_no_band() -> None:
    """No shaded band, matching the paper's own vertex-vs-energy plot."""
    ax = plot_vertex_resolution_by_topology(_make_vertex_topology_result())
    assert len(ax.collections) == 0


def test_plot_vertex_resolution_by_topology_uses_log_xscale() -> None:
    """True energy spans orders of magnitude, so the x-axis should be log."""
    ax = plot_vertex_resolution_by_topology(_make_vertex_topology_result())
    assert ax.get_xscale() == "log"


def test_plot_vertex_resolution_by_topology_reuses_axes() -> None:
    """Passing an existing `ax` should draw onto it, not create a new one."""
    fig, ax = plt.subplots()
    returned_ax = plot_vertex_resolution_by_topology(
        _make_vertex_topology_result(), ax=ax
    )
    assert returned_ax is ax


def _make_vertex_topology_predictions_df(
    energy: np.ndarray, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    n = len(energy)
    x = rng.uniform(-500, 500, size=n)
    y = rng.uniform(-500, 500, size=n)
    z = rng.uniform(-500, 500, size=n)
    return pd.DataFrame(
        {
            "tx": x,
            "ty": y,
            "tz": z,
            "px": x,
            "py": y,
            "pz": z,
            "energy": energy,
            "is_track": rng.random(size=n) < 0.5,
        }
    )


def test_plot_vertex_resolution_by_topology_comparison_lines() -> None:
    """Each model contributes a track+cascade line pair."""
    energy = np.linspace(10, 1000, 300)
    predictions = {
        "ModelA": _make_vertex_topology_predictions_df(energy, seed=20),
        "ModelB": _make_vertex_topology_predictions_df(energy, seed=21),
    }

    ax = plot_vertex_resolution_by_topology_comparison(
        predictions,
        truth_x_col="tx",
        truth_y_col="ty",
        truth_z_col="tz",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        energy_col="energy",
        is_track_col="is_track",
    )

    labels = [line.get_label() for line in ax.lines]
    for model_name in predictions:
        assert f"{model_name} (track)" in labels
        assert f"{model_name} (cascade)" in labels


def test_plot_vertex_resolution_by_topology_comparison_shares_bins() -> (
    None
):
    """Every model must be binned with the same shared edges.

    Same disjoint-ranges trick as elsewhere: with correctly shared bins
    (built from the combined energy range), each model's track line must
    show NaNs where only the *other* model has data.
    """
    energy_low = np.linspace(10, 100, 200)
    energy_high = np.linspace(500, 1000, 200)
    predictions = {
        "Low": _make_vertex_topology_predictions_df(energy_low, seed=22),
        "High": _make_vertex_topology_predictions_df(energy_high, seed=23),
    }

    ax = plot_vertex_resolution_by_topology_comparison(
        predictions,
        truth_x_col="tx",
        truth_y_col="ty",
        truth_z_col="tz",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        energy_col="energy",
        is_track_col="is_track",
        n_bins=10,
    )

    for model_name in predictions:
        track_line = next(
            line
            for line in ax.lines
            if line.get_label() == f"{model_name} (track)"
        )
        assert np.isnan(track_line.get_data()[1]).any()


def test_plot_vertex_resolution_by_topology_comparison_reuses_axes() -> (
    None
):
    """Passing an existing `ax` should draw onto it, not create a new one."""
    fig, ax = plt.subplots()
    energy = np.linspace(10, 1000, 100)
    predictions = {
        "ModelA": _make_vertex_topology_predictions_df(energy, seed=24)
    }

    returned_ax = plot_vertex_resolution_by_topology_comparison(
        predictions,
        truth_x_col="tx",
        truth_y_col="ty",
        truth_z_col="tz",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        energy_col="energy",
        is_track_col="is_track",
        ax=ax,
    )

    assert returned_ax is ax


_ContourResult = Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]


def _make_vertex_contour_result() -> Dict[str, _ContourResult]:
    x = np.linspace(-5, 5, 5)
    y = np.linspace(0, 5, 5)
    X, Y = np.meshgrid(x, y)
    H_track = np.exp(-(X**2 + Y**2))
    H_cascade = np.exp(-((X - 1) ** 2 + Y**2))
    return {
        "track": (X, Y, H_track, 0.2, 1.0, 2.0),
        "cascade": (X, Y, H_cascade, 0.2, 3.0, 4.0),
    }


def test_plot_vertex_contour_by_topology_returns_axes() -> None:
    """The function should return a matplotlib Axes."""
    ax = plot_vertex_contour_by_topology(_make_vertex_contour_result())
    assert isinstance(ax, plt.Axes)


def test_plot_vertex_contour_by_topology_draws_contours_and_markers() -> (
    None
):
    """Two contours (track/cascade) and two median markers should appear."""
    ax = plot_vertex_contour_by_topology(_make_vertex_contour_result())
    assert len(ax.collections) == 4


def test_plot_vertex_contour_by_topology_marker_positions() -> None:
    """Median markers should sit at each topology's known (depth, radial)."""
    ax = plot_vertex_contour_by_topology(_make_vertex_contour_result())
    marker_positions = [
        tuple(c.get_offsets()[0])
        for c in ax.collections
        if isinstance(c, PathCollection)
    ]
    assert (1.0, 2.0) in marker_positions
    assert (3.0, 4.0) in marker_positions


def test_plot_vertex_contour_by_topology_legend_labels() -> None:
    """With a `label`, two legend-only proxy lines should carry it."""
    ax = plot_vertex_contour_by_topology(
        _make_vertex_contour_result(), label="DynEdge"
    )
    track_lines = [
        line for line in ax.lines if line.get_label() == "DynEdge (track)"
    ]
    cascade_lines = [
        line for line in ax.lines if line.get_label() == "DynEdge (cascade)"
    ]
    assert len(track_lines) == 1
    assert track_lines[0].get_linestyle() == "-"
    assert len(cascade_lines) == 1
    assert cascade_lines[0].get_linestyle() == "--"


def test_plot_vertex_contour_by_topology_no_lines_when_label_none() -> None:
    """Without a `label`, no legend-only proxy lines should be drawn."""
    ax = plot_vertex_contour_by_topology(_make_vertex_contour_result())
    assert len(ax.lines) == 0


def test_plot_vertex_contour_by_topology_reuses_axes() -> None:
    """Passing an existing `ax` should draw onto it, not create a new one."""
    fig, ax = plt.subplots()
    returned_ax = plot_vertex_contour_by_topology(
        _make_vertex_contour_result(), ax=ax
    )
    assert returned_ax is ax


def _make_vertex_contour_predictions_df(
    seed: int, n: int = 400
) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    return pd.DataFrame(
        {
            "tx": np.zeros(n),
            "ty": np.zeros(n),
            "tz": np.zeros(n),
            "px": rng.normal(scale=5.0, size=n),
            "py": rng.normal(scale=5.0, size=n),
            "pz": rng.normal(scale=5.0, size=n),
            "is_track": rng.random(size=n) < 0.5,
        }
    )


def test_plot_vertex_contour_by_topology_comparison_legend_labels() -> None:
    """Each model should get its own labelled track/cascade legend entries."""
    predictions = {
        "ModelA": _make_vertex_contour_predictions_df(seed=10),
        "ModelB": _make_vertex_contour_predictions_df(seed=11),
    }

    ax = plot_vertex_contour_by_topology_comparison(
        predictions,
        truth_x_col="tx",
        truth_y_col="ty",
        truth_z_col="tz",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        is_track_col="is_track",
        bins=10,
    )

    labels = [line.get_label() for line in ax.lines]
    for model_name in predictions:
        assert f"{model_name} (track)" in labels
        assert f"{model_name} (cascade)" in labels


def test_plot_vertex_contour_by_topology_comparison_draws_per_model() -> (
    None
):
    """Each model should contribute 2 contours + 2 median markers."""
    predictions = {
        "ModelA": _make_vertex_contour_predictions_df(seed=10),
        "ModelB": _make_vertex_contour_predictions_df(seed=11),
    }

    ax = plot_vertex_contour_by_topology_comparison(
        predictions,
        truth_x_col="tx",
        truth_y_col="ty",
        truth_z_col="tz",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        is_track_col="is_track",
        bins=10,
    )

    assert len(ax.collections) == 4 * len(predictions)


def test_plot_vertex_contour_by_topology_comparison_reuses_axes() -> None:
    """Passing an existing `ax` should draw onto it, not create a new one."""
    fig, ax = plt.subplots()
    predictions = {"ModelA": _make_vertex_contour_predictions_df(seed=12)}

    returned_ax = plot_vertex_contour_by_topology_comparison(
        predictions,
        truth_x_col="tx",
        truth_y_col="ty",
        truth_z_col="tz",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        is_track_col="is_track",
        bins=10,
        ax=ax,
    )

    assert returned_ax is ax
