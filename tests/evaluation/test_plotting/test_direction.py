"""Unit tests for `nubench.evaluation.plotting.direction`."""

from typing import Dict

import matplotlib

matplotlib.use("Agg")  # No display available/needed for tests.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nubench.evaluation.plotting import (  # noqa: E402
    plot_direction_error_distribution_by_topology,
    plot_direction_error_distribution_by_topology_comparison,
    plot_direction_figure,
    plot_direction_resolution_by_topology,
    plot_direction_resolution_by_topology_comparison,
)


def _make_topology_result(
    include_muon: bool = True,
) -> Dict[str, pd.DataFrame]:
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
            "p16": [0.5, 1.0, 1.5],
            "p50": [2.0, 3.0, 4.0],
            "p84": [4.0, 6.0, 8.0],
        }
    )
    result = {"track": track, "cascade": cascade}
    if include_muon:
        result["muon"] = pd.DataFrame(
            {
                "bin_center": [10.0, 100.0, 1000.0],
                "p16": [0.1, 0.2, 0.3],
                "p50": [0.5, 0.6, 0.7],
                "p84": [1.0, 1.1, 1.2],
            }
        )
    return result


def test_plot_direction_resolution_by_topology_returns_axes() -> None:
    """The function should return a matplotlib Axes."""
    ax = plot_direction_resolution_by_topology(_make_topology_result())
    assert isinstance(ax, plt.Axes)


def test_plot_direction_resolution_by_topology_track_is_solid() -> None:
    """The track line should carry the track data and use a solid style."""
    result = _make_topology_result()
    ax = plot_direction_resolution_by_topology(result, label="DynEdge")
    track_lines = [
        line for line in ax.lines if line.get_label() == "DynEdge (track)"
    ]
    assert len(track_lines) == 1
    assert track_lines[0].get_linestyle() == "-"
    x_data, y_data = track_lines[0].get_data()
    assert np.allclose(x_data, result["track"]["bin_center"])
    assert np.allclose(y_data, result["track"]["p50"])


def test_plot_direction_resolution_by_topology_cascade_is_dashed() -> None:
    """The cascade line should carry the cascade data and be dashed."""
    result = _make_topology_result()
    ax = plot_direction_resolution_by_topology(result, label="DynEdge")
    cascade_lines = [
        line for line in ax.lines if line.get_label() == "DynEdge (cascade)"
    ]
    assert len(cascade_lines) == 1
    assert cascade_lines[0].get_linestyle() == "--"
    x_data, y_data = cascade_lines[0].get_data()
    assert np.allclose(x_data, result["cascade"]["bin_center"])
    assert np.allclose(y_data, result["cascade"]["p50"])


def test_plot_direction_resolution_by_topology_muon_is_red_solid() -> None:
    """The muon baseline, when given, is a fixed red solid line."""
    result = _make_topology_result(include_muon=True)
    ax = plot_direction_resolution_by_topology(result, label="DynEdge")
    muon_lines = [
        line for line in ax.lines if line.get_label() == "Kinematic Angle"
    ]
    assert len(muon_lines) == 1
    assert muon_lines[0].get_color() == "red"
    assert muon_lines[0].get_linestyle() == "-"


def test_plot_direction_resolution_by_topology_omits_muon_when_absent() -> (
    None
):
    """No "muon" key in `result` means no muon line gets drawn."""
    result = _make_topology_result(include_muon=False)
    ax = plot_direction_resolution_by_topology(result)
    labels = [line.get_label() for line in ax.lines]
    assert "Kinematic Angle" not in labels


def test_plot_direction_resolution_by_topology_no_label_prefix_when_none() -> (
    None
):
    """Without a `label`, lines shouldn't get a "None (...)" placeholder."""
    ax = plot_direction_resolution_by_topology(_make_topology_result())
    labels = [line.get_label() for line in ax.lines]
    assert not any("None" in str(text) for text in labels)


def test_plot_direction_resolution_by_topology_draws_no_band() -> None:
    """Unlike other direction/energy plots, there's no shaded 68% band."""
    ax = plot_direction_resolution_by_topology(_make_topology_result())
    assert len(ax.collections) == 0


def test_plot_direction_resolution_by_topology_uses_log_xscale() -> None:
    """True energy spans orders of magnitude, so the x-axis should be log."""
    ax = plot_direction_resolution_by_topology(_make_topology_result())
    assert ax.get_xscale() == "log"


def test_plot_direction_resolution_by_topology_xlim_matches_data_range() -> (
    None
):
    """The x-axis should span the data's own range, not a fixed window."""
    ax = plot_direction_resolution_by_topology(_make_topology_result())

    xmin, xmax = ax.get_xlim()

    assert xmin == 10.0
    assert xmax == 1000.0


def test_plot_direction_resolution_by_topology_xlim_adapts_to_range() -> (
    None
):
    """A different energy range in the data should shift the x-axis too -
    not stay pinned to any fixed window.
    """
    result = _make_topology_result()
    for key in result:
        result[key]["bin_center"] = [10.0, 1000.0, 100000.0]

    ax = plot_direction_resolution_by_topology(result)

    xmin, xmax = ax.get_xlim()
    assert xmin == 10.0
    assert xmax == 100000.0


def test_plot_direction_resolution_by_topology_reuses_provided_axes() -> (
    None
):
    """Passing an existing `ax` should draw onto it, not create a new one."""
    fig, ax = plt.subplots()
    returned_ax = plot_direction_resolution_by_topology(
        _make_topology_result(), ax=ax
    )
    assert returned_ax is ax


def _make_topology_predictions_df(
    energy: np.ndarray, seed: int, include_muon: bool = False
) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    n = len(energy)
    zenith = rng.uniform(0, np.pi, size=n)
    azimuth = rng.uniform(0, 2 * np.pi, size=n)
    df = pd.DataFrame(
        {
            "zenith": zenith,
            "azimuth": azimuth,
            "px": np.sin(zenith) * np.cos(azimuth),
            "py": np.sin(zenith) * np.sin(azimuth),
            "pz": np.cos(zenith),
            "energy": energy,
            "is_track": rng.random(size=n) < 0.5,
        }
    )
    if include_muon:
        df["muon_zenith"] = rng.uniform(0, np.pi, size=n)
        df["muon_azimuth"] = rng.uniform(0, 2 * np.pi, size=n)
    return df


def test_plot_direction_resolution_by_topology_comparison_two_lines_per_model() -> (  # noqa: E501
    None
):
    """Each model, without muon columns, contributes a track+cascade line."""
    energy = np.linspace(10, 1000, 300)
    predictions = {
        "ModelA": _make_topology_predictions_df(energy, seed=10),
        "ModelB": _make_topology_predictions_df(energy, seed=11),
    }

    ax = plot_direction_resolution_by_topology_comparison(
        predictions,
        truth_zenith_col="zenith",
        truth_azimuth_col="azimuth",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        energy_col="energy",
        is_track_col="is_track",
    )

    assert len(ax.lines) == 4
    for model_name in predictions:
        assert (
            f"{model_name} (track)"
            in [line.get_label() for line in ax.lines]
        )
        assert (
            f"{model_name} (cascade)"
            in [line.get_label() for line in ax.lines]
        )


def test_plot_direction_resolution_by_topology_comparison_with_muon() -> (
    None
):
    """With muon columns given, each model also draws a "muon" line."""
    energy = np.linspace(10, 1000, 300)
    predictions = {
        "ModelA": _make_topology_predictions_df(
            energy, seed=10, include_muon=True
        ),
        "ModelB": _make_topology_predictions_df(
            energy, seed=11, include_muon=True
        ),
    }

    ax = plot_direction_resolution_by_topology_comparison(
        predictions,
        truth_zenith_col="zenith",
        truth_azimuth_col="azimuth",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        energy_col="energy",
        is_track_col="is_track",
        muon_zenith_col="muon_zenith",
        muon_azimuth_col="muon_azimuth",
    )

    labels = [line.get_label() for line in ax.lines]
    assert labels.count("Kinematic Angle") == len(predictions)


def test_plot_direction_resolution_by_topology_comparison_shares_bins() -> (
    None
):
    """Every model must be binned with the same shared edges.

    Same disjoint-ranges trick as the other comparisons: with correctly
    shared bins (built from the combined energy range), each model's
    track line must show NaNs where only the *other* model has data.
    """
    energy_low = np.linspace(10, 100, 200)
    energy_high = np.linspace(500, 1000, 200)
    predictions = {
        "Low": _make_topology_predictions_df(energy_low, seed=20),
        "High": _make_topology_predictions_df(energy_high, seed=21),
    }

    ax = plot_direction_resolution_by_topology_comparison(
        predictions,
        truth_zenith_col="zenith",
        truth_azimuth_col="azimuth",
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


def test_plot_direction_resolution_by_topology_comparison_reuses_axes() -> (
    None
):
    """Passing an existing `ax` should draw onto it, not create a new one."""
    fig, ax = plt.subplots()
    energy = np.linspace(10, 1000, 100)
    predictions = {"ModelA": _make_topology_predictions_df(energy, seed=30)}

    returned_ax = plot_direction_resolution_by_topology_comparison(
        predictions,
        truth_zenith_col="zenith",
        truth_azimuth_col="azimuth",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        energy_col="energy",
        is_track_col="is_track",
        ax=ax,
    )

    assert returned_ax is ax


def _make_error_distribution_topology_result(
    include_muon: bool = True,
) -> Dict[str, pd.DataFrame]:
    track = pd.DataFrame(
        {"bin_left": [0.0, 1.0, 2.0], "percentage": [50.0, 30.0, 20.0]}
    )
    cascade = pd.DataFrame(
        {"bin_left": [0.0, 1.0, 2.0], "percentage": [10.0, 40.0, 50.0]}
    )
    result = {"track": track, "cascade": cascade}
    if include_muon:
        result["muon"] = pd.DataFrame(
            {"bin_left": [0.0, 1.0, 2.0], "percentage": [80.0, 15.0, 5.0]}
        )
    return result


def test_plot_direction_error_distribution_by_topology_returns_axes() -> (
    None
):
    """The function should return a matplotlib Axes."""
    ax = plot_direction_error_distribution_by_topology(
        _make_error_distribution_topology_result()
    )
    assert isinstance(ax, plt.Axes)


def test_plot_direction_error_distribution_by_topology_track_is_solid() -> (
    None
):
    """The track line should carry the track data and use a solid style."""
    result = _make_error_distribution_topology_result()
    ax = plot_direction_error_distribution_by_topology(result, label="DynEdge")
    track_lines = [
        line for line in ax.lines if line.get_label() == "DynEdge (track)"
    ]
    assert len(track_lines) == 1
    assert track_lines[0].get_linestyle() == "-"
    x_data, y_data = track_lines[0].get_data()
    assert np.allclose(x_data, result["track"]["bin_left"])
    assert np.allclose(y_data, result["track"]["percentage"])


def test_plot_direction_error_distribution_by_topology_cascade_is_dashed() -> (
    None
):
    """The cascade line should carry the cascade data and be dashed."""
    result = _make_error_distribution_topology_result()
    ax = plot_direction_error_distribution_by_topology(result, label="DynEdge")
    cascade_lines = [
        line for line in ax.lines if line.get_label() == "DynEdge (cascade)"
    ]
    assert len(cascade_lines) == 1
    assert cascade_lines[0].get_linestyle() == "--"
    x_data, y_data = cascade_lines[0].get_data()
    assert np.allclose(x_data, result["cascade"]["bin_left"])
    assert np.allclose(y_data, result["cascade"]["percentage"])


def test_plot_direction_error_distribution_by_topology_muon_is_red_solid() -> (
    None
):
    """The muon baseline, when given, is a fixed red solid line."""
    result = _make_error_distribution_topology_result(include_muon=True)
    ax = plot_direction_error_distribution_by_topology(
        result, label="DynEdge"
    )
    muon_lines = [
        line for line in ax.lines if line.get_label() == "Kinematic Angle"
    ]
    assert len(muon_lines) == 1
    assert muon_lines[0].get_color() == "red"
    assert muon_lines[0].get_linestyle() == "-"


def test_plot_direction_error_distribution_by_topology_omits_muon() -> None:
    """No "muon" key in `result` means no muon line gets drawn."""
    result = _make_error_distribution_topology_result(include_muon=False)
    ax = plot_direction_error_distribution_by_topology(result)
    labels = [line.get_label() for line in ax.lines]
    assert "Kinematic Angle" not in labels


def test_plot_direction_error_distribution_by_topology_no_none_label() -> (
    None
):
    """Without a `label`, lines shouldn't get a "None (...)" placeholder."""
    ax = plot_direction_error_distribution_by_topology(
        _make_error_distribution_topology_result()
    )
    labels = [line.get_label() for line in ax.lines]
    assert not any("None" in str(text) for text in labels)


def test_plot_direction_error_distribution_by_topology_draws_no_band() -> (
    None
):
    """There's no shaded band on this plot, same as the vs-energy version."""
    ax = plot_direction_error_distribution_by_topology(
        _make_error_distribution_topology_result()
    )
    assert len(ax.collections) == 0


def test_plot_direction_error_distribution_by_topology_linear_xscale() -> (
    None
):
    """Unlike the vs-energy plot, the opening-angle axis stays linear."""
    ax = plot_direction_error_distribution_by_topology(
        _make_error_distribution_topology_result()
    )
    assert ax.get_xscale() == "linear"


def test_plot_direction_error_distribution_by_topology_reuses_axes() -> None:
    """Passing an existing `ax` should draw onto it, not create a new one."""
    fig, ax = plt.subplots()
    returned_ax = plot_direction_error_distribution_by_topology(
        _make_error_distribution_topology_result(), ax=ax
    )
    assert returned_ax is ax


def test_plot_direction_error_distribution_by_topology_comparison_lines() -> (
    None
):
    """Each model, without muon columns, contributes a track+cascade line."""
    energy = np.linspace(10, 1000, 300)
    predictions = {
        "ModelA": _make_topology_predictions_df(energy, seed=10),
        "ModelB": _make_topology_predictions_df(energy, seed=11),
    }

    ax = plot_direction_error_distribution_by_topology_comparison(
        predictions,
        truth_zenith_col="zenith",
        truth_azimuth_col="azimuth",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        is_track_col="is_track",
    )

    # Not `len(ax.lines) == 4`: a decorative reference line (unlabeled,
    # like the "ideal"/"chance" lines elsewhere) may also be drawn once
    # per model, so check the *labeled* model lines specifically instead.
    labels = [line.get_label() for line in ax.lines]
    for model_name in predictions:
        assert f"{model_name} (track)" in labels
        assert f"{model_name} (cascade)" in labels


def test_plot_direction_error_distribution_by_topology_comparison_muon() -> (
    None
):
    """With muon columns given, each model also draws a "muon" line."""
    energy = np.linspace(10, 1000, 300)
    predictions = {
        "ModelA": _make_topology_predictions_df(
            energy, seed=10, include_muon=True
        ),
        "ModelB": _make_topology_predictions_df(
            energy, seed=11, include_muon=True
        ),
    }

    ax = plot_direction_error_distribution_by_topology_comparison(
        predictions,
        truth_zenith_col="zenith",
        truth_azimuth_col="azimuth",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        is_track_col="is_track",
        muon_zenith_col="muon_zenith",
        muon_azimuth_col="muon_azimuth",
    )

    labels = [line.get_label() for line in ax.lines]
    assert labels.count("Kinematic Angle") == len(predictions)


def test_plot_direction_error_distribution_by_topology_comparison_bins() -> (
    None
):
    """Every model must reuse the same bin edges as the first model.

    With correctly shared bins, each model's track line must use
    identical "bin_left" x-values - not independently recomputed edges
    per model.
    """
    energy = np.linspace(10, 1000, 300)
    predictions = {
        "ModelA": _make_topology_predictions_df(energy, seed=10),
        "ModelB": _make_topology_predictions_df(energy, seed=11),
    }

    ax = plot_direction_error_distribution_by_topology_comparison(
        predictions,
        truth_zenith_col="zenith",
        truth_azimuth_col="azimuth",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        is_track_col="is_track",
        n_bins=10,
    )

    track_a = next(
        line for line in ax.lines if line.get_label() == "ModelA (track)"
    )
    track_b = next(
        line for line in ax.lines if line.get_label() == "ModelB (track)"
    )
    assert np.allclose(track_a.get_data()[0], track_b.get_data()[0])


def test_plot_direction_error_distribution_by_topology_comparison_axes() -> (
    None
):
    """Passing an existing `ax` should draw onto it, not create a new one."""
    fig, ax = plt.subplots()
    energy = np.linspace(10, 1000, 100)
    predictions = {"ModelA": _make_topology_predictions_df(energy, seed=30)}

    returned_ax = plot_direction_error_distribution_by_topology_comparison(
        predictions,
        truth_zenith_col="zenith",
        truth_azimuth_col="azimuth",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        is_track_col="is_track",
        ax=ax,
    )

    assert returned_ax is ax


def test_plot_direction_figure_returns_fig_and_two_axes() -> None:
    """The figure should have two distinct Axes, one per panel."""
    energy = np.linspace(10, 1000, 300)
    predictions = {"ModelA": _make_topology_predictions_df(energy, seed=10)}

    fig, (ax_resolution, ax_distribution) = plot_direction_figure(
        predictions,
        truth_zenith_col="zenith",
        truth_azimuth_col="azimuth",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        energy_col="energy",
        is_track_col="is_track",
    )

    assert isinstance(fig, plt.Figure)
    assert ax_resolution is not ax_distribution


def test_plot_direction_figure_left_panel_is_resolution() -> None:
    """Left panel: the log-scale resolution-vs-energy plot."""
    energy = np.linspace(10, 1000, 300)
    predictions = {"ModelA": _make_topology_predictions_df(energy, seed=10)}

    _, (ax_resolution, _) = plot_direction_figure(
        predictions,
        truth_zenith_col="zenith",
        truth_azimuth_col="azimuth",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        energy_col="energy",
        is_track_col="is_track",
    )

    assert ax_resolution.get_xscale() == "log"
    labels = [line.get_label() for line in ax_resolution.lines]
    assert "ModelA (track)" in labels


def test_plot_direction_figure_right_panel_is_distribution() -> None:
    """Right panel: the linear-scale error-distribution plot."""
    energy = np.linspace(10, 1000, 300)
    predictions = {"ModelA": _make_topology_predictions_df(energy, seed=10)}

    _, (_, ax_distribution) = plot_direction_figure(
        predictions,
        truth_zenith_col="zenith",
        truth_azimuth_col="azimuth",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        energy_col="energy",
        is_track_col="is_track",
    )

    assert ax_distribution.get_xscale() == "linear"
    labels = [line.get_label() for line in ax_distribution.lines]
    assert "ModelA (track)" in labels
