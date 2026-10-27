"""Direction reconstruction plots: angular resolution vs. energy and the
opening-angle error distribution, both split by track/cascade/muon, plus
the combined figure putting the two side by side.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nubench.evaluation.metrics import (
    direction_error_distribution_by_topology,
    direction_resolution_by_topology,
)
from nubench.evaluation.resolution import combined_log_bins
from nubench.style import model_color


def _plot_topology_lines(
    ax: plt.Axes,
    result: Dict[str, pd.DataFrame],
    x_col: str,
    y_col: str,
    label: Optional[str],
    color: Optional[str],
) -> None:
    """Draw track solid, cascade dashed, muon red - the paper's convention."""
    for key, linestyle in (("track", "-"), ("cascade", "--")):
        ax.plot(
            result[key][x_col], result[key][y_col],
            label=f"{label} ({key})" if label is not None else None,
            color=color, linestyle=linestyle, linewidth=2,
        )
    if "muon" in result:
        ax.plot(
            result["muon"][x_col], result["muon"][y_col],
            label="Kinematic Angle", color="red", linestyle="-", linewidth=2,
        )


def plot_direction_resolution_by_topology(
    result: Dict[str, pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> plt.Axes:
    """Plot one model's angular resolution vs. energy, by topology.

    Median lines only. The x range comes from whatever energy range
    `result` spans, so only the decade ticks inside it show.
    """
    if ax is None:
        fig, ax = plt.subplots()
    _plot_topology_lines(ax, result, "bin_center", "p50", label, color)
    bin_centers = pd.concat([
        result[key]["bin_center"] for key in result
    ])
    xmin, xmax = bin_centers.min(), bin_centers.max()
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 10)
    # Only the decades the data actually spans. A fixed list covering
    # more would both widen the limits set above and make matplotlib
    # drop the first and last labels, losing the range endpoints.
    decades = range(
        int(np.floor(np.log10(xmin))), int(np.ceil(np.log10(xmax))) + 1
    )
    ax.set_xticks([10.0**p for p in decades])
    ax.set_yticks([1, 5, 10])
    ax.grid(which="both", alpha=0.3)
    ax.set_xlabel("True Energy [GeV]")
    ax.set_ylabel("Opening Angle [deg]")
    return ax


def plot_direction_resolution_by_topology_comparison(
    predictions: Dict[str, pd.DataFrame],
    truth_zenith_col: str,
    truth_azimuth_col: str,
    pred_x_col: str,
    pred_y_col: str,
    pred_z_col: str,
    energy_col: str,
    is_track_col: str,
    muon_zenith_col: Optional[str] = None,
    muon_azimuth_col: Optional[str] = None,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 30,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Overlay every model's track/cascade/muon angular resolution.

    `bins` defaults to `n_bins` log-spaced edges over the combined
    true-energy range across all models.
    """
    if bins is None:
        bins = combined_log_bins(
            (df[energy_col] for df in predictions.values()), n_bins
        )
    if ax is None:
        fig, ax = plt.subplots()
    for model_name, df in predictions.items():
        plot_direction_resolution_by_topology(
            direction_resolution_by_topology(
                df, truth_zenith_col, truth_azimuth_col,
                pred_x_col, pred_y_col, pred_z_col,
                energy_col, is_track_col,
                muon_zenith_col, muon_azimuth_col,
                bins=bins, n_bins=n_bins,
            ),
            ax=ax,
            label=model_name,
            color=model_color(model_name),
        )
    ax.legend()
    return ax


def plot_direction_error_distribution_by_topology(
    result: Dict[str, pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> plt.Axes:
    """Plot one model's opening-angle distribution, by topology.

    x is the opening angle itself (linear) and y a percentage of events.
    """
    if ax is None:
        fig, ax = plt.subplots()
    _plot_topology_lines(ax, result, "bin_left", "percentage", label, color)
    ax.set_xlim(-0.1, 5)
    ax.set_ylim(0, 5)
    ax.set_xticks([0, 1, 5])
    ax.set_yticks(np.arange(0, 6, 1))
    ax.xaxis.minorticks_on()
    ax.yaxis.minorticks_on()
    ax.plot([1, 1], [0, 5], color="grey", alpha=0.4, linewidth=2)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.grid(which="both", alpha=0.3)
    ax.set_xlabel("Opening Angle [deg]")
    ax.set_ylabel("Percentage [%]")
    return ax


def plot_direction_error_distribution_by_topology_comparison(
    predictions: Dict[str, pd.DataFrame],
    truth_zenith_col: str,
    truth_azimuth_col: str,
    pred_x_col: str,
    pred_y_col: str,
    pred_z_col: str,
    is_track_col: str,
    muon_zenith_col: Optional[str] = None,
    muon_azimuth_col: Optional[str] = None,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 120,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Overlay every model's opening-angle distribution."""
    if ax is None:
        fig, ax = plt.subplots()
    for model_name, df in predictions.items():
        result = direction_error_distribution_by_topology(
            df, truth_zenith_col, truth_azimuth_col,
            pred_x_col, pred_y_col, pred_z_col, is_track_col,
            muon_zenith_col, muon_azimuth_col,
            bins=bins, n_bins=n_bins,
        )
        # The first model's auto-built bins are recover from its result
        # and used for all models, so every model is binned identically.
        if bins is None:
            bin_left = result["track"]["bin_left"].to_numpy()
            width = bin_left[1] - bin_left[0]
            bins = np.append(bin_left, bin_left[-1] + width)
        plot_direction_error_distribution_by_topology(
            result,
            ax=ax,
            label=model_name,
            color=model_color(model_name),
        )
    ax.legend()
    return ax


def plot_direction_figure(
    predictions: Dict[str, pd.DataFrame],
    truth_zenith_col: str,
    truth_azimuth_col: str,
    pred_x_col: str,
    pred_y_col: str,
    pred_z_col: str,
    energy_col: str,
    is_track_col: str,
    muon_zenith_col: Optional[str] = None,
    muon_azimuth_col: Optional[str] = None,
    distribution_bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    distribution_n_bins: int = 120,
    ax_resolution: Optional[plt.Axes] = None,
    ax_distribution: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """The paper's two direction panels side by side: resolution, then
    error distribution.

    Pass both `ax_*` to draw into an existing grid (see
    `nubench.multipanel`); otherwise a new 1x2 Figure is created.

    `distribution_bins` applies to the right panel only. Its auto-built
    default spans the full opening-angle range, which is far wider than
    the ~0-5 deg window that panel displays - pass e.g.
    `np.linspace(0, 10, 120)` for the paper-matching zoom.
    """
    if ax_resolution is None or ax_distribution is None:
        fig, (ax_resolution, ax_distribution) = plt.subplots(
            1, 2, figsize=(6, 3), constrained_layout=True
        )
    else:
        # Every Axes this receives comes from `plt.subplots()`, so
        # `.figure` is always a plain Figure, never a SubFigure.
        fig = ax_resolution.figure  # type: ignore[assignment]
    plot_direction_resolution_by_topology_comparison(
        predictions, truth_zenith_col, truth_azimuth_col,
        pred_x_col, pred_y_col, pred_z_col, energy_col, is_track_col,
        muon_zenith_col=muon_zenith_col,
        muon_azimuth_col=muon_azimuth_col,
        ax=ax_resolution,
    )
    plot_direction_error_distribution_by_topology_comparison(
        predictions, truth_zenith_col, truth_azimuth_col,
        pred_x_col, pred_y_col, pred_z_col, is_track_col,
        muon_zenith_col=muon_zenith_col,
        muon_azimuth_col=muon_azimuth_col,
        bins=distribution_bins,
        n_bins=distribution_n_bins,
        ax=ax_distribution,
    )
    return fig, (ax_resolution, ax_distribution)
