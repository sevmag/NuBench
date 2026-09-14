"""Direction reconstruction plots: angular resolution vs. energy and the
opening-angle error distribution, both split by track/cascade/muon, plus
the combined figure putting both side by side.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nubench.evaluation.metrics import (
    direction_error_distribution_by_topology,
    direction_resolution_by_topology,
)
from nubench.style import model_color


def log_decade_ticks(vmin: float, vmax: float) -> List[float]:
    """Whole powers of ten spanning [vmin, vmax], for a log-scale axis.

    Used to build axis limits/ticks from whatever range the data
    actually spans, rather than a fixed window - some NuBench detectors
    were only simulated up to 10^3 GeV rather than 10^5 GeV, and a
    retrained model (see Stage 2) could span any range at all, so a
    fixed constant would either clip real data or leave a mostly-empty
    plot depending on which detector's data is passed in.
    """
    lo = int(np.floor(np.log10(vmin)))
    hi = int(np.ceil(np.log10(vmax)))
    return [10.0 ** p for p in range(lo, hi + 1)]


def plot_direction_resolution_by_topology(
    result: Dict[str, pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    degrees: bool = True,
) -> plt.Axes:
    """Plot angular resolution vs. energy, split into track/cascade/muon.

    Matches the paper's convention: track solid, cascade dashed, both in
    the same color for one model; the muon baseline (if present) always
    red and solid, since it's a physical reference rather than a model
    prediction. Unlike the plain energy calibration plot, there is no
    shaded 68% band here - just the median lines. The x-axis range/ticks
    are built from whatever energy range `result` actually spans (not a
    fixed window) - some NuBench detectors are only simulated up to
    10^3 GeV rather than 10^5 GeV, and a retrained model could span any
    range at all, so the axis always matches the real data instead of
    assuming any particular detector's range.

    Args:
        result: Output of `direction_resolution_by_topology` - a dict
            with keys "track"/"cascade" (each a DataFrame with columns
            "bin_center", "p16", "p50", "p84"), plus "muon" if computed.
        ax: Existing Axes to draw onto. If None, a new Figure/Axes pair is
            created. Passing the same `ax` across multiple calls is how
            several models end up overlaid on one figure.
        label: Legend label prefix for this model, e.g. "DynEdge" becomes
            "DynEdge (track)" / "DynEdge (cascade)" on the two lines.
        color: Color shared by this model's track and cascade lines. If
            None, matplotlib picks one automatically.
        degrees: Whether the angle values in `result` are in degrees
            (True, matching `direction_resolution_by_topology`'s own
            default) or radians - only used to label the y-axis.

    Returns:
        The Axes the plot was drawn on (either the one passed in, or a
        newly created one).
    """
    if ax is None:
        fig, ax = plt.subplots()
    track_label = f"{label} (track)" if label is not None else None
    ax.plot(
        result["track"]["bin_center"],
        result["track"]["p50"],
        label=track_label,
        color=color,
        linestyle="-",
        linewidth=2,
    )
    cascade_label = f"{label} (cascade)" if label is not None else None
    ax.plot(
        result["cascade"]["bin_center"],
        result["cascade"]["p50"],
        label=cascade_label,
        color=color,
        linestyle="--",
        linewidth=2,
    )
    if "muon" in result:
        ax.plot(
            result["muon"]["bin_center"],
            result["muon"]["p50"],
            label="Kinematic Angle",
            color="red",
            linestyle="-",
            linewidth=2,
        )
    bin_centers = pd.concat(
        [result["track"]["bin_center"], result["cascade"]["bin_center"]]
        + ([result["muon"]["bin_center"]] if "muon" in result else [])
    )
    xmin, xmax = bin_centers.min(), bin_centers.max()
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 10)
    ax.set_xticks(log_decade_ticks(xmin, xmax))
    ax.set_yticks([1, 5, 10])
    ax.grid(which="both", alpha=0.3)
    ax.set_xlabel("True Energy [GeV]")
    ax.set_ylabel("Opening Angle [deg]" if degrees else "Opening Angle [rad]")
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
    degrees: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot track/cascade/muon angular resolution for several models.

    Args:
        predictions: Mapping from model name (used as its legend label) to
            its prediction DataFrame. Every DataFrame must contain the
            angle/vector columns below, plus `energy_col` and
            `is_track_col` (and the muon columns, if given).
        truth_zenith_col: Name of the column holding the true zenith angle
            (radians) - the same across every model's DataFrame.
        truth_azimuth_col: Name of the column holding the true azimuth
            angle (radians) - the same across every model's DataFrame.
        pred_x_col: Name of the column holding the predicted direction
            vector's x-component - the same across every model's
            DataFrame.
        pred_y_col: Name of the column holding the predicted direction
            vector's y-component - the same across every model's
            DataFrame.
        pred_z_col: Name of the column holding the predicted direction
            vector's z-component - the same across every model's
            DataFrame.
        energy_col: Name of the column holding the true energy - the same
            across every model's DataFrame.
        is_track_col: Name of the boolean track/cascade column - the same
            across every model's DataFrame.
        muon_zenith_col: Name of the column holding the outgoing muon's
            own zenith angle, in radians. If either this or
            `muon_azimuth_col` is None, no muon line is drawn.
        muon_azimuth_col: Name of the column holding the outgoing muon's
            own azimuth angle, in radians.
        bins: Bin edges to use for every model. If None, edges are built
            automatically to span the *combined* true-energy range across
            all models, so every model is binned identically for a fair
            comparison.
        n_bins: Number of bins to construct when `bins` is not given.
        degrees: If True (default), report/label the opening angle in
            degrees. Otherwise radians.
        ax: Existing Axes to draw onto. If None, a new Figure/Axes pair is
            created.

    Returns:
        The Axes with one track+cascade line pair per model (plus a muon
        line, if computed), with a legend identifying each by its key in
        `predictions`.
    """
    if bins is None:
        combined_min = min(df[energy_col].min() for df in predictions.values())
        combined_max = max(df[energy_col].max() for df in predictions.values())
        bins = np.logspace(
            np.log10(combined_min), np.log10(combined_max), n_bins + 1)
    if ax is None:
        fig, ax = plt.subplots()
    for model_name, df in predictions.items():
        result = direction_resolution_by_topology(
            df,
            truth_zenith_col,
            truth_azimuth_col,
            pred_x_col,
            pred_y_col,
            pred_z_col,
            energy_col,
            is_track_col,
            muon_zenith_col,
            muon_azimuth_col,
            bins=bins,
            n_bins=n_bins,
            degrees=degrees
        )
        plot_direction_resolution_by_topology(
            result,
            ax=ax,
            label=model_name,
            color=model_color(model_name),
            degrees=degrees,
        )
    ax.legend()
    return ax


def plot_direction_error_distribution_by_topology(
    result: Dict[str, pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    degrees: bool = True,
) -> plt.Axes:
    """Plot the opening-angle distribution, split into track/cascade/muon.

    Same track-solid/cascade-dashed/muon-red-solid convention as
    `plot_direction_resolution_by_topology`, but the x-axis here is the
    opening angle itself (linear scale) rather than energy, and the
    y-axis is a percentage of events rather than a resolution.

    Args:
        result: Output of `direction_error_distribution_by_topology` - a
            dict with keys "track"/"cascade" (each a DataFrame with
            columns "bin_left", "percentage"), plus "muon" if computed.
        ax: Existing Axes to draw onto. If None, a new Figure/Axes pair is
            created. Passing the same `ax` across multiple calls is how
            several models end up overlaid on one figure.
        label: Legend label prefix for this model, e.g. "DynEdge" becomes
            "DynEdge (track)" / "DynEdge (cascade)" on the two lines.
        color: Color shared by this model's track and cascade lines. If
            None, matplotlib picks one automatically.
        degrees: Whether the angle values in `result` are in degrees
            (True, matching `direction_error_distribution_by_topology`'s
            own default) or radians - only used to label the x-axis.

    Returns:
        The Axes the plot was drawn on (either the one passed in, or a
        newly created one).
    """
    if ax is None:
        fig, ax = plt.subplots()
    track_label = f"{label} (track)" if label is not None else None
    ax.plot(
        result["track"]["bin_left"],
        result["track"]["percentage"],
        label=track_label,
        color=color,
        linestyle="-",
        linewidth=2,
    )
    cascade_label = f"{label} (cascade)" if label is not None else None
    ax.plot(
        result["cascade"]["bin_left"],
        result["cascade"]["percentage"],
        label=cascade_label,
        color=color,
        linestyle="--",
        linewidth=2,
    )
    if "muon" in result:
        ax.plot(
            result["muon"]["bin_left"],
            result["muon"]["percentage"],
            label="Kinematic Angle",
            color="red",
            linestyle="-",
            linewidth=2,
        )
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
    ax.set_xlabel("Opening Angle [deg]" if degrees else "Opening Angle [rad]")
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
    degrees: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot the opening-angle distribution for several models.

    Args:
        predictions: Mapping from model name (used as its legend label) to
            its prediction DataFrame. Every DataFrame must contain the
            angle/vector columns below, plus `is_track_col` (and the muon
            columns, if given).
        truth_zenith_col: Name of the column holding the true zenith angle
            (radians) - the same across every model's DataFrame.
        truth_azimuth_col: Name of the column holding the true azimuth
            angle (radians) - the same across every model's DataFrame.
        pred_x_col: Name of the column holding the predicted direction
            vector's x-component - the same across every model's
            DataFrame.
        pred_y_col: Name of the column holding the predicted direction
            vector's y-component - the same across every model's
            DataFrame.
        pred_z_col: Name of the column holding the predicted direction
            vector's z-component - the same across every model's
            DataFrame.
        is_track_col: Name of the boolean track/cascade column - the same
            across every model's DataFrame.
        muon_zenith_col: Name of the column holding the outgoing muon's
            own zenith angle, in radians. If either this or
            `muon_azimuth_col` is None, no muon line is drawn.
        muon_azimuth_col: Name of the column holding the outgoing muon's
            own azimuth angle, in radians.
        bins: Bin edges to use for every model. If None, edges are built
            automatically to span the *combined* opening-angle range
            across all models, so every model is binned identically for a
            fair comparison.
        n_bins: Number of bins to construct when `bins` is not given.
        degrees: If True (default), compute/label the opening angle in
            degrees. Otherwise radians.
        ax: Existing Axes to draw onto. If None, a new Figure/Axes pair is
            created.

    Returns:
        The Axes with one track+cascade line pair per model (plus a muon
        line, if computed), with a legend identifying each by its key in
        `predictions`.
    """
    if ax is None:
        fig, ax = plt.subplots()
    for model_name, df in predictions.items():
        result = direction_error_distribution_by_topology(
            df,
            truth_zenith_col,
            truth_azimuth_col,
            pred_x_col,
            pred_y_col,
            pred_z_col,
            is_track_col,
            muon_zenith_col,
            muon_azimuth_col,
            bins=bins,
            n_bins=n_bins,
            degrees=degrees,
        )
        if bins is None:
            bin_left = result["track"]["bin_left"].to_numpy()
            bin_width = bin_left[1] - bin_left[0]
            bins = np.append(bin_left, bin_left[-1] + bin_width)
        plot_direction_error_distribution_by_topology(
            result,
            ax=ax,
            label=model_name,
            color=model_color(model_name),
            degrees=degrees,
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
    degrees: bool = True,
    distribution_bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    distribution_n_bins: int = 120,
    ax_resolution: Optional[plt.Axes] = None,
    ax_distribution: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Build the combined direction figure: resolution + error distribution.

    A 1x2 `Figure` - pure composition, no new metric math - putting the
    paper's two direction plots side by side so a user sees both without
    calling two functions and arranging them by hand: left panel is
    `plot_direction_resolution_by_topology_comparison` (opening angle
    vs. energy), right panel is
    `plot_direction_error_distribution_by_topology_comparison` (the
    opening-angle-distribution histogram), each drawn via its own `ax`
    parameter onto one half of the same Figure.

    Args:
        predictions: Mapping from model name (used as its legend label) to
            its prediction DataFrame - the same dict both underlying
            comparison functions expect.
        truth_zenith_col: Name of the column holding the true zenith
            angle (radians) - the same across every model's DataFrame.
        truth_azimuth_col: Name of the column holding the true azimuth
            angle (radians) - the same across every model's DataFrame.
        pred_x_col: Name of the column holding the predicted direction
            vector's x-component - the same across every model's
            DataFrame.
        pred_y_col: Name of the column holding the predicted direction
            vector's y-component - the same across every model's
            DataFrame.
        pred_z_col: Name of the column holding the predicted direction
            vector's z-component - the same across every model's
            DataFrame.
        energy_col: Name of the column holding the true energy - only
            used by the left (resolution-vs-energy) panel.
        is_track_col: Name of the boolean track/cascade column - the same
            across every model's DataFrame.
        muon_zenith_col: Name of the column holding the outgoing muon's
            own zenith angle, in radians. If either this or
            `muon_azimuth_col` is None, no muon line is drawn in either
            panel.
        muon_azimuth_col: Name of the column holding the outgoing muon's
            own azimuth angle, in radians.
        degrees: If True (default), report/label the opening angle in
            degrees in both panels. Otherwise radians.
        distribution_bins: Bin edges for the right (error-distribution)
            panel only - the left panel always uses its own combined
            energy range. If None, edges are built automatically to span
            the *combined* opening-angle range across all models, which
            (matching `plot_direction_error_distribution_by_topology`'s
            own fixed axis limits) may be much wider than the ~0-10 deg.
            window actually shown - pass e.g. `np.linspace(0, 10, 120)`
            for a paper-matching zoomed-in view.
        distribution_n_bins: Number of bins to construct when
            `distribution_bins` is not given.
        ax_resolution: Existing Axes for the left (resolution-vs-energy)
            panel. Must be given together with `ax_distribution` (both
            or neither) - if either is None, a new Figure with both
            panels is created instead. Passing both in is how a multi-
            detector grid (see `nubench.multipanel`) places one
            detector's pair into its own slice of a bigger grid.
        ax_distribution: Existing Axes for the right (error-
            distribution) panel.

    Returns:
        `(fig, (ax_resolution, ax_distribution))` - the Figure, the
        resolution-vs-energy Axes, and the error-distribution Axes.
    """
    if ax_resolution is None or ax_distribution is None:
        fig, (ax_resolution, ax_distribution) = plt.subplots(
            1, 2, figsize=(6, 3), constrained_layout=True
        )
    else:
        # Same reasoning as `plot_energy_calibration_figure`'s own
        # `ax_hist`/`ax_main` reuse: every Axes this function ever
        # receives comes from `plt.subplots()`, so `.figure` is always a
        # plain `Figure`, never a `SubFigure`.
        fig = ax_resolution.figure  # type: ignore[assignment]
    plot_direction_resolution_by_topology_comparison(
        predictions,
        truth_zenith_col,
        truth_azimuth_col,
        pred_x_col,
        pred_y_col,
        pred_z_col,
        energy_col,
        is_track_col,
        muon_zenith_col=muon_zenith_col,
        muon_azimuth_col=muon_azimuth_col,
        degrees=degrees,
        ax=ax_resolution
    )
    plot_direction_error_distribution_by_topology_comparison(
        predictions,
        truth_zenith_col,
        truth_azimuth_col,
        pred_x_col,
        pred_y_col,
        pred_z_col,
        is_track_col,
        muon_zenith_col=muon_zenith_col,
        muon_azimuth_col=muon_azimuth_col,
        bins=distribution_bins,
        n_bins=distribution_n_bins,
        degrees=degrees,
        ax=ax_distribution
    )
    return fig, (ax_resolution, ax_distribution)
