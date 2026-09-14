"""Vertex reconstruction plots: position resolution vs. energy, and the 2D
depth/radial containment contour - both split by track/cascade.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nubench.evaluation.metrics import (
    vertex_contour_by_topology,
    vertex_resolution_by_topology,
)
from nubench.style import model_color


def plot_vertex_resolution_by_topology(
    result: Dict[str, pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> plt.Axes:
    """Plot vertex-position resolution vs. energy, split by track/cascade.

    Same track-solid/cascade-dashed convention as
    `plot_direction_resolution_by_topology`, but there's no muon line
    here - vertex has no equivalent physical baseline. No shaded band,
    matching the paper's own vertex-vs-energy plot.

    Args:
        result: Output of `vertex_resolution_by_topology` - a dict with
            keys "track"/"cascade" (each a DataFrame with columns
            "bin_center", "p16", "p50", "p84").
        ax: Existing Axes to draw onto. If None, a new Figure/Axes pair is
            created. Passing the same `ax` across multiple calls is how
            several models end up overlaid on one figure.
        label: Legend label prefix for this model, e.g. "DynEdge" becomes
            "DynEdge (track)" / "DynEdge (cascade)" on the two lines.
        color: Color shared by this model's track and cascade lines. If
            None, matplotlib picks one automatically.

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
    ax.set_xscale("log")
    ax.yaxis.minorticks_on()
    ax.grid(which="both", alpha=0.3)
    ax.set_xlabel("True Energy [GeV]")
    ax.set_ylabel("Euclidean Distance [m]")
    return ax


def plot_vertex_resolution_by_topology_comparison(
    predictions: Dict[str, pd.DataFrame],
    truth_x_col: str,
    truth_y_col: str,
    truth_z_col: str,
    pred_x_col: str,
    pred_y_col: str,
    pred_z_col: str,
    energy_col: str,
    is_track_col: str,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 30,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot track/cascade vertex-position resolution for several models.

    Args:
        predictions: Mapping from model name (used as its legend label) to
            its prediction DataFrame. Every DataFrame must contain the
            position columns below, plus `energy_col` and `is_track_col`.
        truth_x_col: Name of the column holding the true vertex
            x-coordinate - the same across every model's DataFrame.
        truth_y_col: Name of the column holding the true vertex
            y-coordinate - the same across every model's DataFrame.
        truth_z_col: Name of the column holding the true vertex
            z-coordinate - the same across every model's DataFrame.
        pred_x_col: Name of the column holding the predicted vertex
            x-coordinate - the same across every model's DataFrame.
        pred_y_col: Name of the column holding the predicted vertex
            y-coordinate - the same across every model's DataFrame.
        pred_z_col: Name of the column holding the predicted vertex
            z-coordinate - the same across every model's DataFrame.
        energy_col: Name of the column holding the true energy - the same
            across every model's DataFrame.
        is_track_col: Name of the boolean track/cascade column - the same
            across every model's DataFrame.
        bins: Bin edges to use for every model. If None, edges are built
            automatically to span the *combined* true-energy range across
            all models, so every model is binned identically for a fair
            comparison.
        n_bins: Number of bins to construct when `bins` is not given.
        ax: Existing Axes to draw onto. If None, a new Figure/Axes pair is
            created.

    Returns:
        The Axes with one track+cascade line pair per model, with a
        legend identifying each by its key in `predictions`.
    """
    if bins is None:
        combined_min = min(df[energy_col].min() for df in predictions.values())
        combined_max = max(df[energy_col].max() for df in predictions.values())
        bins = np.logspace(
            np.log10(combined_min), np.log10(combined_max), n_bins + 1)
    if ax is None:
        fig, ax = plt.subplots()
    for model_name, df in predictions.items():
        result = vertex_resolution_by_topology(
            df, truth_x_col, truth_y_col, truth_z_col,
            pred_x_col, pred_y_col, pred_z_col,
            energy_col, is_track_col, bins=bins
        )
        plot_vertex_resolution_by_topology(
            result,
            ax=ax,
            label=model_name,
            color=model_color(model_name)
        )
    ax.legend()
    return ax


def plot_vertex_contour_by_topology(
    result: Dict[
        str, Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]
    ],
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> plt.Axes:
    """Plot a depth-vs-radial 68% containment contour, split track/cascade.

    Draws each topology's 68% containment contour (track solid, cascade
    dashed, same convention as `plot_vertex_resolution_by_topology`) plus
    a marker at that topology's median (depth, radial) point - track as
    a star, cascade as a dot, matching the paper's own vertex contour
    plot.

    Note: `Axes.contour` doesn't support a `label` argument the way
    `Axes.plot` does (it's silently ignored - no legend entry), so a
    legend needs a separate trick: draw an invisible zero-length line
    with the same color/linestyle purely to carry the legend entry.

    Args:
        result: Output of `vertex_contour_by_topology` - a dict with keys
            "track"/"cascade", each mapping to a tuple
            `(X, Y, H, level, median_depth, median_radial)`.
        ax: Existing Axes to draw onto. If None, a new Figure/Axes pair is
            created. Passing the same `ax` across multiple calls is how
            several models end up overlaid on one figure.
        label: Legend label prefix for this model, e.g. "DynEdge" becomes
            "DynEdge (track)" / "DynEdge (cascade)". If None, no legend
            entries are added for this model.
        color: Color shared by this model's track and cascade contours
            and markers. If None, matplotlib picks one automatically.

    Returns:
        The Axes the plot was drawn on (either the one passed in, or a
        newly created one).
    """
    if ax is None:
        fig, ax = plt.subplots()

    (
        track_X, track_Y, track_H, track_level, track_depth, track_radial,
    ) = result["track"]
    (
        cascade_X, cascade_Y, cascade_H, cascade_level,
        cascade_depth, cascade_radial,
    ) = result["cascade"]

    ax.contour(
        track_X,
        track_Y,
        track_H,
        levels=[track_level],
        colors=color,
        linewidths=3,
        alpha=0.7,
    )
    ax.contour(
        cascade_X,
        cascade_Y,
        cascade_H,
        levels=[cascade_level],
        colors=color,
        linestyles="dashed",
        linewidths=3,
    )

    if label is not None:
        ax.plot(
            [], [], color=color, linestyle="-", label=f"{label} (track)"
        )
        ax.plot(
            [], [], color=color, linestyle="--", label=f"{label} (cascade)"
        )

    ax.scatter(track_depth, track_radial, color=color, marker="*", s=70)
    ax.scatter(cascade_depth, cascade_radial, color=color, marker=".", s=70)

    ax.set_xlabel("Vertical Distance [m]")
    ax.set_ylabel("Radial Distance [m]")
    ax.yaxis.minorticks_on()
    ax.xaxis.minorticks_on()
    ax.grid(alpha=0.3, which="both")
    return ax


def plot_vertex_contour_by_topology_comparison(
    predictions: Dict[str, pd.DataFrame],
    truth_x_col: str,
    truth_y_col: str,
    truth_z_col: str,
    pred_x_col: str,
    pred_y_col: str,
    pred_z_col: str,
    is_track_col: str,
    bins: Union[int, Sequence[int], Sequence[np.ndarray]] = 100,
    sigma: float = 3.0,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot depth-vs-radial 68% containment contours for several models.

    Unlike the other `*_comparison` functions here, this doesn't build a
    combined bin range across models automatically - `bins` is passed
    through unchanged to every model's `vertex_contour_by_topology` call.
    If you want every model's contour on directly comparable axes, pass
    explicit bin edges (`bins=[x_edges, y_edges]`) rather than a plain
    int - that's what the paper's own script does, tailored per detector
    geometry.

    Args:
        predictions: Mapping from model name (used as its legend label) to
            its prediction DataFrame. Every DataFrame must contain the
            position columns below, plus `is_track_col`.
        truth_x_col: Name of the column holding the true vertex
            x-coordinate - the same across every model's DataFrame.
        truth_y_col: Name of the column holding the true vertex
            y-coordinate - the same across every model's DataFrame.
        truth_z_col: Name of the column holding the true vertex
            z-coordinate - the same across every model's DataFrame.
        pred_x_col: Name of the column holding the predicted vertex
            x-coordinate - the same across every model's DataFrame.
        pred_y_col: Name of the column holding the predicted vertex
            y-coordinate - the same across every model's DataFrame.
        pred_z_col: Name of the column holding the predicted vertex
            z-coordinate - the same across every model's DataFrame.
        is_track_col: Name of the boolean track/cascade column - the same
            across every model's DataFrame.
        bins: Passed through to `vertex_contour_by_topology`'s own `bins`
            argument, identically for every model.
        sigma: Passed through to `vertex_contour_by_topology`'s own
            `sigma` argument.
        ax: Existing Axes to draw onto. If None, a new Figure/Axes pair is
            created.

    Returns:
        The Axes with one track+cascade contour pair per model, with a
        legend identifying each by its key in `predictions`.
    """
    if ax is None:
        fig, ax = plt.subplots()
    for model_name, df in predictions.items():
        result = vertex_contour_by_topology(
            df,
            truth_x_col,
            truth_y_col,
            truth_z_col,
            pred_x_col,
            pred_y_col,
            pred_z_col,
            is_track_col,
            bins=bins,
            sigma=sigma
        )
        plot_vertex_contour_by_topology(
            result,
            ax=ax,
            label=model_name,
            color=model_color(model_name)
        )
    ax.legend()
    return ax
