"""Vertex reconstruction plots: position resolution vs. energy and the 2D
depth/radial containment contour, both split by track/cascade.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nubench.evaluation.metrics import (
    vertex_contour_by_topology,
    vertex_resolution_by_topology,
)
from nubench.evaluation.metrics.vertex import Contour
from nubench.evaluation.resolution import combined_log_bins, radial_and_depth
from nubench.style import model_color


def plot_vertex_resolution_by_topology(
    result: Dict[str, pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> plt.Axes:
    """Plot one model's vertex resolution vs. energy, by topology.

    Track solid, cascade dashed.
    """
    if ax is None:
        fig, ax = plt.subplots()
    for key, linestyle in (("track", "-"), ("cascade", "--")):
        ax.plot(
            result[key]["bin_center"], result[key]["p50"],
            label=f"{label} ({key})" if label is not None else None,
            color=color, linestyle=linestyle, linewidth=2,
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
    """Overlay every model's track/cascade vertex resolution."""
    if bins is None:
        bins = combined_log_bins(
            (df[energy_col] for df in predictions.values()), n_bins
        )
    if ax is None:
        fig, ax = plt.subplots()
    for model_name, df in predictions.items():
        plot_vertex_resolution_by_topology(
            vertex_resolution_by_topology(
                df, truth_x_col, truth_y_col, truth_z_col,
                pred_x_col, pred_y_col, pred_z_col,
                energy_col, is_track_col, bins=bins,
            ),
            ax=ax,
            label=model_name,
            color=model_color(model_name),
        )
    ax.legend()
    return ax


def plot_vertex_contour_by_topology(
    result: Dict[str, Contour],
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> plt.Axes:
    """Plot one model's 68% containment contours, by topology.

    Track solid with a star at its median point, cascade dashed with a
    dot - and only the track contour translucent, an asymmetry copied
    from the paper. `Axes.contour` silently ignores `label`, legend
    entries come from zero-length dummy lines instead.
    """
    if ax is None:
        fig, ax = plt.subplots()
    for key, linestyle, marker in (
        ("track", "solid", "*"), ("cascade", "dashed", "."),
    ):
        X, Y, H, level, median_depth, median_radial = result[key]
        ax.contour(
            X, Y, H, levels=[level], colors=color,
            linestyles=linestyle, linewidths=3,
            alpha=0.7 if key == "track" else None,
        )
        ax.scatter(median_depth, median_radial, color=color, marker=marker,
                   s=70)
        if label is not None:
            ax.plot(
                [], [], color=color,
                linestyle="-" if key == "track" else "--",
                label=f"{label} ({key})",
            )
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
    """Overlay every model's 68% containment contours.

    `bins` passes straight through to every model unchanged - no combined
    range is built. For directly comparable axes pass explicit edges
    (`bins=[x_edges, y_edges]`); `nubench.multipanel` derives those per
    detector.
    """
    if ax is None:
        fig, ax = plt.subplots()
    for model_name, df in predictions.items():
        plot_vertex_contour_by_topology(
            vertex_contour_by_topology(
                df, truth_x_col, truth_y_col, truth_z_col,
                pred_x_col, pred_y_col, pred_z_col, is_track_col,
                bins=bins, sigma=sigma,
            ),
            ax=ax,
            label=model_name,
            color=model_color(model_name),
        )
    ax.legend()
    return ax


def vertex_contour_axis_ranges(
    predictions: Dict[str, pd.DataFrame],
    truth_x_col: str,
    truth_y_col: str,
    truth_z_col: str,
    pred_x_col: str,
    pred_y_col: str,
    pred_z_col: str,
    is_track_col: str,
    padding: float = 0.25,
    percentiles: Tuple[float, float] = (1.0, 99.0),
    sigma: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Data-driven `(depth_edges, radial_edges)` for one contour panel.

    `vertex_contour_by_topology`'s `bins=100` default spans the full
    min-max residual range, where a handful of outliers stretch the axes
    until the 68% contour shrinks to a dot. Build one generous
    first-pass binning, compute every model's 68% contour on it, and
    crop to the union of *those* bounding boxes plus `padding`.
    """
    depths = []
    radials = []
    for df in predictions.values():
        radial, depth = radial_and_depth(
            df[truth_x_col], df[truth_y_col], df[truth_z_col],
            df[pred_x_col], df[pred_y_col], df[pred_z_col],
        )
        depths.append(np.asarray(depth))
        radials.append(np.asarray(radial))
    depth_lo, depth_hi = np.percentile(np.concatenate(depths), percentiles)
    radial_hi_broad = np.percentile(np.concatenate(radials), percentiles[1])
    broad_bins = [
        np.linspace(depth_lo, depth_hi, 150),
        np.linspace(0.0, radial_hi_broad, 150),
    ]

    depth_min = depth_max = radial_max = None
    for df in predictions.values():
        result = vertex_contour_by_topology(
            df, truth_x_col, truth_y_col, truth_z_col,
            pred_x_col, pred_y_col, pred_z_col, is_track_col,
            bins=broad_bins, sigma=sigma,
        )
        for X, Y, H, level, median_depth, median_radial in result.values():
            mask = H >= level
            xs = np.append(X[mask], median_depth)
            ys = np.append(Y[mask], median_radial)
            depth_min = min(depth_min, xs.min()) if depth_min is not None \
                else xs.min()
            depth_max = max(depth_max, xs.max()) if depth_max is not None \
                else xs.max()
            radial_max = max(radial_max, ys.max()) if radial_max is not None \
                else ys.max()
    if depth_min is None:
        # Too little data for any bin to clear the containment threshold.
        depth_min, depth_max = depth_lo, depth_hi
        radial_max = radial_hi_broad
    assert depth_max is not None and radial_max is not None

    depth_pad = (depth_max - depth_min) * padding
    # Radial distance is non-negative and the paper's radial axis starts
    # at exactly 0; only its upper edge needs padding.
    return (
        np.linspace(depth_min - depth_pad, depth_max + depth_pad, 100),
        np.linspace(0.0, radial_max * (1 + padding), 100),
    )
