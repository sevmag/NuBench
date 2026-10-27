"""Vertex reconstruction metrics: position resolution vs. energy and the
2D depth/radial containment contour, both split by track/cascade.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from nubench.evaluation.resolution import (
    binned_percentiles,
    combined_log_bins,
    contour_68_level,
    euclidean_distance,
    radial_and_depth,
)

Contour = Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]


def vertex_resolution(
    df: pd.DataFrame,
    truth_x_col: str,
    truth_y_col: str,
    truth_z_col: str,
    pred_x_col: str,
    pred_y_col: str,
    pred_z_col: str,
    energy_col: str,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 30,
) -> pd.DataFrame:
    """Percentiles of the true-to-predicted vertex distance, per energy bin.

    The binning variable (`energy_col`) is separate from the residual's own
    columns. Bins default to `n_bins` log-spaced edges over `df[energy_col]`.
    """
    distances = euclidean_distance(
        df[truth_x_col], df[truth_y_col], df[truth_z_col],
        df[pred_x_col], df[pred_y_col], df[pred_z_col],
    )
    if bins is None:
        bins = combined_log_bins([df[energy_col]], n_bins)
    return binned_percentiles(x=df[energy_col], y=distances, bins=bins)


def vertex_resolution_by_topology(
    df: pd.DataFrame,
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
) -> Dict[str, pd.DataFrame]:
    """`vertex_resolution` for track and cascade, on shared bins."""
    if bins is None:
        bins = combined_log_bins([df[energy_col]], n_bins)
    return {
        name: vertex_resolution(
            subset, truth_x_col, truth_y_col, truth_z_col,
            pred_x_col, pred_y_col, pred_z_col, energy_col, bins=bins,
        )
        for name, subset in (
            ("track", df[df[is_track_col]]),
            ("cascade", df[~df[is_track_col]]),
        )
    }


def vertex_contour_by_topology(
    df: pd.DataFrame,
    truth_x_col: str,
    truth_y_col: str,
    truth_z_col: str,
    pred_x_col: str,
    pred_y_col: str,
    pred_z_col: str,
    is_track_col: str,
    bins: Union[int, Sequence[int], Sequence[np.ndarray]] = 100,
    sigma: float = 3.0,
) -> Dict[str, Contour]:
    """68%-containment depth/radial density contours per topology.

    Returns {"track", "cascade"} of `(X, Y, H, level, median_depth,
    median_radial)` - `contour_68_level`'s output plus the median point,
    since the contour plot draws both the outline and a marker at
    the typical point. `bins` and `sigma` pass through, identically for
    both topologies, so the contours stay comparable.
    """
    result = {}
    for name, subset in (
        ("track", df[df[is_track_col]]),
        ("cascade", df[~df[is_track_col]]),
    ):
        radial, depth = radial_and_depth(
            subset[truth_x_col], subset[truth_y_col], subset[truth_z_col],
            subset[pred_x_col], subset[pred_y_col], subset[pred_z_col],
        )
        X, Y, H, level = contour_68_level(
            depth, radial, bins=bins, sigma=sigma
        )
        result[name] = (X, Y, H, level, np.median(depth), np.median(radial))
    return result
