"""Vertex reconstruction metrics: position resolution vs. energy, and the
2D depth/radial containment contour - both split by track/cascade.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from nubench.evaluation.resolution import (
    binned_percentiles,
    contour_68_level,
    euclidean_distance,
    radial_and_depth,
)


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
    """Compute vertex-position resolution in bins of true energy.

    Same shape as `direction_resolution`: the binning variable
    (`energy_col`) is a separate column from the ones the residual itself
    is computed from (the position columns).

    Args:
        df: DataFrame containing the position columns below, plus
            `energy_col`.
        truth_x_col: Name of the column holding the true vertex
            x-coordinate.
        truth_y_col: Name of the column holding the true vertex
            y-coordinate.
        truth_z_col: Name of the column holding the true vertex
            z-coordinate.
        pred_x_col: Name of the column holding the predicted vertex
            x-coordinate.
        pred_y_col: Name of the column holding the predicted vertex
            y-coordinate.
        pred_z_col: Name of the column holding the predicted vertex
            z-coordinate.
        energy_col: Name of the column holding the true energy - the
            variable to bin by.
        bins: Bin edges for the true energy. If None, `n_bins` log-spaced
            bin edges are constructed automatically from the range of
            `df[energy_col]`.
        n_bins: Number of bins to construct when `bins` is not given.
            Ignored if `bins` is given.

    Returns:
        A DataFrame as returned by `binned_percentiles`: "bin_center" is
        the median true energy per bin, and "p16"/"p50"/"p84" bound the
        distribution of the Euclidean distance between the true and
        predicted vertex position in that bin.
    """
    truth_x = df[truth_x_col]
    truth_y = df[truth_y_col]
    truth_z = df[truth_z_col]
    pred_x = df[pred_x_col]
    pred_y = df[pred_y_col]
    pred_z = df[pred_z_col]
    distances = euclidean_distance(
        truth_x, truth_y, truth_z, pred_x, pred_y, pred_z
    )
    if bins is None:
        bins = np.logspace(
            np.log10(df[energy_col].min()),
            np.log10(df[energy_col].max()),
            n_bins + 1,
        )
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
    """Compute vertex-position resolution vs. energy, split by track/cascade.

    Reuses `vertex_resolution` on track-only and cascade-only subsets, the
    same reasoning as `direction_resolution_by_topology`. Unlike direction,
    there's no muon-baseline entry here - a reconstructed muon track has
    no equivalent physical "true vertex" of its own to compare against.

    Args:
        df: DataFrame containing the position columns below, plus
            `energy_col` and `is_track_col`.
        truth_x_col: Name of the column holding the true vertex
            x-coordinate.
        truth_y_col: Name of the column holding the true vertex
            y-coordinate.
        truth_z_col: Name of the column holding the true vertex
            z-coordinate.
        pred_x_col: Name of the column holding the predicted vertex
            x-coordinate.
        pred_y_col: Name of the column holding the predicted vertex
            y-coordinate.
        pred_z_col: Name of the column holding the predicted vertex
            z-coordinate.
        energy_col: Name of the column holding the true energy - the
            variable to bin by.
        is_track_col: Name of a boolean column marking track (True) vs.
            cascade (False) events.
        bins: Bin edges for the true energy, shared across both track and
            cascade. If None, `n_bins` log-spaced bin edges are
            constructed automatically from the range of `df[energy_col]`.
        n_bins: Number of bins to construct when `bins` is not given.

    Returns:
        A dict with keys "track" and "cascade" (each a DataFrame in the
        same shape `vertex_resolution` returns).
    """
    if bins is None:
        bins = np.logspace(
            np.log10(df[energy_col].min()),
            np.log10(df[energy_col].max()),
            n_bins + 1,
        )
    track = vertex_resolution(
        df[df[is_track_col]],
        truth_x_col,
        truth_y_col,
        truth_z_col,
        pred_x_col,
        pred_y_col,
        pred_z_col,
        energy_col,
        bins=bins,
    )
    cascade = vertex_resolution(
        df[~df[is_track_col]],
        truth_x_col,
        truth_y_col,
        truth_z_col,
        pred_x_col,
        pred_y_col,
        pred_z_col,
        energy_col,
        bins=bins,
    )
    return {"track": track, "cascade": cascade}


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
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]]:
    """Compute a 2D depth-vs-radial density contour, split by track/cascade.

    For each topology, computes the radial/depth position residual (via
    `radial_and_depth`) and its 68%-containment density contour (via
    `contour_68_level`), plus the median depth/radial as a single
    representative point - the paper's own vertex contour plot draws both
    the contour outline and this median point per topology.

    Args:
        df: DataFrame containing the position columns below, plus
            `is_track_col`.
        truth_x_col: Name of the column holding the true vertex
            x-coordinate.
        truth_y_col: Name of the column holding the true vertex
            y-coordinate.
        truth_z_col: Name of the column holding the true vertex
            z-coordinate.
        pred_x_col: Name of the column holding the predicted vertex
            x-coordinate.
        pred_y_col: Name of the column holding the predicted vertex
            y-coordinate.
        pred_z_col: Name of the column holding the predicted vertex
            z-coordinate.
        is_track_col: Name of a boolean column marking track (True) vs.
            cascade (False) events.
        bins: Passed through to `contour_68_level`'s own `bins` argument -
            the same value is used for both track and cascade, so their
            contours are directly comparable.
        sigma: Passed through to `contour_68_level`'s own `sigma`
            argument.

    Returns:
        A dict with keys "track" and "cascade", each mapping to a tuple
        `(X, Y, H, level, median_depth, median_radial)` - the first four
        as returned by `contour_68_level`, plus the median depth and
        median radial residual for that topology (for a marker at the
        "typical" point).
    """
    result = {}
    for name, subset in [
        ("track", df[df[is_track_col]]),
        ("cascade", df[~df[is_track_col]])
    ]:
        radial, depth = radial_and_depth(
            subset[truth_x_col],
            subset[truth_y_col],
            subset[truth_z_col],
            subset[pred_x_col],
            subset[pred_y_col],
            subset[pred_z_col])
        X, Y, H, level = contour_68_level(
            depth,
            radial,
            bins=bins,
            sigma=sigma)
        result[name] = (X, Y, H, level, np.median(depth), np.median(radial))
    return result
