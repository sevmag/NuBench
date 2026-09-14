"""Direction reconstruction metrics: angular resolution vs. energy, and the
opening-angle error distribution - both plain and split by
track/cascade/muon.
"""

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from nubench.evaluation.resolution import (
    binned_percentiles,
    histogram_percentage,
    opening_angle,
    spherical_to_cartesian,
)


def direction_resolution(
    df: pd.DataFrame,
    truth_zenith_col: str,
    truth_azimuth_col: str,
    pred_x_col: str,
    pred_y_col: str,
    pred_z_col: str,
    energy_col: str,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 30,
    degrees: bool = True,
) -> pd.DataFrame:
    """Compute angular resolution in bins of true energy.

    The binning variable (`energy_col`) is a *different* column from the
    ones the residual itself is computed from (the angle/direction
    columns) - true energy is what direction resolution is typically
    reported as a function of, not direction itself.

    Args:
        df: DataFrame containing the angle/vector columns below, plus
            `energy_col`.
        truth_zenith_col: Name of the column holding the true zenith angle,
            in radians.
        truth_azimuth_col: Name of the column holding the true azimuth
            angle, in radians.
        pred_x_col: Name of the column holding the predicted direction
            vector's x-component.
        pred_y_col: Name of the column holding the predicted direction
            vector's y-component.
        pred_z_col: Name of the column holding the predicted direction
            vector's z-component.
        energy_col: Name of the column holding the true energy - the
            variable to bin by.
        bins: Bin edges for the true energy. If None, `n_bins` log-spaced
            bin edges are constructed automatically from the range of
            `df[energy_col]`.
        n_bins: Number of bins to construct when `bins` is not given.
            Ignored if `bins` is given.
        degrees: If True (default), report the opening angle in degrees.
            Otherwise radians.

    Returns:
        A DataFrame as returned by `binned_percentiles`: "bin_center" is
        the median true energy per bin, and "p16"/"p50"/"p84" bound the
        distribution of the opening angle between the true and predicted
        direction in that bin.
    """
    truth_x, truth_y, truth_z = spherical_to_cartesian(
        df[truth_zenith_col], df[truth_azimuth_col]
    )
    opening_angles = opening_angle(
        truth_x, truth_y, truth_z,
        df[pred_x_col], df[pred_y_col], df[pred_z_col],
        degrees=degrees
    )
    if bins is None:
        bins = np.logspace(
            np.log10(df[energy_col].min()),
            np.log10(df[energy_col].max()),
            n_bins + 1,
        )
    return binned_percentiles(x=df[energy_col], y=opening_angles, bins=bins)


def direction_resolution_by_topology(
    df: pd.DataFrame,
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
) -> Dict[str, pd.DataFrame]:
    """Compute angular resolution vs. energy, split by track/cascade.

    Reuses `direction_resolution` on different subsets/columns rather than
    computing anything new: once on track-only events, once on
    cascade-only events, and (if muon columns are given) once comparing
    the true direction against the outgoing muon's own kinematic
    direction for track events - a physical lower bound on achievable
    resolution, not a model prediction at all.

    Args:
        df: DataFrame containing all the columns referenced below.
        truth_zenith_col: Name of the column holding the true zenith
            angle, in radians.
        truth_azimuth_col: Name of the column holding the true azimuth
            angle, in radians.
        pred_x_col: Name of the column holding the predicted direction
            vector's x-component.
        pred_y_col: Name of the column holding the predicted direction
            vector's y-component.
        pred_z_col: Name of the column holding the predicted direction
            vector's z-component.
        energy_col: Name of the column holding the true energy - the
            variable to bin by.
        is_track_col: Name of a boolean column marking track (True) vs.
            cascade (False) events. Build this yourself before calling -
            e.g. NuBench's own convention is
            `(df["interaction"] == 1) & (df["initial_state_type"].abs() == 14)`
            - since it's dataset-specific selection logic, not something
            this function should assume.
        muon_zenith_col: Name of the column holding the outgoing muon's
            own zenith angle, in radians. If either this or
            `muon_azimuth_col` is None, the "muon" entry is omitted from
            the result.
        muon_azimuth_col: Name of the column holding the outgoing muon's
            own azimuth angle, in radians.
        bins: Bin edges for the true energy, shared across all computed
            entries. If None, `n_bins` log-spaced bin edges are
            constructed automatically from the range of `df[energy_col]`.
        n_bins: Number of bins to construct when `bins` is not given.
        degrees: If True (default), report the opening angle in degrees.
            Otherwise radians.

    Returns:
        A dict with keys "track" and "cascade" (each a DataFrame in the
        same shape `direction_resolution` returns), plus "muon" if both
        muon columns were given.
    """
    if bins is None:
        bins = np.logspace(
            np.log10(df[energy_col].min()),
            np.log10(df[energy_col].max()),
            n_bins + 1,
            )
    track = direction_resolution(
        df[df[is_track_col]],
        truth_zenith_col,
        truth_azimuth_col,
        pred_x_col,
        pred_y_col,
        pred_z_col,
        energy_col,
        bins=bins,
        degrees=degrees
        )
    cascade = direction_resolution(
        df[~df[is_track_col]],
        truth_zenith_col,
        truth_azimuth_col,
        pred_x_col,
        pred_y_col,
        pred_z_col,
        energy_col,
        bins=bins,
        degrees=degrees
        )
    result = {"track": track, "cascade": cascade}
    if muon_zenith_col is not None and muon_azimuth_col is not None:
        track_df = df[df[is_track_col]].copy()
        muon_x, muon_y, muon_z = spherical_to_cartesian(
            track_df[muon_zenith_col], track_df[muon_azimuth_col]
            )
        track_df["_muon_x"] = muon_x
        track_df["_muon_y"] = muon_y
        track_df["_muon_z"] = muon_z
        muon = direction_resolution(
            track_df,
            truth_zenith_col,
            truth_azimuth_col,
            "_muon_x",
            "_muon_y",
            "_muon_z",
            energy_col,
            bins=bins,
            degrees=degrees,
            )
        result["muon"] = muon
    return result


def direction_error_distribution(
    df: pd.DataFrame,
    truth_zenith_col: str,
    truth_azimuth_col: str,
    pred_x_col: str,
    pred_y_col: str,
    pred_z_col: str,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 120,
    degrees: bool = True,
) -> pd.DataFrame:
    """Compute what percentage of events fall in each opening-angle bin.

    Unlike `direction_resolution`, this isn't binned by energy at all - it
    histograms the opening angle between the true and predicted direction
    itself, reporting what percentage of events land in each angle bin.

    Args:
        df: DataFrame containing the angle/vector columns below.
        truth_zenith_col: Name of the column holding the true zenith angle,
            in radians.
        truth_azimuth_col: Name of the column holding the true azimuth
            angle, in radians.
        pred_x_col: Name of the column holding the predicted direction
            vector's x-component.
        pred_y_col: Name of the column holding the predicted direction
            vector's y-component.
        pred_z_col: Name of the column holding the predicted direction
            vector's z-component.
        bins: Bin edges for the opening angle. If None, `n_bins`
            linearly-spaced bin edges are constructed automatically from
            the range of the computed opening angles.
        n_bins: Number of bins to construct when `bins` is not given.
            Ignored if `bins` is given.
        degrees: If True (default), compute the opening angle in degrees.
            Otherwise radians.

    Returns:
        A DataFrame as returned by `histogram_percentage`: "bin_left" is
        each bin's left edge, and "percentage" is the percentage of
        events whose opening angle falls in that bin.
    """
    truth_x, truth_y, truth_z = spherical_to_cartesian(
        df[truth_zenith_col], df[truth_azimuth_col]
    )
    opening_angles = opening_angle(
        truth_x, truth_y, truth_z,
        df[pred_x_col], df[pred_y_col], df[pred_z_col],
        degrees=degrees
    )
    if bins is None:
        bins = np.linspace(
            opening_angles.min(),
            opening_angles.max(),
            n_bins + 1,
        )
    return histogram_percentage(opening_angles, bins)


def direction_error_distribution_by_topology(
    df: pd.DataFrame,
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
) -> Dict[str, pd.DataFrame]:
    """Compute the opening-angle distribution, split by track/cascade/muon.

    Reuses `direction_error_distribution` on different subsets/columns,
    the same way `direction_resolution_by_topology` reuses
    `direction_resolution`: once on track-only events, once on
    cascade-only events, and (if muon columns are given) once comparing
    the true direction against the outgoing muon's own kinematic
    direction for track events.

    Args:
        df: DataFrame containing all the columns referenced below.
        truth_zenith_col: Name of the column holding the true zenith
            angle, in radians.
        truth_azimuth_col: Name of the column holding the true azimuth
            angle, in radians.
        pred_x_col: Name of the column holding the predicted direction
            vector's x-component.
        pred_y_col: Name of the column holding the predicted direction
            vector's y-component.
        pred_z_col: Name of the column holding the predicted direction
            vector's z-component.
        is_track_col: Name of a boolean column marking track (True) vs.
            cascade (False) events.
        muon_zenith_col: Name of the column holding the outgoing muon's
            own zenith angle, in radians. If either this or
            `muon_azimuth_col` is None, the "muon" entry is omitted from
            the result.
        muon_azimuth_col: Name of the column holding the outgoing muon's
            own azimuth angle, in radians.
        bins: Bin edges for the opening angle, shared across all computed
            entries. If None, built once from the opening angle over the
            *whole* DataFrame (which - since track and cascade partition
            it - already spans both), so every topology is binned
            identically.
        n_bins: Number of bins to construct when `bins` is not given.
        degrees: If True (default), compute the opening angle in degrees.
            Otherwise radians.

    Returns:
        A dict with keys "track" and "cascade" (each a DataFrame in the
        same shape `direction_error_distribution` returns), plus "muon"
        if both muon columns were given.
    """
    if bins is None:
        truth_x, truth_y, truth_z = spherical_to_cartesian(
            df[truth_zenith_col], df[truth_azimuth_col]
        )
        opening_angles = opening_angle(
            truth_x, truth_y, truth_z,
            df[pred_x_col], df[pred_y_col], df[pred_z_col],
            degrees=degrees
        )
        bins = np.linspace(
            opening_angles.min(),
            opening_angles.max(),
            n_bins + 1,
        )
    track = direction_error_distribution(
        df[df[is_track_col]],
        truth_zenith_col,
        truth_azimuth_col,
        pred_x_col,
        pred_y_col,
        pred_z_col,
        bins=bins,
        degrees=degrees
    )
    cascade = direction_error_distribution(
        df[~df[is_track_col]],
        truth_zenith_col,
        truth_azimuth_col,
        pred_x_col,
        pred_y_col,
        pred_z_col,
        bins=bins,
        degrees=degrees
    )
    result = {"track": track, "cascade": cascade}
    if muon_zenith_col is not None and muon_azimuth_col is not None:
        track_df = df[df[is_track_col]].copy()
        muon_x, muon_y, muon_z = spherical_to_cartesian(
            track_df[muon_zenith_col], track_df[muon_azimuth_col]
            )
        track_df["_muon_x"] = muon_x
        track_df["_muon_y"] = muon_y
        track_df["_muon_z"] = muon_z
        muon = direction_error_distribution(
            track_df,
            truth_zenith_col,
            truth_azimuth_col,
            "_muon_x",
            "_muon_y",
            "_muon_z",
            bins=bins,
            degrees=degrees,
            )
        result["muon"] = muon
    return result
