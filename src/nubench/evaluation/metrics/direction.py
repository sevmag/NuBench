"""Direction reconstruction metrics for angular resolution vs. energy and the
opening-angle error distribution, plain and split by track/cascade/muon.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from nubench.evaluation.resolution import (
    binned_percentiles,
    combined_log_bins,
    histogram_percentage,
    opening_angle,
    spherical_to_cartesian,
)

MUON_COLS = ("_muon_x", "_muon_y", "_muon_z")


def _opening_angles(
    df: pd.DataFrame,
    truth_zenith_col: str,
    truth_azimuth_col: str,
    pred_x_col: str,
    pred_y_col: str,
    pred_z_col: str,
    degrees: bool,
) -> np.ndarray:
    """Opening angle between the true (spherical) and predicted direction."""
    truth_x, truth_y, truth_z = spherical_to_cartesian(
        df[truth_zenith_col], df[truth_azimuth_col]
    )
    return opening_angle(
        truth_x, truth_y, truth_z,
        df[pred_x_col], df[pred_y_col], df[pred_z_col],
        degrees=degrees,
    )


def _track_events_with_muon_direction(
    df: pd.DataFrame,
    is_track_col: str,
    muon_zenith_col: str,
    muon_azimuth_col: str,
) -> pd.DataFrame:
    """Track-only events with the muon direction added as `MUON_COLS`.

    Lets the muon's own kinematic direction - a physical lower bound on
    achievable resolution, not a model prediction - be passed through the
    same plain metric functions as any predicted direction.
    """
    track_df = df[df[is_track_col]].copy()
    muon = spherical_to_cartesian(
        track_df[muon_zenith_col], track_df[muon_azimuth_col]
    )
    for col, values in zip(MUON_COLS, muon):
        track_df[col] = values
    return track_df


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
    """Opening-angle percentiles in bins of true energy.

    Returns `binned_percentiles`' columns; bins default to `n_bins`
    log-spaced edges over `df[energy_col]`.
    """
    angles = _opening_angles(
        df, truth_zenith_col, truth_azimuth_col,
        pred_x_col, pred_y_col, pred_z_col, degrees,
    )
    if bins is None:
        bins = combined_log_bins([df[energy_col]], n_bins)
    return binned_percentiles(x=df[energy_col], y=angles, bins=bins)


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
    """`direction_resolution` for track, cascade and (optionally) muon.

    `is_track_col` must already exist - it is dataset-specific selection
    logic (NuBench's own convention is
    `(interaction == 1) & (initial_state_type.abs() == 14)`), so build it
    with `nubench.data.add_is_track_column` first. "muon" is included
    only when both muon columns are given. Bins are shared across all
    entries.
    """
    if bins is None:
        bins = combined_log_bins([df[energy_col]], n_bins)

    def resolution(subset: pd.DataFrame, pred: Tuple[str, str, str]):
        return direction_resolution(
            subset, truth_zenith_col, truth_azimuth_col, *pred,
            energy_col, bins=bins, degrees=degrees,
        )

    pred_cols = (pred_x_col, pred_y_col, pred_z_col)
    result = {
        "track": resolution(df[df[is_track_col]], pred_cols),
        "cascade": resolution(df[~df[is_track_col]], pred_cols),
    }
    if muon_zenith_col is not None and muon_azimuth_col is not None:
        result["muon"] = resolution(
            _track_events_with_muon_direction(
                df, is_track_col, muon_zenith_col, muon_azimuth_col
            ),
            MUON_COLS,
        )
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
    """Percentage of events per opening-angle bin.

    Histograms the opening angle itself. Returns `histogram_percentage`'s
    columns ("bin_left", "percentage"); bins default to `n_bins` linear
    edges over the computed angles.
    """
    angles = _opening_angles(
        df, truth_zenith_col, truth_azimuth_col,
        pred_x_col, pred_y_col, pred_z_col, degrees,
    )
    if bins is None:
        bins = np.linspace(angles.min(), angles.max(), n_bins + 1)
    return histogram_percentage(angles, bins)


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
    """`direction_error_distribution` per topology, on shared bins."""
    if bins is None:
        angles = _opening_angles(
            df, truth_zenith_col, truth_azimuth_col,
            pred_x_col, pred_y_col, pred_z_col, degrees,
        )
        bins = np.linspace(angles.min(), angles.max(), n_bins + 1)

    def distribution(subset: pd.DataFrame, pred: Tuple[str, str, str]):
        return direction_error_distribution(
            subset, truth_zenith_col, truth_azimuth_col, *pred,
            bins=bins, degrees=degrees,
        )

    pred_cols = (pred_x_col, pred_y_col, pred_z_col)
    result = {
        "track": distribution(df[df[is_track_col]], pred_cols),
        "cascade": distribution(df[~df[is_track_col]], pred_cols),
    }
    if muon_zenith_col is not None and muon_azimuth_col is not None:
        result["muon"] = distribution(
            _track_events_with_muon_direction(
                df, is_track_col, muon_zenith_col, muon_azimuth_col
            ),
            MUON_COLS,
        )
    return result
