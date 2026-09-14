"""Energy reconstruction metrics: calibration-style resolution, plain and
split by track/cascade (CC/NC for muon-neutrino events).
"""

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from nubench.evaluation.resolution import binned_percentiles


def energy_calibration(
    df: pd.DataFrame,
    truth_col: str,
    pred_col: str,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 60,
) -> pd.DataFrame:
    """Compute reconstructed-vs-true energy calibration in bins of true energy.

    Reports the raw predicted energy directly - median reconstructed
    energy and its 68% band per true-energy bin - suitable for a
    calibration-style plot with a y=x "perfect reconstruction" reference
    line.

    Args:
        df: DataFrame containing at least `truth_col` and `pred_col`.
        truth_col: Name of the column holding the true energy.
        pred_col: Name of the column holding the predicted energy.
        bins: Bin edges for the true energy. If None, `n_bins` log-spaced
            bin edges are constructed automatically from the range of
            `df[truth_col]`.
        n_bins: Number of bins to construct when `bins` is not given.
            Ignored if `bins` is given.

    Returns:
        A DataFrame as returned by `binned_percentiles`: "bin_center" is
        the median true energy per bin, and "p16"/"p50"/"p84" bound the
        distribution of `df[pred_col]` (the raw reconstructed energy, not
        a residual) in that bin.
    """
    if bins is None:
        bins = np.logspace(
            np.log10(df[truth_col].min()),
            np.log10(df[truth_col].max()),
            n_bins + 1,
        )
    return binned_percentiles(x=df[truth_col], y=df[pred_col], bins=bins)


def energy_calibration_by_topology(
    df: pd.DataFrame,
    truth_col: str,
    pred_col: str,
    is_track_col: str,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 60,
) -> Dict[str, pd.DataFrame]:
    """Compute energy calibration split by track/cascade.

    Reuses `energy_calibration` on track-only and cascade-only subsets,
    same reasoning as `vertex_resolution_by_topology`. Restricted to
    muon-neutrino events (the usual `is_track_col` convention used
    everywhere else in this module), "track" and "cascade" here are the
    same thing as CC and NC: a muon-neutrino CC interaction produces a
    track (the outgoing muon), while NC produces a cascade - so this
    dict's two keys are literally the CC/NC split, not a new concept.

    Args:
        df: DataFrame containing `truth_col`, `pred_col`, and
            `is_track_col`.
        truth_col: Name of the column holding the true energy.
        pred_col: Name of the column holding the predicted energy.
        is_track_col: Name of a boolean column marking track/CC (True)
            vs. cascade/NC (False) events. Build this yourself before
            calling, same as every other `*_by_topology` function here.
        bins: Bin edges for the true energy, shared across both track and
            cascade. If None, `n_bins` log-spaced bin edges are
            constructed automatically from the range of `df[truth_col]`.
        n_bins: Number of bins to construct when `bins` is not given.

    Returns:
        A dict with keys "track" and "cascade" (each a DataFrame in the
        same shape `energy_calibration` returns).
    """
    if bins is None:
        bins = np.logspace(
            np.log10(df[truth_col].min()),
            np.log10(df[truth_col].max()),
            n_bins + 1,
        )
    track = energy_calibration(
        df[df[is_track_col]], truth_col, pred_col, bins=bins
    )
    cascade = energy_calibration(
        df[~df[is_track_col]], truth_col, pred_col, bins=bins
    )
    return {"track": track, "cascade": cascade}
