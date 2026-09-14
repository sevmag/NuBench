"""Track/cascade classification metrics: ROC curves and AUC (plain and
split by energy regime), and the raw classifier score distribution split
by track/cascade.
"""

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from nubench.evaluation.resolution import histogram_counts


def roc_curve_data(
    df: pd.DataFrame,
    truth_col: str,
    score_col: str,
) -> pd.DataFrame:
    """Compute ROC curve points for a binary classifier.

    Unlike every other metric function here, this isn't binned by energy
    at all - a ROC curve is a property of the whole score distribution,
    evaluated across every possible decision threshold.

    Args:
        df: DataFrame containing `truth_col` and `score_col`.
        truth_col: Name of the column holding the true binary label (0/1,
            or bool) - e.g. "is this event a track".
        score_col: Name of the column holding the classifier's score
            (higher = more confident of the positive/"1" class). Real
            NuBench files sometimes store this as a raw logit rather than
            a probability - apply `torch.sigmoid` yourself first if so;
            this function does not guess.

    Returns:
        A DataFrame with columns "fpr", "tpr", "threshold", one row per
        distinct threshold `sklearn.metrics.roc_curve` considers, ordered
        so that "fpr" is monotonically increasing (suitable for plotting
        directly as a curve).
    """
    fpr, tpr, threshold = roc_curve(df[truth_col], df[score_col])
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": threshold})


def auc_score(
    df: pd.DataFrame,
    truth_col: str,
    score_col: str,
) -> float:
    """Compute the area under the ROC curve (AUC) for a binary classifier.

    Args:
        df: DataFrame containing `truth_col` and `score_col`.
        truth_col: Name of the column holding the true binary label (0/1,
            or bool).
        score_col: Name of the column holding the classifier's score
            (higher = more confident of the positive/"1" class).

    Returns:
        The AUC, a single number in [0, 1] (0.5 = no better than chance,
        1.0 = perfect separation).
    """
    auc = roc_auc_score(df[truth_col], df[score_col])
    return auc


def track_score_distribution_by_topology(
    df: pd.DataFrame,
    score_col: str,
    is_track_col: str,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 50,
) -> Dict[str, pd.DataFrame]:
    """Compute the classifier score distribution, split by track/cascade.

    Histograms the raw classifier score itself (not a residual, and with
    no "truth" counterpart - there's no true score to compare against),
    split into track-only and cascade-only events. Useful for seeing how
    cleanly a model's score separates the two classes, beyond what a
    single ROC curve or AUC number shows.

    Args:
        df: DataFrame containing `score_col` and `is_track_col`.
        score_col: Name of the column holding the classifier's score
            (higher = more confident the event is a track). Real NuBench
            files sometimes store this as a raw logit rather than a
            probability - apply `torch.sigmoid` yourself first if so,
            same as `roc_curve_data`; this function does not guess.
        is_track_col: Name of a boolean column marking track (True) vs.
            cascade (False) events.
        bins: Bin edges for the score. If None, `n_bins` linearly-spaced
            edges are constructed automatically from the range of
            `df[score_col]`, shared between both groups.
        n_bins: Number of bins to construct when `bins` is not given.

    Returns:
        A dict with keys "track" and "cascade", each a DataFrame as
        returned by `histogram_counts` ("bin_left"/"count") - raw counts,
        not a percentage, since the paper's own version of this plot
        shows absolute "Log Counts" rather than a normalized shape.
    """
    if bins is None:
        bins = np.linspace(
            df[score_col].min(),
            df[score_col].max(),
            n_bins + 1,
        )
    track = histogram_counts(df[score_col][df[is_track_col]], bins)
    cascade = histogram_counts(df[score_col][~df[is_track_col]], bins)
    return {
        "track": track,
        "cascade": cascade
    }


def roc_curve_data_by_energy_regime(
    df: pd.DataFrame,
    truth_col: str,
    score_col: str,
    energy_col: str,
    low_threshold: float = 100.0,
    high_threshold: float = 1000.0,
) -> Dict[str, pd.DataFrame]:
    """Compute ROC curves separately for three energy regimes.

    Reuses `roc_curve_data` on three energy-sliced subsets rather than
    computing anything new: low energy (<= `low_threshold`), mid energy
    (`low_threshold` < E <= `high_threshold`), and high energy
    (> `high_threshold`) - matching the paper's own three-regime ROC
    comparison, which shows a classifier's separation power changes with
    energy in a way a single overall ROC curve or AUC number can't.

    Args:
        df: DataFrame containing `truth_col`, `score_col`, `energy_col`.
        truth_col: Name of the column holding the true binary label.
        score_col: Name of the column holding the classifier's score.
        energy_col: Name of the column holding the true energy - the
            variable used to split into regimes.
        low_threshold: Upper bound (inclusive) of the "low energy"
            regime.
        high_threshold: Upper bound (inclusive) of the "mid energy"
            regime; everything above this is "high energy".

    Returns:
        A dict with keys "low_energy", "mid_energy", "high_energy", each
        a DataFrame as returned by `roc_curve_data`
        ("fpr"/"tpr"/"threshold").
    """
    is_low = df[energy_col] <= low_threshold
    is_mid = (df[energy_col] > low_threshold) & (
        df[energy_col] <= high_threshold
    )
    is_high = df[energy_col] > high_threshold
    low_energy = roc_curve_data(df[is_low], truth_col, score_col)
    mid_energy = roc_curve_data(df[is_mid], truth_col, score_col)
    high_energy = roc_curve_data(df[is_high], truth_col, score_col)
    return {
        "low_energy": low_energy,
        "mid_energy": mid_energy,
        "high_energy": high_energy
    }
