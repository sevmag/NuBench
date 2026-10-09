"""Track/cascade classification metrics: ROC curves and the raw
classifier score distribution.

Scores are used as-is. Some NuBench files store a raw logit rather than
a probability, so load through `nubench.data.load_feature_predictions`
(or call `nubench.data.normalize_score_column` yourself) to get those
squashed to [0, 1] first.
"""

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from nubench.evaluation.resolution import histogram_counts


def roc_curve_data(
    df: pd.DataFrame,
    truth_col: str,
    score_col: str,
) -> pd.DataFrame:
    """ROC curve points as a DataFrame with "fpr", "tpr", "threshold"."""
    fpr, tpr, threshold = roc_curve(df[truth_col], df[score_col])
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": threshold})


def track_score_distribution_by_topology(
    df: pd.DataFrame,
    score_col: str,
    is_track_col: str,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 50,
) -> Dict[str, pd.DataFrame]:
    """Histogram the classifier score for track and cascade events.

    Returns {"track", "cascade"} of `histogram_counts` frames
    ("bin_left"/"count") - raw counts, since the paper plots absolute
    log-counts rather than a normalized shape. Bins default to `n_bins`
    linear edges over `df[score_col]`, shared by both groups.
    """
    if bins is None:
        bins = np.linspace(
            df[score_col].min(),
            df[score_col].max(),
            n_bins + 1,
        )
    return {
        "track": histogram_counts(df[score_col][df[is_track_col]], bins),
        "cascade": histogram_counts(df[score_col][~df[is_track_col]], bins),
    }


def roc_curve_data_by_energy_regime(
    df: pd.DataFrame,
    truth_col: str,
    score_col: str,
    energy_col: str,
    low_threshold: float = 100.0,
    high_threshold: float = 1000.0,
) -> Dict[str, pd.DataFrame]:
    """ROC curves for three energy regimes.

    Returns {"low_energy", "mid_energy", "high_energy"}, split at
    `low_threshold` and `high_threshold` (both inclusive upper bounds).
    """
    energy = df[energy_col]
    is_mid = (energy > low_threshold) & (energy <= high_threshold)
    return {
        "low_energy": roc_curve_data(
            df[energy <= low_threshold], truth_col, score_col
        ),
        "mid_energy": roc_curve_data(df[is_mid], truth_col, score_col),
        "high_energy": roc_curve_data(
            df[energy > high_threshold], truth_col, score_col
        ),
    }
