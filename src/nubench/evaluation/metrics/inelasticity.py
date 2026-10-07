"""Inelasticity reconstruction metrics: resolution vs. energy and the raw
value distribution split by energy regime.
"""

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from nubench.evaluation.resolution import (
    binned_percentiles,
    combined_log_bins,
    histogram_percentage,
)


def inelasticity_resolution(
    df: pd.DataFrame,
    truth_col: str,
    pred_col: str,
    energy_col: str,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 30,
) -> pd.DataFrame:
    """Percentiles of `|pred - truth|` per true-energy bin.

    Inelasticity is already dimensionless in [0, 1], so the residual is
    the absolute difference, not a relative percentage.
    """
    residual = (df[pred_col] - df[truth_col]).abs()
    if bins is None:
        bins = combined_log_bins([df[energy_col]], n_bins)
    return binned_percentiles(x=df[energy_col], y=residual, bins=bins)


def inelasticity_distribution_by_energy_regime(
    df: pd.DataFrame,
    pred_col: str,
    truth_col: str,
    energy_col: str,
    energy_threshold: float = 100.0,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 50,
) -> Dict[str, pd.DataFrame]:
    """Histogram the raw inelasticity value, split low/high energy.

    It histograms the predicted *and* true values, to compare the shape
    of a model's distribution against the truth rather than just its accuracy.
    """
    if bins is None:
        bins = np.linspace(
            min(df[pred_col].min(), df[truth_col].min()),
            max(df[pred_col].max(), df[truth_col].max()),
            n_bins + 1,
        )
    is_low = df[energy_col] <= energy_threshold
    return {
        "low_energy_pred": histogram_percentage(df[pred_col][is_low], bins),
        "low_energy_truth": histogram_percentage(df[truth_col][is_low], bins),
        "high_energy_pred": histogram_percentage(df[pred_col][~is_low], bins),
        "high_energy_truth": histogram_percentage(
            df[truth_col][~is_low], bins
        ),
    }
