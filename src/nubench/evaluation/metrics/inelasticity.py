"""Inelasticity reconstruction metrics: resolution vs. energy, and the raw
value distribution split by energy regime.
"""

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from nubench.evaluation.resolution import (
    binned_percentiles,
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
    """Compute inelasticity resolution in bins of true energy.

    Inelasticity is already a dimensionless quantity in [0, 1], so the
    residual here is the absolute difference `|pred - truth|`, not a
    relative percentage. Like direction/vertex, the binning variable
    (`energy_col`) is a separate column from the truth/pred columns.

    Args:
        df: DataFrame containing `truth_col`, `pred_col`, and `energy_col`.
        truth_col: Name of the column holding the true (visible)
            inelasticity.
        pred_col: Name of the column holding the predicted inelasticity.
            Note: real NuBench files are inconsistent about this column's
            name across models (e.g. `visible_inelasticity_pred` vs.
            `inelasticity_pred`) - pass whichever applies to your file.
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
        distribution of `|df[pred_col] - df[truth_col]|` in that bin.
    """
    residual = (df[pred_col] - df[truth_col]).abs()
    if bins is None:
        bins = np.logspace(
            np.log10(df[energy_col].min()),
            np.log10(df[energy_col].max()),
            n_bins + 1,
        )
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
    """Compute the inelasticity value distribution, split by energy regime.

    Unlike `inelasticity_resolution` (a residual binned by energy), this
    histograms the raw inelasticity *value* itself - both the predicted
    and the true value - split into two energy regimes rather than by
    track/cascade. Useful for comparing the overall *shape* of a model's
    predicted distribution against the true distribution's shape, not
    just its accuracy.

    Args:
        df: DataFrame containing `pred_col`, `truth_col`, `energy_col`.
        pred_col: Name of the column holding the predicted inelasticity.
        truth_col: Name of the column holding the true (visible)
            inelasticity.
        energy_col: Name of the column holding the true energy - the
            variable used to split into regimes.
        energy_threshold: Energy value separating "low" (<=) from "high"
            (>) energy events.
        bins: Bin edges for the inelasticity value. If None, `n_bins`
            linearly-spaced edges are constructed automatically from the
            combined range of `df[pred_col]` and `df[truth_col]`, so
            predicted and true distributions share the same bins.
        n_bins: Number of bins to construct when `bins` is not given.

    Returns:
        A dict with keys "low_energy_pred", "high_energy_pred",
        "low_energy_truth", "high_energy_truth" - each a DataFrame as
        returned by `histogram_percentage` ("bin_left"/"percentage").
    """
    if bins is None:
        combined_min = min(df[pred_col].min(), df[truth_col].min())
        combined_max = max(df[pred_col].max(), df[truth_col].max())
        bins = np.linspace(combined_min, combined_max, n_bins + 1)
    is_le = df[energy_col] <= energy_threshold
    is_he = ~is_le
    low_energy_pred = histogram_percentage(df[pred_col][is_le], bins)
    low_energy_truth = histogram_percentage(df[truth_col][is_le], bins)
    high_energy_pred = histogram_percentage(df[pred_col][is_he], bins)
    high_energy_truth = histogram_percentage(df[truth_col][is_he], bins)
    return {
        "low_energy_pred": low_energy_pred,
        "low_energy_truth": low_energy_truth,
        "high_energy_pred": high_energy_pred,
        "high_energy_truth": high_energy_truth
        }
