"""Energy reconstruction metrics."""

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from nubench.evaluation.resolution import (
    binned_percentiles,
    combined_log_bins,
)


def energy_calibration(
    df: pd.DataFrame,
    truth_col: str,
    pred_col: str,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 60,
) -> pd.DataFrame:
    """Median reconstructed energy and its 68% band per true-energy bin.

    Returns `binned_percentiles`' columns ("bin_center", "p16", "p50", "p84").
    Bins default to `n_bins` log-spaced edges over `df[truth_col]`.
    """
    if bins is None:
        bins = combined_log_bins([df[truth_col]], n_bins)
    return binned_percentiles(x=df[truth_col], y=df[pred_col], bins=bins)
