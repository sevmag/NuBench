"""Unit tests for `nubench.evaluation.metrics.energy`."""

import numpy as np
import pandas as pd

from nubench.evaluation.metrics import (
    energy_calibration,
    energy_calibration_by_topology,
)
from nubench.evaluation.resolution import binned_percentiles


def test_energy_calibration_matches_manual_binned_percentiles() -> None:
    """Should report raw predicted energy per bin, not a residual."""
    rng = np.random.default_rng(seed=10)
    truth = rng.uniform(10, 1000, size=500)
    pred = truth * (1 + rng.normal(scale=0.1, size=500))
    df = pd.DataFrame({"true_energy": truth, "reco_energy": pred})
    bins = np.array([10.0, 100.0, 1000.0])

    result = energy_calibration(
        df, truth_col="true_energy", pred_col="reco_energy", bins=bins
    )

    expected = binned_percentiles(truth, pred, bins)

    for column in ("bin_center", "p16", "p50", "p84"):
        assert np.allclose(
            result[column], expected[column], equal_nan=True
        )


def test_energy_calibration_is_column_name_agnostic() -> None:
    """The function must not assume any particular column names."""
    df = pd.DataFrame(
        {
            "weird_truth_name": [10.0, 10.0, 100.0, 100.0],
            "weird_pred_name": [11.0, 9.0, 90.0, 110.0],
        }
    )
    bins = np.array([1.0, 50.0, 200.0])

    result = energy_calibration(
        df,
        truth_col="weird_truth_name",
        pred_col="weird_pred_name",
        bins=bins,
    )

    assert list(result.columns) == ["bin_center", "p16", "p50", "p84"]
    assert len(result) == 2


def test_energy_calibration_default_bins() -> None:
    """Without `bins`, `n_bins` log-spaced edges should be built, and a
    perfect predictor should trace the y=x line exactly (median reco ==
    median truth in every populated bin, since reco == truth everywhere).
    """
    rng = np.random.default_rng(seed=11)
    truth = rng.uniform(10, 10_000, size=5000)
    df = pd.DataFrame({"energy": truth, "energy_pred": truth})

    result = energy_calibration(
        df, truth_col="energy", pred_col="energy_pred", n_bins=12
    )

    assert len(result) == 12
    populated = result.dropna()
    assert len(populated) > 0
    assert np.allclose(populated["p50"], populated["bin_center"])


def test_energy_calibration_by_topology_known_values() -> None:
    """Track/cascade should each reflect their own known calibration."""
    is_track = np.array([True] * 10 + [False] * 10)
    df = pd.DataFrame(
        {
            "truth": np.where(is_track, 100.0, 2000.0),
            "pred": np.where(is_track, 150.0, 1600.0),
            "is_track": is_track,
        }
    )
    bins = np.array([10.0, 500.0, 10000.0])

    result = energy_calibration_by_topology(
        df,
        truth_col="truth",
        pred_col="pred",
        is_track_col="is_track",
        bins=bins,
    )

    assert set(result.keys()) == {"track", "cascade"}
    assert np.isclose(result["track"]["p50"].iloc[0], 150.0)
    assert np.isclose(result["cascade"]["p50"].iloc[1], 1600.0)


def test_energy_calibration_by_topology_shares_bins() -> None:
    """Track and cascade must be binned with the same shared edges.

    Same disjoint-ranges trick used throughout: with correctly shared
    bins, each topology must show NaNs where only the *other* topology
    has data.
    """
    n_low, n_high = 100, 100
    truth_low = np.linspace(10, 100, n_low)
    truth_high = np.linspace(500, 1000, n_high)
    truth = np.concatenate([truth_low, truth_high])
    is_track = np.array([True] * n_low + [False] * n_high)
    df = pd.DataFrame(
        {
            "truth": truth,
            "pred": truth,
            "is_track": is_track,
        }
    )

    result = energy_calibration_by_topology(
        df,
        truth_col="truth",
        pred_col="pred",
        is_track_col="is_track",
        n_bins=10,
    )

    assert result["track"]["p50"].isna().any()
    assert result["cascade"]["p50"].isna().any()
