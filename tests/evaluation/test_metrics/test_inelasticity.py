"""Unit tests for `nubench.evaluation.metrics.inelasticity`."""

import numpy as np
import pandas as pd

from nubench.evaluation.metrics import (
    inelasticity_distribution_by_energy_regime,
    inelasticity_resolution,
)
from nubench.evaluation.resolution import binned_percentiles


def test_inelasticity_resolution_matches_manual_residual() -> None:
    """The residual should match a manually-computed absolute difference."""
    rng = np.random.default_rng(seed=8)
    n = 500
    truth = rng.uniform(0, 1, size=n)
    pred = np.clip(truth + rng.normal(scale=0.1, size=n), 0.0, 1.0)
    energy = rng.uniform(10, 1000, size=n)
    df = pd.DataFrame(
        {"true_y": truth, "pred_y": pred, "energy": energy}
    )
    bins = np.array([10.0, 100.0, 1000.0])

    result = inelasticity_resolution(
        df,
        truth_col="true_y",
        pred_col="pred_y",
        energy_col="energy",
        bins=bins,
    )

    residual = np.abs(pred - truth)
    expected = binned_percentiles(energy, residual, bins)

    for column in ("bin_center", "p16", "p50", "p84"):
        assert np.allclose(
            result[column], expected[column], equal_nan=True
        )


def test_inelasticity_resolution_is_column_name_agnostic() -> None:
    """The function must not assume any particular column names."""
    df = pd.DataFrame(
        {
            "weird_truth": [0.2, 0.2, 0.8, 0.8],
            "weird_pred": [0.3, 0.1, 0.6, 1.0],
            "weird_energy": [10.0, 20.0, 90.0, 95.0],
        }
    )
    bins = np.array([1.0, 50.0, 200.0])

    result = inelasticity_resolution(
        df,
        truth_col="weird_truth",
        pred_col="weird_pred",
        energy_col="weird_energy",
        bins=bins,
    )

    assert list(result.columns) == ["bin_center", "p16", "p50", "p84"]
    assert len(result) == 2


def test_inelasticity_resolution_binned_by_energy_not_value() -> None:
    """The binning variable must be `energy_col`, not the inelasticity columns.

    Constructs data where inelasticity values ([0, 1]) and energy values
    ([10, 10000]) are unrelated, with a perfect predictor (residual ~0
    everywhere) - a bug that accidentally binned by inelasticity instead
    of energy would produce `bin_center` values within [0, 1] instead of
    the (very different) energy range.
    """
    rng = np.random.default_rng(seed=9)
    n = 2000
    truth = rng.uniform(0, 1, size=n)
    energy = rng.uniform(10, 10_000, size=n)
    df = pd.DataFrame({"truth": truth, "pred": truth, "energy": energy})

    result = inelasticity_resolution(
        df,
        truth_col="truth",
        pred_col="pred",
        energy_col="energy",
        n_bins=10,
    )

    assert result["bin_center"].min() >= 10.0
    assert result["bin_center"].max() <= 10_000.0
    assert result["p50"].notna().any()
    assert np.allclose(result["p50"].dropna(), 0.0, atol=1e-9)


def test_inelasticity_distribution_by_energy_regime_known_values() -> None:
    """Each regime's pred/truth histogram should land in its known bin."""
    # Values sit safely mid-bin (not on a bin edge) to avoid floating-point
    # boundary ambiguity between the literal value and the computed edge.
    is_le = np.array([True] * 10 + [False] * 10)
    df = pd.DataFrame(
        {
            "pred": np.where(is_le, 0.15, 0.65),
            "truth": np.where(is_le, 0.35, 0.85),
            "energy": np.where(is_le, 50.0, 500.0),
        }
    )
    bins = np.linspace(0, 1, 11)  # 10 bins, width 0.1.

    result = inelasticity_distribution_by_energy_regime(
        df,
        pred_col="pred",
        truth_col="truth",
        energy_col="energy",
        energy_threshold=100.0,
        bins=bins,
    )

    assert set(result.keys()) == {
        "low_energy_pred",
        "high_energy_pred",
        "low_energy_truth",
        "high_energy_truth",
    }

    def pct_at(df: pd.DataFrame, bin_left: float) -> float:
        return df[np.isclose(df["bin_left"], bin_left)]["percentage"].iloc[0]

    assert np.isclose(pct_at(result["low_energy_pred"], 0.1), 100.0)
    assert np.isclose(pct_at(result["low_energy_truth"], 0.3), 100.0)
    assert np.isclose(pct_at(result["high_energy_pred"], 0.6), 100.0)
    assert np.isclose(pct_at(result["high_energy_truth"], 0.8), 100.0)


def test_inelasticity_distribution_by_energy_regime_shares_bins() -> None:
    """All four histograms should use identical bin edges."""
    rng = np.random.default_rng(seed=5)
    n = 400
    df = pd.DataFrame(
        {
            "pred": rng.uniform(0, 1, size=n),
            "truth": rng.uniform(0, 1, size=n),
            "energy": rng.uniform(10, 10_000, size=n),
        }
    )

    result = inelasticity_distribution_by_energy_regime(
        df,
        pred_col="pred",
        truth_col="truth",
        energy_col="energy",
        n_bins=10,
    )

    reference = result["low_energy_pred"]["bin_left"]
    for key in (
        "high_energy_pred", "low_energy_truth", "high_energy_truth"
    ):
        assert np.allclose(result[key]["bin_left"], reference)
