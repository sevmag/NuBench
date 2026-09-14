"""Unit tests for `nubench.evaluation.metrics.classification`."""

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from nubench.evaluation.metrics import (
    auc_score,
    roc_curve_data,
    roc_curve_data_by_energy_regime,
    track_score_distribution_by_topology,
)


def test_roc_curve_data_has_expected_columns() -> None:
    """The result should have exactly the fpr/tpr/threshold columns."""
    df = pd.DataFrame(
        {"truth": [0, 0, 0, 1, 1, 1], "score": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]}
    )

    result = roc_curve_data(df, truth_col="truth", score_col="score")

    assert list(result.columns) == ["fpr", "tpr", "threshold"]


def test_roc_curve_data_is_monotonic_in_fpr() -> None:
    """fpr should be monotonically non-decreasing, suitable for plotting."""
    rng = np.random.default_rng(seed=40)
    n = 500
    truth = rng.integers(0, 2, size=n)
    score = rng.uniform(0, 1, size=n)
    df = pd.DataFrame({"truth": truth, "score": score})

    result = roc_curve_data(df, truth_col="truth", score_col="score")

    assert np.all(np.diff(result["fpr"]) >= 0)


def test_roc_curve_data_perfect_separation_reaches_top_left() -> None:
    """Perfectly separated classes should include the (fpr=0, tpr=1) point."""
    df = pd.DataFrame(
        {
            "truth": [0, 0, 0, 1, 1, 1],
            "score": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        }
    )

    result = roc_curve_data(df, truth_col="truth", score_col="score")

    assert np.any((result["fpr"] == 0.0) & (result["tpr"] == 1.0))


def test_auc_score_perfect_separation_is_one() -> None:
    """Perfectly separated classes should score a perfect AUC of 1.0."""
    df = pd.DataFrame(
        {
            "truth": [0, 0, 0, 1, 1, 1],
            "score": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        }
    )

    assert np.isclose(auc_score(df, truth_col="truth", score_col="score"), 1.0)


def test_auc_score_matches_sklearn() -> None:
    """The result should match a direct `sklearn.metrics.roc_auc_score`."""
    rng = np.random.default_rng(seed=41)
    n = 500
    truth = rng.integers(0, 2, size=n)
    score = rng.uniform(0, 1, size=n)
    df = pd.DataFrame({"weird_truth": truth, "weird_score": score})

    result = auc_score(df, truth_col="weird_truth", score_col="weird_score")

    assert np.isclose(result, roc_auc_score(truth, score))


def test_track_score_distribution_by_topology_known_values() -> None:
    """Track/cascade should each reflect their own known score value."""
    is_track = np.array([True] * 10 + [False] * 10)
    df = pd.DataFrame(
        {
            "score": np.where(is_track, 0.85, 0.15),
            "is_track": is_track,
        }
    )
    bins = np.linspace(0, 1, 11)  # 10 bins, width 0.1.

    result = track_score_distribution_by_topology(
        df, score_col="score", is_track_col="is_track", bins=bins
    )

    assert set(result.keys()) == {"track", "cascade"}
    track = result["track"]
    cascade = result["cascade"]
    assert (
        track[np.isclose(track["bin_left"], 0.8)]["count"].iloc[0] == 10
    )
    assert (
        cascade[np.isclose(cascade["bin_left"], 0.1)]["count"].iloc[0]
        == 10
    )


def test_track_score_distribution_by_topology_shares_bins() -> None:
    """Track and cascade must be binned with the same shared edges.

    Track events confined to a small score range and cascade events to a
    disjoint, large score range - with correctly shared bins, each
    topology must show a count of 0 in the range only the *other*
    occupies.
    """
    n_side = 100
    track_scores = np.linspace(0.7, 0.8, n_side)
    cascade_scores = np.linspace(0.1, 0.2, n_side)
    scores = np.concatenate([track_scores, cascade_scores])
    is_track = np.array([True] * n_side + [False] * n_side)
    df = pd.DataFrame({"score": scores, "is_track": is_track})

    result = track_score_distribution_by_topology(
        df, score_col="score", is_track_col="is_track", n_bins=10
    )

    assert np.allclose(
        result["track"]["bin_left"], result["cascade"]["bin_left"]
    )
    assert result["track"]["count"].iloc[0] == 0
    assert result["cascade"]["count"].iloc[-1] == 0


def test_roc_curve_data_by_energy_regime_uses_correct_subset() -> None:
    """Each regime's ROC curve should only reflect its own energy slice."""
    # Low-energy events: perfectly separable (score matches truth exactly).
    low_energy = np.full(20, 50.0)
    low_truth = np.array([0] * 10 + [1] * 10)
    low_score = low_truth.astype(float)

    # Mid-energy events: perfectly separable in the *opposite* direction
    # (inverted score) - if the low-energy curve leaked in here by
    # mistake, this regime would incorrectly reach perfect separation too.
    mid_energy = np.full(20, 500.0)
    mid_truth = np.array([0] * 10 + [1] * 10)
    mid_score = 1 - mid_truth.astype(float)

    # High-energy events: exist just so the masks are exhaustive.
    high_energy = np.full(20, 5000.0)
    high_truth = np.array([0] * 10 + [1] * 10)
    high_score = high_truth.astype(float)

    df = pd.DataFrame(
        {
            "truth": np.concatenate([low_truth, mid_truth, high_truth]),
            "score": np.concatenate([low_score, mid_score, high_score]),
            "energy": np.concatenate(
                [low_energy, mid_energy, high_energy]
            ),
        }
    )

    result = roc_curve_data_by_energy_regime(
        df,
        truth_col="truth",
        score_col="score",
        energy_col="energy",
        low_threshold=100.0,
        high_threshold=1000.0,
    )

    assert set(result.keys()) == {
        "low_energy",
        "mid_energy",
        "high_energy",
    }
    # Low energy: perfect separation -> reaches (fpr=0, tpr=1).
    low = result["low_energy"]
    assert np.any((low["fpr"] == 0.0) & (low["tpr"] == 1.0))
    # Mid energy: perfectly *inverted* score -> never reaches (0, 1).
    mid = result["mid_energy"]
    assert not np.any((mid["fpr"] == 0.0) & (mid["tpr"] == 1.0))


def test_roc_curve_data_by_energy_regime_boundary_is_inclusive() -> None:
    """An event exactly at `low_threshold` should count as low energy."""
    df = pd.DataFrame(
        {
            "truth": [0, 0, 0, 1, 0, 1, 0, 1],
            "score": [0.1, 0.2, 0.3, 0.9, 0.4, 0.6, 0.5, 0.5],
            # First 4 events are the low-energy regime under test; the
            # rest just need to exist so the mid/high masks aren't empty
            # (an empty slice would make `roc_curve_data` itself error).
            "energy": [50.0, 60.0, 70.0, 100.0, 500.0, 500.0, 5000.0, 5000.0],
        }
    )

    result = roc_curve_data_by_energy_regime(
        df,
        truth_col="truth",
        score_col="score",
        energy_col="energy",
        low_threshold=100.0,
        high_threshold=1000.0,
    )

    # If the boundary event (energy=100.0, the only positive-class row)
    # were excluded from "low_energy", sklearn's roc_curve would never
    # report tpr=1.0 there - so reaching it proves it was included.
    assert np.any(result["low_energy"]["tpr"] == 1.0)
