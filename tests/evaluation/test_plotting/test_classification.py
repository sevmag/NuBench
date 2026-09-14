"""Unit tests for `nubench.evaluation.plotting.classification`."""

from typing import Dict

import matplotlib

matplotlib.use("Agg")  # No display available/needed for tests.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nubench.evaluation.plotting import (  # noqa: E402
    plot_roc_curve,
    plot_roc_curve_by_energy_regime,
    plot_roc_curve_by_energy_regime_comparison,
    plot_roc_curve_comparison,
    plot_track_score_distribution_by_topology,
    plot_track_score_distribution_by_topology_comparison,
)


def _make_roc_result() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fpr": [0.0, 0.1, 0.3, 0.6, 1.0],
            "tpr": [0.0, 0.4, 0.7, 0.9, 1.0],
            "threshold": [1.0, 0.8, 0.5, 0.2, 0.0],
        }
    )


def test_plot_roc_curve_returns_axes() -> None:
    """The function should return a matplotlib Axes."""
    ax = plot_roc_curve(_make_roc_result())
    assert isinstance(ax, plt.Axes)


def test_plot_roc_curve_draws_correct_line() -> None:
    """One of the plotted lines should carry the fpr/tpr data exactly."""
    result = _make_roc_result()
    ax = plot_roc_curve(result)
    data_lines = [
        line
        for line in ax.lines
        if len(line.get_data()[0]) == len(result)
        and np.allclose(line.get_data()[0], result["fpr"])
        and np.allclose(line.get_data()[1], result["tpr"])
    ]
    assert len(data_lines) == 1


def test_plot_roc_curve_reuses_provided_axes() -> None:
    """Passing an existing `ax` should draw onto it, not create a new one."""
    fig, ax = plt.subplots()
    returned_ax = plot_roc_curve(_make_roc_result(), ax=ax)
    assert returned_ax is ax


def test_plot_roc_curve_legend_label_includes_auc() -> None:
    """When `auc` is given, it should be folded into the legend label."""
    ax = plot_roc_curve(_make_roc_result(), auc=0.942, label="DynEdge")
    labels = [line.get_label() for line in ax.lines]
    assert any("DynEdge" in lbl and "0.94" in lbl for lbl in labels)


def test_plot_roc_curve_legend_label_without_auc() -> None:
    """Without `auc`, the label should be used exactly as given."""
    ax = plot_roc_curve(_make_roc_result(), label="DynEdge")
    labels = [line.get_label() for line in ax.lines]
    assert "DynEdge" in labels


def _make_classification_predictions_df(
    seed: int, n: int = 500
) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    truth = rng.integers(0, 2, size=n)
    score = truth + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"truth": truth, "score": score})


def test_plot_roc_curve_comparison_one_curve_per_model() -> None:
    """Each model in `predictions` should get its own labelled ROC curve."""
    predictions = {
        "ModelA": _make_classification_predictions_df(seed=50),
        "ModelB": _make_classification_predictions_df(seed=51),
    }

    ax = plot_roc_curve_comparison(
        predictions, truth_col="truth", score_col="score"
    )

    labels = [line.get_label() for line in ax.lines]
    assert any("ModelA" in lbl for lbl in labels)
    assert any("ModelB" in lbl for lbl in labels)


def test_plot_roc_curve_comparison_labels_include_auc() -> None:
    """Every model's legend label should include its AUC value."""
    predictions = {
        "ModelA": _make_classification_predictions_df(seed=52),
        "ModelB": _make_classification_predictions_df(seed=53),
    }

    ax = plot_roc_curve_comparison(
        predictions, truth_col="truth", score_col="score"
    )

    labels = [line.get_label() for line in ax.lines]
    assert any("AUC=" in lbl for lbl in labels)


def test_plot_roc_curve_comparison_reuses_provided_axes() -> None:
    """Passing an existing `ax` should draw onto it, not create a new one."""
    fig, ax = plt.subplots()
    predictions = {"ModelA": _make_classification_predictions_df(seed=54)}

    returned_ax = plot_roc_curve_comparison(
        predictions, truth_col="truth", score_col="score", ax=ax
    )

    assert returned_ax is ax


def _make_roc_energy_regime_result() -> Dict[str, pd.DataFrame]:
    return {
        "low_energy": pd.DataFrame(
            {
                "fpr": [0.0, 0.2, 1.0],
                "tpr": [0.0, 0.6, 1.0],
                "threshold": [1.0, 0.5, 0.0],
            }
        ),
        "mid_energy": pd.DataFrame(
            {
                "fpr": [0.0, 0.3, 1.0],
                "tpr": [0.0, 0.5, 1.0],
                "threshold": [1.0, 0.5, 0.0],
            }
        ),
        "high_energy": pd.DataFrame(
            {
                "fpr": [0.0, 0.4, 1.0],
                "tpr": [0.0, 0.4, 1.0],
                "threshold": [1.0, 0.5, 0.0],
            }
        ),
    }


def test_plot_roc_curve_by_energy_regime_returns_axes() -> None:
    """The function should return a matplotlib Axes."""
    ax = plot_roc_curve_by_energy_regime(_make_roc_energy_regime_result())
    assert isinstance(ax, plt.Axes)


def test_plot_roc_curve_by_energy_regime_low_solid_and_labeled() -> None:
    """The low-energy line carries the model's data and label, solid."""
    result = _make_roc_energy_regime_result()
    ax = plot_roc_curve_by_energy_regime(result, label="DynEdge")
    lines = [line for line in ax.lines if line.get_label() == "DynEdge"]
    assert len(lines) == 1
    assert lines[0].get_linestyle() == "-"
    x_data, y_data = lines[0].get_data()
    assert np.allclose(x_data, result["low_energy"]["fpr"])
    assert np.allclose(y_data, result["low_energy"]["tpr"])


def test_plot_roc_curve_by_energy_regime_mid_dashed_unlabeled() -> None:
    """The mid-energy line carries its data, dashed, with no label."""
    result = _make_roc_energy_regime_result()
    ax = plot_roc_curve_by_energy_regime(result, label="DynEdge")
    mid_lines = [
        line
        for line in ax.lines
        if line.get_linestyle() == "--" and len(line._dash_pattern[1]) == 2
    ]
    assert len(mid_lines) == 1
    assert mid_lines[0].get_label() != "DynEdge"
    x_data, y_data = mid_lines[0].get_data()
    assert np.allclose(x_data, result["mid_energy"]["fpr"])
    assert np.allclose(y_data, result["mid_energy"]["tpr"])


def test_plot_roc_curve_by_energy_regime_high_dashdotdot_unlabeled() -> None:
    """The high-energy line uses a distinct dash-dot-dot pattern.

    `get_linestyle()` reports both a plain dashed line and a custom
    dash-dot-dot tuple as the same generic string "--", so this checks
    the underlying dash pattern's length instead (2 segments for plain
    dashed, 4 for dash-dot-dot) to actually tell them apart.
    """
    result = _make_roc_energy_regime_result()
    ax = plot_roc_curve_by_energy_regime(result, label="DynEdge")
    high_lines = [
        line
        for line in ax.lines
        if line.get_linestyle() == "--" and len(line._dash_pattern[1]) == 4
    ]
    assert len(high_lines) == 1
    assert high_lines[0].get_label() != "DynEdge"
    x_data, y_data = high_lines[0].get_data()
    assert np.allclose(x_data, result["high_energy"]["fpr"])
    assert np.allclose(y_data, result["high_energy"]["tpr"])


def test_plot_roc_curve_by_energy_regime_reuses_axes() -> None:
    """Passing an existing `ax` should draw onto it, not create a new one."""
    fig, ax = plt.subplots()
    returned_ax = plot_roc_curve_by_energy_regime(
        _make_roc_energy_regime_result(), ax=ax
    )
    assert returned_ax is ax


def _make_roc_energy_regime_predictions_df(
    seed: int, n: int = 600
) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    truth = rng.integers(0, 2, size=n)
    score = truth + rng.normal(scale=0.5, size=n)
    energy = rng.uniform(10, 10_000, size=n)
    return pd.DataFrame({"truth": truth, "score": score, "energy": energy})


def test_roc_curve_by_energy_regime_comparison_model_legend() -> None:
    """Each model should get exactly one (low-energy) legend entry."""
    predictions = {
        "ModelA": _make_roc_energy_regime_predictions_df(seed=60),
        "ModelB": _make_roc_energy_regime_predictions_df(seed=61),
    }

    ax = plot_roc_curve_by_energy_regime_comparison(
        predictions,
        truth_col="truth",
        score_col="score",
        energy_col="energy",
    )

    labels = [line.get_label() for line in ax.lines]
    assert labels.count("ModelA") == 1
    assert labels.count("ModelB") == 1


def test_roc_curve_by_energy_regime_comparison_energy_regime_legend() -> (
    None
):
    """A separate grey legend group should explain the three linestyles."""
    predictions = {"ModelA": _make_roc_energy_regime_predictions_df(seed=60)}

    ax = plot_roc_curve_by_energy_regime_comparison(
        predictions,
        truth_col="truth",
        score_col="score",
        energy_col="energy",
        low_threshold=100.0,
        high_threshold=1000.0,
    )

    labels = [line.get_label() for line in ax.lines]
    assert any("100" in lbl for lbl in labels)
    assert any("1000" in lbl for lbl in labels)


def test_roc_curve_by_energy_regime_comparison_reuses_axes() -> None:
    """Passing an existing `ax` should draw onto it, not create a new one."""
    fig, ax = plt.subplots()
    predictions = {"ModelA": _make_roc_energy_regime_predictions_df(seed=62)}

    returned_ax = plot_roc_curve_by_energy_regime_comparison(
        predictions,
        truth_col="truth",
        score_col="score",
        energy_col="energy",
        ax=ax,
    )

    assert returned_ax is ax


def _make_track_score_result() -> Dict[str, pd.DataFrame]:
    bin_left = [0.0, 0.2, 0.4, 0.6, 0.8]
    return {
        "track": pd.DataFrame(
            {"bin_left": bin_left, "count": [5, 10, 15, 30, 40]}
        ),
        "cascade": pd.DataFrame(
            {"bin_left": bin_left, "count": [40, 30, 15, 10, 5]}
        ),
    }


def test_plot_track_score_distribution_by_topology_returns_axes() -> None:
    """The function should return a matplotlib Axes."""
    ax = plot_track_score_distribution_by_topology(_make_track_score_result())
    assert isinstance(ax, plt.Axes)


def test_plot_track_score_distribution_by_topology_track_dotted() -> None:
    """The track line should carry the track data and be dotted."""
    result = _make_track_score_result()
    ax = plot_track_score_distribution_by_topology(result, label="DynEdge")
    lines = [
        line for line in ax.lines if line.get_label() == "DynEdge (track)"
    ]
    assert len(lines) == 1
    assert lines[0].get_linestyle() == ":"
    x_data, y_data = lines[0].get_data()
    assert np.allclose(x_data, result["track"]["bin_left"])
    assert np.allclose(y_data, result["track"]["count"])


def test_plot_track_score_distribution_by_topology_cascade_solid() -> None:
    """The cascade line should carry the cascade data and be solid."""
    result = _make_track_score_result()
    ax = plot_track_score_distribution_by_topology(result, label="DynEdge")
    lines = [
        line for line in ax.lines if line.get_label() == "DynEdge (cascade)"
    ]
    assert len(lines) == 1
    assert lines[0].get_linestyle() == "-"
    x_data, y_data = lines[0].get_data()
    assert np.allclose(x_data, result["cascade"]["bin_left"])
    assert np.allclose(y_data, result["cascade"]["count"])


def test_plot_track_score_distribution_by_topology_no_none_label() -> None:
    """Without a `label`, lines shouldn't get a "None (...)" placeholder."""
    ax = plot_track_score_distribution_by_topology(_make_track_score_result())
    labels = [line.get_label() for line in ax.lines]
    assert not any("None" in str(text) for text in labels)


def test_plot_track_score_distribution_by_topology_log_yscale() -> None:
    """The y-axis should be log-scaled ("Log Counts" in the paper)."""
    ax = plot_track_score_distribution_by_topology(_make_track_score_result())
    assert ax.get_yscale() == "log"


def test_plot_track_score_distribution_by_topology_no_band() -> None:
    """There's no shaded band on this plot."""
    ax = plot_track_score_distribution_by_topology(_make_track_score_result())
    assert len(ax.collections) == 0


def test_plot_track_score_distribution_by_topology_reuses_axes() -> None:
    """Passing an existing `ax` should draw onto it, not create a new one."""
    fig, ax = plt.subplots()
    returned_ax = plot_track_score_distribution_by_topology(
        _make_track_score_result(), ax=ax
    )
    assert returned_ax is ax


def _make_track_score_predictions_df(
    seed: int, n: int = 400
) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    return pd.DataFrame(
        {
            "score": rng.uniform(0, 1, size=n),
            "is_track": rng.random(size=n) < 0.5,
        }
    )


def test_track_score_distribution_comparison_lines_per_model() -> None:
    """Each model contributes a track+cascade line pair."""
    predictions = {
        "ModelA": _make_track_score_predictions_df(seed=10),
        "ModelB": _make_track_score_predictions_df(seed=11),
    }

    ax = plot_track_score_distribution_by_topology_comparison(
        predictions, score_col="score", is_track_col="is_track", n_bins=10
    )

    labels = [line.get_label() for line in ax.lines]
    for model_name in predictions:
        assert f"{model_name} (track)" in labels
        assert f"{model_name} (cascade)" in labels


def test_track_score_distribution_comparison_shares_bins() -> None:
    """Every model must be binned with the same shared edges."""
    predictions = {
        "ModelA": _make_track_score_predictions_df(seed=10),
        "ModelB": _make_track_score_predictions_df(seed=11),
    }

    ax = plot_track_score_distribution_by_topology_comparison(
        predictions, score_col="score", is_track_col="is_track", n_bins=10
    )

    line_a = next(
        line for line in ax.lines if line.get_label() == "ModelA (track)"
    )
    line_b = next(
        line for line in ax.lines if line.get_label() == "ModelB (track)"
    )
    assert np.allclose(line_a.get_data()[0], line_b.get_data()[0])


def test_track_score_distribution_comparison_reuses_axes() -> None:
    """Passing an existing `ax` should draw onto it, not create a new one."""
    fig, ax = plt.subplots()
    predictions = {"ModelA": _make_track_score_predictions_df(seed=12)}

    returned_ax = plot_track_score_distribution_by_topology_comparison(
        predictions,
        score_col="score",
        is_track_col="is_track",
        n_bins=10,
        ax=ax,
    )

    assert returned_ax is ax
