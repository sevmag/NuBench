"""Track/cascade classification plots: ROC curves (plain and split by
energy regime) and the classifier score distribution split by
track/cascade.
"""

from typing import Dict, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nubench.evaluation.metrics import (
    auc_score,
    roc_curve_data,
    roc_curve_data_by_energy_regime,
    track_score_distribution_by_topology,
)
from nubench.style import model_color, model_linestyle


def plot_roc_curve(
    result: pd.DataFrame,
    auc: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    linestyle: Optional[str] = None,
) -> plt.Axes:
    """Plot a ROC curve (true positive rate vs. false positive rate).

    Args:
        result: Output of `roc_curve_data` - a DataFrame with columns
            "fpr", "tpr", "threshold".
        auc: If given, folded into the legend label as e.g.
            "DynEdge (AUC=0.94)". Typically the output of `auc_score` on
            the same data used to build `result`.
        ax: Existing Axes to draw onto. If None, a new Figure/Axes pair is
            created. Passing the same `ax` across multiple calls is how
            several models end up overlaid on one figure.
        label: Legend label for this curve (e.g. a model name).
        color: Line color. If None, matplotlib picks one automatically.
        linestyle: Line style. If None, matplotlib's default ("-").

    Returns:
        The Axes the plot was drawn on (either the one passed in, or a
        newly created one).
    """
    if ax is None:
        fig, ax = plt.subplots()
    plot_label: Optional[str]
    if label is not None and auc is not None:
        plot_label = f"{label} (AUC={auc:.3f})"
    else:
        plot_label = label
    ax.plot(
        result["fpr"],
        result["tpr"],
        label=plot_label,
        color=color,
        linestyle=linestyle,
        linewidth=2,
    )
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    ax.grid(which="both", alpha=0.3)
    return ax


def plot_roc_curve_comparison(
    predictions: Dict[str, pd.DataFrame],
    truth_col: str,
    score_col: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot ROC curves for several models on one figure.

    Unlike every other `*_comparison` function here, there's no "shared
    bins" concern - a ROC curve isn't binned by anything, so each model's
    curve is computed entirely independently of the others.

    Args:
        predictions: Mapping from model name (used as its legend label) to
            its prediction DataFrame. Every DataFrame must contain
            `truth_col` and `score_col`.
        truth_col: Name of the column holding the true binary label - the
            same across every model's DataFrame.
        score_col: Name of the column holding the classifier's score - the
            same across every model's DataFrame.
        ax: Existing Axes to draw onto. If None, a new Figure/Axes pair is
            created.

    Returns:
        The Axes with one ROC curve per model (AUC folded into each
        legend label), with a legend identifying each by its key in
        `predictions`.
    """
    if ax is None:
        fig, ax = plt.subplots()
    for model_name, df in predictions.items():
        result = roc_curve_data(df, truth_col, score_col)
        auc = auc_score(df, truth_col, score_col)
        plot_roc_curve(
            result,
            auc=auc,
            ax=ax,
            label=model_name,
            color=model_color(model_name),
            linestyle=model_linestyle(model_name),
        )
    ax.legend()
    return ax


def plot_roc_curve_by_energy_regime(
    result: Dict[str, pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> plt.Axes:
    """Plot ROC curves for one model, split into three energy regimes.

    Low-energy solid, mid-energy dashed, high-energy dash-dot-dot -
    matching the paper's own three-regime ROC comparison. Only the
    low-energy line carries `label` (so each model gets exactly one
    legend entry from this function); the mid/high lines are always
    unlabeled - `plot_roc_curve_by_energy_regime_comparison` adds a
    second, separate legend explaining what the three linestyles mean,
    since that's shared across every model rather than being one more
    thing to repeat per model.

    Args:
        result: Output of `roc_curve_data_by_energy_regime` - a dict with
            keys "low_energy"/"mid_energy"/"high_energy", each a
            DataFrame with columns "fpr", "tpr", "threshold".
        ax: Existing Axes to draw onto. If None, a new Figure/Axes pair is
            created. Passing the same `ax` across multiple calls is how
            several models end up overlaid on one figure.
        label: Legend label for this model (put on the low-energy line
            only).
        color: Color shared by all three of this model's lines. If None,
            matplotlib picks one automatically.

    Returns:
        The Axes the plot was drawn on (either the one passed in, or a
        newly created one).
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(
        result["low_energy"]["fpr"],
        result["low_energy"]["tpr"],
        color=color,
        linestyle="-",
        linewidth=2.5,
        label=label
    )
    ax.plot(
        result["mid_energy"]["fpr"],
        result["mid_energy"]["tpr"],
        color=color,
        linestyle="--",
        linewidth=2.5,
        label=None
    )
    ax.plot(
        result["high_energy"]["fpr"],
        result["high_energy"]["tpr"],
        color=color,
        linestyle=(0, (3, 1, 1, 1)),
        linewidth=2.5,
        label=None
    )
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    ax.yaxis.minorticks_on()
    ax.xaxis.minorticks_on()
    ax.grid(which="both", alpha=0.3)
    return ax


def plot_roc_curve_by_energy_regime_comparison(
    predictions: Dict[str, pd.DataFrame],
    truth_col: str,
    score_col: str,
    energy_col: str,
    low_threshold: float = 100.0,
    high_threshold: float = 1000.0,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot energy-regime-split ROC curves for several models.

    Unlike every other `*_comparison` function here, the legend needs two
    separate groups of entries to be readable: one entry per model
    (color-coded, via the low-energy line's `label`), and one entry per
    energy regime (fixed grey, varying only by linestyle) explaining what
    solid/dashed/dash-dot-dot mean - since that meaning is shared across
    every model rather than being model-specific.

    Args:
        predictions: Mapping from model name (used as its legend label) to
            its prediction DataFrame. Every DataFrame must contain
            `truth_col`, `score_col`, and `energy_col`.
        truth_col: Name of the column holding the true binary label - the
            same across every model's DataFrame.
        score_col: Name of the column holding the classifier's score -
            the same across every model's DataFrame.
        energy_col: Name of the column holding the true energy - the same
            across every model's DataFrame.
        low_threshold: Upper bound (inclusive) of the "low energy"
            regime.
        high_threshold: Upper bound (inclusive) of the "mid energy"
            regime; everything above this is "high energy".
        ax: Existing Axes to draw onto. If None, a new Figure/Axes pair is
            created.

    Returns:
        The Axes with three ROC curves per model, plus a legend
        identifying both the models (by color) and the energy regimes
        (by linestyle).
    """
    if ax is None:
        fig, ax = plt.subplots()
    for model_name, df in predictions.items():
        result = roc_curve_data_by_energy_regime(
            df,
            truth_col,
            score_col,
            energy_col,
            low_threshold=low_threshold,
            high_threshold=high_threshold
        )
        plot_roc_curve_by_energy_regime(
            result,
            ax=ax,
            label=model_name,
            color=model_color(model_name)
        )
    ax.plot(
        [], [], color="grey", linestyle="-",
        label=f"E <= {low_threshold:g} GeV",
    )
    ax.plot(
        [], [], color="grey", linestyle="--",
        label=f"{low_threshold:g} <= E <= {high_threshold:g} GeV",
    )
    ax.plot(
        [], [], color="grey", linestyle=(0, (3, 1, 1, 1)),
        label=f"E >= {high_threshold:g} GeV",
    )
    ax.legend()
    return ax


def plot_track_score_distribution_by_topology(
    result: Dict[str, pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> plt.Axes:
    """Plot the classifier score distribution, split by track/cascade.

    Track dotted, cascade solid - note this is a *different* linestyle
    convention from the track/cascade plots elsewhere (which use solid
    for track, dashed for cascade). This matches the paper's own score-
    histogram figure specifically; other plots in the paper use the
    other convention, so this isn't an inconsistency on our part. There's
    no "truth" reference line here (unlike the inelasticity distribution
    plot) - there's no true score to compare against.

    Args:
        result: Output of `track_score_distribution_by_topology` - a
            dict with keys "track"/"cascade", each a DataFrame with
            columns "bin_left", "count".
        ax: Existing Axes to draw onto. If None, a new Figure/Axes pair is
            created. Passing the same `ax` across multiple calls is how
            several models end up overlaid on one figure.
        label: Legend label prefix for this model, e.g. "DynEdge" becomes
            "DynEdge (track)" / "DynEdge (cascade)" on the two lines.
        color: Color shared by this model's track and cascade lines. If
            None, matplotlib picks one automatically.

    Returns:
        The Axes the plot was drawn on (either the one passed in, or a
        newly created one).
    """
    if ax is None:
        fig, ax = plt.subplots()
    track_label = f"{label} (track)" if label is not None else None
    ax.step(
        result["track"]["bin_left"],
        result["track"]["count"],
        where="post",
        label=track_label,
        color=color,
        linestyle=":",
        linewidth=2,
    )
    cascade_label = f"{label} (cascade)" if label is not None else None
    ax.step(
        result["cascade"]["bin_left"],
        result["cascade"]["count"],
        where="post",
        label=cascade_label,
        color=color,
        linestyle="-",
        linewidth=2,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Track Score")
    ax.set_ylabel("Log Counts")
    ax.grid(which="both", alpha=0.3)
    ax.xaxis.minorticks_on()
    ax.yaxis.minorticks_on()
    return ax


def plot_track_score_distribution_by_topology_comparison(
    predictions: Dict[str, pd.DataFrame],
    score_col: str,
    is_track_col: str,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 50,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot the classifier score distribution for several models.

    Args:
        predictions: Mapping from model name (used as its legend label) to
            its prediction DataFrame. Every DataFrame must contain
            `score_col` and `is_track_col`.
        score_col: Name of the column holding the classifier's score -
            the same across every model's DataFrame.
        is_track_col: Name of the boolean track/cascade column - the same
            across every model's DataFrame.
        bins: Bin edges to use for every model. If None, edges are built
            automatically to span the *combined* score range across all
            models, so every model is binned identically for a fair
            comparison.
        n_bins: Number of bins to construct when `bins` is not given.
        ax: Existing Axes to draw onto. If None, a new Figure/Axes pair is
            created.

    Returns:
        The Axes with one track+cascade line pair per model, with a
        legend identifying each by its key in `predictions`.
    """
    if bins is None:
        combined_min = min(df[score_col].min() for df in predictions.values())
        combined_max = max(df[score_col].max() for df in predictions.values())
        bins = np.linspace(combined_min, combined_max, n_bins + 1)
    if ax is None:
        fig, ax = plt.subplots()
    for model_name, df in predictions.items():
        result = track_score_distribution_by_topology(
            df,
            score_col,
            is_track_col,
            bins=bins
        )
        plot_track_score_distribution_by_topology(
            result,
            ax=ax,
            label=model_name,
            color=model_color(model_name)
        )
    ax.legend()
    return ax
