"""Track/cascade classification plots: ROC curves (plain and split by
energy regime) and the classifier score distribution.
"""

from typing import Dict, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator
from sklearn.metrics import auc

from nubench.evaluation.metrics import (
    roc_curve_data,
    roc_curve_data_by_energy_regime,
    track_score_distribution_by_topology,
)
from nubench.style import model_color, model_linestyle

HIGH_ENERGY_LINESTYLE = (0, (3, 1, 1, 1))  # densely dash-dotted


def plot_roc_curve(
    result: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    linestyle: Optional[str] = None,
) -> plt.Axes:
    """Plot one ROC curve from `roc_curve_data`'s output. s"""
    if ax is None:
        fig, ax = plt.subplots()
    plot_label: Optional[str] = label
    if label is not None:
        plot_label = f"{label} (AUC={auc(result['fpr'], result['tpr']):.3f})"
    ax.plot(
        result["fpr"], result["tpr"], label=plot_label, color=color,
        linestyle=linestyle, linewidth=2,
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
    """Overlay one ROC curve per model, AUC folded into each label."""
    if ax is None:
        fig, ax = plt.subplots()
    for model_name, df in predictions.items():
        plot_roc_curve(
            roc_curve_data(df, truth_col, score_col),
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
    """Plot one model's three energy-regime ROC curves."""
    if ax is None:
        fig, ax = plt.subplots()
    for key, linestyle, line_label in (
        ("low_energy", "-", label),
        ("mid_energy", "--", None),
        ("high_energy", HIGH_ENERGY_LINESTYLE, None),
    ):
        ax.plot(
            result[key]["fpr"], result[key]["tpr"], color=color,
            linestyle=linestyle, linewidth=2.5, label=line_label,
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
    """Overlay every model's three energy-regime ROC curves."""
    if ax is None:
        fig, ax = plt.subplots()
    for model_name, df in predictions.items():
        plot_roc_curve_by_energy_regime(
            roc_curve_data_by_energy_regime(
                df, truth_col, score_col, energy_col,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
            ),
            ax=ax,
            label=model_name,
            color=model_color(model_name),
        )
    for linestyle, regime_label in (
        ("-", f"E <= {low_threshold:g} GeV"),
        ("--", f"{low_threshold:g} < E <= {high_threshold:g} GeV"),
        (HIGH_ENERGY_LINESTYLE, f"E > {high_threshold:g} GeV"),
    ):
        ax.plot([], [], color="grey", linestyle=linestyle, label=regime_label)
    ax.legend()
    return ax


def plot_track_score_distribution_by_topology(
    result: Dict[str, pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> plt.Axes:
    """Plot one model's track/cascade score histograms as step lines."""
    if ax is None:
        fig, ax = plt.subplots()
    for key, linestyle in (("track", ":"), ("cascade", "-")):
        ax.step(
            result[key]["bin_left"], result[key]["count"], where="post",
            label=f"{label} ({key})" if label is not None else None,
            color=color, linestyle=linestyle, linewidth=2,
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathcal{T}$-score")
    ax.set_ylabel("Log Counts")
    ax.grid(which="both", alpha=0.3)
    ax.xaxis.minorticks_on()
    ax.set_yticks([])
    ax.yaxis.set_minor_locator(NullLocator())
    return ax


def plot_track_score_distribution_by_topology_comparison(
    predictions: Dict[str, pd.DataFrame],
    score_col: str,
    is_track_col: str,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 50,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Overlay every model's track/cascade score histograms.

    `bins` defaults to `n_bins` edges spanning the combined score range
    across all models, so every model is binned identically.
    """
    if bins is None:
        combined_min = min(df[score_col].min() for df in predictions.values())
        combined_max = max(df[score_col].max() for df in predictions.values())
        bins = np.linspace(combined_min, combined_max, n_bins + 1)
    if ax is None:
        fig, ax = plt.subplots()
    for model_name, df in predictions.items():
        plot_track_score_distribution_by_topology(
            track_score_distribution_by_topology(
                df, score_col, is_track_col, bins=bins
            ),
            ax=ax,
            label=model_name,
            color=model_color(model_name),
        )
    ax.legend()
    return ax
