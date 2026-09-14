"""Inelasticity reconstruction plots: resolution vs. energy and the value
distribution split by energy regime, plus the combined figure putting
both side by side.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nubench.evaluation.metrics import (
    inelasticity_distribution_by_energy_regime,
    inelasticity_resolution,
)
from nubench.style import model_color


def plot_inelasticity_resolution(
    result: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> plt.Axes:
    """Plot inelasticity resolution vs. neutrino energy.

    Args:
        result: Output of `inelasticity_resolution` - a DataFrame with columns
            "bin_center", "p16", "p50", "p84".
        ax: Existing Axes to draw onto. If None, a new Figure/Axes pair is
            created. Passing the same `ax` across multiple calls is how
            several models end up overlaid on one figure.
        label: Legend label for this line (e.g. a model name).
        color: Line/band color. If None, matplotlib picks one
            automatically (its usual color-cycling behavior).

    Returns:
        The Axes the plot was drawn on (either the one passed in, or a
        newly created one).
    """
    if ax is None:
        fig, ax = plt.subplots()
    (line,) = ax.plot(
        result["bin_center"], result["p50"], label=label, color=color,
        linewidth=2,
    )
    ax.fill_between(
        result["bin_center"],
        result["p16"],
        result["p84"],
        alpha=0.3,
        color=line.get_color(),
    )
    ax.set_xscale("log")
    ax.yaxis.minorticks_on()
    ax.xaxis.minorticks_on()
    ax.grid(which='minor', alpha=0.3)
    ax.set_xlabel("Neutrino Energy [GeV]")
    ax.set_ylabel('$|y_\\text{vis.} - y_\\text{reco.}|$')
    return ax


def plot_inelasticity_resolution_comparison(
    predictions: Dict[str, pd.DataFrame],
    truth_col: str,
    pred_col: str,
    energy_col: str,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 30,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot inelasticity resolution for several models on one figure.

    Args:
        predictions: Mapping from model name (used as its legend label) to
            its prediction DataFrame. Every DataFrame must contain
            `truth_col` and `pred_col`.
        truth_col: Name of the column holding the true inelasticity - the same
            across every model's DataFrame.
        pred_col: Name of the column holding the predicted inelasticity - the
            same across every model's DataFrame.
        energy_col: Name of the column holding the true energy - the same
                    across every model's DataFrame.
        bins: Bin edges to use for every model. If None, edges are built
            automatically to span the *combined* true-energy range across
            all models, so every model is binned identically for a fair
            comparison.
        n_bins: Number of bins to construct when `bins` is not given.
        ax: Existing Axes to draw onto. If None, a new Figure/Axes pair is
            created.

    Returns:
        The Axes with one bias/resolution line+band per model, with a
        legend identifying each by its key in `predictions`.
    """
    if bins is None:
        combined_min = min(
            df[energy_col].min() for df in predictions.values()
        )
        combined_max = max(
            df[energy_col].max() for df in predictions.values()
        )
        bins = np.logspace(
            np.log10(combined_min), np.log10(combined_max), n_bins + 1
        )
    if ax is None:
        fig, ax = plt.subplots()
    for model_name, df in predictions.items():
        result = inelasticity_resolution(
            df, truth_col, pred_col, energy_col=energy_col, bins=bins
        )
        plot_inelasticity_resolution(
            result, ax=ax, label=model_name, color=model_color(model_name)
        )
    ax.legend()
    return ax


def plot_inelasticity_distribution_by_energy_regime(
    result: Dict[str, pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> plt.Axes:
    """Plot the inelasticity value distribution, split by energy regime.

    Low-energy solid, high-energy dashed - matching the paper's own
    figure legend exactly (opposite of the track/cascade convention
    used elsewhere in this codebase, which is solid/dashed too but for
    a different split) - colored by model for the predicted
    distribution. The true distribution is drawn the same way but
    always in black, since it's a fixed reference, not a model
    prediction (same reasoning as the "muon" baseline in
    `plot_direction_resolution_by_topology`).

    Args:
        result: Output of `inelasticity_distribution_by_energy_regime` -
            a dict with keys "low_energy_pred", "high_energy_pred",
            "low_energy_truth", "high_energy_truth", each a DataFrame
            with columns "bin_left", "percentage".
        ax: Existing Axes to draw onto. If None, a new Figure/Axes pair is
            created. Passing the same `ax` across multiple calls is how
            several models end up overlaid on one figure.
        label: Legend label prefix for this model's predicted-distribution
            lines, e.g. "DynEdge" becomes "DynEdge (low energy)" /
            "DynEdge (high energy)". If None, no legend entries are added
            for the predicted lines.
        color: Color for this model's predicted-distribution lines. If
            None, matplotlib picks one automatically.

    Returns:
        The Axes the plot was drawn on (either the one passed in, or a
        newly created one).
    """
    if ax is None:
        fig, ax = plt.subplots()
    low_energy_label = f"{label} (low energy)" if label is not None else None
    ax.step(
        result["low_energy_pred"]["bin_left"],
        result["low_energy_pred"]["percentage"],
        where="post",
        color=color,
        linestyle="-",
        linewidth=2,
        label=low_energy_label,
    )
    high_energy_label = f"{label} (high energy)" if label is not None else None
    ax.step(
        result["high_energy_pred"]["bin_left"],
        result["high_energy_pred"]["percentage"],
        where="post",
        color=color,
        linestyle="--",
        linewidth=2,
        label=high_energy_label,
    )
    ax.step(
        result["low_energy_truth"]["bin_left"],
        result["low_energy_truth"]["percentage"],
        where="post",
        color="black",
        linestyle="-",
        linewidth=2,
        label="truth (low energy)",
    )
    ax.step(
        result["high_energy_truth"]["bin_left"],
        result["high_energy_truth"]["percentage"],
        where="post",
        color="black",
        linestyle="--",
        linewidth=2,
        label="truth (high energy)",
    )
    ax.set_xlabel("Visible Inelasticity [arb.]")
    ax.set_ylabel("Percentage [%]")
    return ax


def plot_inelasticity_distribution_by_energy_regime_comparison(
    predictions: Dict[str, pd.DataFrame],
    pred_col: str,
    truth_col: str,
    energy_col: str,
    energy_threshold: float = 100.0,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 50,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot the inelasticity value distribution for several models.

    Args:
        predictions: Mapping from model name (used as its legend label) to
            its prediction DataFrame. Every DataFrame must contain
            `pred_col`, `truth_col`, and `energy_col`.
        pred_col: Name of the column holding the predicted inelasticity -
            the same across every model's DataFrame.
        truth_col: Name of the column holding the true (visible)
            inelasticity - the same across every model's DataFrame.
        energy_col: Name of the column holding the true energy - the same
            across every model's DataFrame.
        energy_threshold: Energy value separating "low" (<=) from "high"
            (>) energy events.
        bins: Bin edges to use for every model. If None, edges are built
            automatically to span the *combined* inelasticity-value range
            across every model's predicted and true columns together, so
            every model is binned identically for a fair comparison.
        n_bins: Number of bins to construct when `bins` is not given.
        ax: Existing Axes to draw onto. If None, a new Figure/Axes pair is
            created.

    Returns:
        The Axes with one low+high-energy line pair per model (plus the
        shared black truth reference), with a legend identifying each by
        its key in `predictions`.
    """
    if bins is None:
        combined_min = min(
            min(df[pred_col].min(), df[truth_col].min())
            for df in predictions.values()
        )
        combined_max = max(
            max(df[pred_col].max(), df[truth_col].max())
            for df in predictions.values()
        )
        bins = np.linspace(combined_min, combined_max, n_bins + 1)
    if ax is None:
        fig, ax = plt.subplots()
    for model_name, df in predictions.items():
        result = inelasticity_distribution_by_energy_regime(
            df,
            pred_col,
            truth_col,
            energy_col,
            energy_threshold=energy_threshold,
            bins=bins,
        )
        plot_inelasticity_distribution_by_energy_regime(
            result,
            ax=ax,
            label=model_name,
            color=model_color(model_name),
        )
    ax.legend()
    return ax


def plot_inelasticity_figure(
    predictions: Dict[str, pd.DataFrame],
    truth_col: str,
    pred_col: str,
    energy_col: str,
    energy_threshold: float = 100.0,
    distribution_bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    distribution_n_bins: int = 50,
    ax_distribution: Optional[plt.Axes] = None,
    ax_resolution: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Build the combined inelasticity figure: distribution + resolution.

    A 1x2 `Figure` - pure composition, no new metric math - putting the
    paper's two inelasticity plots side by side: left panel is
    `plot_inelasticity_distribution_by_energy_regime_comparison` (the
    predicted/true value-distribution histograms, split low/high
    energy), right panel is `plot_inelasticity_resolution_comparison`
    (residual vs. energy), each drawn via its own `ax` parameter onto
    one half of the same Figure.

    Args:
        predictions: Mapping from model name (used as its legend label) to
            its prediction DataFrame - the same dict both underlying
            comparison functions expect.
        truth_col: Name of the column holding the true (visible)
            inelasticity - the same across every model's DataFrame.
        pred_col: Name of the column holding the predicted inelasticity -
            the same across every model's DataFrame.
        energy_col: Name of the column holding the true energy - used by
            both panels (as the low/high-energy split on the left, and
            as the binning variable on the right).
        energy_threshold: Energy value separating "low" (<=) from "high"
            (>) energy events, used only by the left (distribution)
            panel.
        distribution_bins: Bin edges for the left (distribution) panel
            only - see `plot_inelasticity_distribution_by_energy_regime_
            comparison`'s own `bins` parameter.
        distribution_n_bins: Number of bins to construct when
            `distribution_bins` is not given.
        ax_distribution: Existing Axes for the left (distribution)
            panel. Must be given together with `ax_resolution` (both or
            neither) - if either is None, a new Figure with both panels
            is created instead. Passing both in is how a multi-detector
            grid (see `nubench.multipanel`) places one detector's pair
            into its own slice of a bigger grid.
        ax_resolution: Existing Axes for the right (resolution-vs-
            energy) panel.

    Returns:
        `(fig, (ax_distribution, ax_resolution))` - the Figure, the
        value-distribution Axes, and the resolution-vs-energy Axes.
    """
    if ax_distribution is None or ax_resolution is None:
        fig, (ax_distribution, ax_resolution) = plt.subplots(
            1, 2, figsize=(6, 3), constrained_layout=True
        )
    else:
        # Same reasoning as `plot_energy_calibration_figure`'s own
        # `ax_hist`/`ax_main` reuse: every Axes this function ever
        # receives comes from `plt.subplots()`, so `.figure` is always a
        # plain `Figure`, never a `SubFigure`.
        fig = ax_distribution.figure  # type: ignore[assignment]
    plot_inelasticity_distribution_by_energy_regime_comparison(
            predictions,
            pred_col,
            truth_col,
            energy_col,
            energy_threshold=energy_threshold,
            bins=distribution_bins,
            n_bins=distribution_n_bins,
            ax=ax_distribution
    )
    plot_inelasticity_resolution_comparison(
            predictions, truth_col, pred_col, energy_col, ax=ax_resolution
    )
    return fig, (ax_distribution, ax_resolution)
