"""Inelasticity reconstruction plots: resolution vs. energy and the value
distribution split by energy regime, plus the combined figure putting the
two side by side.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nubench.evaluation.metrics import (
    inelasticity_distribution_by_energy_regime,
    inelasticity_resolution,
)
from nubench.evaluation.resolution import combined_log_bins
from nubench.style import model_color


def plot_inelasticity_resolution(
    result: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> plt.Axes:
    """Plot one model's inelasticity resolution vs. energy, with 68% band."""
    if ax is None:
        fig, ax = plt.subplots()
    (line,) = ax.plot(
        result["bin_center"], result["p50"], label=label, color=color,
        linewidth=2,
    )
    ax.fill_between(
        result["bin_center"], result["p16"], result["p84"],
        alpha=0.3, color=line.get_color(),
    )
    ax.set_xscale("log")
    ax.yaxis.minorticks_on()
    ax.xaxis.minorticks_on()
    ax.grid(which="both", alpha=0.3)
    ax.set_xlabel("Neutrino Energy [GeV]")
    ax.set_ylabel("$|y_\\text{vis.} - y_\\text{reco.}|$")
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
    """Overlay every model's inelasticity resolution."""
    if bins is None:
        bins = combined_log_bins(
            (df[energy_col] for df in predictions.values()), n_bins
        )
    if ax is None:
        fig, ax = plt.subplots()
    for model_name, df in predictions.items():
        plot_inelasticity_resolution(
            inelasticity_resolution(
                df, truth_col, pred_col, energy_col=energy_col, bins=bins
            ),
            ax=ax, label=model_name, color=model_color(model_name),
        )
    ax.legend()
    return ax


def plot_inelasticity_distribution_by_energy_regime(
    result: Dict[str, pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> plt.Axes:
    """Plot one model's inelasticity distribution as step histograms.

    High energy solid, low energy dashed, as the paper's own notebook
    draws it. Note its published caption says the reverse - the figure
    and the code agree with each other, the caption is the odd one out.
    Truth is black, being a fixed reference rather than a prediction.
    """
    if ax is None:
        fig, ax = plt.subplots()
    for key, linestyle, line_label in (
        ("high_energy_pred", "-",
         f"{label} (high energy)" if label is not None else None),
        ("low_energy_pred", "--",
         f"{label} (low energy)" if label is not None else None),
        ("high_energy_truth", "-", "truth (high energy)"),
        ("low_energy_truth", "--", "truth (low energy)"),
    ):
        ax.step(
            result[key]["bin_left"], result[key]["percentage"], where="post",
            color="black" if key.endswith("truth") else color,
            linestyle=linestyle, linewidth=2, label=line_label,
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
    """Overlay every model's inelasticity distribution, plus black truth.
    """
    if bins is None:
        bins = np.linspace(
            min(
                min(df[pred_col].min(), df[truth_col].min())
                for df in predictions.values()
            ),
            max(
                max(df[pred_col].max(), df[truth_col].max())
                for df in predictions.values()
            ),
            n_bins + 1,
        )
    if ax is None:
        fig, ax = plt.subplots()
    for model_name, df in predictions.items():
        plot_inelasticity_distribution_by_energy_regime(
            inelasticity_distribution_by_energy_regime(
                df, pred_col, truth_col, energy_col,
                energy_threshold=energy_threshold, bins=bins,
            ),
            ax=ax, label=model_name, color=model_color(model_name),
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
    """Two inelasticity panels side by side: value distribution,
      then resolution vs. energy.

    Pass both `ax_*` to draw into an existing grid (see
    `nubench.multipanel`); otherwise a new 1x2 Figure is created.
    `distribution_bins`/`distribution_n_bins` apply to the left panel.
    """
    if ax_distribution is None or ax_resolution is None:
        fig, (ax_distribution, ax_resolution) = plt.subplots(
            1, 2, figsize=(6, 3), constrained_layout=True
        )
    else:
        # Every Axes this receives comes from `plt.subplots()`, so
        # `.figure` is always a plain Figure, never a SubFigure.
        fig = ax_distribution.figure  # type: ignore[assignment]
    plot_inelasticity_distribution_by_energy_regime_comparison(
        predictions, pred_col, truth_col, energy_col,
        energy_threshold=energy_threshold,
        bins=distribution_bins,
        n_bins=distribution_n_bins,
        ax=ax_distribution,
    )
    plot_inelasticity_resolution_comparison(
        predictions, truth_col, pred_col, energy_col, ax=ax_resolution
    )
    return fig, (ax_distribution, ax_resolution)
