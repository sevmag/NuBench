"""Energy reconstruction plots: the calibration plot, the two-panel
calibration figure, and the CC/NC two-column comparison.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nubench.evaluation.metrics import energy_calibration
from nubench.evaluation.resolution import combined_log_bins
from nubench.style import model_color, model_linestyle


def plot_energy_calibration(
    result: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    linestyle: Optional[str] = None,
) -> plt.Axes:
    """Plot median reconstructed energy and its 68% band vs. true energy,
    against a black dashed y=x reference line for perfect calibration.
    """
    if ax is None:
        fig, ax = plt.subplots()
    (line,) = ax.plot(
        result["bin_center"], result["p50"], label=label, color=color,
        linestyle=linestyle, linewidth=2,
    )
    ax.fill_between(
        result["bin_center"], result["p16"], result["p84"],
        alpha=0.3, color=line.get_color(),
    )
    ax.plot(
        result["bin_center"], result["bin_center"], label="ideal",
        color="black", linestyle="--", linewidth=0.5,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True Energy [GeV]")
    ax.set_ylabel("Reco. Energy [GeV]")
    return ax


def plot_energy_calibration_comparison(
    predictions: Dict[str, pd.DataFrame],
    truth_col: str,
    pred_col: str,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 60,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Overlay every model's energy calibration.

    Model names matching one of the four paper models (see
    `nubench.style`) get that model's canonical color and linestyle.
    `bins` defaults to the combined true-energy range across all models.
    """
    if bins is None:
        bins = combined_log_bins(
            (df[truth_col] for df in predictions.values()), n_bins
        )
    if ax is None:
        fig, ax = plt.subplots()
    for model_name, df in predictions.items():
        plot_energy_calibration(
            energy_calibration(df, truth_col, pred_col, bins=bins),
            ax=ax,
            label=model_name,
            color=model_color(model_name),
            linestyle=model_linestyle(model_name),
        )
    ax.grid(which="both", alpha=0.3)
    # Each model drew its own identical "ideal" line. They have to be
    # removed as Line2D objects, not just filtered out of this legend:
    # `nubench.multipanel` later rebuilds the legend from
    # `ax.get_legend_handles_labels()`, which reads the Axes' lines
    # directly rather than whatever the last `ax.legend()` kept.
    seen_ideal = False
    for line in list(ax.lines):
        if line.get_label() == "ideal":
            if seen_ideal:
                line.remove()
            seen_ideal = True
    ax.legend()
    return ax


def plot_energy_calibration_figure(
    predictions: Dict[str, pd.DataFrame],
    truth_col: str,
    pred_col: str,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 60,
    ax_hist: Optional[plt.Axes] = None,
    ax_main: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """The paper's energy figure: a marginal histogram located above the
    calibration plot, sharing an x axis.

    The top panel step-histograms each model's predicted energy plus the
    combined truth in black. Pass both `ax_*` to draw into an existing
    grid; otherwise a new Figure is created.
    """
    if bins is None:
        bins = combined_log_bins(
            (df[truth_col] for df in predictions.values()), n_bins
        )
    if ax_hist is None or ax_main is None:
        fig, (ax_hist, ax_main) = plt.subplots(
            2, 1,
            figsize=(3, 3 + 3 * 0.2),
            gridspec_kw={"height_ratios": [0.2, 1]},
            sharex=True,
            constrained_layout=True,
        )
    else:
        # Every Axes this receives comes from `plt.subplots()`, so
        # `.figure` is always a plain Figure, never a SubFigure.
        fig = ax_hist.figure  # type: ignore[assignment]

    # mypy's matplotlib stub doesn't accept np.ndarray for `hist`'s
    # `bins`, though it works fine at runtime.
    for model_name, df in predictions.items():
        ax_hist.hist(
            df[pred_col],
            bins=bins,  # type: ignore[arg-type]
            histtype="step",
            color=model_color(model_name),
        )
    ax_hist.hist(
        pd.concat(
            [df[truth_col] for df in predictions.values()], ignore_index=True
        ),
        bins=bins,  # type: ignore[arg-type]
        histtype="step",
        color="black",
    )
    ax_hist.set_xscale("log")
    ax_hist.set_yscale("log")
    ax_hist.yaxis.tick_right()
    ax_hist.xaxis.minorticks_on()
    ax_hist.spines[["right", "top", "left"]].set_visible(False)
    ax_hist.yaxis.set_visible(False)
    plot_energy_calibration_comparison(
        predictions, truth_col, pred_col, bins, n_bins, ax=ax_main
    )
    return fig, (ax_hist, ax_main)


def plot_energy_calibration_by_topology_figure(
    predictions: Dict[str, pd.DataFrame],
    truth_col: str,
    pred_col: str,
    is_track_col: str,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 60,
    axes: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """The paper's CC/NC energy figure: two `plot_energy_calibration_
    figure` columns side by side, track/CC left and cascade/NC right.

    Restricted to muon-neutrino events (which is what `is_track_col`
    encodes), track *is* CC and cascade *is* NC.

    `axes` is a 2x2 array (row 0 histograms, row 1 calibration; column 0
    CC, column 1 NC), as `plt.subplots(2, 2)` returns. Pass one to draw
    into an existing grid; otherwise a new Figure is created. Bins come
    from the full `predictions` before the split, so both columns share
    them and stay comparable.
    """
    if bins is None:
        bins = combined_log_bins(
            (df[truth_col] for df in predictions.values()), n_bins
        )
    if axes is None:
        fig, axes = plt.subplots(
            2, 2,
            figsize=(6, 3 + 3 * 0.2),
            gridspec_kw={"height_ratios": [0.2, 1]},
            sharex=True,
            constrained_layout=True,
        )
    else:
        # Every Axes this receives comes from `plt.subplots()`, so
        # `.figure` is always a plain Figure, never a SubFigure.
        fig = axes[0, 0].figure  # type: ignore[assignment]
    label_bbox = dict(facecolor="white", edgecolor="none", alpha=0.3)
    for col, (is_cc, topology_label) in enumerate(
        ((True, r"$\nu_\mu^{CC}$"), (False, r"$\nu_\mu^{NC}$"))
    ):
        subset = {
            name: df[df[is_track_col] if is_cc else ~df[is_track_col]]
            for name, df in predictions.items()
        }
        plot_energy_calibration_figure(
            subset, truth_col, pred_col, bins=bins, n_bins=n_bins,
            ax_hist=axes[0, col], ax_main=axes[1, col],
        )
        axes[1, col].text(
            0.05, 0.85, topology_label, transform=axes[1, col].transAxes,
            fontsize=14, bbox=label_bbox,
        )
    return fig, axes
