"""Energy reconstruction plots: calibration-style resolution, plain and the
CC/NC (track/cascade) two-column comparison figure.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nubench.evaluation.metrics import energy_calibration
from nubench.style import model_color, model_linestyle


def plot_energy_calibration(
    result: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    linestyle: Optional[str] = None,
) -> plt.Axes:
    """Plot reconstructed vs. true energy (a calibration-style plot).

    Median reconstructed energy and its 68% band per true-energy bin,
    against a black dashed y=x reference line marking perfect
    reconstruction. When overlaying several models via
    `plot_energy_calibration_comparison`, the reference line is redrawn
    once per model - harmless visually, since it's identical every time.

    Args:
        result: Output of `energy_calibration` - a DataFrame with columns
            "bin_center", "p16", "p50", "p84".
        ax: Existing Axes to draw onto. If None, a new Figure/Axes pair is
            created. Passing the same `ax` across multiple calls is how
            several models end up overlaid on one figure.
        label: Legend label for this line (e.g. a model name).
        color: Line/band color. If None, matplotlib picks one
            automatically (its usual color-cycling behavior).
        linestyle: Line style. If None, matplotlib's default ("-").

    Returns:
        The Axes the plot was drawn on (either the one passed in, or a
        newly created one).
    """
    if ax is None:
        fig, ax = plt.subplots()
    (line,) = ax.plot(
        result["bin_center"],
        result["p50"],
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=2,
    )
    ax.fill_between(
        result["bin_center"],
        result["p16"],
        result["p84"],
        alpha=0.3,
        color=line.get_color(),
    )
    ax.plot(
        result["bin_center"],
        result["bin_center"],
        label="ideal",
        color="black",
        linestyle="--",
        linewidth=0.5,
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
    """Plot reconstructed-vs-true energy calibration for several models.

    Args:
        predictions: Mapping from model name (used as its legend label) to
            its prediction DataFrame. Every DataFrame must contain
            `truth_col` and `pred_col`. Model names matching one of the
            four paper models (see `nubench.style`) are drawn in that
            model's canonical color/linestyle; other names fall back to
            matplotlib's automatic styling.
        truth_col: Name of the column holding the true energy - the same
            across every model's DataFrame.
        pred_col: Name of the column holding the predicted energy - the
            same across every model's DataFrame.
        bins: Bin edges to use for every model. If None, edges are built
            automatically to span the *combined* true-energy range across
            all models, so every model is binned identically for a fair
            comparison.
        n_bins: Number of bins to construct when `bins` is not given.
        ax: Existing Axes to draw onto. If None, a new Figure/Axes pair is
            created.

    Returns:
        The Axes with one calibration line+band per model, with a legend
        identifying each by its key in `predictions`.
    """
    if bins is None:
        combined_min = min(df[truth_col].min() for df in predictions.values())
        combined_max = max(df[truth_col].max() for df in predictions.values())
        bins = np.logspace(
            np.log10(combined_min), np.log10(combined_max), n_bins + 1)
    if ax is None:
        fig, ax = plt.subplots()
    for model_name, df in predictions.items():
        result = energy_calibration(df, truth_col, pred_col, bins=bins)
        plot_energy_calibration(
            result,
            ax=ax,
            label=model_name,
            color=model_color(model_name),
            linestyle=model_linestyle(model_name),
        )
    ax.grid(which="both", alpha=0.3)
    # `plot_energy_calibration` redraws the "ideal" reference line once
    # per model (harmless on the plot itself, since every model's own
    # copy is identical) - but with several models overlaid, that means
    # several duplicate "ideal" *lines* on the Axes, not just duplicate
    # legend entries. A plain legend-level dedup isn't enough: this
    # figure's own legend later gets discarded and rebuilt from
    # `ax.get_legend_handles_labels()` when several detectors are
    # combined into one multi-panel grid (see `nubench.multipanel`),
    # which re-derives its handles straight from the Axes' lines, not
    # from whatever the last `ax.legend()` call filtered down to. So
    # the duplicates have to be removed as actual Line2D objects, not
    # just hidden from this one legend.
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
    """Build the two-panel energy calibration figure.

    A small top panel with each model's predicted-energy distribution as
    a step histogram (plus the true-energy distribution in black), and a
    main bottom panel with the calibration line+band+ideal-line per model,
    sharing the x-axis. Unlike every other plotting function here, this
    one needs two Axes, so by default it builds its own Figure rather
    than accepting a single `ax` - there's no single `ax` to fit the
    usual pattern into. Passing `ax_hist`/`ax_main` in together (e.g. one
    column of a larger grid built elsewhere) draws onto those instead,
    the same "reuse an existing Axes" idea as every other plot function's
    `ax` parameter, just doubled since this one needs two.

    Args:
        predictions: Mapping from model name (used as its legend label) to
            its prediction DataFrame. Every DataFrame must contain
            `truth_col` and `pred_col`.
        truth_col: Name of the column holding the true energy - the same
            across every model's DataFrame.
        pred_col: Name of the column holding the predicted energy - the
            same across every model's DataFrame.
        bins: Bin edges to use for both the histogram and the calibration
            statistics, for every model. If None, edges are built
            automatically to span the *combined* true-energy range across
            all models.
        n_bins: Number of bins to construct when `bins` is not given.
        ax_hist: Existing Axes for the top histogram panel. Must be given
            together with `ax_main` (both or neither) - if either is
            None, a new Figure with both panels is created instead.
        ax_main: Existing Axes for the bottom calibration panel.

    Returns:
        `(fig, (ax_hist, ax_main))` - the Figure, the top histogram Axes,
        and the bottom calibration Axes, in that order.
    """
    if bins is None:
        combined_min = min(df[truth_col].min() for df in predictions.values())
        combined_max = max(df[truth_col].max() for df in predictions.values())
        bins = np.logspace(
            np.log10(combined_min), np.log10(combined_max), n_bins + 1)
    if ax_hist is None or ax_main is None:
        fig, (ax_hist, ax_main) = plt.subplots(
            2,
            1,
            figsize=(3, 3 + 3 * 0.2),
            gridspec_kw={"height_ratios": [0.2, 1]},
            sharex=True,
            constrained_layout=True,
        )
    else:
        # `Axes.figure` is typed `Figure | SubFigure` since it could in
        # principle belong to a subfigure - always a plain `Figure` here,
        # since every Axes this function ever receives comes from
        # `plt.subplots()`, never from a subfigure.
        fig = ax_hist.figure  # type: ignore[assignment]

    for model_name, df in predictions.items():
        # mypy's matplotlib stub for `Axes.hist`'s `bins` doesn't
        # recognise `np.ndarray` as a valid `Sequence[float]`, even
        # though it works fine at runtime - same array type
        # `binned_percentiles`/`np.histogram` accept without complaint
        # elsewhere in this codebase.
        ax_hist.hist(
            df[pred_col],
            bins=bins,  # type: ignore[arg-type]
            histtype='step',
            color=model_color(model_name)
            )
    truth_col_combined = pd.concat(
        [df[truth_col] for df in predictions.values()], ignore_index=True
        )
    ax_hist.hist(
        truth_col_combined,
        bins=bins,  # type: ignore[arg-type]
        histtype='step',
        color='black'
    )
    ax_hist.set_xscale('log')
    ax_hist.set_yscale('log')
    ax_hist.yaxis.tick_right()
    ax_hist.xaxis.minorticks_on()
    ax_hist.spines[['right', 'top', 'left']].set_visible(False)
    ax_hist.yaxis.set_visible(False)
    plot_energy_calibration_comparison(
        predictions,
        truth_col,
        pred_col,
        bins,
        n_bins,
        ax=ax_main
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
    """Build the CC/NC two-column energy calibration figure.

    Two side-by-side copies of `plot_energy_calibration_figure`'s own
    two-panel layout (marginal histogram on top, calibration plot
    below), reusing it directly via its `ax_hist`/`ax_main` parameters -
    left column restricted to track/CC events, right column to
    cascade/NC events, for every model in `predictions`. Same physical
    meaning as `energy_calibration_by_topology`'s "track"/"cascade" keys:
    once restricted to muon-neutrino events, this *is* the CC/NC split.

    Args:
        predictions: Mapping from model name (used as its legend label) to
            its prediction DataFrame. Every DataFrame must contain
            `truth_col`, `pred_col`, and `is_track_col`.
        truth_col: Name of the column holding the true energy - the same
            across every model's DataFrame.
        pred_col: Name of the column holding the predicted energy - the
            same across every model's DataFrame.
        is_track_col: Name of the boolean track/CC vs. cascade/NC column
            - the same across every model's DataFrame.
        bins: Bin edges to use for both the histograms and the
            calibration statistics, for every model and both columns. If
            None, edges are built automatically to span the *combined*
            true-energy range across all models (both topologies
            together, so track and cascade share identical bins too).
        n_bins: Number of bins to construct when `bins` is not given.
        axes: Existing 2x2 array of Axes to draw onto (row 0 =
            histograms, row 1 = calibration plots; column 0 = track/CC,
            column 1 = cascade/NC) - the same shape `plt.subplots(2, 2)`
            itself returns. If None, a new Figure with its own 2x2 grid
            is created. Passing an existing block in is how a multi-
            detector grid (see `nubench.multipanel`) places one
            detector's CC/NC pair into its own slice of a bigger grid.

    Returns:
        `(fig, axes)` - the Figure, and the 2x2 array of Axes
        `plt.subplots(2, 2, ...)` itself returns (row 0 = histograms,
        row 1 = calibration plots; column 0 = track/CC, column 1 =
        cascade/NC).
    """
    if bins is None:
        combined_min = min(df[truth_col].min() for df in predictions.values())
        combined_max = max(df[truth_col].max() for df in predictions.values())
        bins = np.logspace(
            np.log10(combined_min), np.log10(combined_max), n_bins + 1)
    if axes is None:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(6, 3 + 3 * 0.2),
            gridspec_kw={"height_ratios": [0.2, 1]},
            sharex=True,
            constrained_layout=True,
        )
    else:
        # Same reasoning as `plot_energy_calibration_figure`'s own
        # `ax_hist`/`ax_main` reuse: every Axes this function ever
        # receives comes from `plt.subplots()`, so `.figure` is always a
        # plain `Figure`, never a `SubFigure`.
        fig = axes[0, 0].figure  # type: ignore[assignment]
    track_predictions = {
        name: df[df[is_track_col]] for name, df in predictions.items()
        }
    cascade_predictions = {
        name: df[~df[is_track_col]] for name, df in predictions.items()
        }
    plot_energy_calibration_figure(
        track_predictions,
        truth_col,
        pred_col,
        bins=bins,
        n_bins=n_bins,
        ax_hist=axes[0, 0],
        ax_main=axes[1, 0],
    )
    plot_energy_calibration_figure(
        cascade_predictions,
        truth_col,
        pred_col,
        bins=bins,
        n_bins=n_bins,
        ax_hist=axes[0, 1],
        ax_main=axes[1, 1],
    )
    label_bbox = dict(facecolor="white", edgecolor="none", alpha=0.3)
    axes[1, 0].text(
        0.05, 0.85, r"$\nu_\mu^{CC}$",
        transform=axes[1, 0].transAxes, fontsize=14, bbox=label_bbox,
    )
    axes[1, 1].text(
        0.05, 0.85, r"$\nu_\mu^{NC}$",
        transform=axes[1, 1].transAxes, fontsize=14, bbox=label_bbox,
    )
    return fig, axes
