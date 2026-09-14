"""Multi-detector paper-figure reproduction.

Reuses the single-detector plotting functions from
`nubench.evaluation.plotting`, arranging them into a grid with one slot
per detector, to reproduce the paper's own literal multi-detector
figures. Two different grid shapes are needed, matching how many Axes
each task's own combined figure needs per detector:

- energy, direction, inelasticity each have a *combined* two-panel
  figure (`plot_energy_calibration_by_topology_figure`,
  `plot_direction_figure`, `plot_inelasticity_figure`) - the paper packs
  two detectors' worth of these side by side per row (four panels wide),
  wrapping to additional rows as needed. `plot_energy_multi_detector_
  figure`/`plot_direction_multi_detector_figure`/`plot_inelasticity_
  multi_detector_figure` reproduce this by slicing a bigger Axes grid
  and passing each detector's own slice into the existing combined
  figure's `axes`/`ax_*` reuse parameters - the same "pass in an
  existing Axes" pattern already used everywhere else in this codebase.
- vertex and classification only have a single representative plot per
  detector, so the paper instead packs four detectors per row.
  `plot_roc_multi_detector_figure`/`plot_vertex_resolution_multi_
  detector_figure` build this grid directly and are what
  `make_multi_panel_figure`'s own default dispatch uses for these two
  features; `plot_track_score_multi_detector_figure`/`plot_vertex_
  contour_multi_detector_figure` build the same shape for the paper's
  other classification/vertex figures, not reachable from
  `make_multi_panel_figure` itself (call them directly). Each has its
  own consolidated legend, axis-label dedup, and (for the vertex
  contour) a data-driven axis crop - a plainer one-panel-per-dataset
  grid is also available generically as `nubench.evaluation.plotting.
  plot_multi_panel`, for any other single comparison plot not covered
  by one of these.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from nubench.data import (
    DEFAULT_COLUMNS,
    FEATURES_NEEDING_IS_TRACK,
    FEATURES_NEEDING_TRACK_ONLY_FILTER,
    add_is_track_column,
    filter_to_track_events,
    find_dataset_dir,
    load_predictions,
    normalize_inelasticity_pred_column,
    normalize_score_column,
)
from nubench.evaluation.metrics import vertex_contour_by_topology
from nubench.evaluation.plotting import (
    plot_direction_figure,
    plot_energy_calibration_by_topology_figure,
    plot_inelasticity_figure,
    plot_roc_curve_by_energy_regime_comparison,
    plot_track_score_distribution_by_topology_comparison,
    plot_vertex_contour_by_topology_comparison,
    plot_vertex_resolution_by_topology_comparison,
)
from nubench.evaluation.resolution import radial_and_depth
from nubench.style import detector_color, detector_display_name, model_color


def _display_name(detector: str) -> str:
    """The paper's display name for a detector key, e.g. "arca" -> "Flower L".

    Falls back to the key itself for anything not in
    `nubench.style.DETECTOR_DISPLAY_NAMES` - e.g. a directory name that
    doesn't follow NuBench's own detector-key convention - rather than
    raising, since an unrecognized name is still perfectly fine as a
    subplot title.
    """
    try:
        return detector_display_name(detector)
    except KeyError:
        return detector


def _color_for_display_name(display_name: str) -> str:
    """The paper's canonical text color for a detector's *display* name.

    Falls back to black for anything not in
    `nubench.style.DETECTOR_COLORS` - e.g. a directory name that doesn't
    follow NuBench's own detector-key convention - rather than raising.
    """
    try:
        return detector_color(display_name)
    except KeyError:
        return "black"


def _label_color(detector: str) -> str:
    """The paper's canonical text color for a detector *key*, e.g. "arca"."""
    return _color_for_display_name(_display_name(detector))


def _remove_all_legends(fig: plt.Figure) -> None:
    """Remove every per-panel legend the underlying plotting functions
    added - each detector's block reuses those functions, and each one
    calls its own `ax.legend()`, so left alone every panel gets its own
    copy of an identical legend. Used before building one consolidated,
    figure-level legend instead.
    """
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()


def _place_legend(
    fig: plt.Figure,
    axes: Optional[np.ndarray],
    handles: list,
    labels: list,
    loc: str = "lower right",
) -> None:
    """Place a figure-level legend, centered on the grid's empty (hidden)
    Axes if any exist - matching the paper's own figures, which tuck
    their single shared legend into whatever empty grid cell a jagged
    final row leaves behind - otherwise falling back to a fixed corner.

    A long legend (many models, e.g. once a detector with all four
    models is involved) can be taller than a single hidden grid row.
    `ax.get_position()` alone isn't enough to catch that: it's just the
    bare plot rectangle, not the tick-label text rendered just outside
    it, so a legend that "fits" by that measure can still visually
    overlap a real panel's axis numbers above it. Checking the actual
    rendered bounding boxes (`get_tightbbox`, which does include tick
    labels) and growing the legend into more columns - there's usually
    plenty of *width* to spare in an empty row, just not height - until
    it no longer overlaps anything real is more robust than guessing a
    fixed row count or column count up front.
    """
    if not handles:
        return
    if axes is not None:
        hidden = [ax for ax in axes.flat if not ax.get_visible()]
        visible = [ax for ax in axes.flat if ax.get_visible()]
        if hidden:
            # `constrained_layout` only finalizes Axes positions at draw
            # time - force one now so the hidden Axes' positions below
            # are accurate rather than stale placeholders.
            fig.canvas.draw()
            x0 = min(ax.get_position().x0 for ax in hidden)
            x1 = max(ax.get_position().x1 for ax in hidden)
            y0 = min(ax.get_position().y0 for ax in hidden)
            y1 = max(ax.get_position().y1 for ax in hidden)
            renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
            visible_bboxes = [ax.get_tightbbox(renderer) for ax in visible]
            ncol = 1
            legend = fig.legend(
                handles, labels, loc="center",
                bbox_to_anchor=((x0 + x1) / 2, (y0 + y1) / 2),
                bbox_transform=fig.transFigure,
                ncol=ncol,
            )
            fig.canvas.draw()
            while ncol < len(labels):
                legend_bbox = legend.get_window_extent(renderer)
                if not any(
                    legend_bbox.overlaps(vb) for vb in visible_bboxes
                ):
                    break
                legend.remove()
                ncol += 1
                legend = fig.legend(
                    handles, labels, loc="center",
                    bbox_to_anchor=((x0 + x1) / 2, (y0 + y1) / 2),
                    bbox_transform=fig.transFigure,
                    ncol=ncol,
                )
                fig.canvas.draw()
            return
    fig.legend(handles, labels, loc=loc)


def _consolidate_legends(
    fig: plt.Figure,
    axes: Optional[np.ndarray] = None,
    loc: str = "lower right",
) -> None:
    """Keep exactly one legend for the whole multi-detector grid.

    Each detector's block reuses the underlying single-detector
    plotting functions, and each one calls its own `ax.legend()` -
    identical in content across every detector and panel (same models,
    same fixed reference lines like "ideal"), so showing all of them
    clutters the figure badly and can collide with the detector-name
    labels. Removes every per-panel legend and adds back exactly one,
    built from whichever panel's handles were found first - matching
    the paper's own figures, which carry a single shared legend rather
    than one per panel.
    """
    handles: list = []
    labels: list = []
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            if not handles:
                handles, labels = ax.get_legend_handles_labels()
            legend.remove()
    _place_legend(fig, axes, handles, labels, loc=loc)


def _hide_unused_axes(axes: np.ndarray) -> None:
    """Hide every Axes marked invisible, in a way `constrained_layout`
    actually ignores.

    `set_visible(False)` alone still leaves an Axes counted towards its
    own column/row's size allocation - with a jagged final row (fewer
    detectors than the grid is wide), that starves the *other*, real
    rows sharing that column of the space their own labels need,
    clipping e.g. a y-axis label that has plenty of room in a fully
    filled grid. `set_in_layout(False)` tells `constrained_layout` to
    disregard the Axes entirely, while `get_position()` (used to center
    the legend in the freed-up space) still works normally.
    """
    for ax in axes.flat:
        if not ax.get_visible():
            ax.set_in_layout(False)


def _dedupe_shared_xlabels(axes: np.ndarray) -> None:
    """Keep the x-axis label text and tick labels only on the bottom-most
    *visible* Axes in each column, clearing/hiding them elsewhere.

    `sharex=True`'s own automatic tick-label hiding assumes a column's
    last grid row is always the visible one - that breaks as soon as a
    jagged final row (fewer detectors than the grid is wide) leaves a
    column's true last visible Axes higher up than that, duplicating
    the axis label down every detector row instead of showing it once
    at the bottom, as the paper's own figures do.
    """
    n_rows, n_cols = axes.shape
    for col in range(n_cols):
        visible_rows = [
            r for r in range(n_rows) if axes[r, col].get_visible()
        ]
        if not visible_rows:
            continue
        last_row = max(visible_rows)
        for r in visible_rows:
            ax = axes[r, col]
            if r == last_row:
                ax.tick_params(labelbottom=True)
            else:
                ax.tick_params(labelbottom=False)
                if ax.get_xlabel():
                    ax.set_xlabel("")


def _dedupe_shared_ylabels(axes: np.ndarray) -> None:
    """Keep the y-axis label text and tick labels only on the left-most
    *visible* Axes in each row, clearing/hiding them elsewhere.

    Mirrors `_dedupe_shared_xlabels`'s own column-based approach,
    transposed to rows - meant for a single-panel-per-detector grid
    (track score, ROC, vertex resolution/contour), where every panel
    shares the same y-quantity, so repeating the label/tick numbers on
    every column wastes space the paper's own figures don't spend.
    """
    n_rows, n_cols = axes.shape
    for row in range(n_rows):
        visible_cols = [
            c for c in range(n_cols) if axes[row, c].get_visible()
        ]
        if not visible_cols:
            continue
        first_col = min(visible_cols)
        for c in visible_cols:
            ax = axes[row, c]
            if c == first_col:
                ax.tick_params(labelleft=True)
            else:
                ax.tick_params(labelleft=False)
                if ax.get_ylabel():
                    ax.set_ylabel("")


def _finalize_single_panel_grid(
    fig: plt.Figure,
    axes: np.ndarray,
    handles: list,
    labels: list,
    dedupe_ylabels: bool = True,
) -> None:
    """Shared post-processing for every single-panel-per-detector grid
    (track score, ROC, vertex resolution, vertex contour): hide unused
    slots in a way `constrained_layout` actually respects, dedupe the
    repeated x axis label down to one per column, strip every per-panel
    legend, and place one consolidated legend instead - the same
    cleanup `plot_energy_multi_detector_figure`/`plot_direction_multi_
    detector_figure`/`plot_inelasticity_multi_detector_figure` each do
    for their own paired-panel grids.

    Args:
        dedupe_ylabels: If True (the default - track score, ROC), also
            dedupe the y axis label/ticks down to one per row, since
            every panel shares the exact same y-quantity and range
            there. Vertex resolution and vertex contour pass False
            instead: their y-axis *range* genuinely differs from panel
            to panel (different detectors, different typical
            resolutions), so hiding the numbers on all but the first
            column would make the panels impossible to compare.
    """
    _hide_unused_axes(axes)
    _dedupe_shared_xlabels(axes)
    if dedupe_ylabels:
        _dedupe_shared_ylabels(axes)
    _remove_all_legends(fig)
    _place_legend(fig, axes, handles, labels)


def _detector_grid_shape(n_detectors: int, detectors_per_row: int):
    """How many detector-rows a grid of `n_detectors` needs, and the
    (block_row, block_col) each detector index falls into.

    Always leaves at least one hidden slot for `_place_legend` to sit
    in: when `n_detectors` divides `detectors_per_row` exactly, there
    would otherwise be no empty grid space at all, and `_place_legend`
    falls back to a fixed figure corner that can overlap a real panel.
    Adding one full spare row guarantees a legend-sized empty region
    every time, at the cost of a bit of extra figure height only in
    this exactly-full case.
    """
    detectors_per_row = max(1, detectors_per_row)
    n_detector_rows = -(-n_detectors // detectors_per_row)  # ceil division
    if n_detector_rows * detectors_per_row == n_detectors:
        n_detector_rows += 1
    positions = [divmod(i, detectors_per_row) for i in range(n_detectors)]
    return n_detector_rows, positions


def load_multi_panel_predictions(
    data_root: str,
    detectors: List[str],
    feature: str,
    models: List[str],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load one subplot's worth of predictions per detector.

    Args:
        data_root: Directory containing one subdirectory per detector.
        detectors: Detector keys, one per subplot in the resulting grid.
        feature: One of the five reconstruction tasks this package
            supports.
        models: Model names to overlay within each detector's subplot.

    Returns:
        Mapping from detector name to that detector's own
        `{model_name: DataFrame}` dict - the shape `plot_multi_panel`
        expects as its own first argument.
    """
    predictions_by_dataset = {}
    for detector in detectors:
        dataset_dir = find_dataset_dir(data_root, detector)
        predictions = {}
        for model in models:
            df = load_predictions(dataset_dir, feature, model=model)
            if feature in FEATURES_NEEDING_IS_TRACK:
                df = add_is_track_column(df)
            if feature in FEATURES_NEEDING_TRACK_ONLY_FILTER:
                df = filter_to_track_events(df)
            if feature == "classification":
                df = normalize_score_column(df)
            if feature == "inelasticity":
                df = normalize_inelasticity_pred_column(df)
            predictions[model] = df
        predictions_by_dataset[detector] = predictions
    return predictions_by_dataset


def plot_energy_multi_detector_figure(
    predictions_by_detector: Dict[str, Dict[str, pd.DataFrame]],
    truth_col: str,
    pred_col: str,
    is_track_col: str = "is_track",
    ncols: int = 4,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 60,
) -> plt.Figure:
    """Build the paper's multi-detector CC/NC energy calibration grid.

    Each detector gets its own 2x2 block - the same one
    `plot_energy_calibration_by_topology_figure` builds for a single
    detector (marginal histogram + calibration, one column per CC/NC) -
    packed `ncols // 2` detectors per row, wrapping to further rows as
    needed. Bins are computed per detector (from that detector's own
    combined true-energy range), matching
    `plot_energy_calibration_by_topology_figure`'s own default - pass
    `bins` explicitly for identical bins across every detector too.

    Args:
        predictions_by_detector: Mapping from detector key (e.g. "arca")
            to that detector's own per-model predictions dict. Every
            DataFrame must contain `truth_col`, `pred_col`, and
            `is_track_col`.
        truth_col: Name of the column holding the true energy - the same
            across every detector/model's DataFrame.
        pred_col: Name of the column holding the predicted energy - the
            same across every detector/model's DataFrame.
        is_track_col: Name of the boolean track/CC vs. cascade/NC column
            - the same across every detector/model's DataFrame.
        ncols: Total Axes columns in the grid - `ncols // 2` detectors
            per row, since each detector needs two columns (CC/NC).
        bins: Bin edges to use for every detector. If None (default),
            each detector gets its own bins built from its own combined
            true-energy range.
        n_bins: Number of bins to construct when `bins` is not given.

    Returns:
        The Figure containing one CC/NC block per detector, each
        labelled with the paper's display name for that detector.
    """
    detectors = list(predictions_by_detector.keys())
    detectors_per_row = max(1, ncols // 2)
    n_detector_rows, positions = _detector_grid_shape(
        len(detectors), detectors_per_row
    )
    total_cols = detectors_per_row * 2
    total_rows = n_detector_rows * 2
    height_ratios = [0.2, 1] * n_detector_rows
    fig, axes = plt.subplots(
        total_rows,
        total_cols,
        figsize=(total_cols * 3, n_detector_rows * (3 + 3 * 0.2)),
        gridspec_kw={"height_ratios": height_ratios},
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for (block_row, block_col), detector in zip(positions, detectors):
        row0, col0 = block_row * 2, block_col * 2
        block_axes = axes[row0:row0 + 2, col0:col0 + 2]
        plot_energy_calibration_by_topology_figure(
            predictions_by_detector[detector],
            truth_col,
            pred_col,
            is_track_col,
            bins=bins,
            n_bins=n_bins,
            axes=block_axes,
        )
        for c in (col0, col0 + 1):
            axes[row0 + 1, c].text(
                0.05, 0.98, _display_name(detector),
                transform=axes[row0 + 1, c].transAxes,
                fontsize=14, va="top", color=_label_color(detector),
            )
    n_slots = n_detector_rows * detectors_per_row
    for j in range(len(detectors), n_slots):
        block_row, block_col = divmod(j, detectors_per_row)
        row0, col0 = block_row * 2, block_col * 2
        for r in range(row0, row0 + 2):
            for c in range(col0, col0 + 2):
                axes[r, c].set_visible(False)
    _hide_unused_axes(axes)
    _dedupe_shared_xlabels(axes)
    _consolidate_legends(fig, axes=axes)
    return fig


def _direction_legend_handles(
    models: List[str], has_kinematic_angle: bool
) -> Tuple[list, list]:
    """Build (handles, labels) for the paper's own direction legend.

    One solid colored line per model (labelled by model name only - not
    the underlying per-panel "ModelA (track)"/"ModelA (cascade)" labels,
    which exist so a *single* detector's own two topology lines stay
    distinguishable, but would otherwise dominate the shared legend with
    every detector's copy of the same names), plus two shared grey
    entries spelling out what solid/dashed mean
    (r"$\\nu_\\mu^{CC}$"/r"$\\nu_\\mu^{NC}$", matching the paper's own
    wording rather than "track"/"cascade"), plus "Kinematic Angle" if a
    reference line is present.
    """
    handles: list = []
    labels: list = []
    for model in models:
        handles.append(Line2D([], [], color=model_color(model), linestyle="-"))
        labels.append(model)
    handles.append(Line2D([], [], color="grey", linestyle="-"))
    labels.append(r"$\nu_\mu^{CC}$")
    handles.append(Line2D([], [], color="grey", linestyle="--"))
    labels.append(r"$\nu_\mu^{NC}$")
    if has_kinematic_angle:
        handles.append(Line2D([], [], color="red", linestyle="-"))
        labels.append("Kinematic Angle")
    return handles, labels


def plot_direction_multi_detector_figure(
    predictions_by_detector: Dict[str, Dict[str, pd.DataFrame]],
    truth_zenith_col: str,
    truth_azimuth_col: str,
    pred_x_col: str,
    pred_y_col: str,
    pred_z_col: str,
    energy_col: str,
    is_track_col: str = "is_track",
    muon_zenith_col: Optional[str] = None,
    muon_azimuth_col: Optional[str] = None,
    degrees: bool = True,
    distribution_bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    distribution_n_bins: int = 120,
    ncols: int = 4,
) -> plt.Figure:
    """Build the paper's multi-detector direction resolution/distribution grid.

    Each detector gets its own 1x2 block - the same one
    `plot_direction_figure` builds for a single detector (resolution-vs-
    energy + error distribution) - packed `ncols // 2` detectors per
    row, wrapping to further rows as needed.

    Args:
        predictions_by_detector: Mapping from detector key to that
            detector's own per-model predictions dict.
        truth_zenith_col: Name of the column holding the true zenith
            angle (radians).
        truth_azimuth_col: Name of the column holding the true azimuth
            angle (radians).
        pred_x_col: Name of the column holding the predicted direction
            vector's x-component.
        pred_y_col: Name of the column holding the predicted direction
            vector's y-component.
        pred_z_col: Name of the column holding the predicted direction
            vector's z-component.
        energy_col: Name of the column holding the true energy.
        is_track_col: Name of the boolean track/cascade column.
        muon_zenith_col: Name of the column holding the outgoing muon's
            own zenith angle, in radians. If either this or
            `muon_azimuth_col` is None, no muon line is drawn.
        muon_azimuth_col: Name of the column holding the outgoing muon's
            own azimuth angle, in radians.
        degrees: If True (default), report/label the opening angle in
            degrees. Otherwise radians.
        distribution_bins: Bin edges for each detector's right (error-
            distribution) panel - see `plot_direction_figure`'s own
            parameter of the same name.
        distribution_n_bins: Number of bins to construct when
            `distribution_bins` is not given.
        ncols: Total Axes columns in the grid - `ncols // 2` detectors
            per row, since each detector needs two columns.

    Returns:
        The Figure containing one resolution+distribution block per
        detector, each labelled with the paper's display name for that
        detector.
    """
    detectors = list(predictions_by_detector.keys())
    detectors_per_row = max(1, ncols // 2)
    n_detector_rows, positions = _detector_grid_shape(
        len(detectors), detectors_per_row
    )
    total_cols = detectors_per_row * 2
    fig, axes = plt.subplots(
        n_detector_rows,
        total_cols,
        figsize=(total_cols * 3, n_detector_rows * 3),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for (block_row, block_col), detector in zip(positions, detectors):
        col0 = block_col * 2
        ax_resolution = axes[block_row, col0]
        ax_distribution = axes[block_row, col0 + 1]
        plot_direction_figure(
            predictions_by_detector[detector],
            truth_zenith_col,
            truth_azimuth_col,
            pred_x_col,
            pred_y_col,
            pred_z_col,
            energy_col,
            is_track_col,
            muon_zenith_col=muon_zenith_col,
            muon_azimuth_col=muon_azimuth_col,
            degrees=degrees,
            distribution_bins=distribution_bins,
            distribution_n_bins=distribution_n_bins,
            ax_resolution=ax_resolution,
            ax_distribution=ax_distribution,
        )
        for ax in (ax_resolution, ax_distribution):
            ax.text(
                0.05, 0.95, _display_name(detector),
                transform=ax.transAxes, fontsize=14, va="top",
                color=_label_color(detector),
            )
    n_slots = n_detector_rows * detectors_per_row
    for j in range(len(detectors), n_slots):
        block_row, block_col = divmod(j, detectors_per_row)
        col0 = block_col * 2
        axes[block_row, col0].set_visible(False)
        axes[block_row, col0 + 1].set_visible(False)
    _hide_unused_axes(axes)
    _dedupe_shared_xlabels(axes)
    _remove_all_legends(fig)
    models = list(next(iter(predictions_by_detector.values())).keys())
    has_kinematic_angle = (
        muon_zenith_col is not None and muon_azimuth_col is not None
    )
    handles, labels = _direction_legend_handles(models, has_kinematic_angle)
    _place_legend(fig, axes, handles, labels)
    return fig


def _inelasticity_legend_handles(
    models: List[str], energy_threshold: float
) -> Tuple[list, list]:
    """Build (handles, labels) for the paper's own inelasticity legend.

    One solid colored line per model (by name only - not the underlying
    per-panel "ModelA (low energy)"/"ModelA (high energy)" labels), a
    black "Truth" entry, plus two shared grey entries spelling out what
    solid/dashed mean in terms of the actual energy threshold (matching
    the paper's own "E <= ... GeV"/"E > ... GeV" wording rather than
    "low energy"/"high energy") - the same pattern already used by
    `plot_roc_curve_by_energy_regime_comparison`. Solid is low energy,
    dashed is high energy - matching the paper's own legend exactly
    (see `plot_inelasticity_distribution_by_energy_regime`'s own
    docstring).
    """
    handles: list = []
    labels: list = []
    for model in models:
        handles.append(Line2D([], [], color=model_color(model), linestyle="-"))
        labels.append(model)
    handles.append(Line2D([], [], color="black", linestyle="-"))
    labels.append("Truth")
    handles.append(Line2D([], [], color="grey", linestyle="-"))
    labels.append(f"E <= {energy_threshold:g} GeV")
    handles.append(Line2D([], [], color="grey", linestyle="--"))
    labels.append(f"E > {energy_threshold:g} GeV")
    return handles, labels


def plot_inelasticity_multi_detector_figure(
    predictions_by_detector: Dict[str, Dict[str, pd.DataFrame]],
    truth_col: str,
    pred_col: str,
    energy_col: str,
    energy_threshold: float = 100.0,
    distribution_bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    distribution_n_bins: int = 49,
    ncols: int = 4,
) -> plt.Figure:
    """Build the paper's multi-detector inelasticity distribution/
    resolution grid.

    Each detector gets its own 1x2 block - the same one
    `plot_inelasticity_figure` builds for a single detector (value
    distribution + resolution-vs-energy) - packed `ncols // 2` detectors
    per row, wrapping to further rows as needed.

    Args:
        predictions_by_detector: Mapping from detector key to that
            detector's own per-model predictions dict.
        truth_col: Name of the column holding the true (visible)
            inelasticity.
        pred_col: Name of the column holding the predicted inelasticity.
        energy_col: Name of the column holding the true energy.
        energy_threshold: Energy value separating "low" (<=) from "high"
            (>) energy events.
        distribution_bins: Bin edges for each detector's left
            (distribution) panel - see `plot_inelasticity_figure`'s own
            parameter of the same name.
        distribution_n_bins: Number of bins to construct when
            `distribution_bins` is not given. Defaults to 49, matching
            the paper's own notebook exactly (`np.linspace(0, 1, 50)`
            is 50 edges = 49 bins) - the same value regardless of grid
            size, since the original notebook doesn't coarsen this for
            any detector.
        ncols: Total Axes columns in the grid - `ncols // 2` detectors
            per row, since each detector needs two columns.

    Returns:
        The Figure containing one distribution+resolution block per
        detector, each labelled with the paper's display name for that
        detector.
    """
    detectors = list(predictions_by_detector.keys())
    detectors_per_row = max(1, ncols // 2)
    n_detector_rows, positions = _detector_grid_shape(
        len(detectors), detectors_per_row
    )
    total_cols = detectors_per_row * 2
    fig, axes = plt.subplots(
        n_detector_rows,
        total_cols,
        figsize=(total_cols * 3, n_detector_rows * 3),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for (block_row, block_col), detector in zip(positions, detectors):
        col0 = block_col * 2
        ax_distribution = axes[block_row, col0]
        ax_resolution = axes[block_row, col0 + 1]
        plot_inelasticity_figure(
            predictions_by_detector[detector],
            truth_col,
            pred_col,
            energy_col,
            energy_threshold=energy_threshold,
            distribution_bins=distribution_bins,
            distribution_n_bins=distribution_n_bins,
            ax_distribution=ax_distribution,
            ax_resolution=ax_resolution,
        )
        # Matches the paper's own open-histogram style exactly (its own
        # notebook does this too: no box, no y-axis label, and no y-axis
        # ticks/numbers at all - a percentage isn't shown as a labelled
        # axis, just a relative shape between the drawn lines).
        ax_distribution.spines[["right", "top", "left"]].set_visible(False)
        ax_distribution.set_yticks([])
        ax_distribution.set_ylabel("")
        for ax, x, ha in (
            (ax_distribution, 0.95, "right"), (ax_resolution, 0.05, "left")
        ):
            ax.text(
                x, 0.95, _display_name(detector),
                transform=ax.transAxes,
                fontsize=14, va="top", ha=ha,
                color=_label_color(detector),
            )
    n_slots = n_detector_rows * detectors_per_row
    for j in range(len(detectors), n_slots):
        block_row, block_col = divmod(j, detectors_per_row)
        col0 = block_col * 2
        axes[block_row, col0].set_visible(False)
        axes[block_row, col0 + 1].set_visible(False)
    _hide_unused_axes(axes)
    _dedupe_shared_xlabels(axes)
    _remove_all_legends(fig)
    models = list(next(iter(predictions_by_detector.values())).keys())
    handles, labels = _inelasticity_legend_handles(models, energy_threshold)
    _place_legend(fig, axes, handles, labels)
    return fig


def _track_score_legend_handles(models: List[str]) -> Tuple[list, list]:
    """Build (handles, labels) for the paper's own track-score legend.

    One solid colored line per model (by name only - not the underlying
    per-panel "ModelA (track)"/"ModelA (cascade)" labels), plus two
    shared grey entries spelling out what dotted/solid mean in terms of
    the actual physics - "track" is muon-neutrino CC, "cascade" is NC
    (same convention used everywhere else in this codebase) - matching
    the paper's own $\\nu_\\mu^{CC}$/$\\nu_\\mu^{NC}$ wording rather than
    "track"/"cascade", the same pattern already used by
    `_direction_legend_handles`.
    """
    handles: list = []
    labels: list = []
    for model in models:
        handles.append(
            Line2D([], [], color=model_color(model), linestyle="-")
        )
        labels.append(model)
    handles.append(Line2D([], [], color="grey", linestyle=":"))
    labels.append(r"$\nu_\mu^{CC}$")
    handles.append(Line2D([], [], color="grey", linestyle="-"))
    labels.append(r"$\nu_\mu^{NC}$")
    return handles, labels


def plot_track_score_multi_detector_figure(
    predictions_by_detector: Dict[str, Dict[str, pd.DataFrame]],
    score_col: str,
    is_track_col: str,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 50,
    ncols: int = 4,
) -> plt.Figure:
    """Build the paper's multi-detector track-score log-count grid.

    One panel per detector (`ncols` per row, wrapping to further rows as
    needed) - each showing `plot_track_score_distribution_by_topology_
    comparison`'s own step-histogram lines for every model.

    Args:
        predictions_by_detector: Mapping from detector key to that
            detector's own per-model predictions dict.
        score_col: Name of the column holding the classifier's score.
        is_track_col: Name of the boolean track/cascade column.
        bins: Bin edges for every detector's panel - see
            `plot_track_score_distribution_by_topology_comparison`'s own
            `bins` parameter.
        n_bins: Number of bins to construct when `bins` is not given.
        ncols: Detectors per row (one panel each).

    Returns:
        The Figure containing one track-score panel per detector,
        titled with the paper's display name for that detector.
    """
    detectors = list(predictions_by_detector.keys())
    n_rows, positions = _detector_grid_shape(len(detectors), ncols)
    fig, axes = plt.subplots(
        n_rows, ncols,
        figsize=(ncols * 3, n_rows * 3),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for (row, col), detector in zip(positions, detectors):
        ax = axes[row, col]
        plot_track_score_distribution_by_topology_comparison(
            predictions_by_detector[detector],
            score_col=score_col,
            is_track_col=is_track_col,
            bins=bins,
            n_bins=n_bins,
            ax=ax,
        )
        # A text annotation inside the panel, not a title above it -
        # matching the paper's own figure and every paired-panel grid
        # elsewhere in this module (energy/direction/inelasticity).
        ax.text(
            0.05, 0.95, _display_name(detector),
            transform=ax.transAxes, fontsize=14, va="top", ha="left",
            color=_label_color(detector),
        )
    n_slots = n_rows * ncols
    for j in range(len(detectors), n_slots):
        row, col = divmod(j, ncols)
        axes[row, col].set_visible(False)
    models = list(next(iter(predictions_by_detector.values())).keys())
    handles, labels = _track_score_legend_handles(models)
    _finalize_single_panel_grid(fig, axes, handles, labels)
    return fig


def _roc_energy_regime_legend_handles(
    models: List[str], low_threshold: float, high_threshold: float
) -> Tuple[list, list]:
    """Build (handles, labels) for the paper's own energy-regime-split ROC
    legend: one solid colored line per model, plus three shared grey
    entries spelling out what solid/dashed/dash-dot-dot mean in terms of
    the actual energy thresholds - the same "model color, shared grey
    linestyle meaning" pattern as `_inelasticity_legend_handles`.
    """
    handles: list = []
    labels: list = []
    for model in models:
        handles.append(
            Line2D([], [], color=model_color(model), linestyle="-")
        )
        labels.append(model)
    handles.append(Line2D([], [], color="grey", linestyle="-"))
    labels.append(f"E <= {low_threshold:g} GeV")
    handles.append(Line2D([], [], color="grey", linestyle="--"))
    labels.append(f"{low_threshold:g} < E <= {high_threshold:g} GeV")
    handles.append(Line2D([], [], color="grey", linestyle=(0, (3, 1, 1, 1))))
    labels.append(f"E > {high_threshold:g} GeV")
    return handles, labels


def plot_roc_multi_detector_figure(
    predictions_by_detector: Dict[str, Dict[str, pd.DataFrame]],
    truth_col: str,
    score_col: str,
    energy_col: str,
    low_threshold: float = 100.0,
    high_threshold: float = 1000.0,
    ncols: int = 4,
) -> plt.Figure:
    """Build the paper's multi-detector, energy-regime-split ROC grid.

    One panel per detector (`ncols` per row, wrapping to further rows as
    needed) - each showing `plot_roc_curve_by_energy_regime_comparison`'s
    own three-linestyle-per-model curves (solid/dashed/dash-dot-dot for
    low/mid/high energy), matching the paper's own multi-detector ROC
    figure - unlike `make_multi_panel_figure`'s own "classification"
    dispatch, which uses the single plain ROC curve instead.

    Args:
        predictions_by_detector: Mapping from detector key to that
            detector's own per-model predictions dict.
        truth_col: Name of the column holding the true binary label.
        score_col: Name of the column holding the classifier's score.
        energy_col: Name of the column holding the true energy.
        low_threshold: Upper bound (inclusive) of the "low energy"
            regime.
        high_threshold: Upper bound (inclusive) of the "mid energy"
            regime; everything above this is "high energy".
        ncols: Detectors per row (one panel each).

    Returns:
        The Figure containing one ROC panel per detector, titled with
        the paper's display name for that detector.
    """
    detectors = list(predictions_by_detector.keys())
    n_rows, positions = _detector_grid_shape(len(detectors), ncols)
    fig, axes = plt.subplots(
        n_rows, ncols,
        figsize=(ncols * 3, n_rows * 3),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for (row, col), detector in zip(positions, detectors):
        ax = axes[row, col]
        plot_roc_curve_by_energy_regime_comparison(
            predictions_by_detector[detector],
            truth_col=truth_col,
            score_col=score_col,
            energy_col=energy_col,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            ax=ax,
        )
        # A text annotation inside the panel, not a title above it -
        # matching the paper's own figure. Bottom-right, not top-left
        # like the other multi-detector grids: a good ROC curve hugs
        # the top-left corner, so that's the one place a label here
        # would collide with the curves themselves.
        ax.text(
            0.95, 0.05, _display_name(detector),
            transform=ax.transAxes, fontsize=14, va="bottom", ha="right",
            color=_label_color(detector),
        )
    n_slots = n_rows * ncols
    for j in range(len(detectors), n_slots):
        row, col = divmod(j, ncols)
        axes[row, col].set_visible(False)
    models = list(next(iter(predictions_by_detector.values())).keys())
    handles, labels = _roc_energy_regime_legend_handles(
        models, low_threshold, high_threshold
    )
    _finalize_single_panel_grid(fig, axes, handles, labels)
    return fig


def _vertex_resolution_legend_handles(
    models: List[str],
) -> Tuple[list, list]:
    """Build (handles, labels) for the paper's own vertex-resolution
    legend: one solid colored line per model, plus two shared grey
    entries spelling out solid/dashed as muon-neutrino CC/NC - the same
    pattern as `_direction_legend_handles`'s own CC/NC entries.
    """
    handles: list = []
    labels: list = []
    for model in models:
        handles.append(
            Line2D([], [], color=model_color(model), linestyle="-")
        )
        labels.append(model)
    handles.append(Line2D([], [], color="grey", linestyle="-"))
    labels.append(r"$\nu_\mu^{CC}$")
    handles.append(Line2D([], [], color="grey", linestyle="--"))
    labels.append(r"$\nu_\mu^{NC}$")
    return handles, labels


def plot_vertex_resolution_multi_detector_figure(
    predictions_by_detector: Dict[str, Dict[str, pd.DataFrame]],
    truth_x_col: str,
    truth_y_col: str,
    truth_z_col: str,
    pred_x_col: str,
    pred_y_col: str,
    pred_z_col: str,
    energy_col: str,
    is_track_col: str,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 30,
    ncols: int = 4,
) -> plt.Figure:
    """Build the paper's multi-detector vertex-resolution grid.

    One panel per detector (`ncols` per row, wrapping to further rows as
    needed) - each showing `plot_vertex_resolution_by_topology_
    comparison`'s own Euclidean-distance-vs-energy lines for every
    model. Unlike every other single-panel grid in this module, the y
    axis label/ticks are kept on *every* panel, not just the first
    column of each row - each detector's Euclidean Distance range
    genuinely differs, so hiding the numbers would make the panels
    impossible to compare against each other.

    Args:
        predictions_by_detector: Mapping from detector key to that
            detector's own per-model predictions dict.
        truth_x_col: Name of the column holding the true vertex
            x-coordinate.
        truth_y_col: Name of the column holding the true vertex
            y-coordinate.
        truth_z_col: Name of the column holding the true vertex
            z-coordinate.
        pred_x_col: Name of the column holding the predicted vertex
            x-coordinate.
        pred_y_col: Name of the column holding the predicted vertex
            y-coordinate.
        pred_z_col: Name of the column holding the predicted vertex
            z-coordinate.
        energy_col: Name of the column holding the true energy.
        is_track_col: Name of the boolean track/cascade column.
        bins: Bin edges for every detector's panel - see
            `plot_vertex_resolution_by_topology_comparison`'s own `bins`
            parameter.
        n_bins: Number of bins to construct when `bins` is not given.
        ncols: Detectors per row (one panel each).

    Returns:
        The Figure containing one vertex-resolution panel per detector,
        titled with the paper's display name for that detector.
    """
    detectors = list(predictions_by_detector.keys())
    n_rows, positions = _detector_grid_shape(len(detectors), ncols)
    fig, axes = plt.subplots(
        n_rows, ncols,
        figsize=(ncols * 3, n_rows * 3),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for (row, col), detector in zip(positions, detectors):
        ax = axes[row, col]
        plot_vertex_resolution_by_topology_comparison(
            predictions_by_detector[detector],
            truth_x_col=truth_x_col,
            truth_y_col=truth_y_col,
            truth_z_col=truth_z_col,
            pred_x_col=pred_x_col,
            pred_y_col=pred_y_col,
            pred_z_col=pred_z_col,
            energy_col=energy_col,
            is_track_col=is_track_col,
            bins=bins,
            n_bins=n_bins,
            ax=ax,
        )
        # A text annotation inside the panel, not a title above it -
        # matching the paper's own figure and every other single-panel
        # grid in this module except the vertex-contour one, which
        # keeps its title (per explicit request).
        ax.text(
            0.05, 0.95, _display_name(detector),
            transform=ax.transAxes, fontsize=14, va="top", ha="left",
            color=_label_color(detector),
        )
    n_slots = n_rows * ncols
    for j in range(len(detectors), n_slots):
        row, col = divmod(j, ncols)
        axes[row, col].set_visible(False)
    models = list(next(iter(predictions_by_detector.values())).keys())
    handles, labels = _vertex_resolution_legend_handles(models)
    _finalize_single_panel_grid(
        fig, axes, handles, labels, dedupe_ylabels=False
    )
    return fig


def _vertex_contour_axis_ranges(
    predictions: Dict[str, pd.DataFrame],
    truth_x_col: str,
    truth_y_col: str,
    truth_z_col: str,
    pred_x_col: str,
    pred_y_col: str,
    pred_z_col: str,
    is_track_col: str,
    padding: float = 0.25,
    percentiles: Tuple[float, float] = (1.0, 99.0),
    sigma: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Data-driven bin edges for one detector's vertex-contour panel.

    `vertex_contour_by_topology`'s own default (`bins=100`, a plain int)
    spans the raw depth/radial residual's full min-max range - a
    handful of badly-reconstructed outlier events stretch that range so
    wide the actual 68% contour shrinks to a barely-visible dot, which
    is exactly the "plotted area is way too big" symptom compared to
    the paper.

    Cropping to a raw-percentile range of the *scatter* (as an earlier
    version of this function did) still leaves a lot of empty axis
    space, since a density contour's actual footprint is far smaller
    than where even 99% of individual (long-tailed) residuals happen to
    fall. Instead, this builds one generous first-pass set of bins,
    computes every model's own 68%-containment contour on them (the
    same computation `plot_vertex_contour_by_topology_comparison` does
    for real, just done here first to measure it), and crops to the
    union of *those* contours' own bounding boxes - the region where
    the smoothed density actually clears the containment threshold -
    with a fixed padding fraction around it. No per-detector hardcoded
    lookup table - computed fresh from whatever data is passed in,
    since a Stage 2 retraining run could shift a detector's own
    reconstruction quality (and therefore this natural range)
    arbitrarily.

    Returns:
        `(depth_edges, radial_edges)` - bin edge arrays pooling every
        model's own track+cascade residuals for this one detector.
    """
    depths = []
    radials = []
    for df in predictions.values():
        radial, depth = radial_and_depth(
            df[truth_x_col], df[truth_y_col], df[truth_z_col],
            df[pred_x_col], df[pred_y_col], df[pred_z_col],
        )
        depths.append(np.asarray(depth))
        radials.append(np.asarray(radial))
    depth_all = np.concatenate(depths)
    radial_all = np.concatenate(radials)
    depth_lo, depth_hi = np.percentile(depth_all, percentiles)
    radial_hi_broad = np.percentile(radial_all, percentiles[1])
    broad_depth_edges = np.linspace(depth_lo, depth_hi, 150)
    broad_radial_edges = np.linspace(0.0, radial_hi_broad, 150)

    depth_min = depth_max = radial_max = None
    for df in predictions.values():
        result = vertex_contour_by_topology(
            df, truth_x_col, truth_y_col, truth_z_col,
            pred_x_col, pred_y_col, pred_z_col, is_track_col,
            bins=[broad_depth_edges, broad_radial_edges], sigma=sigma,
        )
        for X, Y, H, level, median_depth, median_radial in result.values():
            mask = H >= level
            xs = np.append(X[mask], median_depth)
            ys = np.append(Y[mask], median_radial)
            depth_min = xs.min() if depth_min is None else min(
                depth_min, xs.min()
            )
            depth_max = xs.max() if depth_max is None else max(
                depth_max, xs.max()
            )
            radial_max = ys.max() if radial_max is None else max(
                radial_max, ys.max()
            )
    if depth_min is None:
        # Degenerate case (e.g. too little data for any bin to clear
        # the containment threshold) - fall back to the broad range.
        depth_min, depth_max = depth_lo, depth_hi
        radial_max = radial_hi_broad
    assert depth_max is not None and radial_max is not None

    depth_pad = (depth_max - depth_min) * padding
    depth_edges = np.linspace(
        depth_min - depth_pad, depth_max + depth_pad, 100
    )
    # Radial distance is non-negative, and the paper's own radial axis
    # always starts exactly at 0 - only the upper edge needs padding.
    radial_pad = radial_max * padding
    radial_edges = np.linspace(0.0, radial_max + radial_pad, 100)
    return depth_edges, radial_edges


def _vertex_contour_legend_handles(models: List[str]) -> Tuple[list, list]:
    """Build (handles, labels) for the paper's own vertex-contour legend:
    one solid colored line per model, plus two shared grey entries that
    combine linestyle *and* marker in a single handle (solid + star for
    CC, dashed + dot for NC) - matching the paper's own legend, which
    shows the median-point marker alongside the contour linestyle in
    one combined entry rather than as two separate ones.
    """
    handles: list = []
    labels: list = []
    for model in models:
        handles.append(
            Line2D([], [], color=model_color(model), linestyle="-")
        )
        labels.append(model)
    handles.append(
        Line2D(
            [], [], color="grey", linestyle="-", marker="*", markersize=10,
        )
    )
    labels.append(r"$\nu_\mu^{CC}$")
    handles.append(
        Line2D(
            [], [], color="grey", linestyle="--", marker=".", markersize=10,
        )
    )
    labels.append(r"$\nu_\mu^{NC}$")
    return handles, labels


def plot_vertex_contour_multi_detector_figure(
    predictions_by_detector: Dict[str, Dict[str, pd.DataFrame]],
    truth_x_col: str,
    truth_y_col: str,
    truth_z_col: str,
    pred_x_col: str,
    pred_y_col: str,
    pred_z_col: str,
    is_track_col: str,
    sigma: float = 3.0,
    ncols: int = 4,
) -> plt.Figure:
    """Build the paper's multi-detector vertex-contour grid.

    One panel per detector (`ncols` per row, wrapping to further rows as
    needed) - each showing `plot_vertex_contour_by_topology_comparison`'s
    own 68%-containment depth/radial contour for every model, cropped
    to each contour's own bounding box (see `_vertex_contour_axis_
    ranges`) rather than the full outlier-stretched default, with a
    linear radial axis cut at exactly 0, matching the paper. Like
    vertex resolution, the y axis label/ticks are kept on *every*
    panel, not just the first column of each row, since each
    detector's own cropped radial range genuinely differs.

    Args:
        predictions_by_detector: Mapping from detector key to that
            detector's own per-model predictions dict.
        truth_x_col: Name of the column holding the true vertex
            x-coordinate.
        truth_y_col: Name of the column holding the true vertex
            y-coordinate.
        truth_z_col: Name of the column holding the true vertex
            z-coordinate.
        pred_x_col: Name of the column holding the predicted vertex
            x-coordinate.
        pred_y_col: Name of the column holding the predicted vertex
            y-coordinate.
        pred_z_col: Name of the column holding the predicted vertex
            z-coordinate.
        is_track_col: Name of the boolean track/cascade column.
        sigma: Passed through to `vertex_contour_by_topology`'s own
            `sigma` argument.
        ncols: Detectors per row (one panel each).

    Returns:
        The Figure containing one vertex-contour panel per detector,
        titled with the paper's display name for that detector.
    """
    detectors = list(predictions_by_detector.keys())
    n_rows, positions = _detector_grid_shape(len(detectors), ncols)
    fig, axes = plt.subplots(
        n_rows, ncols,
        figsize=(ncols * 3, n_rows * 3),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for (row, col), detector in zip(positions, detectors):
        ax = axes[row, col]
        predictions = predictions_by_detector[detector]
        depth_edges, radial_edges = _vertex_contour_axis_ranges(
            predictions, truth_x_col, truth_y_col, truth_z_col,
            pred_x_col, pred_y_col, pred_z_col, is_track_col,
            sigma=sigma,
        )
        plot_vertex_contour_by_topology_comparison(
            predictions,
            truth_x_col=truth_x_col,
            truth_y_col=truth_y_col,
            truth_z_col=truth_z_col,
            pred_x_col=pred_x_col,
            pred_y_col=pred_y_col,
            pred_z_col=pred_z_col,
            is_track_col=is_track_col,
            bins=[depth_edges, radial_edges],
            sigma=sigma,
            ax=ax,
        )
        ax.set_xlim(depth_edges[0], depth_edges[-1])
        # Radial distance is non-negative and the paper's own radial
        # axis is linear, cut at exactly 0 - not log-scaled.
        ax.set_ylim(0.0, radial_edges[-1])
        ax.set_title(
            _display_name(detector), color=_label_color(detector),
            fontsize=14,
        )
    n_slots = n_rows * ncols
    for j in range(len(detectors), n_slots):
        row, col = divmod(j, ncols)
        axes[row, col].set_visible(False)
    models = list(next(iter(predictions_by_detector.values())).keys())
    handles, labels = _vertex_contour_legend_handles(models)
    _finalize_single_panel_grid(
        fig, axes, handles, labels, dedupe_ylabels=False
    )
    return fig


def make_multi_panel_figure(
    feature: str,
    predictions_by_dataset: Dict[str, Dict[str, pd.DataFrame]],
    ncols: int = 4,
) -> plt.Figure:
    """Build the paper's multi-detector grid for a single feature.

    energy, direction, and inelasticity get the paper's own paired-panel
    layout (two Axes per detector, `ncols // 2` detectors per row - see
    `plot_energy_multi_detector_figure`/`plot_direction_multi_detector_
    figure`/`plot_inelasticity_multi_detector_figure`); vertex and
    classification only have one representative plot per detector
    (resolution-vs-energy, and the energy-regime-split ROC curve
    respectively), one panel per detector (`ncols` detectors per row) -
    see `plot_vertex_resolution_multi_detector_figure`/`plot_roc_multi_
    detector_figure`.

    Args:
        feature: One of the five reconstruction tasks this package
            supports.
        predictions_by_dataset: Mapping from detector *key* (e.g.
            "arca") to its own per-model predictions dict, as returned
            by `load_multi_panel_predictions` - keys are relabeled to
            the paper's own display names (e.g. "Flower L") for the
            subplot titles/labels.
        ncols: Total Axes columns in the grid (energy/direction/
            inelasticity fit `ncols // 2` detectors per row since each
            needs two columns; vertex/classification fit `ncols`
            detectors per row since each needs only one).

    Returns:
        The Figure containing one detector's worth of panels per
        detector, each labelled with the paper's display name.
    """
    if feature not in DEFAULT_COLUMNS:
        raise ValueError(
            f"Unknown feature {feature!r}; must be one of "
            f"{sorted(DEFAULT_COLUMNS)}"
        )
    columns = DEFAULT_COLUMNS[feature]
    if feature == "energy":
        return plot_energy_multi_detector_figure(
            predictions_by_dataset,
            truth_col=columns["truth_col"],
            pred_col=columns["pred_col"],
            is_track_col="is_track",
            ncols=ncols,
        )
    if feature == "direction":
        return plot_direction_multi_detector_figure(
            predictions_by_dataset,
            truth_zenith_col=columns["truth_zenith_col"],
            truth_azimuth_col=columns["truth_azimuth_col"],
            pred_x_col=columns["pred_x_col"],
            pred_y_col=columns["pred_y_col"],
            pred_z_col=columns["pred_z_col"],
            energy_col=columns["energy_col"],
            is_track_col="is_track",
            muon_zenith_col=columns["muon_zenith_col"],
            muon_azimuth_col=columns["muon_azimuth_col"],
            ncols=ncols,
            # Same reasoning as `nubench.cli.make_figure`'s own direction
            # branch: the auto-built bins span the *combined* opening-
            # angle range, far wider than the ~0-10 deg. window each
            # right-hand panel actually displays - without this, every
            # panel renders empty. This zoomed-in range is the paper-
            # matching default.
            distribution_bins=np.linspace(0, 10, 120),
        )
    if feature == "vertex":
        return plot_vertex_resolution_multi_detector_figure(
            predictions_by_dataset,
            truth_x_col=columns["truth_x_col"],
            truth_y_col=columns["truth_y_col"],
            truth_z_col=columns["truth_z_col"],
            pred_x_col=columns["pred_x_col"],
            pred_y_col=columns["pred_y_col"],
            pred_z_col=columns["pred_z_col"],
            energy_col=columns["energy_col"],
            is_track_col="is_track",
            ncols=ncols,
        )
    if feature == "inelasticity":
        return plot_inelasticity_multi_detector_figure(
            predictions_by_dataset,
            truth_col=columns["truth_col"],
            pred_col=columns["pred_col"],
            energy_col=columns["energy_col"],
            ncols=ncols,
        )
    if feature == "classification":
        return plot_roc_multi_detector_figure(
            predictions_by_dataset,
            truth_col=columns["truth_col"],
            score_col=columns["score_col"],
            energy_col=columns["energy_col"],
            ncols=ncols,
        )
    raise ValueError(
        f"Unknown feature {feature!r}; must be one of "
        f"{sorted(DEFAULT_COLUMNS)}"
    )
