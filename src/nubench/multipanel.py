"""Multi-detector paper-figure reproduction.

Arranges the single-detector plots from `nubench.evaluation.plotting`
into a grid with one slot per detector. Two grid shapes are needed:

- energy, direction and inelasticity have a *combined* two-panel figure
  per detector, so the paper packs `ncols // 2` detectors per row.
- vertex and classification have a single plot per detector, so the
  paper packs `ncols` per row.

Each figure gets one consolidated legend placed in the grid's leftover
empty cells, plus axis-label dedup so a shared label appears once.
"""

from typing import (
    Callable, Dict, List, Optional, Sequence, Tuple, Union,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from nubench.data import DEFAULT_COLUMNS, load_feature_predictions
from nubench.evaluation.plotting import (
    plot_direction_figure,
    plot_energy_calibration_by_topology_figure,
    plot_inelasticity_figure,
    plot_roc_curve_by_energy_regime_comparison,
    plot_track_score_distribution_by_topology_comparison,
    plot_vertex_contour_by_topology_comparison,
    plot_vertex_resolution_by_topology_comparison,
    vertex_contour_axis_ranges,
)
from nubench.evaluation.plotting.classification import HIGH_ENERGY_LINESTYLE
from nubench.style import (
    detector_color, detector_display_name, model_color, model_linestyle,
)

CC_LABEL = r"$\nu_\mu^{CC}$"
NC_LABEL = r"$\nu_\mu^{NC}$"

# (x, y, va, ha) for a detector label drawn inside a panel.
TOP_LEFT = (0.05, 0.95, "top", "left")
TOP_RIGHT = (0.95, 0.95, "top", "right")
BOTTOM_RIGHT = (0.95, 0.05, "bottom", "right")

PanelLabel = Tuple[float, float, str, str]


# ----------------------------------------------------------------------
# Detector labelling
# How a detector key becomes the paper's display name and colour.
# ----------------------------------------------------------------------


def _display_name(detector: str) -> str:
    """The paper's display name for a detector key: "arca" -> "Flower L".

    Falls back to the key itself for anything unrecognized.
    """
    try:
        return detector_display_name(detector)
    except KeyError:
        return detector


def _label_color(detector: str) -> str:
    """The paper's text color for a detector key, black if unrecognized."""
    try:
        return detector_color(_display_name(detector))
    except KeyError:
        return "black"


def _label_panel(ax: plt.Axes, detector: str, pos: PanelLabel) -> None:
    """Draw a detector name inside a panel, as the paper does."""
    x, y, va, ha = pos
    ax.text(
        x, y, _display_name(detector), transform=ax.transAxes,
        fontsize=14, va=va, ha=ha, color=_label_color(detector),
    )


# ----------------------------------------------------------------------
# Legend
# Every grid shows one consolidated legend instead of the identical
# copy each reused single-detector plot function draws for itself.
# `_legend_handles` builds the entries, `_remove_all_legends` clears
# the per-panel ones, `_place_legend` positions the survivor.
# ----------------------------------------------------------------------


def _legend_handles(
    models: List[str],
    extras: Sequence[Tuple[str, dict]],
    per_model_linestyle: bool = False,
) -> Tuple[list, list]:
    """One colored line per model, plus shared `extras` entries.

    Models are labelled by name only - the underlying per-panel labels
    ("DynEdge (track)") would otherwise repeat every detector's copy of
    the same names. Each extra is a `(label, Line2D kwargs)` pair,
    defaulting to grey, used to spell out what the linestyles mean.

    Models are drawn solid by default, because linestyle carries a
    second meaning in those figures (CC/NC, energy regime) and the two
    would collide. Energy is the exception: there linestyle is part of
    the model's own identity, so it passes `per_model_linestyle=True`.
    """
    handles = [
        Line2D(
            [], [], color=model_color(model),
            linestyle=model_linestyle(model) if per_model_linestyle else "-",
        )
        for model in models
    ]
    labels = list(models)
    for label, kwargs in extras:
        handles.append(Line2D([], [], **{"color": "grey", **kwargs}))
        labels.append(label)
    return handles, labels


def _models_of(
    predictions_by_detector: Dict[str, Dict[str, pd.DataFrame]],
) -> List[str]:
    """Model names from the first detector's predictions dict."""
    return list(next(iter(predictions_by_detector.values())).keys())


def _remove_all_legends(fig: plt.Figure) -> None:
    """Drop every per-panel legend.

    Called before building one consolidated figure-level
    legend instead.
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
    """Place one figure-level legend in the grid's empty (hidden) cells.

    Matches the paper, which tucks its shared legend into whatever cell a
    jagged final row leaves behind; falls back to a fixed corner if the
    grid is full.

    A long legend can be taller than one hidden row. `get_position()`
    isn't enough to catch that - it's the bare plot rectangle, excluding
    the tick-label text just outside it, so a legend that "fits" by that
    measure can still overlap a real panel's axis numbers. Instead,
    compare real rendered bboxes (`get_tightbbox` includes tick labels)
    and grow into more columns until nothing overlaps: an empty row has
    width to spare, just not height.
    """
    if not handles:
        return
    hidden = [] if axes is None else [
        ax for ax in axes.flat if not ax.get_visible()
    ]
    if axes is None or not hidden:
        fig.legend(handles, labels, loc=loc)
        return
    visible = [ax for ax in axes.flat if ax.get_visible()]
    # `constrained_layout` only finalizes positions at draw time - force
    # one so the positions read below aren't stale placeholders.
    fig.canvas.draw()
    # find center of empty region
    positions = [ax.get_position() for ax in hidden]
    center = (
        (min(p.x0 for p in positions) + max(p.x1 for p in positions)) / 2,
        (min(p.y0 for p in positions) + max(p.y1 for p in positions)) / 2,
    )
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    visible_bboxes = [ax.get_tightbbox(renderer) for ax in visible]
    for ncol in range(1, len(labels) + 1):
        legend = fig.legend(
            handles, labels, loc="center", bbox_to_anchor=center,
            bbox_transform=fig.transFigure, ncol=ncol,
        )
        fig.canvas.draw()
        bbox = legend.get_window_extent(renderer)
        if not any(bbox.overlaps(vb) for vb in visible_bboxes):
            return
        if ncol < len(labels):
            legend.remove()


# ----------------------------------------------------------------------
# Grid layout
#
# Shared machinery for arranging one detector per slot: how many rows
# are needed, which axes to hide, which labels to dedupe, and the two
# grid shapes (one panel per detector, or a side-by-side pair).
# ----------------------------------------------------------------------


def _hide_unused_axes(axes: np.ndarray) -> None:
    """Make hidden Axes invisible to `constrained_layout` as well.

    `set_visible(False)` alone still counts an Axes towards its column's
    size allocation, so a jagged final row starves the real rows sharing
    that column of the space their labels need. `set_in_layout(False)`
    drops it from layout entirely while `get_position()` still works.
    """
    for ax in axes.flat:
        if not ax.get_visible():
            ax.set_in_layout(False)


def _dedupe_shared_xlabels(axes: np.ndarray) -> None:
    """Keep x label and tick labels only on each column's bottom-most
    *visible* Axes.

    `sharex=True`'s own hiding assumes a column's last grid row is the
    visible one, which breaks as soon as a jagged final row leaves the
    true last visible Axes higher up.
    """
    n_rows, n_cols = axes.shape
    for col in range(n_cols):
        visible = [r for r in range(n_rows) if axes[r, col].get_visible()]
        for r in visible:
            ax = axes[r, col]
            if r == max(visible):
                ax.tick_params(labelbottom=True)
            else:
                ax.tick_params(labelbottom=False)
                if ax.get_xlabel():
                    ax.set_xlabel("")


def _dedupe_shared_ylabels(axes: np.ndarray) -> None:
    """Keep y label and tick labels only on each row's left-most visible
    Axes - `_dedupe_shared_xlabels` transposed.
    """
    n_rows, n_cols = axes.shape
    for row in range(n_rows):
        visible = [c for c in range(n_cols) if axes[row, c].get_visible()]
        for c in visible:
            ax = axes[row, c]
            if c == min(visible):
                ax.tick_params(labelleft=True)
            else:
                ax.tick_params(labelleft=False)
                if ax.get_ylabel():
                    ax.set_ylabel("")


def _finalize_grid(
    fig: plt.Figure,
    axes: np.ndarray,
    handles: list,
    labels: list,
    dedupe_xlabels: bool = True,
    dedupe_ylabels: bool = True,
) -> None:
    """Shared post-processing for every multi-detector grid.

    Pass either flag as False where that axis' *range* differs panel to
    panel, since a reader would otherwise read a panel's scale off a
    neighbour that does not share it. Both are False for the vertex
    contour, whose axes are cropped per detector.
    """
    _hide_unused_axes(axes)
    if dedupe_xlabels:
        _dedupe_shared_xlabels(axes)
    if dedupe_ylabels:
        _dedupe_shared_ylabels(axes)
    _remove_all_legends(fig)
    _place_legend(fig, axes, handles, labels)


def _detector_grid_shape(
    n_detectors: int, detectors_per_row: int
) -> Tuple[int, List[Tuple[int, int]]]:
    """Rows needed, and the (block_row, block_col) of each detector.

    Always leaves at least one hidden slot for `_place_legend` to sit in.
    """
    detectors_per_row = max(1, detectors_per_row)
    n_rows = -(-n_detectors // detectors_per_row)  # ceil division
    if n_rows * detectors_per_row == n_detectors:
        n_rows += 1
    return n_rows, [divmod(i, detectors_per_row) for i in range(n_detectors)]


def _single_panel_grid(
    predictions_by_detector: Dict[str, Dict[str, pd.DataFrame]],
    draw: Callable[[plt.Axes, Dict[str, pd.DataFrame]], None],
    handles: list,
    labels: list,
    ncols: int = 4,
    panel_label: Optional[PanelLabel] = TOP_LEFT,
    dedupe_xlabels: bool = True,
    dedupe_ylabels: bool = True,
) -> plt.Figure:
    """One panel per detector, `ncols` per row.

    `draw(ax, predictions)` renders one detector's panel. `panel_label`
    places the detector name inside the panel; None uses a title above
    it instead.
    """
    detectors = list(predictions_by_detector)
    n_rows, positions = _detector_grid_shape(len(detectors), ncols)
    fig, axes = plt.subplots(
        n_rows, ncols,
        figsize=(ncols * 3, n_rows * 3),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for (row, col), detector in zip(positions, detectors):
        ax = axes[row, col]
        draw(ax, predictions_by_detector[detector])
        if panel_label is None:
            ax.set_title(
                _display_name(detector), color=_label_color(detector),
                fontsize=14,
            )
        else:
            _label_panel(ax, detector, panel_label)
    for j in range(len(detectors), n_rows * ncols):
        axes[divmod(j, ncols)].set_visible(False)
    _finalize_grid(
        fig, axes, handles, labels,
        dedupe_xlabels=dedupe_xlabels, dedupe_ylabels=dedupe_ylabels,
    )
    return fig


def _paired_panel_grid(
    predictions_by_detector: Dict[str, Dict[str, pd.DataFrame]],
    draw: Callable[[plt.Axes, plt.Axes, Dict[str, pd.DataFrame]], None],
    handles: list,
    labels: list,
    panel_labels: Tuple[PanelLabel, PanelLabel],
    ncols: int = 4,
    dedupe_xlabels: bool = True,
) -> plt.Figure:
    """Two side-by-side panels per detector, `ncols // 2` detectors per row.

    `draw(left, right, predictions)` renders one detector's pair. Pass
    `dedupe_xlabels=False` where each detector's x *range* differs, so
    every panel shows the range it is actually drawn on.
    """
    detectors = list(predictions_by_detector)
    per_row = max(1, ncols // 2)
    n_rows, positions = _detector_grid_shape(len(detectors), per_row)
    total_cols = per_row * 2
    fig, axes = plt.subplots(
        n_rows, total_cols,
        figsize=(total_cols * 3, n_rows * 3),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for (block_row, block_col), detector in zip(positions, detectors):
        col0 = block_col * 2
        pair = (axes[block_row, col0], axes[block_row, col0 + 1])
        draw(pair[0], pair[1], predictions_by_detector[detector])
        for ax, pos in zip(pair, panel_labels):
            _label_panel(ax, detector, pos)
    for j in range(len(detectors), n_rows * per_row):
        block_row, block_col = divmod(j, per_row)
        for c in (block_col * 2, block_col * 2 + 1):
            axes[block_row, c].set_visible(False)
    _finalize_grid(
        fig, axes, handles, labels,
        dedupe_xlabels=dedupe_xlabels, dedupe_ylabels=False,
    )
    return fig


# ----------------------------------------------------------------------
# Public API: loading, then one builder per paper figure
#
# Each builder supplies a `draw` callback and its legend entries to
# one of the two grid helpers above. `make_multi_panel_figure` is the
# feature-keyed dispatch the CLI uses.
# ----------------------------------------------------------------------


def load_multi_panel_predictions(
    data_root: str,
    detectors: List[str],
    feature: str,
    models: List[str],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Every detector's `{model: DataFrame}` predictions for `feature`."""
    return {
        detector: load_feature_predictions(
            data_root, detector, feature, models
        )
        for detector in detectors
    }


def plot_energy_multi_detector_figure(
    predictions_by_detector: Dict[str, Dict[str, pd.DataFrame]],
    truth_col: str,
    pred_col: str,
    is_track_col: str = "is_track",
    ncols: int = 4,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 60,
) -> plt.Figure:
    """The paper's multi-detector CC/NC energy calibration grid.

    Each detector gets the 2x2 block `plot_energy_calibration_by_topology
    _figure` builds (marginal histogram over calibration, one column per
    CC/NC), `ncols // 2` detectors per row. Unlike the other grids this
    needs two *rows* per detector, so it builds its own gridspec rather
    than going through `_paired_panel_grid`. Bins default to per-detector
    ranges; pass `bins` for identical bins everywhere.
    """
    detectors = list(predictions_by_detector)
    per_row = max(1, ncols // 2)
    n_rows, positions = _detector_grid_shape(len(detectors), per_row)
    total_cols = per_row * 2
    fig, axes = plt.subplots(
        n_rows * 2,
        total_cols,
        figsize=(total_cols * 3, n_rows * (3 + 3 * 0.2)),
        gridspec_kw={"height_ratios": [0.2, 1] * n_rows},
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for (block_row, block_col), detector in zip(positions, detectors):
        row0, col0 = block_row * 2, block_col * 2
        plot_energy_calibration_by_topology_figure(
            predictions_by_detector[detector],
            truth_col,
            pred_col,
            is_track_col,
            bins=bins,
            n_bins=n_bins,
            axes=axes[row0:row0 + 2, col0:col0 + 2],
        )
        for c in (col0, col0 + 1):
            _label_panel(
                axes[row0 + 1, c], detector, (0.05, 0.98, "top", "left")
            )
    for j in range(len(detectors), n_rows * per_row):
        block_row, block_col = divmod(j, per_row)
        row0, col0 = block_row * 2, block_col * 2
        for r in range(row0, row0 + 2):
            for c in (col0, col0 + 1):
                axes[r, c].set_visible(False)
    handles, labels = _legend_handles(
        _models_of(predictions_by_detector),
        [("ideal", {"color": "black", "linestyle": "--", "linewidth": 0.5})],
        per_model_linestyle=True,
    )
    _finalize_grid(fig, axes, handles, labels, dedupe_ylabels=False)
    return fig


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
    distribution_bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    distribution_n_bins: int = 120,
    ncols: int = 4,
) -> plt.Figure:
    """The paper's multi-detector direction resolution/distribution grid.

    Each detector gets the 1x2 block `plot_direction_figure` builds. The
    muon "Kinematic Angle" reference line is drawn only when both muon
    columns are given.
    """
    def draw(
        ax_resolution: plt.Axes,
        ax_distribution: plt.Axes,
        predictions: Dict[str, pd.DataFrame],
    ) -> None:
        plot_direction_figure(
            predictions,
            truth_zenith_col,
            truth_azimuth_col,
            pred_x_col,
            pred_y_col,
            pred_z_col,
            energy_col,
            is_track_col,
            muon_zenith_col=muon_zenith_col,
            muon_azimuth_col=muon_azimuth_col,
            distribution_bins=distribution_bins,
            distribution_n_bins=distribution_n_bins,
            ax_resolution=ax_resolution,
            ax_distribution=ax_distribution,
        )

    extras = [(CC_LABEL, {"linestyle": "-"}), (NC_LABEL, {"linestyle": "--"})]
    if muon_zenith_col is not None and muon_azimuth_col is not None:
        extras.append(
            ("Kinematic Angle", {"color": "red", "linestyle": "-"})
        )
    handles, labels = _legend_handles(
        _models_of(predictions_by_detector), extras
    )
    return _paired_panel_grid(
        predictions_by_detector, draw, handles, labels,
        panel_labels=(TOP_LEFT, TOP_LEFT), ncols=ncols,
        dedupe_xlabels=False,
    )


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
    """The paper's multi-detector inelasticity distribution/resolution grid.

    Each detector gets the 1x2 block `plot_inelasticity_figure` builds.
    """
    def draw(
        ax_distribution: plt.Axes,
        ax_resolution: plt.Axes,
        predictions: Dict[str, pd.DataFrame],
    ) -> None:
        plot_inelasticity_figure(
            predictions,
            truth_col,
            pred_col,
            energy_col,
            energy_threshold=energy_threshold,
            distribution_bins=distribution_bins,
            distribution_n_bins=distribution_n_bins,
            ax_distribution=ax_distribution,
            ax_resolution=ax_resolution,
        )
        # The paper's open-histogram style: no box, no y label, no y
        # ticks - a percentage shown as a relative shape, not an axis.
        ax_distribution.spines[["right", "top", "left"]].set_visible(False)
        ax_distribution.set_yticks([])
        ax_distribution.set_ylabel("")

    handles, labels = _legend_handles(
        _models_of(predictions_by_detector),
        [
            ("Truth", {"color": "black", "linestyle": "-"}),
            (f"E > {energy_threshold:g} GeV", {"linestyle": "-"}),
            (f"E <= {energy_threshold:g} GeV", {"linestyle": "--"}),
        ],
    )
    return _paired_panel_grid(
        predictions_by_detector, draw, handles, labels,
        panel_labels=(TOP_RIGHT, TOP_LEFT), ncols=ncols,
        dedupe_xlabels=False,
    )


def plot_track_score_multi_detector_figure(
    predictions_by_detector: Dict[str, Dict[str, pd.DataFrame]],
    score_col: str,
    is_track_col: str,
    bins: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_bins: int = 50,
    ncols: int = 4,
) -> plt.Figure:
    """The paper's multi-detector track-score log-count grid."""
    def draw(ax: plt.Axes, predictions: Dict[str, pd.DataFrame]) -> None:
        plot_track_score_distribution_by_topology_comparison(
            predictions, score_col=score_col, is_track_col=is_track_col,
            bins=bins, n_bins=n_bins, ax=ax,
        )

    handles, labels = _legend_handles(
        _models_of(predictions_by_detector),
        # Dotted/solid here, not solid/dashed - see
        # `plot_track_score_distribution_by_topology`.
        [(CC_LABEL, {"linestyle": ":"}), (NC_LABEL, {"linestyle": "-"})],
    )
    return _single_panel_grid(
        predictions_by_detector, draw, handles, labels, ncols=ncols,
    )


def plot_roc_multi_detector_figure(
    predictions_by_detector: Dict[str, Dict[str, pd.DataFrame]],
    truth_col: str,
    score_col: str,
    energy_col: str,
    low_threshold: float = 100.0,
    high_threshold: float = 1000.0,
    ncols: int = 4,
) -> plt.Figure:
    """The paper's multi-detector, energy-regime-split ROC grid.

    The detector label sits bottom-right rather than top-left: a good
    ROC curve hugs the top-left corner.
    """
    def draw(ax: plt.Axes, predictions: Dict[str, pd.DataFrame]) -> None:
        plot_roc_curve_by_energy_regime_comparison(
            predictions, truth_col=truth_col, score_col=score_col,
            energy_col=energy_col, low_threshold=low_threshold,
            high_threshold=high_threshold, ax=ax,
        )

    handles, labels = _legend_handles(
        _models_of(predictions_by_detector),
        [
            (f"E <= {low_threshold:g} GeV", {"linestyle": "-"}),
            (
                f"{low_threshold:g} < E <= {high_threshold:g} GeV",
                {"linestyle": "--"},
            ),
            (
                f"E > {high_threshold:g} GeV",
                {"linestyle": HIGH_ENERGY_LINESTYLE},
            ),
        ],
    )
    return _single_panel_grid(
        predictions_by_detector, draw, handles, labels, ncols=ncols,
        panel_label=BOTTOM_RIGHT,
    )


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
    """The paper's multi-detector vertex-resolution grid.

    Y labels and ticks are kept on every panel, because each
    detector's Euclidean-distance range genuinely differs.
    """
    def draw(ax: plt.Axes, predictions: Dict[str, pd.DataFrame]) -> None:
        plot_vertex_resolution_by_topology_comparison(
            predictions,
            truth_x_col=truth_x_col, truth_y_col=truth_y_col,
            truth_z_col=truth_z_col, pred_x_col=pred_x_col,
            pred_y_col=pred_y_col, pred_z_col=pred_z_col,
            energy_col=energy_col, is_track_col=is_track_col,
            bins=bins, n_bins=n_bins, ax=ax,
        )

    handles, labels = _legend_handles(
        _models_of(predictions_by_detector),
        [(CC_LABEL, {"linestyle": "-"}), (NC_LABEL, {"linestyle": "--"})],
    )
    return _single_panel_grid(
        predictions_by_detector, draw, handles, labels, ncols=ncols,
        dedupe_xlabels=False, dedupe_ylabels=False,
    )


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
    """The paper's multi-detector vertex-contour grid.

    Each panel shows every model's 68%-containment depth/radial contour,
    cropped per detector by `vertex_contour_axis_ranges` rather than the
    outlier-stretched default, with a linear radial axis cut at 0. Keeps
    its title above the panel, and both axes' labels and ticks on every
    panel: the crop differs per detector, so a shared axis would be read
    off a neighbour that does not share it.
    """
    def draw(ax: plt.Axes, predictions: Dict[str, pd.DataFrame]) -> None:
        depth_edges, radial_edges = vertex_contour_axis_ranges(
            predictions, truth_x_col, truth_y_col, truth_z_col,
            pred_x_col, pred_y_col, pred_z_col, is_track_col, sigma=sigma,
        )
        plot_vertex_contour_by_topology_comparison(
            predictions,
            truth_x_col=truth_x_col, truth_y_col=truth_y_col,
            truth_z_col=truth_z_col, pred_x_col=pred_x_col,
            pred_y_col=pred_y_col, pred_z_col=pred_z_col,
            is_track_col=is_track_col, bins=[depth_edges, radial_edges],
            sigma=sigma, ax=ax,
        )
        ax.set_xlim(depth_edges[0], depth_edges[-1])
        ax.set_ylim(0.0, radial_edges[-1])

    # Linestyle and marker combined in one handle per topology.
    handles, labels = _legend_handles(
        _models_of(predictions_by_detector),
        [
            (CC_LABEL, {"linestyle": "-", "marker": "*", "markersize": 10}),
            (NC_LABEL, {"linestyle": "--", "marker": ".", "markersize": 10}),
        ],
    )
    return _single_panel_grid(
        predictions_by_detector, draw, handles, labels, ncols=ncols,
        panel_label=None, dedupe_xlabels=False, dedupe_ylabels=False,
    )


# Which feature's data each figure needs. Mostly one figure per
# feature, but vertex and classification each have a second paper
# figure drawn from the same columns, which is why figure and feature
# are not the same thing.
FIGURE_FEATURE: Dict[str, str] = {
    "energy": "energy",
    "direction": "direction",
    "inelasticity": "inelasticity",
    "vertex": "vertex",
    "vertex-contour": "vertex",
    "classification": "classification",
    "track-score": "classification",
}


def make_multi_panel_figure(
    feature: str,
    predictions_by_dataset: Dict[str, Dict[str, pd.DataFrame]],
    ncols: int = 4,
    figure: Optional[str] = None,
) -> plt.Figure:
    """The paper's multi-detector grid for one figure.

    `figure` defaults to `feature`, which covers the five figures named
    after their feature. Pass "vertex-contour" or "track-score" for the
    paper's other two, which use the same loaded data as "vertex" and
    "classification" respectively.
    """
    figure = figure or feature
    if figure not in FIGURE_FEATURE:
        raise ValueError(
            f"Unknown figure {figure!r}; must be one of "
            f"{sorted(FIGURE_FEATURE)}"
        )
    columns = DEFAULT_COLUMNS[FIGURE_FEATURE[figure]]
    if figure == "vertex-contour":
        return plot_vertex_contour_multi_detector_figure(
            predictions_by_dataset,
            truth_x_col=columns["truth_x_col"],
            truth_y_col=columns["truth_y_col"],
            truth_z_col=columns["truth_z_col"],
            pred_x_col=columns["pred_x_col"],
            pred_y_col=columns["pred_y_col"],
            pred_z_col=columns["pred_z_col"],
            is_track_col="is_track",
            ncols=ncols,
        )
    if figure == "track-score":
        return plot_track_score_multi_detector_figure(
            predictions_by_dataset,
            score_col=columns["score_col"],
            is_track_col="is_track",
            ncols=ncols,
        )
    if figure == "energy":
        return plot_energy_multi_detector_figure(
            predictions_by_dataset,
            truth_col=columns["truth_col"],
            pred_col=columns["pred_col"],
            ncols=ncols,
        )
    if figure == "direction":
        return plot_direction_multi_detector_figure(
            predictions_by_dataset,
            truth_zenith_col=columns["truth_zenith_col"],
            truth_azimuth_col=columns["truth_azimuth_col"],
            pred_x_col=columns["pred_x_col"],
            pred_y_col=columns["pred_y_col"],
            pred_z_col=columns["pred_z_col"],
            energy_col=columns["energy_col"],
            muon_zenith_col=columns["muon_zenith_col"],
            muon_azimuth_col=columns["muon_azimuth_col"],
            ncols=ncols,
            distribution_bins=np.linspace(0, 5, 120),
        )
    if figure == "vertex":
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
    if figure == "inelasticity":
        return plot_inelasticity_multi_detector_figure(
            predictions_by_dataset,
            truth_col=columns["truth_col"],
            pred_col=columns["pred_col"],
            energy_col=columns["energy_col"],
            ncols=ncols,
        )
    return plot_roc_multi_detector_figure(
        predictions_by_dataset,
        truth_col=columns["truth_col"],
        score_col=columns["score_col"],
        energy_col=columns["energy_col"],
        ncols=ncols,
    )
