"""Unit tests for `nubench.multipanel`."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.lines import Line2D

matplotlib.use("Agg")  # No display available/needed for tests.

import matplotlib.pyplot as plt  # noqa: E402

from nubench.multipanel import (  # noqa: E402
    _detector_grid_shape,
    _hide_unused_axes,
    _place_legend,
    load_multi_panel_predictions,
    make_multi_panel_figure,
    plot_roc_multi_detector_figure,
    plot_track_score_multi_detector_figure,
    plot_vertex_contour_multi_detector_figure,
    plot_vertex_resolution_multi_detector_figure,
)


def _make_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A synthetic DataFrame with every column every feature needs.

    Same shape as `tests/test_cli.py`'s own `_make_df` - kept as a
    separate copy so this test module doesn't depend on that one.
    """
    rng = np.random.default_rng(seed=seed)
    energy = rng.uniform(10, 10_000, size=n)
    zenith = rng.uniform(0, np.pi, size=n)
    azimuth = rng.uniform(0, 2 * np.pi, size=n)
    dir_x = np.sin(zenith) * np.cos(azimuth)
    dir_y = np.sin(zenith) * np.sin(azimuth)
    dir_z = np.cos(zenith)
    noise = rng.normal(scale=0.05, size=n)
    x = rng.uniform(-100, 100, size=n)
    y = rng.uniform(-100, 100, size=n)
    z = rng.uniform(-100, 100, size=n)
    visible_inelasticity = rng.uniform(0, 1, size=n)
    interaction = rng.integers(1, 3, size=n)  # 1 == CC, 2 == NC
    initial_state_type = rng.choice([14, -14], size=n)
    return pd.DataFrame(
        {
            "initial_state_energy": energy,
            "energy_pred": energy * (1 + noise),
            "initial_state_zenith": zenith,
            "initial_state_azimuth": azimuth,
            "dir_x_pred": dir_x + noise,
            "dir_y_pred": dir_y + noise,
            "dir_z_pred": dir_z + noise,
            "muon_zenith": zenith + noise,
            "muon_azimuth": azimuth + noise,
            "initial_state_x": x,
            "initial_state_y": y,
            "initial_state_z": z,
            "position_x_pred": x + noise * 10,
            "position_y_pred": y + noise * 10,
            "position_z_pred": z + noise * 10,
            "visible_inelasticity": visible_inelasticity,
            "inelasticity_pred": np.clip(
                visible_inelasticity + noise, 0.0, 1.0
            ),
            "target_pred": rng.uniform(0, 1, size=n),
            "interaction": interaction,
            "initial_state_type": initial_state_type,
        }
    )


def _make_detector_dir(root: Path, detector: str, seed: int) -> None:
    dataset_dir = root / f"nubench_{detector}_dynedge"
    dataset_dir.mkdir(parents=True)
    df = _make_df(seed=seed)
    for feature, target in [
        ("energy", "initial_state_energy"),
        ("direction", "direction"),
        ("vertex", "position"),
        ("inelasticity", "visible_inelasticity"),
        ("classification", "track"),
    ]:
        df.to_parquet(
            dataset_dir
            / f"training_{detector}_full_{target}_test_results.parquet"
        )


@pytest.fixture
def multi_detector_root(tmp_path: Path) -> Path:
    _make_detector_dir(tmp_path, "arca", seed=1)
    _make_detector_dir(tmp_path, "orca", seed=2)
    return tmp_path


def test_load_multi_panel_predictions_one_entry_per_detector(
    multi_detector_root: Path,
) -> None:
    """Should return one sub-dict per detector, keyed by detector name."""
    predictions_by_dataset = load_multi_panel_predictions(
        str(multi_detector_root), ["arca", "orca"], "energy", ["DynEdge"]
    )

    assert set(predictions_by_dataset.keys()) == {"arca", "orca"}
    assert set(predictions_by_dataset["arca"].keys()) == {"DynEdge"}


def test_load_multi_panel_predictions_adds_is_track_where_needed(
    multi_detector_root: Path,
) -> None:
    """Same `is_track` rule as the single-detector loader should apply."""
    predictions_by_dataset = load_multi_panel_predictions(
        str(multi_detector_root), ["arca"], "energy", ["DynEdge"]
    )

    assert "is_track" in predictions_by_dataset["arca"]["DynEdge"].columns


def test_load_multi_panel_predictions_filters_inelasticity_to_track_only(
    multi_detector_root: Path,
) -> None:
    """Same track-only-filter rule as the single-detector loader should
    apply, matching the original NuBench_Plots notebooks' own
    `plot_inelasticity_vs_energy`/`plot_inelasticity_dist`.
    """
    predictions_by_dataset = load_multi_panel_predictions(
        str(multi_detector_root), ["arca"], "inelasticity", ["DynEdge"]
    )

    df = predictions_by_dataset["arca"]["DynEdge"]
    assert len(df) < 200
    assert (df["interaction"] == 1).all()
    assert (df["initial_state_type"].abs() == 14).all()


@pytest.mark.parametrize("feature", ["vertex", "classification"])
def test_make_multi_panel_figure_one_axes_per_detector(
    multi_detector_root: Path, feature: str
) -> None:
    """vertex/classification (one representative plot per detector)
    should produce one visible subplot per detector, labelled with the
    paper's display name (an in-panel text annotation, not a title)
    rather than the raw key.
    """
    predictions_by_dataset = load_multi_panel_predictions(
        str(multi_detector_root), ["arca", "orca"], feature, ["DynEdge"]
    )

    fig = make_multi_panel_figure(feature, predictions_by_dataset)

    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 2
    all_text = {t.get_text() for ax in visible for t in ax.texts}
    assert all_text == {"Flower L", "Flower S"}


@pytest.mark.parametrize(
    "feature,axes_per_detector",
    [("energy", 4), ("direction", 2), ("inelasticity", 2)],
)
def test_make_multi_panel_figure_paired_panel_axes_per_detector(
    multi_detector_root: Path, feature: str, axes_per_detector: int
) -> None:
    """energy/direction/inelasticity (a two-panel combined figure per
    detector) should produce one whole block of Axes per detector -
    not just one - each block labelled with the paper's display name
    via a text annotation rather than a title.
    """
    predictions_by_dataset = load_multi_panel_predictions(
        str(multi_detector_root), ["arca", "orca"], feature, ["DynEdge"]
    )

    fig = make_multi_panel_figure(feature, predictions_by_dataset)

    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 2 * axes_per_detector
    all_text = " ".join(t.get_text() for ax in visible for t in ax.texts)
    assert "Flower L" in all_text
    assert "Flower S" in all_text


def test_make_multi_panel_figure_paired_panels_wrap_to_second_row(
    tmp_path: Path,
) -> None:
    """A third detector (default ncols=4 -> 2 detectors/row for paired-
    panel features) should wrap to a second row, matching the paper's
    own layout, rather than widening the first row indefinitely.
    """
    for i, detector in enumerate(["arca", "orca", "trident"]):
        _make_detector_dir(tmp_path, detector, seed=i)
    predictions_by_dataset = load_multi_panel_predictions(
        str(tmp_path), ["arca", "orca", "trident"], "direction", ["DynEdge"]
    )

    fig = make_multi_panel_figure("direction", predictions_by_dataset)

    # 2 detectors/row * 2 axes/detector = 4 in a full row; a third
    # detector alone starts a second row (2 more visible), leaving that
    # row's other 2 slots hidden.
    visible = [ax for ax in fig.axes if ax.get_visible()]
    hidden = [ax for ax in fig.axes if not ax.get_visible()]
    assert len(visible) == 6
    assert len(hidden) == 2


def test_make_multi_panel_figure_unknown_feature_raises(
    multi_detector_root: Path,
) -> None:
    """An unrecognized feature should raise, not silently do nothing."""
    predictions_by_dataset = load_multi_panel_predictions(
        str(multi_detector_root), ["arca"], "energy", ["DynEdge"]
    )

    with pytest.raises(ValueError):
        make_multi_panel_figure("not_a_real_feature", predictions_by_dataset)


def test_make_multi_panel_figure_falls_back_to_raw_key_for_unknown_detector(
    multi_detector_root: Path,
) -> None:
    """A detector key with no known display name should keep its own name
    as its text label, rather than raising.
    """
    predictions_by_dataset = load_multi_panel_predictions(
        str(multi_detector_root), ["arca"], "energy", ["DynEdge"]
    )
    predictions_by_dataset["some_future_detector"] = (
        predictions_by_dataset.pop("arca")
    )

    fig = make_multi_panel_figure("energy", predictions_by_dataset)

    all_text = " ".join(
        t.get_text() for ax in fig.axes if ax.get_visible() for t in ax.texts
    )
    assert "some_future_detector" in all_text


def test_make_multi_panel_figure_colors_detector_labels_paired_panel(
    multi_detector_root: Path,
) -> None:
    """Detector-name labels should use the paper's own per-detector color
    (e.g. "arca" -> "Flower L" -> tab:blue), for a paired-panel feature.
    """
    predictions_by_dataset = load_multi_panel_predictions(
        str(multi_detector_root), ["arca"], "energy", ["DynEdge"]
    )

    fig = make_multi_panel_figure("energy", predictions_by_dataset)

    label = next(
        t for ax in fig.axes for t in ax.texts if t.get_text() == "Flower L"
    )
    assert label.get_color() == "tab:blue"


def test_make_multi_panel_figure_colors_detector_labels_single_panel(
    multi_detector_root: Path,
) -> None:
    """vertex/classification subplot in-panel labels should use the same
    colors as the paired-panel features' own text annotations.
    """
    predictions_by_dataset = load_multi_panel_predictions(
        str(multi_detector_root), ["arca"], "vertex", ["DynEdge"]
    )

    fig = make_multi_panel_figure("vertex", predictions_by_dataset)

    label = next(
        t for ax in fig.axes for t in ax.texts if t.get_text() == "Flower L"
    )
    assert label.get_color() == "tab:blue"


def test_make_multi_panel_figure_label_color_falls_back_to_black(
    multi_detector_root: Path,
) -> None:
    """An unrecognized detector should get a black label, not raise."""
    predictions_by_dataset = load_multi_panel_predictions(
        str(multi_detector_root), ["arca"], "energy", ["DynEdge"]
    )
    predictions_by_dataset["some_future_detector"] = (
        predictions_by_dataset.pop("arca")
    )

    fig = make_multi_panel_figure("energy", predictions_by_dataset)

    label = next(
        t for ax in fig.axes for t in ax.texts
        if t.get_text() == "some_future_detector"
    )
    assert label.get_color() == "black"


def _make_combined_file_detector_dir(
    root: Path, detector: str, seed: int
) -> None:
    """A detector directory in the real "combined file per model, one
    parquet holds every task" download layout, using "track_pred" for
    the classifier score - the convention that differs from
    `_make_detector_dir`'s own per-feature-file layout ("target_pred").
    """
    dataset_dir = root / f"nubench_{detector}_dynedge"
    dataset_dir.mkdir(parents=True)
    df = _make_df(seed=seed).rename(columns={"target_pred": "track_pred"})
    df.to_parquet(dataset_dir / f"DynEdge_v1.0_{detector}.parquet")


def test_make_multi_panel_figure_classification_mixed_score_conventions(
    tmp_path: Path,
) -> None:
    """Two detectors from *different* download sources - one using
    "target_pred", one using "track_pred" - should both work together
    in the same multi-panel call. Regression test: normalizing the
    score column once per whole call (from a single representative
    DataFrame) breaks as soon as a detector using the *other*
    convention is mixed in - each DataFrame must be normalized on its
    own, at load time.
    """
    _make_detector_dir(tmp_path, "arca", seed=1)
    _make_combined_file_detector_dir(tmp_path, "trident", seed=2)
    predictions_by_dataset = load_multi_panel_predictions(
        str(tmp_path), ["arca", "trident"], "classification", ["DynEdge"]
    )

    fig = make_multi_panel_figure("classification", predictions_by_dataset)

    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 2


def _make_combined_file_model_dir(
    root: Path,
    detector: str,
    model: str,
    seed: int,
    inelasticity_pred_col: str = "inelasticity_pred",
) -> None:
    """One model's parquet in the real combined-file-per-model layout,
    added to a detector directory that may already hold other models'
    files - lets a single detector mix models using *different* real-
    world column-naming conventions for the same target (e.g. GRIT's
    own "visible_inelasticity_pred" vs. DynEdge's "inelasticity_pred").
    """
    dataset_dir = root / f"nubench_{detector}_dynedge"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    df = _make_df(seed=seed)
    if inelasticity_pred_col != "inelasticity_pred":
        df = df.rename(
            columns={"inelasticity_pred": inelasticity_pred_col}
        )
    df.to_parquet(dataset_dir / f"{model}_v1.0_{detector}.parquet")


def test_make_multi_panel_figure_inelasticity_mixed_pred_conventions(
    tmp_path: Path,
) -> None:
    """Two models from *different* download sources - one using
    "inelasticity_pred" (DynEdge), one using "visible_inelasticity_pred"
    (GRIT) - should both work together in the same multi-panel call.
    Regression test for the same class of bug the score-column mixed-
    convention test above catches: each DataFrame must be normalized on
    its own, at load time, not once per call from a single
    representative DataFrame.
    """
    _make_combined_file_model_dir(tmp_path, "arca", "DynEdge", seed=1)
    _make_combined_file_model_dir(
        tmp_path, "arca", "GRIT", seed=2,
        inelasticity_pred_col="visible_inelasticity_pred",
    )
    predictions_by_dataset = load_multi_panel_predictions(
        str(tmp_path), ["arca"], "inelasticity", ["DynEdge", "GRIT"]
    )

    fig = make_multi_panel_figure("inelasticity", predictions_by_dataset)

    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 2  # one distribution + one resolution panel


def test_plot_track_score_multi_detector_figure_titles_and_legend(
    multi_detector_root: Path,
) -> None:
    """One in-panel-labelled block per detector (a text annotation, not
    a title above the axes - matching the paper), step-style lines, one
    consolidated CC/NC-worded legend rather than one legend per panel.
    """
    predictions_by_dataset = load_multi_panel_predictions(
        str(multi_detector_root), ["arca", "orca"], "classification",
        ["DynEdge"],
    )

    fig = plot_track_score_multi_detector_figure(
        predictions_by_dataset, score_col="target_pred",
        is_track_col="is_track",
    )

    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 2
    all_text = {t.get_text() for ax in visible for t in ax.texts}
    assert all_text == {"Flower L", "Flower S"}
    assert all(ax.get_legend() is None for ax in fig.axes)
    assert len(fig.legends) == 1
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "DynEdge" in labels
    assert any("CC" in label for label in labels)
    assert any("NC" in label for label in labels)


def test_plot_roc_multi_detector_figure_three_energy_regime_lines(
    multi_detector_root: Path,
) -> None:
    """Each panel gets three lines per model (low/mid/high energy), one
    consolidated legend with no AUC and no per-panel duplicates.
    """
    predictions_by_dataset = load_multi_panel_predictions(
        str(multi_detector_root), ["arca", "orca"], "classification",
        ["DynEdge"],
    )

    fig = plot_roc_multi_detector_figure(
        predictions_by_dataset,
        truth_col="is_track",
        score_col="target_pred",
        energy_col="initial_state_energy",
    )

    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 2
    all_text = {t.get_text() for ax in visible for t in ax.texts}
    assert all_text == {"Flower L", "Flower S"}
    # 3 real low/mid/high-energy curves per model, plus 3 zero-length
    # "grey" lines `plot_roc_curve_by_energy_regime_comparison` draws
    # purely to carry its own (since-stripped) per-panel legend entries.
    data_lines = [line for line in visible[0].lines if len(line.get_xdata())]
    assert len(data_lines) == 3
    assert all(ax.get_legend() is None for ax in fig.axes)
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert not any("AUC" in label for label in labels)
    assert any("GeV" in label for label in labels)


def test_plot_vertex_resolution_multi_detector_figure_linear_yscale(
    multi_detector_root: Path,
) -> None:
    """Linear y-axis (each detector's own Euclidean Distance range can
    differ a lot, so log decade ticks aren't wanted here), an in-panel
    detector-name text annotation rather than a title, and one
    consolidated CC/NC-worded legend, rather than one legend per panel.
    """
    predictions_by_dataset = load_multi_panel_predictions(
        str(multi_detector_root), ["arca", "orca"], "vertex", ["DynEdge"]
    )

    fig = plot_vertex_resolution_multi_detector_figure(
        predictions_by_dataset,
        truth_x_col="initial_state_x",
        truth_y_col="initial_state_y",
        truth_z_col="initial_state_z",
        pred_x_col="position_x_pred",
        pred_y_col="position_y_pred",
        pred_z_col="position_z_pred",
        energy_col="initial_state_energy",
        is_track_col="is_track",
    )

    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 2
    assert all(ax.get_yscale() == "linear" for ax in visible)
    all_text = {t.get_text() for ax in visible for t in ax.texts}
    assert all_text == {"Flower L", "Flower S"}
    assert all(ax.get_legend() is None for ax in fig.axes)
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert any("CC" in label for label in labels)
    assert any("NC" in label for label in labels)


def test_plot_vertex_resolution_multi_detector_figure_ylabel_on_every_panel(
    multi_detector_root: Path,
) -> None:
    """Unlike every other single-panel grid, the y-axis label/ticks are
    kept on every panel (not just the first column of each row), since
    each detector's Euclidean Distance range genuinely differs.
    """
    predictions_by_dataset = load_multi_panel_predictions(
        str(multi_detector_root), ["arca", "orca"], "vertex", ["DynEdge"]
    )

    fig = plot_vertex_resolution_multi_detector_figure(
        predictions_by_dataset,
        truth_x_col="initial_state_x",
        truth_y_col="initial_state_y",
        truth_z_col="initial_state_z",
        pred_x_col="position_x_pred",
        pred_y_col="position_y_pred",
        pred_z_col="position_z_pred",
        energy_col="initial_state_energy",
        is_track_col="is_track",
    )

    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert all(ax.get_ylabel() != "" for ax in visible)


def test_plot_vertex_contour_multi_detector_figure_cropped_zero_baseline(
    multi_detector_root: Path,
) -> None:
    """The radial axis is linear, cut at exactly 0 (matching the paper),
    and cropped to each contour's own bounding box rather than the
    outlier-stretched full range - the real symptom being fixed here is
    a contour shrunk to a barely-visible dot inside a huge, mostly-
    empty axes. The y-axis label/ticks stay on every panel too, since
    each detector's own cropped radial range genuinely differs.
    """
    predictions_by_dataset = load_multi_panel_predictions(
        str(multi_detector_root), ["arca", "orca"], "vertex", ["DynEdge"]
    )

    fig = plot_vertex_contour_multi_detector_figure(
        predictions_by_dataset,
        truth_x_col="initial_state_x",
        truth_y_col="initial_state_y",
        truth_z_col="initial_state_z",
        pred_x_col="position_x_pred",
        pred_y_col="position_y_pred",
        pred_z_col="position_z_pred",
        is_track_col="is_track",
    )

    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 2
    assert all(ax.get_yscale() == "linear" for ax in visible)
    for ax in visible:
        ymin, ymax = ax.get_ylim()
        assert ymin == 0.0
        # Cropped, not the full outlier-stretched min/max range.
        assert ymax < 1000
    # This is the one single-panel grid that keeps its title (per
    # explicit request) rather than an in-panel text annotation.
    assert {ax.get_title() for ax in visible} == {"Flower L", "Flower S"}
    assert all(ax.get_ylabel() != "" for ax in visible)
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert any("CC" in label for label in labels)
    assert any("NC" in label for label in labels)


def test_plot_track_score_multi_detector_figure_dedupes_labels(
    tmp_path: Path,
) -> None:
    """x-label only on the bottom-most visible panel of each grid column,
    y-label only on the left-most visible panel of each grid row - a
    third detector wrapping to a new row (ncols=2 -> 2 detectors/row)
    exercises both a column with two visible rows and one with only one.
    """
    for i, detector in enumerate(["arca", "orca", "trident"]):
        _make_detector_dir(tmp_path, detector, seed=i)
    predictions_by_dataset = load_multi_panel_predictions(
        str(tmp_path), ["arca", "orca", "trident"], "classification",
        ["DynEdge"],
    )

    fig = plot_track_score_multi_detector_figure(
        predictions_by_dataset, score_col="target_pred",
        is_track_col="is_track", ncols=2,
    )

    axes = np.asarray(fig.axes).reshape(2, 2)
    # Column 0: arca (row 0) + trident (row 1) - only trident (the
    # bottom-most visible row) keeps the x label.
    assert axes[0, 0].get_xlabel() == ""
    assert axes[1, 0].get_xlabel() != ""
    # Column 1: orca alone (row 0, no row-1 counterpart) keeps its own
    # x label directly, since it's already the bottom of its column.
    assert axes[0, 1].get_xlabel() != ""
    # Row 0: arca (col 0, the left-most) keeps the y label; orca (col 1)
    # doesn't.
    assert axes[0, 0].get_ylabel() != ""
    assert axes[0, 1].get_ylabel() == ""
    # Row 1: trident is alone (col 0), so it keeps its own y label too.
    assert axes[1, 0].get_ylabel() != ""


def test_detector_grid_shape_leaves_spare_row_when_grid_exactly_full() -> (
    None
):
    """When detectors exactly fill every row (no natural remainder), an
    extra spare row should still be added - otherwise `_place_legend`
    has nowhere to put the legend except a fixed corner that can
    overlap a real panel.
    """
    n_rows, positions = _detector_grid_shape(4, 2)

    assert n_rows == 3  # 2 rows of real detectors + 1 spare row
    assert len(positions) == 4
    assert max(row for row, _ in positions) == 1  # detectors fill rows 0-1


def test_detector_grid_shape_no_extra_row_when_already_jagged() -> None:
    """A grid that already has a natural remainder (and therefore
    already has empty space for the legend) shouldn't get an extra row
    piled on top of that.
    """
    n_rows, positions = _detector_grid_shape(3, 2)

    assert n_rows == 2
    assert len(positions) == 3


def test_plot_track_score_multi_detector_figure_reserves_legend_row(
    tmp_path: Path,
) -> None:
    """4 detectors at ncols=4 exactly fills one row - previously this
    left no empty grid space for the legend, which fell back to a fixed
    corner that could overlap the last panel. There should now be a
    whole spare (hidden) row reserved for it instead.
    """
    for i, detector in enumerate(["arca", "orca", "trident", "gvd"]):
        _make_detector_dir(tmp_path, detector, seed=i)
    predictions_by_dataset = load_multi_panel_predictions(
        str(tmp_path), ["arca", "orca", "trident", "gvd"],
        "classification", ["DynEdge"],
    )

    fig = plot_track_score_multi_detector_figure(
        predictions_by_dataset, score_col="target_pred",
        is_track_col="is_track", ncols=4,
    )

    visible = [ax for ax in fig.axes if ax.get_visible()]
    hidden = [ax for ax in fig.axes if not ax.get_visible()]
    assert len(visible) == 4
    assert len(hidden) == 4


def test_place_legend_avoids_overlapping_real_panel_when_legend_tall() -> (
    None
):
    """A legend with more entries than fit in a single spare grid row's
    height (e.g. once a detector with all four paper models is
    involved) must not overlap the real panel above it.

    `ax.get_position()` alone isn't enough to catch this: it's just the
    bare plot rectangle, not the tick-label text rendered just outside
    it - a legend that "fits" by that measure can still visually
    overlap a real panel's own axis numbers. Regression test for
    exactly that: a 7-entry legend (4 models + 2 topology + 1 physical
    reference line, matching `_direction_legend_handles`'s own count
    once a detector has DeepIce alongside the other three) in a grid
    with only one spare row.
    """
    n_rows = 4
    fig, axes = plt.subplots(
        n_rows, 4, figsize=(12, n_rows * 3), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    for row in range(n_rows):
        for col in range(4):
            ax = axes[row, col]
            ax.plot([1, 100], [1, 100])
            ax.set_xscale("log")
            ax.set_xlabel("True Energy [GeV]")
    # Hide the last row entirely, as an always-spare-row grid would.
    for col in range(4):
        axes[n_rows - 1, col].set_visible(False)
    _hide_unused_axes(axes)

    handles = [Line2D([], [], color="black") for _ in range(7)]
    labels = [
        "DynEdge", "ParticleNeT", "GRIT", "DeepIce",
        r"$\nu_\mu^{CC}$", r"$\nu_\mu^{NC}$", "Kinematic Angle",
    ]
    _place_legend(fig, axes, handles, labels)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend = fig.legends[0]
    legend_bbox = legend.get_window_extent(renderer)
    visible = [ax for ax in axes.flat if ax.get_visible()]
    assert not any(
        legend_bbox.overlaps(ax.get_tightbbox(renderer)) for ax in visible
    )
