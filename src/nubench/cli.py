"""Command-line entry point for regenerating NuBench evaluation plots.

Single-detector mode (--model/--detector/--feature) is the default;
--multi-panel switches to reproducing the paper's literal multi-detector
figures via `nubench.multipanel`.
"""

import argparse
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from nubench.evaluation.plotting import (
    plot_direction_figure,
    plot_energy_calibration_by_topology_figure,
    plot_inelasticity_figure,
    plot_roc_curve_comparison,
    plot_vertex_resolution_by_topology_comparison,
)
from nubench.multipanel import (
    load_multi_panel_predictions,
    make_multi_panel_figure,
)

# Every feature this CLI can plot - the keys of `DEFAULT_COLUMNS`, which
# also defines the real-file column names each one needs.
FEATURES = sorted(DEFAULT_COLUMNS)


def load_feature_predictions(
    data_root: str,
    detector: str,
    feature: str,
    models: List[str],
) -> Dict[str, pd.DataFrame]:
    """Load one DataFrame per model for a given detector/feature selection.

    Args:
        data_root: Directory containing one subdirectory per detector
            (see `nubench.data.find_dataset_dir`).
        detector: Detector key, e.g. "arca".
        feature: One of `FEATURES`.
        models: Model names to load, e.g. `["DynEdge", "ParticleNeT"]`.
            Each becomes one entry (and one legend label) in the
            returned dict.

    Returns:
        Mapping from model name to its loaded prediction DataFrame, with
        the derived `is_track` column already added, or the DataFrame
        already filtered down to track/CC events, where the plot for
        `feature` needs it.
    """
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
    return predictions


def make_figure(
    feature: str, predictions: Dict[str, pd.DataFrame]
) -> plt.Figure:
    """Build the default figure for one feature from its predictions.

    Picks one representative, ready-to-share plot per task - the
    combined CC/NC or resolution+distribution figure where one exists
    (energy/direction/inelasticity), otherwise the single most broadly
    useful comparison plot (vertex/classification). Every column name
    comes from `nubench.data.DEFAULT_COLUMNS`, matching NuBench's own
    real-file schema - use the plotting functions directly (see
    `nubench.evaluation.plotting`) for anything more custom.

    Args:
        feature: One of `FEATURES`.
        predictions: Mapping from model name to its prediction
            DataFrame, as returned by `load_feature_predictions`.

    Returns:
        The Figure ready to save.
    """
    if feature not in DEFAULT_COLUMNS:
        raise ValueError(
            f"Unknown feature {feature!r}; must be one of {FEATURES}"
        )
    columns = DEFAULT_COLUMNS[feature]
    if feature == "energy":
        fig, _ = plot_energy_calibration_by_topology_figure(
            predictions,
            truth_col=columns["truth_col"],
            pred_col=columns["pred_col"],
            is_track_col="is_track",
        )
        return fig
    if feature == "direction":
        fig, _ = plot_direction_figure(
            predictions,
            truth_zenith_col=columns["truth_zenith_col"],
            truth_azimuth_col=columns["truth_azimuth_col"],
            pred_x_col=columns["pred_x_col"],
            pred_y_col=columns["pred_y_col"],
            pred_z_col=columns["pred_z_col"],
            energy_col=columns["energy_col"],
            is_track_col="is_track",
            muon_zenith_col=columns["muon_zenith_col"],
            muon_azimuth_col=columns["muon_azimuth_col"],
            # The right panel's auto-built bins span the *combined*
            # opening-angle range, which is far wider than the ~0-10 deg.
            # window the panel actually displays - without this, the
            # curve gets clipped down to a handful of very coarse bins.
            # This zoomed-in range is the paper-matching default.
            distribution_bins=np.linspace(0, 10, 120),
        )
        return fig
    if feature == "vertex":
        ax = plot_vertex_resolution_by_topology_comparison(
            predictions,
            truth_x_col=columns["truth_x_col"],
            truth_y_col=columns["truth_y_col"],
            truth_z_col=columns["truth_z_col"],
            pred_x_col=columns["pred_x_col"],
            pred_y_col=columns["pred_y_col"],
            pred_z_col=columns["pred_z_col"],
            energy_col=columns["energy_col"],
            is_track_col="is_track",
        )
        # `Axes.figure` is typed `Figure | SubFigure` since it could in
        # principle belong to a subfigure - always a plain `Figure` here,
        # since every Axes this function ever creates comes from
        # `plt.subplots()`, never from a subfigure.
        return ax.figure  # type: ignore[return-value]
    if feature == "inelasticity":
        fig, _ = plot_inelasticity_figure(
            predictions,
            truth_col=columns["truth_col"],
            pred_col=columns["pred_col"],
            energy_col=columns["energy_col"],
        )
        return fig
    if feature == "classification":
        ax = plot_roc_curve_comparison(
            predictions,
            truth_col=columns["truth_col"],
            score_col=columns["score_col"],
        )
        return ax.figure  # type: ignore[return-value]
    raise ValueError(f"Unknown feature {feature!r}; must be one of {FEATURES}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI's argument parser."""
    parser = argparse.ArgumentParser(
        prog="nubench-plot",
        description=(
            "Regenerate a NuBench evaluation plot from local, "
            "already-downloaded prediction files."
        ),
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help=(
            "Directory containing one subdirectory per detector (e.g. "
            "the downloaded Dataset/ folder)."
        ),
    )
    parser.add_argument(
        "--detector",
        dest="detectors",
        action="append",
        required=True,
        help=(
            "Detector key, e.g. 'arca' - matched as a substring of the "
            "dataset subdirectory's name. Repeat --detector (together "
            "with --multi-panel) to reproduce a multi-detector grid, "
            "one subplot per detector."
        ),
    )
    parser.add_argument(
        "--feature",
        required=True,
        choices=FEATURES,
        help="Reconstruction task to plot.",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        required=True,
        help=(
            "Model name, e.g. 'DynEdge'. Repeat --model to overlay "
            "several models on one figure (or, in --multi-panel mode, "
            "within each subplot)."
        ),
    )
    parser.add_argument(
        "--multi-panel",
        action="store_true",
        help=(
            "Reproduce a multi-detector grid (one subplot per "
            "--detector) instead of a single-detector figure."
        ),
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=4,
        help="Grid columns in --multi-panel mode (default: 4).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to '<feature>.png'.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point: parse arguments, build the figure, save it."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.multi_panel:
        predictions_by_dataset = load_multi_panel_predictions(
            args.data_root, args.detectors, args.feature, args.models
        )
        fig = make_multi_panel_figure(
            args.feature, predictions_by_dataset, ncols=args.ncols
        )
    else:
        if len(args.detectors) != 1:
            parser.error(
                "--detector may only be given once unless --multi-panel "
                "is set"
            )
        predictions = load_feature_predictions(
            args.data_root, args.detectors[0], args.feature, args.models
        )
        fig = make_figure(args.feature, predictions)
    output = args.output or f"{args.feature}.png"
    fig.savefig(output, dpi=150)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
