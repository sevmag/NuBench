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

from nubench.data import DEFAULT_COLUMNS, load_feature_predictions
from nubench.evaluation.plotting import (
    plot_direction_figure,
    plot_energy_calibration_by_topology_figure,
    plot_inelasticity_figure,
    plot_roc_curve_comparison,
    plot_track_score_distribution_by_topology_comparison,
    plot_vertex_contour_by_topology_comparison,
    plot_vertex_resolution_by_topology_comparison,
)
from nubench.multipanel import (
    FIGURE_FEATURE,
    load_multi_panel_predictions,
    make_multi_panel_figure,
)

# Every feature this CLI can plot - the keys of `DEFAULT_COLUMNS`, which
# also defines the real-file column names each one needs.
FEATURES = sorted(DEFAULT_COLUMNS)
# Every figure, including the two drawn from a feature whose name they
# do not share (see `FIGURE_FEATURE`).
FIGURES = sorted(FIGURE_FEATURE)


def make_figure(
    feature: str,
    predictions: Dict[str, pd.DataFrame],
    figure: Optional[str] = None,
) -> plt.Figure:
    """The default figure for one feature.

    One representative plot per task: the combined CC/NC or
    resolution+distribution figure where one exists
    (energy/direction/inelasticity), otherwise the single most broadly
    useful comparison plot (verte x/classification). Column names come
    from `DEFAULT_COLUMNS`; call the plotting functions directly for
    anything more custom.
    """
    figure = figure or feature
    if figure not in FIGURE_FEATURE:
        raise ValueError(
            f"Unknown figure {figure!r}; must be one of {FIGURES}"
        )
    columns = DEFAULT_COLUMNS[FIGURE_FEATURE[figure]]
    if figure == "vertex-contour":
        ax = plot_vertex_contour_by_topology_comparison(
            predictions,
            truth_x_col=columns["truth_x_col"],
            truth_y_col=columns["truth_y_col"],
            truth_z_col=columns["truth_z_col"],
            pred_x_col=columns["pred_x_col"],
            pred_y_col=columns["pred_y_col"],
            pred_z_col=columns["pred_z_col"],
            is_track_col="is_track",
        )
        return ax.figure  # type: ignore[return-value]
    if figure == "track-score":
        ax = plot_track_score_distribution_by_topology_comparison(
            predictions,
            score_col=columns["score_col"],
            is_track_col="is_track",
        )
        return ax.figure  # type: ignore[return-value]
    if figure == "energy":
        fig, _ = plot_energy_calibration_by_topology_figure(
            predictions,
            truth_col=columns["truth_col"],
            pred_col=columns["pred_col"],
            is_track_col="is_track",
        )
        return fig
    if figure == "direction":
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
            # opening-angle range, which is far wider than the ~0-5 deg.
            # window the panel actually displays - without this, the
            # curve gets clipped down to a handful of very coarse bins.
            distribution_bins=np.linspace(0, 5, 120)
        )
        return fig
    if figure == "vertex":
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
        return ax.figure  # type: ignore[return-value]
    if figure == "inelasticity":
        fig, _ = plot_inelasticity_figure(
            predictions,
            truth_col=columns["truth_col"],
            pred_col=columns["pred_col"],
            energy_col=columns["energy_col"],
        )
        return fig
    ax = plot_roc_curve_comparison(
        predictions,
        truth_col=columns["truth_col"],
        score_col=columns["score_col"],
    )
    return ax.figure  # type: ignore[return-value]


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
            "dataset subdirectory's name. Give it once for a "
            "single-detector figure; to give it more than once you must "
            "also pass --multi-panel, which puts each detector in its "
            "own subplot."
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
        "--figure",
        default=None,
        choices=FIGURES,
        help=(
            "Which figure to draw. Defaults to --feature, which names "
            "five of the seven. Use 'vertex-contour' or 'track-score' "
            "for the paper's other two; they reuse the data loaded for "
            "--feature vertex and --feature classification."
        ),
    )
    parser.add_argument(
        "--multi-panel",
        action="store_true",
        help=(
            "Reproduce a multi-detector grid, one subplot per "
            "--detector, instead of a single-detector figure. Required "
            "whenever more than one --detector is given."
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
        help=(
            "Output image path. Defaults to '<feature>.png'. The suffix "
            "picks the format - use '.pdf' for vector output, which is "
            "what a paper wants."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point: parse arguments, build the figure, save it."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    # --figure draws from the data --feature loads, so the two have to
    # agree; otherwise the figure fails later on a missing column.
    if args.figure and FIGURE_FEATURE[args.figure] != args.feature:
        parser.error(
            f"--figure {args.figure} needs --feature "
            f"{FIGURE_FEATURE[args.figure]}, not {args.feature}"
        )
    if args.multi_panel:
        predictions_by_dataset = load_multi_panel_predictions(
            args.data_root, args.detectors, args.feature, args.models
        )
        fig = make_multi_panel_figure(
            args.feature, predictions_by_dataset, ncols=args.ncols,
            figure=args.figure,
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
        fig = make_figure(args.feature, predictions, figure=args.figure)
    output = args.output or f"{args.figure or args.feature}.png"
    fig.savefig(output, dpi=300)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
