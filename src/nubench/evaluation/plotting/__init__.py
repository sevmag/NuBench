"""Plots of model-evaluation metrics, one module per reconstruction task.

These take the DataFrames produced by `nubench.evaluation.metrics`, not
raw prediction DataFrames.
"""

from nubench.evaluation.plotting.classification import (
    plot_roc_curve,
    plot_roc_curve_by_energy_regime,
    plot_roc_curve_by_energy_regime_comparison,
    plot_roc_curve_comparison,
    plot_track_score_distribution_by_topology,
    plot_track_score_distribution_by_topology_comparison,
)
from nubench.evaluation.plotting.direction import (
    plot_direction_error_distribution_by_topology,
    plot_direction_error_distribution_by_topology_comparison,
    plot_direction_figure,
    plot_direction_resolution_by_topology,
    plot_direction_resolution_by_topology_comparison,
)
from nubench.evaluation.plotting.energy import (
    plot_energy_calibration,
    plot_energy_calibration_by_topology_figure,
    plot_energy_calibration_comparison,
    plot_energy_calibration_figure,
)
from nubench.evaluation.plotting.inelasticity import (
    plot_inelasticity_distribution_by_energy_regime,
    plot_inelasticity_distribution_by_energy_regime_comparison,
    plot_inelasticity_figure,
    plot_inelasticity_resolution,
    plot_inelasticity_resolution_comparison,
)
from nubench.evaluation.plotting.vertex import (
    plot_vertex_contour_by_topology,
    plot_vertex_contour_by_topology_comparison,
    plot_vertex_resolution_by_topology,
    plot_vertex_resolution_by_topology_comparison,
    vertex_contour_axis_ranges,
)

__all__ = [
    "plot_roc_curve",
    "plot_roc_curve_by_energy_regime",
    "plot_roc_curve_by_energy_regime_comparison",
    "plot_roc_curve_comparison",
    "plot_track_score_distribution_by_topology",
    "plot_track_score_distribution_by_topology_comparison",
    "plot_direction_error_distribution_by_topology",
    "plot_direction_error_distribution_by_topology_comparison",
    "plot_direction_figure",
    "plot_direction_resolution_by_topology",
    "plot_direction_resolution_by_topology_comparison",
    "plot_energy_calibration",
    "plot_energy_calibration_by_topology_figure",
    "plot_energy_calibration_comparison",
    "plot_energy_calibration_figure",
    "plot_inelasticity_distribution_by_energy_regime",
    "plot_inelasticity_distribution_by_energy_regime_comparison",
    "plot_inelasticity_figure",
    "plot_inelasticity_resolution",
    "plot_inelasticity_resolution_comparison",
    "plot_vertex_contour_by_topology",
    "plot_vertex_contour_by_topology_comparison",
    "plot_vertex_resolution_by_topology",
    "plot_vertex_resolution_by_topology_comparison",
    "vertex_contour_axis_ranges",
]
