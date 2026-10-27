"""Model-evaluation metrics, one module per reconstruction task.

Each function takes DataFrame column names as arguments rather than
assuming a fixed schema, so it works whatever a user named their
prediction/truth columns.
"""

from nubench.evaluation.metrics.classification import (
    roc_curve_data,
    roc_curve_data_by_energy_regime,
    track_score_distribution_by_topology,
)
from nubench.evaluation.metrics.direction import (
    direction_error_distribution,
    direction_error_distribution_by_topology,
    direction_resolution,
    direction_resolution_by_topology,
)
from nubench.evaluation.metrics.energy import energy_calibration
from nubench.evaluation.metrics.inelasticity import (
    inelasticity_distribution_by_energy_regime,
    inelasticity_resolution,
)
from nubench.evaluation.metrics.vertex import (
    vertex_contour_by_topology,
    vertex_resolution,
    vertex_resolution_by_topology,
)

__all__ = [
    "roc_curve_data",
    "roc_curve_data_by_energy_regime",
    "track_score_distribution_by_topology",
    "direction_error_distribution",
    "direction_error_distribution_by_topology",
    "direction_resolution",
    "direction_resolution_by_topology",
    "energy_calibration",
    "inelasticity_distribution_by_energy_regime",
    "inelasticity_resolution",
    "vertex_contour_by_topology",
    "vertex_resolution",
    "vertex_resolution_by_topology",
]
