"""Task-specific model-evaluation metric functions.

Each function here knows about one particular reconstruction task (energy,
direction, vertex, ...) and which residual definition applies to it, but
still takes DataFrame column names as arguments rather than assuming a
fixed schema - so the same function works regardless of what a user named
their prediction/truth columns.

Split one file per task (`energy.py`, `direction.py`, `vertex.py`,
`inelasticity.py`, `classification.py`) for readability; everything is
re-exported here so existing call sites
(`from nubench.evaluation.metrics import energy_calibration`, etc.) keep
working unchanged.
"""

from nubench.evaluation.metrics.classification import (
    auc_score,
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
from nubench.evaluation.metrics.energy import (
    energy_calibration,
    energy_calibration_by_topology,
)
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
    "auc_score",
    "roc_curve_data",
    "roc_curve_data_by_energy_regime",
    "track_score_distribution_by_topology",
    "direction_error_distribution",
    "direction_error_distribution_by_topology",
    "direction_resolution",
    "direction_resolution_by_topology",
    "energy_calibration",
    "energy_calibration_by_topology",
    "inelasticity_distribution_by_energy_regime",
    "inelasticity_resolution",
    "vertex_contour_by_topology",
    "vertex_resolution",
    "vertex_resolution_by_topology",
]
