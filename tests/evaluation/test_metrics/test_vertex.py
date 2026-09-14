"""Unit tests for `nubench.evaluation.metrics.vertex`."""

import numpy as np
import pandas as pd

from nubench.evaluation.metrics import (
    vertex_contour_by_topology,
    vertex_resolution,
    vertex_resolution_by_topology,
)


def test_vertex_resolution_known_distance() -> None:
    """A hand-constructed 3-4-5 offset should show up as a distance of 5."""
    n = 10
    df = pd.DataFrame(
        {
            "true_x": np.zeros(n),
            "true_y": np.zeros(n),
            "true_z": np.zeros(n),
            "pred_x": np.full(n, 3.0),
            "pred_y": np.full(n, 4.0),
            "pred_z": np.zeros(n),
            "energy": np.full(n, 50.0),
        }
    )
    bins = np.array([10.0, 100.0])

    result = vertex_resolution(
        df,
        truth_x_col="true_x",
        truth_y_col="true_y",
        truth_z_col="true_z",
        pred_x_col="pred_x",
        pred_y_col="pred_y",
        pred_z_col="pred_z",
        energy_col="energy",
        bins=bins,
    )

    assert np.isclose(result["p50"].iloc[0], 5.0)


def test_vertex_resolution_is_column_name_agnostic() -> None:
    """The function must not assume any particular column names."""
    n = 6
    df = pd.DataFrame(
        {
            "tx": np.zeros(n),
            "ty": np.zeros(n),
            "tz": np.zeros(n),
            "px": np.full(n, 1.0),
            "py": np.zeros(n),
            "pz": np.zeros(n),
            "e": np.linspace(10, 100, n),
        }
    )
    bins = np.array([1.0, 50.0, 200.0])

    result = vertex_resolution(
        df,
        truth_x_col="tx",
        truth_y_col="ty",
        truth_z_col="tz",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        energy_col="e",
        bins=bins,
    )

    assert list(result.columns) == ["bin_center", "p16", "p50", "p84"]
    assert len(result) == 2


def test_vertex_resolution_binned_by_energy_not_by_position() -> None:
    """The binning variable must be `energy_col`, not the position columns.

    Constructs data where position values and energy values are unrelated,
    with a perfect predictor (pred position == truth position, so distance
    is ~zero everywhere) - a bug that accidentally binned by position
    instead of energy would produce `bin_center` values within the
    position range instead of the (very different) energy range.
    """
    rng = np.random.default_rng(seed=6)
    n = 2000
    x = rng.uniform(-500, 500, size=n)
    y = rng.uniform(-500, 500, size=n)
    z = rng.uniform(-500, 500, size=n)
    energy = rng.uniform(10, 10_000, size=n)
    df = pd.DataFrame(
        {
            "tx": x,
            "ty": y,
            "tz": z,
            "px": x,
            "py": y,
            "pz": z,
            "energy": energy,
        }
    )

    result = vertex_resolution(
        df,
        truth_x_col="tx",
        truth_y_col="ty",
        truth_z_col="tz",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        energy_col="energy",
        n_bins=10,
    )

    # bin_center must track the energy range (10-10000), not the position
    # range (-500 to 500).
    assert result["bin_center"].min() >= 10.0
    assert result["bin_center"].max() <= 10_000.0
    # Perfect predictor -> ~zero distance in every populated bin.
    assert result["p50"].notna().any()
    assert np.allclose(result["p50"].dropna(), 0.0, atol=1e-6)


def test_vertex_resolution_by_topology_known_distances() -> None:
    """Track/cascade should each reflect their own known offset."""
    n = 20
    is_track = np.array([True] * 10 + [False] * 10)
    df = pd.DataFrame(
        {
            "true_x": np.zeros(n),
            "true_y": np.zeros(n),
            "true_z": np.zeros(n),
            # Track predictions: 3-4-5 offset. Cascade: 6-8-10 offset.
            "pred_x": np.where(is_track, 3.0, 6.0),
            "pred_y": np.where(is_track, 4.0, 8.0),
            "pred_z": np.zeros(n),
            "energy": np.full(n, 50.0),
            "is_track": is_track,
        }
    )
    bins = np.array([10.0, 100.0])

    result = vertex_resolution_by_topology(
        df,
        truth_x_col="true_x",
        truth_y_col="true_y",
        truth_z_col="true_z",
        pred_x_col="pred_x",
        pred_y_col="pred_y",
        pred_z_col="pred_z",
        energy_col="energy",
        is_track_col="is_track",
        bins=bins,
    )

    assert set(result.keys()) == {"track", "cascade"}
    assert np.isclose(result["track"]["p50"].iloc[0], 5.0)
    assert np.isclose(result["cascade"]["p50"].iloc[0], 10.0)


def test_vertex_resolution_by_topology_does_not_mutate_input() -> None:
    """The caller's DataFrame shouldn't gain new columns as a side effect."""
    n = 20
    is_track = np.array([True] * 10 + [False] * 10)
    df = pd.DataFrame(
        {
            "true_x": np.zeros(n),
            "true_y": np.zeros(n),
            "true_z": np.zeros(n),
            "pred_x": np.full(n, 3.0),
            "pred_y": np.full(n, 4.0),
            "pred_z": np.zeros(n),
            "energy": np.full(n, 50.0),
            "is_track": is_track,
        }
    )
    original_columns = set(df.columns)

    vertex_resolution_by_topology(
        df,
        truth_x_col="true_x",
        truth_y_col="true_y",
        truth_z_col="true_z",
        pred_x_col="pred_x",
        pred_y_col="pred_y",
        pred_z_col="pred_z",
        energy_col="energy",
        is_track_col="is_track",
    )

    assert set(df.columns) == original_columns


def test_vertex_resolution_by_topology_shares_bins() -> None:
    """Track and cascade must be binned with the same shared edges.

    Same disjoint-ranges trick used throughout the project: with
    correctly shared bins, each topology must show NaNs where only the
    *other* topology has data.
    """
    n_low, n_high = 100, 100
    energy = np.concatenate(
        [np.linspace(10, 100, n_low), np.linspace(500, 1000, n_high)]
    )
    is_track = np.array([True] * n_low + [False] * n_high)
    n = n_low + n_high
    df = pd.DataFrame(
        {
            "true_x": np.zeros(n),
            "true_y": np.zeros(n),
            "true_z": np.zeros(n),
            "pred_x": np.full(n, 3.0),
            "pred_y": np.full(n, 4.0),
            "pred_z": np.zeros(n),
            "energy": energy,
            "is_track": is_track,
        }
    )

    result = vertex_resolution_by_topology(
        df,
        truth_x_col="true_x",
        truth_y_col="true_y",
        truth_z_col="true_z",
        pred_x_col="pred_x",
        pred_y_col="pred_y",
        pred_z_col="pred_z",
        energy_col="energy",
        is_track_col="is_track",
        n_bins=10,
    )

    assert result["track"]["p50"].isna().any()
    assert result["cascade"]["p50"].isna().any()


def test_vertex_contour_by_topology_known_medians() -> None:
    """Track/cascade should each reflect their own known median offset."""
    n = 20
    is_track = np.array([True] * 10 + [False] * 10)
    df = pd.DataFrame(
        {
            "true_x": np.zeros(n),
            "true_y": np.zeros(n),
            "true_z": np.zeros(n),
            # Track: dx=3, dy=4 -> radial=5; dz=2 -> depth=2.
            # Cascade: dx=6, dy=8 -> radial=10; dz=10 -> depth=10.
            "pred_x": np.where(is_track, 3.0, 6.0),
            "pred_y": np.where(is_track, 4.0, 8.0),
            "pred_z": np.where(is_track, -2.0, -10.0),
            "is_track": is_track,
        }
    )

    result = vertex_contour_by_topology(
        df,
        truth_x_col="true_x",
        truth_y_col="true_y",
        truth_z_col="true_z",
        pred_x_col="pred_x",
        pred_y_col="pred_y",
        pred_z_col="pred_z",
        is_track_col="is_track",
        bins=5,
        sigma=0.0,
    )

    assert set(result.keys()) == {"track", "cascade"}
    _, _, _, _, track_depth, track_radial = result["track"]
    _, _, _, _, cascade_depth, cascade_radial = result["cascade"]
    assert np.isclose(track_depth, 2.0)
    assert np.isclose(track_radial, 5.0)
    assert np.isclose(cascade_depth, 10.0)
    assert np.isclose(cascade_radial, 10.0)


def test_vertex_contour_by_topology_does_not_mutate_input() -> None:
    """The caller's DataFrame shouldn't gain new columns as a side effect."""
    n = 20
    is_track = np.array([True] * 10 + [False] * 10)
    df = pd.DataFrame(
        {
            "true_x": np.zeros(n),
            "true_y": np.zeros(n),
            "true_z": np.zeros(n),
            "pred_x": np.full(n, 3.0),
            "pred_y": np.full(n, 4.0),
            "pred_z": np.full(n, -2.0),
            "is_track": is_track,
        }
    )
    original_columns = set(df.columns)

    vertex_contour_by_topology(
        df,
        truth_x_col="true_x",
        truth_y_col="true_y",
        truth_z_col="true_z",
        pred_x_col="pred_x",
        pred_y_col="pred_y",
        pred_z_col="pred_z",
        is_track_col="is_track",
        bins=5,
    )

    assert set(df.columns) == original_columns


def test_vertex_contour_by_topology_shapes() -> None:
    """X/Y/H should match the requested bin count for both topologies."""
    rng = np.random.default_rng(seed=0)
    n = 400
    is_track = np.array([True] * 200 + [False] * 200)
    df = pd.DataFrame(
        {
            "true_x": np.zeros(n),
            "true_y": np.zeros(n),
            "true_z": np.zeros(n),
            "pred_x": rng.normal(size=n),
            "pred_y": rng.normal(size=n),
            "pred_z": rng.normal(size=n),
            "is_track": is_track,
        }
    )

    result = vertex_contour_by_topology(
        df,
        truth_x_col="true_x",
        truth_y_col="true_y",
        truth_z_col="true_z",
        pred_x_col="pred_x",
        pred_y_col="pred_y",
        pred_z_col="pred_z",
        is_track_col="is_track",
        bins=12,
    )

    for topology in ("track", "cascade"):
        X, Y, H, level, depth, radial = result[topology]
        assert X.shape == (12, 12)
        assert Y.shape == (12, 12)
        assert H.shape == (12, 12)
