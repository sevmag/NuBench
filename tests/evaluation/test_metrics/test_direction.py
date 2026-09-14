"""Unit tests for `nubench.evaluation.metrics.direction`."""

import numpy as np
import pandas as pd

from nubench.evaluation.metrics import (
    direction_error_distribution,
    direction_error_distribution_by_topology,
    direction_resolution,
    direction_resolution_by_topology,
)


def test_direction_resolution_known_ninety_degree_offset() -> None:
    """A hand-constructed 90-degree offset should show up as such."""
    n = 10
    df = pd.DataFrame(
        {
            # Truth points along +x (zenith=pi/2, azimuth=0).
            "true_zenith": np.full(n, np.pi / 2),
            "true_azimuth": np.zeros(n),
            # Prediction points along +y - a known 90-degree offset.
            "pred_x": np.zeros(n),
            "pred_y": np.ones(n),
            "pred_z": np.zeros(n),
            "energy": np.full(n, 50.0),
        }
    )
    bins = np.array([10.0, 100.0])

    result = direction_resolution(
        df,
        truth_zenith_col="true_zenith",
        truth_azimuth_col="true_azimuth",
        pred_x_col="pred_x",
        pred_y_col="pred_y",
        pred_z_col="pred_z",
        energy_col="energy",
        bins=bins,
    )

    assert np.isclose(result["p50"].iloc[0], 90.0)


def test_direction_resolution_degrees_flag() -> None:
    """`degrees=False` should report the same offset in radians."""
    n = 10
    df = pd.DataFrame(
        {
            "true_zenith": np.full(n, np.pi / 2),
            "true_azimuth": np.zeros(n),
            "pred_x": np.zeros(n),
            "pred_y": np.ones(n),
            "pred_z": np.zeros(n),
            "energy": np.full(n, 50.0),
        }
    )
    bins = np.array([10.0, 100.0])

    result = direction_resolution(
        df,
        truth_zenith_col="true_zenith",
        truth_azimuth_col="true_azimuth",
        pred_x_col="pred_x",
        pred_y_col="pred_y",
        pred_z_col="pred_z",
        energy_col="energy",
        bins=bins,
        degrees=False,
    )

    assert np.isclose(result["p50"].iloc[0], np.pi / 2)


def test_direction_resolution_is_column_name_agnostic() -> None:
    """The function must not assume any particular column names."""
    n = 6
    df = pd.DataFrame(
        {
            "z": np.full(n, np.pi / 2),
            "a": np.zeros(n),
            "px": np.ones(n),
            "py": np.zeros(n),
            "pz": np.zeros(n),
            "e": np.linspace(10, 100, n),
        }
    )
    bins = np.array([1.0, 50.0, 200.0])

    result = direction_resolution(
        df,
        truth_zenith_col="z",
        truth_azimuth_col="a",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        energy_col="e",
        bins=bins,
    )

    assert list(result.columns) == ["bin_center", "p16", "p50", "p84"]
    assert len(result) == 2


def test_direction_resolution_binned_by_energy_not_by_angle() -> None:
    """The binning variable must be `energy_col`, not the angle columns.

    Constructs data where angle values and energy values are unrelated, so
    a bug that accidentally binned by (say) zenith instead of energy would
    produce a visibly different bin structure/values than binning by
    energy correctly does.
    """
    rng = np.random.default_rng(seed=5)
    n = 2000
    zenith = rng.uniform(0, np.pi, size=n)
    azimuth = rng.uniform(0, 2 * np.pi, size=n)
    x = np.sin(zenith) * np.cos(azimuth)
    y = np.sin(zenith) * np.sin(azimuth)
    z = np.cos(zenith)
    energy = rng.uniform(10, 10_000, size=n)
    df = pd.DataFrame(
        {
            "zenith": zenith,
            "azimuth": azimuth,
            "px": x,
            "py": y,
            "pz": z,
            "energy": energy,
        }
    )

    result = direction_resolution(
        df,
        truth_zenith_col="zenith",
        truth_azimuth_col="azimuth",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        energy_col="energy",
        n_bins=10,
    )

    # bin_center must track the energy range (10-10000), not the zenith
    # range (0-pi) - a wrong implementation binning by zenith would put
    # `bin_center` values roughly within [0, pi] instead.
    assert result["bin_center"].min() >= 10.0
    assert result["bin_center"].max() <= 10_000.0
    # Perfect predictor (pred vector == truth vector) -> ~zero opening angle
    # in every populated bin.
    assert result["p50"].notna().any()
    assert np.allclose(result["p50"].dropna(), 0.0, atol=1e-3)


def _make_topology_df() -> pd.DataFrame:
    n = 20
    is_track = np.array([True] * 10 + [False] * 10)
    return pd.DataFrame(
        {
            # Truth always points along +x (zenith=pi/2, azimuth=0).
            "zenith": np.full(n, np.pi / 2),
            "azimuth": np.zeros(n),
            # Track predictions point along +y (90 deg off); cascade
            # predictions point along +x (perfect, 0 deg off).
            "px": np.where(is_track, 0.0, 1.0),
            "py": np.where(is_track, 1.0, 0.0),
            "pz": np.zeros(n),
            "energy": np.full(n, 50.0),
            "is_track": is_track,
            # Muon points along -x (180 deg off truth) - only meaningful
            # for the track rows, but given for all rows since the
            # function should only look at it for track events anyway.
            "muon_zenith": np.full(n, np.pi / 2),
            "muon_azimuth": np.full(n, np.pi),
        }
    )


def test_direction_resolution_by_topology_known_angles() -> None:
    """Track/cascade/muon should each reflect their own known offset."""
    df = _make_topology_df()
    bins = np.array([10.0, 100.0])

    result = direction_resolution_by_topology(
        df,
        truth_zenith_col="zenith",
        truth_azimuth_col="azimuth",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        energy_col="energy",
        is_track_col="is_track",
        muon_zenith_col="muon_zenith",
        muon_azimuth_col="muon_azimuth",
        bins=bins,
    )

    assert set(result.keys()) == {"track", "cascade", "muon"}
    assert np.isclose(result["track"]["p50"].iloc[0], 90.0)
    assert np.isclose(result["cascade"]["p50"].iloc[0], 0.0)
    assert np.isclose(result["muon"]["p50"].iloc[0], 180.0)


def test_direction_resolution_by_topology_omits_muon_when_not_given() -> (
    None
):
    """Without muon columns, the result shouldn't have a "muon" key."""
    df = _make_topology_df()
    bins = np.array([10.0, 100.0])

    result = direction_resolution_by_topology(
        df,
        truth_zenith_col="zenith",
        truth_azimuth_col="azimuth",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        energy_col="energy",
        is_track_col="is_track",
        bins=bins,
    )

    assert set(result.keys()) == {"track", "cascade"}


def test_direction_resolution_by_topology_does_not_mutate_input() -> None:
    """The caller's DataFrame shouldn't gain new columns as a side effect."""
    df = _make_topology_df()
    original_columns = set(df.columns)
    bins = np.array([10.0, 100.0])

    direction_resolution_by_topology(
        df,
        truth_zenith_col="zenith",
        truth_azimuth_col="azimuth",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        energy_col="energy",
        is_track_col="is_track",
        muon_zenith_col="muon_zenith",
        muon_azimuth_col="muon_azimuth",
        bins=bins,
    )

    assert set(df.columns) == original_columns


def test_direction_resolution_by_topology_shares_bins() -> None:
    """Track and cascade must be binned with the same shared edges.

    Track events only span low energies and cascade events only span
    high energies here - with correctly shared bins (spanning the full
    combined range), each topology must show NaNs where only the *other*
    topology has data.
    """
    n_low, n_high = 100, 100
    energy = np.concatenate(
        [np.linspace(10, 100, n_low), np.linspace(500, 1000, n_high)]
    )
    is_track = np.array([True] * n_low + [False] * n_high)
    n = n_low + n_high
    df = pd.DataFrame(
        {
            "zenith": np.full(n, np.pi / 2),
            "azimuth": np.zeros(n),
            "px": np.ones(n),
            "py": np.zeros(n),
            "pz": np.zeros(n),
            "energy": energy,
            "is_track": is_track,
        }
    )

    result = direction_resolution_by_topology(
        df,
        truth_zenith_col="zenith",
        truth_azimuth_col="azimuth",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        energy_col="energy",
        is_track_col="is_track",
        n_bins=10,
    )

    assert result["track"]["p50"].isna().any()
    assert result["cascade"]["p50"].isna().any()


def test_direction_error_distribution_known_angles() -> None:
    """A known 50/50 mix of 0- and 90-degree offsets splits accordingly."""
    n = 20
    df = pd.DataFrame(
        {
            "true_zenith": np.full(n, np.pi / 2),
            "true_azimuth": np.zeros(n),
            # First half of predictions: perfect (0-degree offset).
            # Second half: along +y, a 90-degree offset.
            "pred_x": np.where(np.arange(n) < 10, 1.0, 0.0),
            "pred_y": np.where(np.arange(n) < 10, 0.0, 1.0),
            "pred_z": np.zeros(n),
        }
    )
    bins = np.array([0.0, 45.0, 90.0, 135.0])

    result = direction_error_distribution(
        df,
        truth_zenith_col="true_zenith",
        truth_azimuth_col="true_azimuth",
        pred_x_col="pred_x",
        pred_y_col="pred_y",
        pred_z_col="pred_z",
        bins=bins,
    )

    assert np.isclose(result["percentage"].iloc[0], 50.0)  # [0, 45): 0-deg.
    assert np.isclose(result["percentage"].iloc[1], 0.0)  # [45, 90): none.
    assert np.isclose(result["percentage"].iloc[2], 50.0)  # [90, 135]: 90-deg.


def test_direction_error_distribution_degrees_flag() -> None:
    """`degrees=False` should bin using radians instead of degrees."""
    n = 10
    df = pd.DataFrame(
        {
            "true_zenith": np.full(n, np.pi / 2),
            "true_azimuth": np.zeros(n),
            "pred_x": np.zeros(n),
            "pred_y": np.ones(n),
            "pred_z": np.zeros(n),
        }
    )
    bins = np.array([0.0, np.pi / 4, np.pi / 2, np.pi])

    result = direction_error_distribution(
        df,
        truth_zenith_col="true_zenith",
        truth_azimuth_col="true_azimuth",
        pred_x_col="pred_x",
        pred_y_col="pred_y",
        pred_z_col="pred_z",
        bins=bins,
        degrees=False,
    )

    assert np.isclose(result["percentage"].iloc[2], 100.0)  # bin [pi/2, pi].


def test_direction_error_distribution_default_bins_span_data_range() -> None:
    """Without explicit bins, edges should span the full angle range."""
    n = 50
    rng = np.random.default_rng(seed=1)
    offsets = rng.uniform(0, np.pi / 4, size=n)
    df = pd.DataFrame(
        {
            "true_zenith": np.full(n, np.pi / 2),
            "true_azimuth": np.zeros(n),
            "pred_x": np.cos(offsets),
            "pred_y": np.sin(offsets),
            "pred_z": np.zeros(n),
        }
    )

    result = direction_error_distribution(
        df,
        truth_zenith_col="true_zenith",
        truth_azimuth_col="true_azimuth",
        pred_x_col="pred_x",
        pred_y_col="pred_y",
        pred_z_col="pred_z",
        n_bins=5,
    )

    assert len(result) == 5
    # Every value should fall inside auto-built bins - none excluded.
    assert np.isclose(result["percentage"].sum(), 100.0)


def test_direction_error_distribution_is_column_name_agnostic() -> None:
    """The function must not assume any particular column names."""
    n = 6
    df = pd.DataFrame(
        {
            "z": np.full(n, np.pi / 2),
            "a": np.zeros(n),
            "px": np.ones(n),
            "py": np.zeros(n),
            "pz": np.zeros(n),
        }
    )
    bins = np.array([0.0, 1.0, 2.0])

    result = direction_error_distribution(
        df,
        truth_zenith_col="z",
        truth_azimuth_col="a",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        bins=bins,
    )

    assert list(result.columns) == ["bin_left", "percentage"]
    assert len(result) == 2


def test_direction_error_distribution_by_topology_known_angles() -> None:
    """Track/cascade/muon should each land fully in their known bin."""
    n = 20
    is_track = np.array([True] * 10 + [False] * 10)
    df = pd.DataFrame(
        {
            "zenith": np.full(n, np.pi / 2),
            "azimuth": np.zeros(n),
            # Track predictions: +y (90-deg offset). Cascade: +x (0-deg).
            "px": np.where(is_track, 0.0, 1.0),
            "py": np.where(is_track, 1.0, 0.0),
            "pz": np.zeros(n),
            "is_track": is_track,
            # Muon direction: -x (180-deg offset from truth).
            "muon_zenith": np.full(n, np.pi / 2),
            "muon_azimuth": np.full(n, np.pi),
        }
    )
    bins = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0])

    result = direction_error_distribution_by_topology(
        df,
        truth_zenith_col="zenith",
        truth_azimuth_col="azimuth",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        is_track_col="is_track",
        muon_zenith_col="muon_zenith",
        muon_azimuth_col="muon_azimuth",
        bins=bins,
    )

    assert set(result.keys()) == {"track", "cascade", "muon"}
    track_bin = result["track"][result["track"]["bin_left"] == 90.0]
    assert np.isclose(track_bin["percentage"].iloc[0], 100.0)
    cascade_bin = result["cascade"][result["cascade"]["bin_left"] == 0.0]
    assert np.isclose(cascade_bin["percentage"].iloc[0], 100.0)
    muon_bin = result["muon"][result["muon"]["bin_left"] == 180.0]
    assert np.isclose(muon_bin["percentage"].iloc[0], 100.0)


def test_direction_error_distribution_by_topology_omits_muon() -> None:
    """Without muon columns, the result shouldn't have a "muon" key."""
    n = 20
    is_track = np.array([True] * 10 + [False] * 10)
    df = pd.DataFrame(
        {
            "zenith": np.full(n, np.pi / 2),
            "azimuth": np.zeros(n),
            "px": np.ones(n),
            "py": np.zeros(n),
            "pz": np.zeros(n),
            "is_track": is_track,
        }
    )

    result = direction_error_distribution_by_topology(
        df,
        truth_zenith_col="zenith",
        truth_azimuth_col="azimuth",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        is_track_col="is_track",
    )

    assert set(result.keys()) == {"track", "cascade"}


def test_direction_error_distribution_by_topology_does_not_mutate_input() -> (
    None
):
    """The caller's DataFrame shouldn't gain new columns as a side effect."""
    n = 20
    is_track = np.array([True] * 10 + [False] * 10)
    df = pd.DataFrame(
        {
            "zenith": np.full(n, np.pi / 2),
            "azimuth": np.zeros(n),
            "px": np.ones(n),
            "py": np.zeros(n),
            "pz": np.zeros(n),
            "is_track": is_track,
            "muon_zenith": np.full(n, np.pi / 2),
            "muon_azimuth": np.full(n, np.pi),
        }
    )
    original_columns = set(df.columns)

    direction_error_distribution_by_topology(
        df,
        truth_zenith_col="zenith",
        truth_azimuth_col="azimuth",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        is_track_col="is_track",
        muon_zenith_col="muon_zenith",
        muon_azimuth_col="muon_azimuth",
    )

    assert set(df.columns) == original_columns


def test_direction_error_distribution_by_topology_shares_bins() -> None:
    """Track and cascade must be binned with the same shared edges.

    Track events all have a small opening-angle offset (0-10 deg) and
    cascade events all have a large, disjoint offset (70-90 deg) - with
    correctly shared bins (covering the combined range), track and
    cascade must use identical bin edges, and each must show 0% in the
    range only the *other* topology's data reaches.
    """
    n_side = 100
    track_offsets = np.radians(np.linspace(0, 10, n_side))
    cascade_offsets = np.radians(np.linspace(70, 90, n_side))
    offsets = np.concatenate([track_offsets, cascade_offsets])
    is_track = np.array([True] * n_side + [False] * n_side)
    df = pd.DataFrame(
        {
            "zenith": np.full(2 * n_side, np.pi / 2),
            "azimuth": np.zeros(2 * n_side),
            "px": np.cos(offsets),
            "py": np.sin(offsets),
            "pz": np.zeros(2 * n_side),
            "is_track": is_track,
        }
    )

    result = direction_error_distribution_by_topology(
        df,
        truth_zenith_col="zenith",
        truth_azimuth_col="azimuth",
        pred_x_col="px",
        pred_y_col="py",
        pred_z_col="pz",
        is_track_col="is_track",
        n_bins=10,
    )

    assert np.allclose(
        result["track"]["bin_left"], result["cascade"]["bin_left"]
    )
    # The top bin (near 90 deg) is only reachable by cascade's data.
    assert result["track"]["percentage"].iloc[-1] == 0.0
    # The bottom bin (near 0 deg) is only reachable by track's data.
    assert result["cascade"]["percentage"].iloc[0] == 0.0
