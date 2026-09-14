"""Unit tests for `nubench.evaluation.resolution`."""

import numpy as np
import pandas as pd

from nubench.evaluation.resolution import (
    binned_percentiles,
    contour_68_level,
    euclidean_distance,
    histogram_counts,
    histogram_percentage,
    opening_angle,
    radial_and_depth,
    spherical_to_cartesian,
)


# Unit test(s): `opening_angle`
def test_opening_angle_identical_vectors() -> None:
    """Identical vectors should have zero opening angle."""
    x = np.array([1.0, -2.0, 0.0])
    y = np.array([2.0, 0.0, 5.0])
    z = np.array([3.0, 1.0, -5.0])
    angle = opening_angle(x, y, z, x, y, z)
    assert np.allclose(angle, 0.0, atol=1e-6)


def test_opening_angle_orthogonal_vectors() -> None:
    """Orthogonal unit vectors should be pi/2 radians apart."""
    angle = opening_angle(
        np.array([1.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([0.0]),
    )
    assert np.allclose(angle, np.pi / 2)


def test_opening_angle_opposite_vectors() -> None:
    """Opposite vectors should be pi radians apart."""
    angle = opening_angle(
        np.array([1.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([-1.0]),
        np.array([0.0]),
        np.array([0.0]),
    )
    assert np.allclose(angle, np.pi)


def test_opening_angle_is_magnitude_invariant() -> None:
    """Only the direction of the vectors should matter, not their length."""
    angle = opening_angle(
        np.array([2.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([5.0]),
        np.array([0.0]),
    )
    assert np.allclose(angle, np.pi / 2)


def test_opening_angle_degrees_flag() -> None:
    """`degrees=True` should convert the output from radians to degrees."""
    angle_deg = opening_angle(
        np.array([1.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([0.0]),
        degrees=True,
    )
    assert np.allclose(angle_deg, 90.0)


def test_opening_angle_known_45_degrees() -> None:
    """A 45-degree case that doesn't lie neatly on the coordinate axes."""
    angle = opening_angle(
        np.array([1.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([1.0]),
        np.array([0.0]),
        degrees=True,
    )
    assert np.allclose(angle, 45.0)


def test_opening_angle_batched() -> None:
    """Several event pairs at once should be evaluated independently."""
    x1 = np.array([1.0, 1.0, 1.0])
    y1 = np.array([0.0, 0.0, 0.0])
    z1 = np.array([0.0, 0.0, 0.0])
    x2 = np.array([1.0, 0.0, -1.0])
    y2 = np.array([0.0, 1.0, 0.0])
    z2 = np.array([0.0, 0.0, 0.0])

    angle = opening_angle(x1, y1, z1, x2, y2, z2, degrees=True)

    assert angle.shape == (3,)
    assert np.allclose(angle, [0.0, 90.0, 180.0])


def test_opening_angle_clips_floating_point_error() -> None:
    """Near-parallel vectors should not produce NaN from floating point noise.

    Two exactly identical vectors give a dot-product-over-norms ratio that
    is mathematically exactly 1, but floating point rounding in the sqrt and
    division can push the computed ratio a hair above 1, which would make
    `arccos` return NaN unless the implementation clips its input.
    """
    x1 = np.array([1.0])
    y1 = np.array([1e-9])
    z1 = np.array([0.0])

    angle = opening_angle(x1, y1, z1, x1, y1, z1)

    assert np.all(np.isfinite(angle))


# Unit test(s): `euclidean_distance`
def test_euclidean_distance_identical_points() -> None:
    """Identical points should have zero distance."""
    x = np.array([1.0, -2.0, 0.0])
    y = np.array([2.0, 0.0, 5.0])
    z = np.array([3.0, 1.0, -5.0])
    dist = euclidean_distance(x, y, z, x, y, z)
    assert np.allclose(dist, 0.0)


def test_euclidean_distance_known_three_four_five() -> None:
    """A classic 3-4-5 right triangle, extended into 3D with z=0."""
    dist = euclidean_distance(
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([3.0]),
        np.array([4.0]),
        np.array([0.0]),
    )
    assert np.allclose(dist, 5.0)


def test_euclidean_distance_known_diagonal() -> None:
    """Unit displacement along all three axes gives sqrt(3)."""
    dist = euclidean_distance(
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([1.0]),
        np.array([1.0]),
    )
    assert np.allclose(dist, np.sqrt(3))


def test_euclidean_distance_is_symmetric() -> None:
    """distance(a, b) should equal distance(b, a)."""
    rng = np.random.default_rng(seed=7)
    p1 = rng.uniform(-100, 100, size=(3, 20))
    p2 = rng.uniform(-100, 100, size=(3, 20))

    forward = euclidean_distance(*p1, *p2)
    backward = euclidean_distance(*p2, *p1)

    assert np.allclose(forward, backward)


def test_euclidean_distance_batched() -> None:
    """Several point pairs at once should be evaluated independently."""
    x1 = np.array([0.0, 0.0, 0.0])
    y1 = np.array([0.0, 0.0, 0.0])
    z1 = np.array([0.0, 0.0, 0.0])
    x2 = np.array([1.0, 0.0, 3.0])
    y2 = np.array([0.0, 1.0, 4.0])
    z2 = np.array([0.0, 0.0, 0.0])

    dist = euclidean_distance(x1, y1, z1, x2, y2, z2)

    assert dist.shape == (3,)
    assert np.allclose(dist, [1.0, 1.0, 5.0])


# Unit test(s): `radial_and_depth`
def test_radial_and_depth_known_three_four_five() -> None:
    """A 3-4 horizontal offset with no vertical offset -> radial=5, depth=0."""
    radial, depth = radial_and_depth(
        x1=0.0, y1=0.0, z1=10.0, x2=3.0, y2=4.0, z2=10.0
    )
    assert np.isclose(radial, 5.0)
    assert np.isclose(depth, 0.0)


def test_radial_and_depth_depth_is_signed() -> None:
    """Unlike every other residual here, depth keeps its sign."""
    radial_above, depth_above = radial_and_depth(
        x1=0.0, y1=0.0, z1=10.0, x2=0.0, y2=0.0, z2=4.0
    )
    radial_below, depth_below = radial_and_depth(
        x1=0.0, y1=0.0, z1=4.0, x2=0.0, y2=0.0, z2=10.0
    )
    assert np.isclose(radial_above, 0.0)
    assert np.isclose(depth_above, 6.0)
    assert np.isclose(radial_below, 0.0)
    assert np.isclose(depth_below, -6.0)


def test_radial_and_depth_batched() -> None:
    """Several point pairs at once should be evaluated independently."""
    x1 = np.array([0.0, 0.0])
    y1 = np.array([0.0, 0.0])
    z1 = np.array([0.0, 0.0])
    x2 = np.array([3.0, 0.0])
    y2 = np.array([4.0, 0.0])
    z2 = np.array([0.0, -2.0])

    radial, depth = radial_and_depth(x1, y1, z1, x2, y2, z2)

    assert radial.shape == (2,)
    assert depth.shape == (2,)
    assert np.allclose(radial, [5.0, 0.0])
    assert np.allclose(depth, [0.0, 2.0])


# Unit test(s): `contour_68_level`
def test_contour_68_level_shapes() -> None:
    """X, Y, H should all share the same (n_y_bins, n_x_bins) shape."""
    rng = np.random.default_rng(seed=1)
    x = rng.normal(size=200)
    y = rng.normal(size=200)

    X, Y, H, level = contour_68_level(x, y, bins=[10, 15])

    assert X.shape == (15, 10)
    assert Y.shape == (15, 10)
    assert H.shape == (15, 10)


def test_contour_68_level_axes_orientation() -> None:
    """X should vary along columns (x-values), Y along rows (y-values)."""
    x = np.array([0.0, 0.0, 5.0, 5.0])
    y = np.array([0.0, 10.0, 0.0, 10.0])
    x_edges = np.array([-1.0, 2.0, 6.0])
    y_edges = np.array([-1.0, 5.0, 11.0])

    X, Y, H, level = contour_68_level(x, y, bins=[x_edges, y_edges])

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    assert np.allclose(X[0, :], x_centers)
    assert np.allclose(Y[:, 0], y_centers)


def test_contour_68_level_no_smoothing_matches_raw_histogram() -> None:
    """With sigma=0, H should exactly match the raw (transposed) counts."""
    rng = np.random.default_rng(seed=2)
    x = rng.uniform(0, 10, size=300)
    y = rng.uniform(0, 10, size=300)
    bins = 8

    X, Y, H, level = contour_68_level(x, y, bins=bins, sigma=0.0)

    expected_H, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    assert np.allclose(H, expected_H.T)


def test_contour_68_level_selects_about_68_percent_mass() -> None:
    """The bins with H >= level should together hold ~68% of the mass."""
    rng = np.random.default_rng(seed=3)
    x = rng.normal(0, 1, size=5000)
    y = rng.normal(0, 1, size=5000)

    X, Y, H, level = contour_68_level(x, y, bins=30, sigma=1.0)

    contained_fraction = H[H >= level].sum() / H.sum()
    assert 0.60 <= contained_fraction <= 0.75


# Unit test(s): `spherical_to_cartesian`
def test_spherical_to_cartesian_zenith_zero_points_along_z() -> None:
    """zenith=0 should point along +z, regardless of azimuth."""
    x, y, z = spherical_to_cartesian(
        np.array([0.0, 0.0]), np.array([0.0, 1.7])
    )
    assert np.allclose(x, 0.0, atol=1e-12)
    assert np.allclose(y, 0.0, atol=1e-12)
    assert np.allclose(z, 1.0)


def test_spherical_to_cartesian_zenith_pi_points_along_minus_z() -> None:
    """zenith=pi should point along -z."""
    x, y, z = spherical_to_cartesian(np.array([np.pi]), np.array([0.0]))
    assert np.allclose(x, 0.0, atol=1e-12)
    assert np.allclose(y, 0.0, atol=1e-12)
    assert np.allclose(z, -1.0)


def test_spherical_to_cartesian_equator_known_directions() -> None:
    """zenith=pi/2 lies in the x-y plane; azimuth picks out x or y."""
    x, y, z = spherical_to_cartesian(
        np.array([np.pi / 2, np.pi / 2]), np.array([0.0, np.pi / 2])
    )
    assert np.allclose(x, [1.0, 0.0], atol=1e-12)
    assert np.allclose(y, [0.0, 1.0], atol=1e-12)
    assert np.allclose(z, [0.0, 0.0], atol=1e-12)


def test_spherical_to_cartesian_returns_unit_vectors() -> None:
    """Regardless of angle, the resulting vector should have unit length."""
    rng = np.random.default_rng(seed=4)
    zenith = rng.uniform(0, np.pi, size=50)
    azimuth = rng.uniform(0, 2 * np.pi, size=50)

    x, y, z = spherical_to_cartesian(zenith, azimuth)

    norm = np.sqrt(x**2 + y**2 + z**2)
    assert np.allclose(norm, 1.0)


# Unit test(s): `binned_percentiles`
def test_binned_percentiles_matches_manual_bins() -> None:
    """Compare against `np.percentile`/`np.median` on manually-sliced bins."""
    rng = np.random.default_rng(seed=0)
    x = rng.uniform(0, 3, size=300)
    y = rng.normal(size=300)
    bins = np.array([0.0, 1.0, 2.0, 3.0])
    percentiles = (16, 50, 84)

    result = binned_percentiles(x, y, bins, percentiles=percentiles)

    assert len(result) == len(bins) - 1
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if i == len(bins) - 2:
            # The last bin should include its right edge.
            mask = (x >= bins[i]) & (x <= bins[i + 1])
        assert np.isclose(result["bin_center"].iloc[i], np.median(x[mask]))
        for p in percentiles:
            assert np.isclose(
                result[f"p{p}"].iloc[i], np.percentile(y[mask], p)
            )


def test_binned_percentiles_empty_bin_is_nan() -> None:
    """A bin with no points in it should yield NaN, not raise an error."""
    x = np.array([0.5, 0.5, 2.5, 2.5])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    bins = np.array([0.0, 1.0, 2.0, 3.0])  # Middle bin [1, 2) is empty.

    result = binned_percentiles(x, y, bins)

    assert np.isnan(result["bin_center"].iloc[1])
    assert np.isnan(result["p50"].iloc[1])
    # The non-empty bins should still be populated.
    assert not np.isnan(result["bin_center"].iloc[0])
    assert not np.isnan(result["bin_center"].iloc[2])


def test_binned_percentiles_accepts_pandas_series() -> None:
    """The function should accept `pandas.Series`, not just `numpy.ndarray`."""
    x = pd.Series([0.5, 1.5, 2.5])
    y = pd.Series([10.0, 20.0, 30.0])
    bins = np.array([0.0, 1.0, 2.0, 3.0])

    result = binned_percentiles(x, y, bins)

    assert len(result) == 3
    assert np.isclose(result["p50"].iloc[0], 10.0)


def test_histogram_percentage_known_counts() -> None:
    """Percentages should match a manual count/total*100 calculation."""
    values = np.array([0.5, 0.5, 0.5, 1.5, 2.5, 2.5])
    bins = np.array([0.0, 1.0, 2.0, 3.0])

    result = histogram_percentage(values, bins)

    assert len(result) == 3
    # Bin [0, 1): 3/6 = 50%, bin [1, 2): 1/6, bin [2, 3]: 2/6.
    assert np.isclose(result["percentage"].iloc[0], 50.0)
    assert np.isclose(result["percentage"].iloc[1], 100.0 / 6.0)
    assert np.isclose(result["percentage"].iloc[2], 200.0 / 6.0)


def test_histogram_percentage_bin_left_matches_bin_edges() -> None:
    """"bin_left" should be each bin's left edge, not a center/right edge."""
    values = np.array([0.5, 1.5, 2.5])
    bins = np.array([0.0, 1.0, 2.0, 3.0])

    result = histogram_percentage(values, bins)

    assert np.allclose(result["bin_left"], [0.0, 1.0, 2.0])


def test_histogram_percentage_sums_to_100_when_all_values_in_range() -> None:
    """If every value falls within `bins`, percentages should sum to 100."""
    rng = np.random.default_rng(seed=0)
    values = rng.uniform(0, 10, size=500)
    bins = np.linspace(0, 10, 21)

    result = histogram_percentage(values, bins)

    assert np.isclose(result["percentage"].sum(), 100.0)


def test_histogram_percentage_excludes_out_of_range_values() -> None:
    """Values outside `bins`' range shouldn't be counted anywhere."""
    values = np.array([0.5, 0.5, 100.0])  # 100.0 is way outside the bins.
    bins = np.array([0.0, 1.0, 2.0])

    result = histogram_percentage(values, bins)

    # Both in-range values land in bin [0, 1); the out-of-range one is
    # dropped, but the percentage is still out of the *total* count (3).
    assert np.isclose(result["percentage"].sum(), 200.0 / 3.0)


def test_histogram_counts_known_values() -> None:
    """Counts should match a manual per-bin tally, not a percentage."""
    values = np.array([0.5, 0.5, 0.5, 1.5, 2.5, 2.5])
    bins = np.array([0.0, 1.0, 2.0, 3.0])

    result = histogram_counts(values, bins)

    assert list(result["count"]) == [3, 1, 2]


def test_histogram_counts_bin_left_matches_bin_edges() -> None:
    """"bin_left" should be each bin's left edge, not a center/right edge."""
    values = np.array([0.5, 1.5, 2.5])
    bins = np.array([0.0, 1.0, 2.0, 3.0])

    result = histogram_counts(values, bins)

    assert np.allclose(result["bin_left"], [0.0, 1.0, 2.0])


def test_histogram_counts_sums_to_total_when_all_values_in_range() -> None:
    """If every value falls within `bins`, counts should sum to len(values)."""
    rng = np.random.default_rng(seed=0)
    values = rng.uniform(0, 10, size=500)
    bins = np.linspace(0, 10, 21)

    result = histogram_counts(values, bins)

    assert result["count"].sum() == len(values)


def test_histogram_counts_excludes_out_of_range_values() -> None:
    """Values outside `bins`' range shouldn't be counted anywhere."""
    values = np.array([0.5, 0.5, 100.0])  # 100.0 is way outside the bins.
    bins = np.array([0.0, 1.0, 2.0])

    result = histogram_counts(values, bins)

    # Both in-range values land in bin [0, 1); the out-of-range one is
    # dropped entirely, so the total is 2, not 3.
    assert result["count"].sum() == 2
