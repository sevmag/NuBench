"""Generic, dependency-light statistics used to build evaluation metrics.

These functions take plain arrays in and arrays/DataFrames out. They know
nothing about "energy" or "direction" or any other specific reconstruction
task - task-specific metric functions built on top of these are the ones
that know which DataFrame columns to plug in.
"""

from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

ArrayLike = Union[np.ndarray, pd.Series]


def opening_angle(
    x1: ArrayLike,
    y1: ArrayLike,
    z1: ArrayLike,
    x2: ArrayLike,
    y2: ArrayLike,
    z2: ArrayLike,
    degrees: bool = False,
) -> np.ndarray:
    """Calculate the opening angle between two batches of 3D vectors.

    The vectors do not need to be normalised - only their direction matters.
    This is used, e.g., to compare a true and a reconstructed direction
    vector (the neutrino direction reconstruction task).

    Args:
        x1: x-component of the first vector(s).
        y1: y-component of the first vector(s).
        z1: z-component of the first vector(s).
        x2: x-component of the second vector(s).
        y2: y-component of the second vector(s).
        z2: z-component of the second vector(s).
        degrees: If True, return the angle in degrees. Otherwise radians.

    Returns:
        The opening angle for each pair of vectors, with the same length as
        the inputs.
    """

    dot = x1 * x2 + y1 * y2 + z1 * z2
    norm1 = np.sqrt(x1**2 + y1**2 + z1**2)
    norm2 = np.sqrt(x2**2 + y2**2 + z2**2)
    cos_theta = dot / (norm1 * norm2)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    if degrees:
        theta = np.degrees(theta)
    return theta


def spherical_to_cartesian(
    zenith: ArrayLike,
    azimuth: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert zenith/azimuth angles to a Cartesian unit direction vector.

    Uses the IceCube/Prometheus convention: zenith is measured from the
    +z axis (zenith=0 points along +z), azimuth is measured around z
    starting from +x.

    Args:
        zenith: Zenith angle(s), in radians.
        azimuth: Azimuth angle(s), in radians.

    Returns:
        A tuple `(x, y, z)` of the unit direction vector's components, each
        with the same length as the inputs.
    """
    x = np.sin(zenith) * np.cos(azimuth)
    y = np.sin(zenith) * np.sin(azimuth)
    z = np.cos(zenith)
    return x, y, z


def euclidean_distance(
    x1: ArrayLike,
    y1: ArrayLike,
    z1: ArrayLike,
    x2: ArrayLike,
    y2: ArrayLike,
    z2: ArrayLike,
) -> np.ndarray:
    """Calculate the Euclidean distance between two batches of 3D points.

    This is used, e.g., to compare a true and a reconstructed interaction
    vertex position (the vertex reconstruction task).

    Args:
        x1: x-coordinate of the first point(s).
        y1: y-coordinate of the first point(s).
        z1: z-coordinate of the first point(s).
        x2: x-coordinate of the second point(s).
        y2: y-coordinate of the second point(s).
        z2: z-coordinate of the second point(s).

    Returns:
        The distance between each pair of points, with the same length as
        the inputs.
    """

    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    return d


def radial_and_depth(
    x1: ArrayLike,
    y1: ArrayLike,
    z1: ArrayLike,
    x2: ArrayLike,
    y2: ArrayLike,
    z2: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a 3D position residual into a radial and a depth component.

    Unlike `euclidean_distance` (a single combined scalar), this keeps
    the horizontal-plane offset and the vertical offset separate - used
    for detector geometries with a distinguished vertical axis (z), where
    it's useful to see *which* direction a reconstruction is biased in,
    not just how far off it is overall.

    Args:
        x1: x-coordinate of the first point(s).
        y1: y-coordinate of the first point(s).
        z1: z-coordinate of the first point(s).
        x2: x-coordinate of the second point(s).
        y2: y-coordinate of the second point(s).
        z2: z-coordinate of the second point(s).

    Returns:
        A tuple `(radial, depth)`:
            - radial: the horizontal-plane distance
              `sqrt((x1-x2)^2 + (y1-y2)^2)` - always >= 0.
            - depth: the signed vertical difference `z1 - z2` - can be
              negative, unlike every other residual in this module.
    """
    radial = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    depth = z1 - z2
    return (radial, depth)


def contour_68_level(
    x: ArrayLike,
    y: ArrayLike,
    bins: Union[int, Sequence[int], Sequence[np.ndarray]],
    sigma: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build a smoothed 2D density and its 68% containment level.

    Bins `x` and `y` into a 2D histogram, smooths it with a Gaussian
    filter (turning a noisy per-bin histogram into a continuous-looking
    density), then finds the density threshold above which the enclosed
    bins together hold ~68% of the total (smoothed) mass - the 2D
    analogue of a 68% containment interval, suitable for drawing as a
    single contour line via `Axes.contour(X, Y, H, levels=[level])`.

    Args:
        x: The first variable, e.g. a signed vertical residual ("depth").
            Length N.
        y: The second variable, e.g. a horizontal-plane residual
            ("radial"). Length N, same order as `x`.
        bins: Passed through to `numpy.histogram2d`'s `bins` argument -
            an int (same bin count on both axes), a pair of ints
            `[n_x_bins, n_y_bins]`, or a pair of explicit edge arrays
            `[x_edges, y_edges]`.
        sigma: Standard deviation of the Gaussian smoothing kernel, in
            bins. Larger values smooth the density more aggressively.

    Returns:
        A tuple `(X, Y, H, level)`:
            - X, Y: 2D meshgrid arrays of bin-center coordinates, each
              shape (n_y_bins, n_x_bins) - ready to pass to
              `Axes.contour(X, Y, H, ...)`.
            - H: the smoothed 2D histogram, same shape as X/Y.
            - level: the density threshold enclosing ~68% of the total
              smoothed mass.
    """
    # 2D histogram
    H, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    H = H.T
    H = gaussian_filter(H, sigma=sigma, mode="reflect")
    # Sort bins by density and find 68% containment
    idx = np.argsort(H.flatten())[::-1]
    cumsum = np.cumsum(H.flatten()[idx])
    cumsum = cumsum / cumsum[-1]
    # Density threshold for 68% containment.
    level = H.flatten()[idx][np.searchsorted(cumsum, 0.68)]
    # Contour plot
    X, Y = np.meshgrid(
        (x_edges[:-1] + x_edges[1:]) / 2,
        (y_edges[:-1] + y_edges[1:]) / 2,
    )
    return X, Y, H, level


def binned_percentiles(
    x: ArrayLike,
    y: ArrayLike,
    bins: Union[np.ndarray, Sequence[float]],
    percentiles: Sequence[float] = (16, 50, 84),
) -> pd.DataFrame:
    """Bin `y` by `x` and compute percentiles of `y` within each bin.

    This is the generic building block behind most of the resolution-style
    plots in the NuBench paper: bin events by their true energy, and within
    each bin, summarise the spread of some residual/reconstructed quantity.

    Args:
        x: The variable to bin by (e.g. true energy). Length N.
        y: The variable to summarise within each bin (e.g. a residual or a
            reconstructed value). Length N, same order as `x`.
        bins: Monotonically increasing bin edges, length (n_bins + 1) -
            same convention as `bins` in `numpy.histogram`.
        percentiles: Which percentiles of `y` to compute per bin, e.g.
            `(16, 50, 84)` for a median and a 68% containment band.

    Returns:
        A DataFrame with one row per bin (in the same order as `bins`) and
        columns:
            - "bin_center": the median of `x` within that bin (NaN if the
              bin is empty).
            - one column per requested percentile, named "pXX" (e.g. "p16",
              "p50", "p84"), containing the percentile of `y` within that
              bin (NaN if the bin is empty).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    bins = np.asarray(bins)
    n_bins = len(bins) - 1

    bin_indices = np.digitize(x, bins)

    rows = []
    for i in range(n_bins):
        # `digitize` is 1-indexed for real bins: bucket `i + 1` holds the
        # points with bins[i] <= x < bins[i + 1].
        mask = bin_indices == i + 1
        if i == n_bins - 1:
            # Points exactly on the outer edge land in digitize's overflow
            # bucket (index len(bins)) instead of the last real bucket -
            # fold them into the last bin so it's right-edge inclusive.
            mask = mask | ((bin_indices == len(bins)) & (x == bins[-1]))

        if mask.sum() == 0:
            row = {"bin_center": np.nan}
            row.update({f"p{p}": np.nan for p in percentiles})
        else:
            row = {"bin_center": np.median(x[mask])}
            values = np.percentile(y[mask], percentiles)
            row.update(zip((f"p{p}" for p in percentiles), values))
        rows.append(row)

    return pd.DataFrame(rows)


def histogram_percentage(
    values: ArrayLike,
    bins: Union[np.ndarray, Sequence[float]],
) -> pd.DataFrame:
    """Histogram `values`, expressing each bin's count as a percentage.

    This is the generic building block behind an error-distribution-style
    plot: instead of binning one variable by another (like
    `binned_percentiles` does), this just histograms a single variable
    (e.g. an opening angle) and reports what percentage of all the values
    fall in each bin.

    Args:
        values: The values to histogram (e.g. opening angles). Length N.
        bins: Monotonically increasing bin edges, length (n_bins + 1) -
            same convention as `bins` in `numpy.histogram`.

    Returns:
        A DataFrame with one row per bin (in the same order as `bins`) and
        columns:
            - "bin_left": the left edge of that bin.
            - "percentage": the percentage of `values` (out of the total
              count) falling in that bin. Values outside `bins`' range
              are not counted anywhere, so percentages across all bins
              can sum to less than 100.
    """
    counts, edges = np.histogram(values, bins=bins)
    percentage = counts / len(values) * 100
    return pd.DataFrame({"bin_left": edges[:-1], "percentage": percentage})


def histogram_counts(
    values: ArrayLike,
    bins: Union[np.ndarray, Sequence[float]],
) -> pd.DataFrame:
    """Histogram `values` into raw per-bin counts.

    Unlike `histogram_percentage`, this reports the actual number of
    values in each bin rather than converting to a percentage of the
    total - useful when a plot is meant to show absolute "Log Counts"
    (a log-scaled y-axis of raw counts), not a normalized shape.

    Args:
        values: The values to histogram (e.g. a classifier score).
            Length N.
        bins: Monotonically increasing bin edges, length (n_bins + 1) -
            same convention as `bins` in `numpy.histogram`.

    Returns:
        A DataFrame with one row per bin (in the same order as `bins`)
        and columns:
            - "bin_left": the left edge of that bin.
            - "count": the number of `values` falling in that bin.
    """
    counts, edges = np.histogram(values, bins=bins)
    return pd.DataFrame({"bin_left": edges[:-1], "count": counts})
