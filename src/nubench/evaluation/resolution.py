"""Generic statistics used to build the evaluation metrics.

Plain arrays in, arrays/DataFrames out.
"""

from typing import Iterable, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

ArrayLike = Union[np.ndarray, pd.Series]


def combined_log_bins(
    series: Iterable[ArrayLike], n_bins: int
) -> np.ndarray:
    """`n_bins` log-spaced edges spanning every series given.

    Lets several models share one binning, so a comparison plot bins them
    all identically.
    """
    series = list(series)
    return np.logspace(
        np.log10(min(np.min(s) for s in series)),
        np.log10(max(np.max(s) for s in series)),
        n_bins + 1,
    )


def opening_angle(
    x1: ArrayLike,
    y1: ArrayLike,
    z1: ArrayLike,
    x2: ArrayLike,
    y2: ArrayLike,
    z2: ArrayLike,
    degrees: bool = False,
) -> np.ndarray:
    """Opening angle between two batches of 3D vectors."""
    dot = x1 * x2 + y1 * y2 + z1 * z2
    norm1 = np.sqrt(x1**2 + y1**2 + z1**2)
    norm2 = np.sqrt(x2**2 + y2**2 + z2**2)
    theta = np.arccos(np.clip(dot / (norm1 * norm2), -1.0, 1.0))
    return np.degrees(theta) if degrees else theta


def spherical_to_cartesian(
    zenith: ArrayLike,
    azimuth: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zenith/azimuth (radians) to a Cartesian unit vector `(x, y, z)`.

    IceCube/Prometheus convention: zenith from the +z axis, azimuth
    around z starting from +x.
    """
    return (
        np.sin(zenith) * np.cos(azimuth),
        np.sin(zenith) * np.sin(azimuth),
        np.cos(zenith),
    )


def euclidean_distance(
    x1: ArrayLike,
    y1: ArrayLike,
    z1: ArrayLike,
    x2: ArrayLike,
    y2: ArrayLike,
    z2: ArrayLike,
) -> np.ndarray:
    """Distance between two batches of 3D points."""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def radial_and_depth(
    x1: ArrayLike,
    y1: ArrayLike,
    z1: ArrayLike,
    x2: ArrayLike,
    y2: ArrayLike,
    z2: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """A 3D position residual split into horizontal and vertical parts.

    Shows which direction a reconstruction is biased in. `radial` is the
    horizontal-plane distance (always >= 0); `depth` is the signed
    `z1 - z2`.
    """
    radial = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return radial, z1 - z2


def contour_68_level(
    x: ArrayLike,
    y: ArrayLike,
    bins: Union[int, Sequence[int], Sequence[np.ndarray]],
    sigma: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """A smoothed 2D density and its 68% containment level.

    Bins `x`/`y` into a 2D histogram, smooths it with a Gaussian of
    `sigma` bins, then finds the density threshold above which the
    enclosed bins hold ~68% of the smoothed mass.
    """
    H, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    # Smooth the histogram to avoid jagged contours from binning
    H = gaussian_filter(H.T, sigma=sigma, mode="reflect")
    # Walk bins from densest down until 68% of the mass is enclosed.
    idx = np.argsort(H.flatten())[::-1]
    cumsum = np.cumsum(H.flatten()[idx])
    level = H.flatten()[idx][np.searchsorted(cumsum / cumsum[-1], 0.68)]
    # Bin centers for contour plotting
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
    min_fraction: float = 1e-4,
) -> pd.DataFrame:
    """Bin `y` by `x` and take percentiles of `y` within each bin.

    The building block behind most of the paper's resolution plots: bin
    events by true energy, then summarise a residual's spread per bin.

    Returns one row per bin with "bin_center" (the median `x` in that
    bin) and one "pXX" column per requested percentile.

    A bin holding less than `min_fraction` of all events gives NaN
    instead, which matplotlib draws as a break in the line rather than a
    measurement. Without it a detector simulated to a lower energy than
    the axis spans trails off into a tail built from a handful of
    stragglers. The threshold is a fraction rather than a flat count so
    it scales with the data.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    bins = np.asarray(bins)
    n_bins = len(bins) - 1
    bin_indices = np.digitize(x, bins)
    min_count = max(1, int(len(x) * min_fraction))
    rows = []
    for i in range(n_bins):
        # `digitize` is 1-indexed for real bins
        mask = bin_indices == i + 1
        # Include the rightmost edge in the last bin, so that the last
        # bin isn't empty if a value happens to be exactly equal to it
        if i == n_bins - 1:
            mask = mask | (x == bins[-1])
        # Too few events to summarise: NaN everything, so the line breaks
        if mask.sum() < min_count:
            row = {"bin_center": np.nan}
            row.update({f"p{p}": np.nan for p in percentiles})
        else:
            row = {"bin_center": np.median(x[mask])}
            row.update(
                zip(
                    (f"p{p}" for p in percentiles),
                    np.percentile(y[mask], percentiles),
                )
            )
        rows.append(row)
    return pd.DataFrame(rows)


def histogram_percentage(
    values: ArrayLike,
    bins: Union[np.ndarray, Sequence[float]],
) -> pd.DataFrame:
    """Histogram `values` as a percentage of the total per bin.

    Returns "bin_left"/"percentage". Values outside `bins` are counted
    nowhere, so the percentages can sum to less than 100.
    """
    counts, edges = np.histogram(values, bins=bins)
    return pd.DataFrame({
        "bin_left": edges[:-1],
        "percentage": counts / len(values) * 100,
    })


def histogram_counts(
    values: ArrayLike,
    bins: Union[np.ndarray, Sequence[float]],
) -> pd.DataFrame:
    """Histogram `values` into raw per-bin counts ("bin_left"/"count")."""
    counts, edges = np.histogram(values, bins=bins)
    return pd.DataFrame({"bin_left": edges[:-1], "count": counts})
