import logging

import joblib
import numpy as np

from ma4m4.utils import log_duration


logger = logging.getLogger(__name__)


@log_duration("compute correlations")
def compute_correlations(latitude, longitude, sst_anomaly):
    latitude, longitude, sst_anomaly = unmask_by_reshaping(
        latitude, longitude, sst_anomaly
    )

    correlations = compute_correlation(sst_anomaly)

    return {
        "latitude": latitude,
        "longitude": longitude,
        "correlation": correlations,
    }


def unmask_by_reshaping(lat, long, y):
    """ Convert a 3D masked array to 2D regular array by removing masked values.

    The time dimension is assume to be the first dimension. It is asserted that either
    all or none of a given time-series is masked.

    Args:
        lat: An m-element vector of latitudes.
        long: An n-element vector of longitudes.
        y: A txmxn 3D masked array containing a times series of length t at each of the
            mn spatial locations. Each time series should be either fully masked (e.g.
            for a land location) or never masked.

    Returns:
        Tuple: (lat, long, y) A tuple containing the "unmasked" versions of lat, long
            and y. Letting k represent the number of spacial locations which aren't
            masked, lat and long are k-element vectors while y is a txk element matrix.
    """
    if (y.mask.any(axis=0) != y.mask.all(axis=0)).any():
        num_offenders = (y.mask.any(axis=0) != y.mask.all(axis=0)).sum()
        raise ValueError(
            f"Expected the time series for each spatial location to be either full "
            f"masked or not masked at all. Found {num_offenders} offending spatial "
            f"locations."
        )
    mask = y.mask.any(axis=0)

    y_unmask = y[:, ~mask].filled(np.nan)
    if np.isnan(y_unmask).any():
        # Really this is an internal sanity check that we didn't need to fill anywhere
        # with NaN. However, it is possible that this would get triggered if y contains
        # a NaN to begin with.
        raise ValueError("Did not expect nan values in data!")

    ix_lat, ix_long = np.nonzero(~mask)
    lat_unmask = lat[ix_lat]
    long_unmask = long[ix_long]

    return lat_unmask, long_unmask, y_unmask


def compute_correlation(y, verbose=1):
    """ Compute the Pearson correlation between all columns at lag 0.

    Computation is done in parallel using joblib.

    Warning:
        NaN is returned when one of the time-series is constant.
    """
    y_centered = y - y.mean(axis=0)
    y_std = np.sqrt(np.mean(y_centered ** 2, axis=0))
    n_time, n_space = y.shape

    def compute_cov(x, y):
        return np.mean(x * y, axis=-1)

    logger.info(f"Processing {n_space:,} tasks with joblib")
    cov = joblib.Parallel(n_jobs=-1, prefer="threads", verbose=verbose)(
        joblib.delayed(compute_cov)(y_centered.T, y_centered.T[i])
        for i in range(n_space)
    )
    cov = np.array(cov)
    r = cov / np.outer(y_std, y_std)

    # We clip the result for numerical stability. The correlation can be slightly
    # greater than 1 before doing this thanks to numerical instability.
    return np.clip(r, -1, 1)
