import numpy as np

from ma4m4.constants import DOWNSAMPLE_DEGREES, MAX_LATITUDE, MIN_LATITUDE
from ma4m4.utils import log_duration


@log_duration("downsample anomaly series")
def downsample_anomaly_series(
    latitude,
    longitude,
    sst_anomaly,
    downsample_degrees=DOWNSAMPLE_DEGREES,
    min_latitude=MIN_LATITUDE,
    max_latitude=MAX_LATITUDE,
):
    """ Downsample the data

    WARNING: This method assumes that the input is equally spaced data on a
    latitude-longitude grid, and that the data contains every half-integer grid point.
    """
    latitude_mask = (latitude >= min_latitude) & (latitude <= max_latitude)
    sst_anomaly_downsample = sst_anomaly[
        (
            slice(None),  # Time axis
            *np.ix_(
                (latitude_mask & (latitude % downsample_degrees == 0.5)),
                (longitude % downsample_degrees == 0.5),
            )
        )
    ]
    lat_downsample = latitude[(latitude_mask & (latitude % downsample_degrees == 0.5))]
    long_downsample = longitude[(longitude % downsample_degrees == 0.5)]

    result = {
        "latitude": lat_downsample,
        "longitude": long_downsample,
        "sst_anomaly": sst_anomaly_downsample,
    }
    meta = {
        "downsample_degrees": downsample_degrees,
        "min_latitude": min_latitude,
        "max_latitude": max_latitude,
    }
    return result, meta
