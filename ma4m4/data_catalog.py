import os

import netCDF4 as nc
import numpy as np

from ma4m4.utils import log_duration, safe_unmask_array


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "01_raw")
INTERMEDIATES_DIR = os.path.join(DATA_DIR, "02_intermediates")

FILE_PATHS = {
    "raw_hadisst": os.path.join(RAW_DIR, "HadISST_sst.nc"),
    "correlations": os.path.join(INTERMEDIATES_DIR, "correlations.npz"),
}


@log_duration("load raw sst data")
def load_raw_sst_data():
    """Load the raw SST data and perform low-level type conversions"""

    if not os.path.isfile(FILE_PATHS["raw_hadisst"]):
        raise FileNotFoundError(
            f"Expected raw data at: {FILE_PATHS['raw_hadisst']!r}. This can be "
            f"downloaded in compressed form from"
            f"https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz "
            f"and should be unzipped in the expected location."
        )

    ds = nc.Dataset(FILE_PATHS["raw_hadisst"])

    time_days = safe_unmask_array(ds["time"][:], "time").astype("float64")

    times = nc.num2date(time_days, ds["time"].units, ds["time"].calendar)
    times = times.astype("datetime64[us]")

    latitude = safe_unmask_array(ds["latitude"][:], "latitude").astype("float64")
    longitude = safe_unmask_array(ds["longitude"][:], "longitude").astype("float64")

    sst = ds["sst"][:].astype("float64")

    return {
        "time_days": time_days,
        "time": times,
        "latitude": latitude,
        "longitude": longitude,
        "sst": sst,
    }


@log_duration("save correlations")
def save_correlations(latitude, longitude, correlation, meta):
    """ Save the correlations as an intermediate dataset """

    np.savez(
        FILE_PATHS["correlations"],
        latitude=latitude,
        longitude=longitude,
        correlation=correlation,
        meta=meta,  # Saved using pickle
    )


@log_duration("load correlations")
def load_correlations():
    """ Load the correlations intermediate dataset """

    # We set allow_pickle=True because the metadata is a dictionary stored using pickle
    with np.load(FILE_PATHS["correlations"], allow_pickle=True) as npz:
        keys = ["latitude", "longitude", "correlation"]
        correlations = {k: npz[k] for k in keys}
        meta = npz["meta"].item()

    return correlations, meta
