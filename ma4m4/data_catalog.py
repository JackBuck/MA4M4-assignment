import logging
import os
import pickle

import netCDF4 as nc
import numpy as np

from ma4m4.utils import log_duration, safe_unmask_array


logger = logging.getLogger(__name__)


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "01_raw")
INTERMEDIATES_DIR = os.path.join(DATA_DIR, "02_intermediates")
OUTPUTS_DIR = os.path.join(DATA_DIR, "03_outputs")
REPORTING_DIR = os.path.join(DATA_DIR, "04_reporting")

FILE_PATHS = {
    "raw_hadisst": os.path.join(RAW_DIR, "HadISST_sst.nc"),
    "correlations": os.path.join(INTERMEDIATES_DIR, "correlations.npz"),
    "network": os.path.join(INTERMEDIATES_DIR, "network.pkl"),
    "communities": os.path.join(OUTPUTS_DIR, "communities_{name}.pkl"),
    "community_comparison_plot_eps": os.path.join(REPORTING_DIR, "community_comparison.eps"),
    "community_comparison_plot_jpg": os.path.join(REPORTING_DIR, "community_comparison.jpg"),
}


def setup_directory_structure():
    if not os.path.isdir(DATA_DIR):
        # Safety check so we don't create the tree of directories in a location the user
        # didn't intend!
        raise FileNotFoundError(
            f"Please create base data directory manually at: {DATA_DIR}"
        )

    directories = sorted({os.path.dirname(p) for p in FILE_PATHS.values()})
    for path in directories:
        if not os.path.exists(path):
            logger.info(f"Creating directory: {path}")
            os.makedirs(path, exist_ok=True)


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

    logger.info(f"Loaded correlations with meta data: {meta}")

    return correlations, meta


@log_duration("save network")
def save_network(graph, meta):
    with open(FILE_PATHS["network"], "wb") as f:
        pickle.dump({"graph": graph, "meta": meta}, f)


@log_duration("load network")
def load_network():
    with open(FILE_PATHS["network"], "rb") as f:
        loaded = pickle.load(f)
    logger.info(f"Loaded network with meta data: {loaded['meta']}")
    return loaded["graph"], loaded["meta"]


@log_duration("save communities")
def save_communities(comms, meta, name):
    with open(FILE_PATHS["communities"].format(name=name), "wb") as f:
        pickle.dump({"communities": comms, "meta": meta}, f)


@log_duration("load communities")
def load_communities(name):
    with open(FILE_PATHS["communities"].format(name=name), "rb") as f:
        loaded = pickle.load(f)
    logger.info(f"Loaded communities with meta data: {loaded['meta']}")
    return loaded["communities"], loaded["meta"]


@log_duration("save community comparison plot")
def save_community_comparison_plot(fig):
    fig.savefig(FILE_PATHS["community_comparison_plot_eps"])
    fig.savefig(FILE_PATHS["community_comparison_plot_jpg"], dpi=240, facecolor="white")
