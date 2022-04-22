import numpy as np
import scipy.signal

from ma4m4.constants import LOW_PASS_CUTOFF, LOW_PASS_BUTTER_ORDER, SST_ICE_VAL
from ma4m4.utils import log_duration, safe_unmask_array


@log_duration("generate anomaly series")
def generate_anomaly_series(data):
    """ Generate the SST anomaly time-series

    This involves removing seasonality, detrending and low-pass filtering.
    """

    with log_duration("remove ice values"):
        sst = mask_ice_in_sst(data["sst"])

    with log_duration("de-trend sst"):
        sst = detrend(data["time_days"], sst)

    with log_duration("remove seasonality"):
        sst = remove_seasonal(sst)

    with log_duration("low pass filter"):
        mask = sst.mask.any(axis=0)
        sst[:, ~mask] = butter_lowpass_filter(
            sst[:, ~mask].filled(),
            cutoff=LOW_PASS_CUTOFF,
            order=LOW_PASS_BUTTER_ORDER,
            sample_freq=1,
            axis=0,
        )

    meta = {
        "low_pass_cutoff": LOW_PASS_CUTOFF,
        "low_pass_butter_order": LOW_PASS_BUTTER_ORDER,
    }

    return sst, meta


def mask_ice_in_sst(sst: np.ma.MaskedArray):
    """ Mask locations where there is ever full ice cover in the SST data

    This adds to the current mask, which has masked out values corresponding to land
    locations.
    """
    is_ice = (sst == SST_ICE_VAL).filled(False)
    ever_ice = is_ice.any(axis=0)  # Time is in first axis
    sst_no_ice = np.ma.masked_where(*np.broadcast_arrays(ever_ice, sst, subok=True))
    return sst_no_ice


def detrend(t: np.array, y: np.ma.MaskedArray):
    """ Remove a linear trend from masked data.

    Args:
        t: Time (in days or other unit, but not datetimes) for each point
        y: Masked array of y-values to de-trend
    """
    if not y.ndim == 3:
        raise ValueError(f"Expected 'y' to be three dimensional. Got {y.ndim=}.")

    has_masked_values = y.mask.any(axis=0)
    all_masked_values = y.mask.all(axis=0)

    if (has_masked_values & ~all_masked_values).any():
        # Assuming this simplifies the detrending method. It is also necessary for other
        # parts of the pipeline so we may as well assume it here.
        raise ValueError(
            "Expected each spatial location to have a time-series with either no "
            "masked values or all masked values."
        )

    t_mat = np.stack([t, np.ones_like(t)], axis=-1)
    y_not_masked = safe_unmask_array(y[:, ~has_masked_values])
    beta, _, _, _ = np.linalg.lstsq(t_mat, y_not_masked, rcond=None)

    y_detrend = y.copy()
    y_detrend[:, ~has_masked_values] -= t_mat @ beta

    return y_detrend


def remove_seasonal(y):
    """ Remove annual seasonality from masked data.

    Assumes that the first axis indexes one data point per month.
    """
    n_pts = len(y)
    y_bymonth = [[] for _ in range(12)]
    for i in range(n_pts):
        y_bymonth[i % 12].append(y[i])

    # The following can result in underflow when dividing. I don't _know_ why but
    # suspect that it is numbers which are close to zero being set to zero during the
    # multiply. At any rate, it doesn't seem to make a difference!
    # Interestingly this is only a problem when the data is represented as float64.
    # Using float32 would make it go away.
    with np.errstate(under="ignore"):
        seas_avg = [sum(y_) / len(y_) for y_ in y_bymonth]

    y_noseas = y.copy()
    for i in range(n_pts):
        y_noseas[i] -= seas_avg[i % 12]

    return y_noseas


def butter_lowpass_filter(y, cutoff, order, sample_freq, axis=-1):
    """ Implement a butterworth filter """
    sos = scipy.signal.butter(order, cutoff, fs=sample_freq, output="sos")
    y_filt = scipy.signal.sosfiltfilt(sos, y, axis=axis)
    return y_filt
