import ma4m4.data_catalog as dc
from ma4m4.anomaly_series import generate_anomaly_series
from ma4m4.compute_correlations import compute_correlations
from ma4m4.downsample import downsample_anomaly_series


def run(recalculate_correlations=True):
    """Run the full pipeline to process the raw SST data into plots in the essay"""
    if recalculate_correlations:
        raw_data = dc.load_raw_sst_data()
        sst_anomaly, meta_an = generate_anomaly_series(raw_data)
        downsampled, meta_ds = downsample_anomaly_series(
            raw_data["latitude"], raw_data["longitude"], sst_anomaly
        )
        correlations = compute_correlations(**downsampled)
        meta_corr = dict(**meta_an, **meta_ds)
        dc.save_correlations(**correlations, meta=meta_corr)
    else:
        correlations, meta_corr = dc.load_correlations()
