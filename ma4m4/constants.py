SST_ICE_VAL = -1000
"""The value to use in the sea surface temperature data to represent full ice cover"""

MAX_LATITUDE = 80
"""Maximum latitude to consider (outside this the data is poor quality)"""
MIN_LATITUDE = -50
"""Minimum latitude to consider (outside this the data is poor quality)"""

DOWNSAMPLE_DEGREES = 2
"""
Default number of degrees of latitude and longitude for grid of down-sampled data.

If we do not down-sample then the matrices are huge and we hit both memory and cpu time
issues.
"""

LOW_PASS_CUTOFF = 1/13
"""Cut-off frequency for low pass filter applied to SST data to get anomaly series"""
LOW_PASS_BUTTER_ORDER = 8
"""Order of the low-pass Butterworth filter used when generating the anomaly series"""

CORRELATION_THRESHOLD = 0.4
"""Threshold above which a correlation will be converted to an edge in the network"""


MODULARITY_MAXIMISATION_RESOLUTION = 1
"""Default resolution for detecting communities via modularity maximisation"""


PLOT_MAX_COMMUNITIES = 20
"""The maximum number of communities to show in plots"""
