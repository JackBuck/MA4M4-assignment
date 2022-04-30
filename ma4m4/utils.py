import logging
import time
from contextlib import contextmanager

import numpy as np

durations_logger = logging.getLogger(__name__ + ".log_duration")


@contextmanager
def log_duration(name, logger=None):
    """ Context manager to log on entering and exiting (with duration)

    Can also be used as a function decorator.
    """
    if not logger:
        logger = durations_logger

    logger.info(f"Starting {name!r}")

    t0 = time.monotonic()
    yield
    t1 = time.monotonic()

    mins, secs = divmod(t1 - t0, 60)
    if mins:
        timestr = f"{mins:.0f} mins {secs:.0f} secs"
    else:
        timestr = f"{secs:.2g} secs"

    logger.info(f"Finished {name!r} ({timestr})")


def safe_unmask_array(arr, name="variable"):
    """ Safely unmask a numpy masked array.

    Raises if the array has a non-trivial mask.
    """
    if np.ma.is_masked(arr):
        raise ValueError(f"Expected {name!r} to have a trivial mask")
    return arr.filled()
