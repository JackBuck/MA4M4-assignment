import logging

import numpy as np

from ma4m4 import pipeline


LOG_FORMAT = "%(asctime)s: %(levelname)s - %(name)s - line %(lineno)d - %(message)s"


if __name__ == "__main__":
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
    np.seterr(over="raise", under="raise")

    pipeline.run(
        recalculate_correlations=True,
        rebuild_network=True,
        rerun_community_detection=True,
    )
