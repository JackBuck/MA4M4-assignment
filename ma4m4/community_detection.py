import logging

import cdlib.algorithms

from ma4m4.constants import MODULARITY_MAXIMISATION_RESOLUTION
from ma4m4.utils import log_duration

logger = logging.getLogger(__name__)


# TODO: Specify a seed for these algorithms!!
# TODO: Repeat and average (somehow!)


@log_duration("detect communities via NG modularity maximisation (louvain)")
def detect_communities_via_ngmodmax_louvain(
    graph, resolution=MODULARITY_MAXIMISATION_RESOLUTION
):
    # I think this is the same algorithm as nx.community.louvain_communities, though the
    # implementation is different.
    comms = cdlib.algorithms.louvain(graph, resolution=resolution)
    logger.info(f"Found {len(comms.communities)} communities")
    return comms


@log_duration("detect communities via infomap")
def detect_communities_via_infomap(graph):
    comms = cdlib.algorithms.infomap(graph)
    logger.info(f"Found {len(comms.communities)} communities")
    return comms


@log_duration("detect communities via asymptotic surprise")
def detect_communities_via_asymptotic_surprise(graph, weight: str = None):
    comms = cdlib.algorithms.surprise_communities(graph, weights=weight)
    logger.info(f"Found {len(comms.communities)} communities")
    return comms
