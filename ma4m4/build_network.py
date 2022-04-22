import networkx as nx
import numpy as np

from ma4m4.constants import CORRELATION_THRESHOLD
from ma4m4.utils import log_duration


@log_duration("build network")
def build_network(
    latitude, longitude, correlation, threshold=CORRELATION_THRESHOLD, two_sided=True
):
    if two_sided:
        adj_mat = np.abs(correlation) >= threshold
    else:
        adj_mat = correlation >= threshold
    np.fill_diagonal(adj_mat, 0)

    graph = nx.from_numpy_array(adj_mat)
    nx.set_node_attributes(graph, {i: x for i, x in enumerate(latitude)}, "latitude")
    nx.set_node_attributes(graph, {i: x for i, x in enumerate(longitude)}, "longitude")

    meta = {"corr_threshold": threshold, "corr_two_sided": two_sided}

    return graph, meta


def print_graph_statistics(graph):
    print("Graph statistics")
    print("----------------")
    print(f"Number of nodes: {graph.number_of_nodes():,}")
    print(f"Number of edges: {graph.number_of_edges():,}")
    print(f"Average degree: {np.mean([d for _, d in graph.degree()]):.1f}")
    print(f"Edge density: {nx.density(graph):.1%}")
