import cartopy.crs as ccrs
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from ma4m4.constants import PLOT_MAX_COMMUNITIES
from ma4m4.utils import log_duration


@log_duration("plotting community comparison")
def plot_community_comparison(communities):
    algorithms = {
        "surprise": "asymptotic surprise",
        "infomap": "infomap",
        "modularity": "modularity (NG null network)",
    }

    central_longitudes = [180, 0]

    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    subfigs = fig.subfigures(1, 3, wspace=0.02)

    for j, alg in enumerate(algorithms):
        # Set up axes with projections from cartopy
        gs = subfigs[j].add_gridspec(2)
        axs = np.squeeze(np.empty(gs.get_geometry(), dtype="object"))
        for i in range(axs.size):
            proj = ccrs.Mollweide(central_longitude=central_longitudes[i])
            ax = subfigs[j].add_subplot(gs[i], projection=proj)
            ax.coastlines()
            ax.set_global()
            axs[i] = ax

        # Plot communities
        _plot_communities(communities[alg], subfigs[j], axs)
        title = f"({chr(ord('a') + j)}) {algorithms[alg]}"
        axs[0].set_title(title, pad=25, fontsize="x-large")

    return fig


def _plot_communities(comms, fig, axs):
    nodes = comms.graph.nodes
    communities = comms.communities

    if len(communities) > PLOT_MAX_COMMUNITIES:
        tail_community = set().union(
            *[c for c in communities[PLOT_MAX_COMMUNITIES - 1:]]
        )
        communities = communities[:PLOT_MAX_COMMUNITIES - 1] + [tail_community]
        truncated_community_list = True
    else:
        truncated_community_list = False

    community_membership = {
        n: _singleton([i for i, c in enumerate(communities) if n in c])
        for n in nodes
    }

    cmap, norm = _get_cmap_and_norm(communities)

    node_ids = list(nodes)  # Fix the order (shouldn't be necessary, but just in case!)
    for ax in axs.flat:
        h = ax.scatter(
            [nodes[n]["longitude"] for n in node_ids],
            [nodes[n]["latitude"] for n in node_ids],
            s=1,
            c=[community_membership[n] for n in node_ids],
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
        )

    cbar = fig.colorbar(
        h,
        label="Community: index (size)",
        ticks=np.arange(len(communities)),
        ax=axs.ravel().tolist(),
        shrink=0.8,
    )
    cbar_tick_labels = [f"{i} ({len(c)})" for i, c in enumerate(communities)]
    if truncated_community_list:
        cbar_tick_labels[-1] = f"Other ({len(communities[-1])})"
    cbar.ax.set_yticklabels(cbar_tick_labels)


def _get_cmap_and_norm(communities, cmap=None):
    if cmap is None:
        if len(communities) <= 10:
            cmap = mpl.cm.get_cmap("tab10", len(communities))
        elif len(communities) <= 20:
            cmap = mpl.cm.get_cmap("tab20", len(communities))
        else:
            cmap = mpl.cm.get_cmap("gist_ncar", len(communities))
    elif isinstance(cmap, str):
        cmap = mpl.cm.get_cmap(cmap, len(communities))

    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, len(communities)), cmap.N)

    return cmap, norm


#######  TODO: Convert the below into a plot above...


def plot_communities(
    graph, communities, algorithm_title=None, central_longitudes=None, markersize=10,
    dpi=100, cmap=None, max_communities=float("Inf"),
):
    communities = communities.communities

    if not central_longitudes:
        central_longitudes = [180, 0]

    # Set up axes
    fig = plt.figure(figsize=(9, 6), dpi=dpi, constrained_layout=True)
    gs = fig.add_gridspec(len(central_longitudes), 1)

    axs = np.empty(gs.get_geometry(), dtype="object")
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            proj = ccrs.Mollweide(
                central_longitude=central_longitudes[i * axs.shape[1] + j],
            )
            ax = fig.add_subplot(gs[i, j], projection=proj)
            ax.coastlines()
            ax.set_global()
            axs[i, j] = ax

    # Plot communities
    if communities != sorted(communities, key=len, reverse=True):
        raise ValueError(
            "Expected communities to be sorted in decreasing order of size!"
        )
    if len(communities) > max_communities:
        tail_community = set().union(*[c for c in communities[max_communities - 1:]])
        communities = communities[:max_communities - 1] + [tail_community]
        truncated_community_list = True
    else:
        truncated_community_list = False

    community_membership = {
        n: _singleton([i for i, c in enumerate(communities) if n in c])
        for n in graph.nodes
    }

    if cmap is None:
        if len(communities) <= 10:
            cmap = mpl.cm.get_cmap("tab10", len(communities))
        elif len(communities) <= 20:
            cmap = mpl.cm.get_cmap("tab20", len(communities))
        else:
            cmap = mpl.cm.get_cmap("gist_ncar", len(communities))
    elif isinstance(cmap, str):
        cmap = mpl.cm.get_cmap(cmap, len(communities))

    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, len(communities)), cmap.N)

    nodes = list(graph.nodes)
    for ax in axs.flat:
        h = ax.scatter(
            [graph.nodes[n]["longitude"] for n in nodes],
            [graph.nodes[n]["latitude"] for n in nodes],
            s=markersize,
            c=[community_membership[n] for n in nodes],
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
        )

    cbar = fig.colorbar(
        h,
        label="Community: index (size)",
        ticks=np.arange(len(communities)),
        ax=axs.ravel().tolist(),
    )
    cbar_tick_labels = [f"{i} ({len(communities[i])})" for i in range(len(communities))]
    if truncated_community_list:
        cbar_tick_labels[-1] = f"Other ({len(communities[-1])})"
    cbar.ax.set_yticklabels(cbar_tick_labels)

    title = "Community assignment"
    if algorithm_title:
        if len(algorithm_title) < 16:
            title += f" from {algorithm_title}"
        else:
            title += f" from\n {algorithm_title}"
    axs[0, 0].set_title(title, fontsize="xx-large")

    return fig


def _singleton(l):
    [x] = l
    return x
