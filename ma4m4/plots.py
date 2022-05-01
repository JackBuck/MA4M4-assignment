import cartopy.crs as ccrs
import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from ma4m4.constants import CORRELATION_THRESHOLD, PLOT_MAX_COMMUNITIES
from ma4m4.utils import log_duration


@log_duration("plot correlations")
def plot_correlations_distribution(
    correlations, threshold=CORRELATION_THRESHOLD, two_sided=True
):
    """ Plot the distribution of correlations, along with the threshold

    Args:
        correlations: A symmetric nxn numpy array of correlations with one row and one
            column for each (unmasked) spatial location
        threshold: A threshold to plot on top of the distribution
        two_sided: If true then both the positive and negative of the threshold will be
            drawn, else only the threshold passed will be drawn

    Returns:
        The matplotlib figure generated
    """
    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    upper_entries = correlations[np.triu_indices(correlations.shape[0], k=1)]
    sns.histplot(
        x=upper_entries,
        bins=100,
        color="tab:blue",
        alpha=0.25,
        element="step",
        label="counts",
        legend=False,
        ax=ax,
    )
    ax.axvline(threshold, color="tab:red", label="threshold")
    if two_sided:
        ax.axvline(-threshold, color="tab:red")
    ax.yaxis.set_major_formatter("{x:,.0f}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Pairs of spatial locations")
    ax.legend(loc="upper right")

    return fig


@log_duration("plot community comparison")
def plot_community_comparison(communities):
    algorithms = {
        "surprise": "asymptotic surprise",
        "infomap": "infomap",
        "modularity": "modularity (NG null network)",
    }

    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    subfigs = fig.subfigures(1, 3, wspace=0.02)

    for j, alg in enumerate(algorithms):
        _, axs, _ = _plot_communities(communities[alg], subfigs[j])
        title = f"({chr(ord('a') + j)}) {algorithms[alg]}"
        axs[0].set_title(title, pad=25, fontsize="x-large")

    return fig


@log_duration("plot communities")
def plot_communities(communities, title=None):
    fig, axs, _ = _plot_communities(communities)
    if title:
        axs[0].set_title(title, pad=15, fontsize="x-large")
    return fig


def _plot_communities(comms, fig=None, central_longitudes=(180, 0)):
    if not fig:
        fig = plt.figure(figsize=(6, 5), constrained_layout=True)

    # Set up axes with projections from cartopy
    gs = fig.add_gridspec(2)
    axs = np.squeeze(np.empty(gs.get_geometry(), dtype="object"))
    for i in range(axs.size):
        proj = ccrs.Mollweide(central_longitude=central_longitudes[i])
        ax = fig.add_subplot(gs[i], projection=proj)
        ax.coastlines()
        ax.set_global()
        axs[i] = ax

    # Extract community info (membership, truncation, ...)
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

    # Plot communities on axes
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

    return fig, axs, cbar


def _singleton(l):
    [x] = l
    return x


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
