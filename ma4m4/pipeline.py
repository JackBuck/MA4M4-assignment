import ma4m4.data_catalog as dc
from ma4m4.anomaly_series import generate_anomaly_series
from ma4m4.build_network import build_network, print_graph_statistics
from ma4m4.community_detection import (
    detect_communities_via_asymptotic_surprise,
    detect_communities_via_infomap,
    detect_communities_via_ngmodmax_louvain,
)
from ma4m4.compute_correlations import compute_correlations
from ma4m4.downsample import downsample_anomaly_series
from ma4m4.plots import plot_communities, plot_community_comparison


def run(
    recalculate_correlations=True,
    rebuild_network=True,
    rerun_community_detection=True,
    regenerate_community_comparison_plot=True,
    regenerate_communities_plot_for_weighted_asymptotic_surprise=True,
):
    """Run the full pipeline to process the raw SST data into plots in the essay"""

    dc.setup_directory_structure()

    if recalculate_correlations:
        run_step_calculate_correlations()
    if rebuild_network:
        run_step_build_network()
    if rerun_community_detection:
        run_step_detect_communities()
    if regenerate_community_comparison_plot:
        run_step_plot_community_comparison()
    if regenerate_communities_plot_for_weighted_asymptotic_surprise:
        run_step_plot_communities_from_weighted_asymptotic_surprise()


def run_step_calculate_correlations():
    raw_data = dc.load_raw_sst_data()
    sst_anomaly, meta_an = generate_anomaly_series(raw_data)
    downsampled, meta_ds = downsample_anomaly_series(
        raw_data["latitude"], raw_data["longitude"], sst_anomaly
    )
    correlations = compute_correlations(**downsampled)
    meta = dict(**meta_an, **meta_ds)
    dc.save_correlations(**correlations, meta=meta)


def run_step_build_network():
    correlations, meta_corr = dc.load_correlations()

    graph, meta_net = build_network(**correlations)
    dc.save_network(graph, dict(**meta_corr, **meta_net))

    print_graph_statistics(graph)


def run_step_detect_communities():
    graph, meta = dc.load_network()

    comms_modularity = detect_communities_via_ngmodmax_louvain(graph)
    meta_modularity = {**meta, "community_algo": "modularity"}
    dc.save_communities(comms_modularity, meta_modularity, name="modularity")

    comms_infomap = detect_communities_via_infomap(graph)
    meta_infomap = {**meta, "community_algo": "infomap"}
    dc.save_communities(comms_infomap, meta_infomap, name="infomap")

    comms_surprise = detect_communities_via_asymptotic_surprise(graph)
    meta_surprise = {**meta, "community_algo": "surprise"}
    dc.save_communities(comms_surprise, meta_surprise, name="surprise")

    comms_surprise = detect_communities_via_asymptotic_surprise(graph, weight="abs_corr")
    meta_surprise = {**meta, "community_algo": "surprise-weighted"}
    dc.save_communities(comms_surprise, meta_surprise, name="surprise-weighted")


def run_step_plot_community_comparison():
    communities = {}
    meta = {}
    for alg in ["modularity", "infomap", "surprise"]:
        communities[alg], meta[alg] = dc.load_communities(alg)

    fig = plot_community_comparison(communities)
    dc.save_community_comparison_plot(fig)


def run_step_plot_communities_from_weighted_asymptotic_surprise():
    communities, meta = dc.load_communities("surprise-weighted")
    fig = plot_communities(communities, title="asymptotic surprise (weighted)")
    dc.save_community_plot(fig, "surprise-weighted")
