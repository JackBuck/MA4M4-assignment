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


def run(
    recalculate_correlations=True,
    rebuild_network=True,
    rerun_community_detection=True,
):
    """Run the full pipeline to process the raw SST data into plots in the essay"""

    dc.setup_directory_structure()

    correlations, meta = run_step_calculate_correlations(recalculate_correlations)
    graph, meta = run_step_build_network(correlations, meta, rebuild_network)
    communities = run_step_detect_communities(graph, meta, rerun_community_detection)


def run_step_calculate_correlations(recalculate_correlations=True):
    if recalculate_correlations:
        raw_data = dc.load_raw_sst_data()
        sst_anomaly, meta_an = generate_anomaly_series(raw_data)
        downsampled, meta_ds = downsample_anomaly_series(
            raw_data["latitude"], raw_data["longitude"], sst_anomaly
        )
        correlations = compute_correlations(**downsampled)
        meta = dict(**meta_an, **meta_ds)
        dc.save_correlations(**correlations, meta=meta)
    else:
        correlations, meta = dc.load_correlations()

    return correlations, meta


def run_step_build_network(correlations, meta, rebuild_network=True):
    if rebuild_network:
        graph, meta_net = build_network(**correlations)
        meta = dict(**meta, **meta_net)
        dc.save_network(graph, dict(**meta, **meta_net))
    else:
        graph, meta = dc.load_network()

    print_graph_statistics(graph)

    return graph, meta


def run_step_detect_communities(graph, meta, rerun_community_detection=True):
    if rerun_community_detection:
        comms_modularity = detect_communities_via_ngmodmax_louvain(graph)
        meta_modularity = {**meta, "community_algo": "modularity"}
        dc.save_communities(comms_modularity, meta_modularity, name="modularity")

        comms_infomap = detect_communities_via_infomap(graph)
        meta_infomap = {**meta, "community_algo": "infomap"}
        dc.save_communities(comms_infomap, meta_infomap, name="infomap")

        comms_surprise = detect_communities_via_asymptotic_surprise(graph)
        meta_surprise = {**meta, "community_algo": "surprise"}
        dc.save_communities(comms_surprise, meta_surprise, name="surprise")

    else:
        comms_modularity, meta_modularity = dc.load_communities("modularity")
        comms_infomap, meta_infomap = dc.load_communities("infomap")
        comms_surprise, meta_surprise = dc.load_communities("surprise")

    return {
        "comms_modularity": comms_modularity,
        "comms_infomap": comms_infomap,
        "comms_surprise": comms_surprise,
        "meta_modularity": meta_modularity,
        "meta_infomap": meta_infomap,
        "meta_surprise": meta_surprise,
    }
