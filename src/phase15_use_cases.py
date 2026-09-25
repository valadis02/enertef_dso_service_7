"""
DSO Service 7 - Phase 15
Test the WINNING model (Phase 14 Config B: deeper MLP, calibrated
synthetic training) on two practical use cases instead of "reconstruct
the whole feeder from scratch":

  1. SMALL CLUSTERS: split each held-out feeder into several smaller
     sub-clusters and run the model on each separately, instead of the
     whole feeder at once. Phase 9 showed accuracy drops with network
     size -- does working on smaller pieces recover some of that?

  2. FLAGGING TOOL: instead of full reconstruction, use the model to
     FLAG suspect edges in an already-documented (but partly wrong)
     topology. Simulate a DSO's asset-management record that is mostly
     correct but has a few errors (some edges "moved" to the wrong
     place), and check whether the model's edge score can tell genuine
     edges from the wrong ones -- a much easier, more realistic task
     than reconstructing everything from zero.
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch

from phase11_full_gnn import TopologyGNN, build_graph_tensors, DEVICE, predict_topology_gnn
from phase1_2_baseline import correlation_mst, evaluate as evaluate_graph
from phase5_simbench import ground_truth_graph, simulate_timeseries_simbench, add_realistic_meter_noise
from phase13_real_training_data import sample_subtree
import simbench as sb

FEEDERS = ["1-LV-rural1--0-sw", "1-LV-semiurb4--0-sw", "1-LV-urban6--0-sw"]


def load_winning_model():
    model = TopologyGNN(edge_mlp_hidden=(64, 32)).to(DEVICE)
    model.load_state_dict(torch.load("results/phase14_best_model_configB.pt"))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Use case 1: small clusters instead of whole feeder
# ---------------------------------------------------------------------------

def partition_into_clusters(G, target_size=12, seed=0):
    """Greedily partition a tree into connected clusters of ~target_size
    nodes each, by repeatedly sampling a subtree from the remaining graph."""
    rng = np.random.default_rng(seed)
    remaining = G.copy()
    clusters = []
    while remaining.number_of_nodes() > 0:
        if remaining.number_of_nodes() <= target_size * 1.5:
            clusters.append(remaining.copy())
            break
        sub = sample_subtree(remaining, rng, min_size=max(4, target_size - 3),
                              max_size=target_size + 3)
        clusters.append(sub)
        remaining = remaining.subgraph([n for n in remaining.nodes() if n not in sub.nodes()]).copy()
        # keep only the largest remaining connected component
        if remaining.number_of_nodes() > 0 and not nx.is_connected(remaining):
            largest_cc = max(nx.connected_components(remaining), key=len)
            remaining = remaining.subgraph(largest_cc).copy()
    return clusters


def test_cluster_approach(model, target_size=12):
    print("\n" + "="*60)
    print("USE CASE 1: small clusters instead of whole feeder")
    print("="*60)
    rows = []
    for code in FEEDERS:
        net = sb.get_simbench_net(code)
        true_graph = ground_truth_graph(net)
        week_start = 96 * 7 * 26
        voltage_df = simulate_timeseries_simbench(net, n_steps=672, start=week_start)
        noisy_df = add_realistic_meter_noise(voltage_df)

        # whole-feeder baseline (already have this, but recompute for consistency)
        mst_whole = predict_topology_gnn(noisy_df, model)
        m_whole = evaluate_graph(mst_whole, true_graph, voltage_df.columns)

        # cluster approach
        clusters = partition_into_clusters(true_graph, target_size=target_size, seed=1)
        cluster_accs = []
        total_correct, total_true = 0, 0
        for cluster in clusters:
            cluster_nodes = [n for n in cluster.nodes() if n in noisy_df.columns]
            if len(cluster_nodes) < 4:
                continue
            cluster_voltage = noisy_df[cluster_nodes]
            mst_cluster = predict_topology_gnn(cluster_voltage, model)
            m_cluster = evaluate_graph(mst_cluster, cluster, cluster_nodes)
            cluster_accs.append(m_cluster["reconstruction_accuracy"])
            total_correct += m_cluster["n_correct_edges"]
            total_true += m_cluster["n_true_edges"]

        overall_cluster_acc = total_correct / total_true if total_true > 0 else 0

        print(f"\n{code} (buses={true_graph.number_of_nodes()}):")
        print(f"  Whole feeder at once      : recon_acc={m_whole['reconstruction_accuracy']:.3f}")
        print(f"  {len(clusters)} clusters (~{target_size} buses each): "
              f"overall recon_acc={overall_cluster_acc:.3f} "
              f"(per-cluster mean={np.mean(cluster_accs):.3f})")

        rows.append({
            "feeder": code, "n_buses": true_graph.number_of_nodes(),
            "whole_feeder_acc": m_whole["reconstruction_accuracy"],
            "n_clusters": len(clusters),
            "cluster_overall_acc": overall_cluster_acc,
            "cluster_mean_acc": np.mean(cluster_accs),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Use case 2: flagging suspect edges in a mostly-correct documented topology
# ---------------------------------------------------------------------------

def corrupt_topology(true_graph, n_corrupt, seed=0):
    """Simulate a DSO record that's mostly right but has n_corrupt edges
    'wrong': remove n_corrupt real edges and add the same number of
    random incorrect edges (keeping node set the same). Returns the
    corrupted graph and the set of specifically WRONG edges in it."""
    rng = np.random.default_rng(seed)
    G = true_graph.copy()
    real_edges = list(G.edges())
    to_remove = [real_edges[i] for i in rng.choice(len(real_edges), size=n_corrupt, replace=False)]
    for e in to_remove:
        G.remove_edge(*e)

    nodes = list(G.nodes())
    wrong_edges = set()
    while len(wrong_edges) < n_corrupt:
        a, b = rng.choice(nodes, size=2, replace=False)
        if not true_graph.has_edge(a, b) and not G.has_edge(a, b):
            G.add_edge(a, b)
            wrong_edges.add(frozenset((a, b)))

    return G, wrong_edges


def test_flagging_approach(model, n_corrupt_fraction=0.15):
    print("\n" + "="*60)
    print("USE CASE 2: flagging suspect edges in a mostly-correct record")
    print("="*60)
    rows = []
    for code in FEEDERS:
        net = sb.get_simbench_net(code)
        true_graph = ground_truth_graph(net)
        week_start = 96 * 7 * 26
        voltage_df = simulate_timeseries_simbench(net, n_steps=672, start=week_start)
        noisy_df = add_realistic_meter_noise(voltage_df)

        n_corrupt = max(1, int(true_graph.number_of_edges() * n_corrupt_fraction))
        corrupted_graph, wrong_edges = corrupt_topology(true_graph, n_corrupt, seed=2)

        with torch.no_grad():
            node_feat, corr, nodes = build_graph_tensors(noisy_df)
            var = torch.tensor(noisy_df.var().fillna(0.0).values, dtype=torch.float32)
            h = model(node_feat, corr)
            logits, i_idx, j_idx = model.score_edges(h, corr, var)
            scores = torch.sigmoid(logits).numpy()

        edge_score = {}
        for a, b, s in zip(i_idx.tolist(), j_idx.tolist(), scores):
            edge_score[frozenset((nodes[a], nodes[b]))] = float(s)

        # for every DOCUMENTED edge (correct + wrong), get the model's score
        documented_edges = [frozenset(e) for e in corrupted_graph.edges()]
        doc_scores = np.array([edge_score.get(e, 0.0) for e in documented_edges])
        is_wrong = np.array([1 if e in wrong_edges else 0 for e in documented_edges])

        # flag the k lowest-scoring documented edges as "suspect" (k = n_corrupt)
        k = n_corrupt
        flagged_idx = np.argsort(doc_scores)[:k]
        flagged_wrong = is_wrong[flagged_idx].sum()
        precision_at_k = flagged_wrong / k
        recall_at_k = flagged_wrong / n_corrupt

        mean_score_correct = doc_scores[is_wrong == 0].mean()
        mean_score_wrong = doc_scores[is_wrong == 1].mean()

        print(f"\n{code} (buses={true_graph.number_of_nodes()}, "
              f"{n_corrupt} wrong edges out of {len(documented_edges)} documented):")
        print(f"  Mean score of CORRECT documented edges: {mean_score_correct:.3f}")
        print(f"  Mean score of WRONG documented edges  : {mean_score_wrong:.3f}")
        print(f"  Flagging the {k} lowest-scoring edges as suspect: "
              f"precision={precision_at_k:.1%}, recall={recall_at_k:.1%}")

        rows.append({
            "feeder": code, "n_buses": true_graph.number_of_nodes(),
            "n_documented": len(documented_edges), "n_wrong": n_corrupt,
            "mean_score_correct": mean_score_correct, "mean_score_wrong": mean_score_wrong,
            "flag_precision_at_k": precision_at_k, "flag_recall_at_k": recall_at_k,
        })
    return pd.DataFrame(rows)


def main():
    model = load_winning_model()

    cluster_results = test_cluster_approach(model, target_size=12)
    cluster_results.to_csv("results/phase15_cluster_results.csv", index=False)

    flagging_results = test_flagging_approach(model, n_corrupt_fraction=0.15)
    flagging_results.to_csv("results/phase15_flagging_results.csv", index=False)

    print("\nSaved results/phase15_cluster_results.csv, results/phase15_flagging_results.csv")


if __name__ == "__main__":
    main()