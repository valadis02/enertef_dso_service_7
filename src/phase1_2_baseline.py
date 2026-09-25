"""
DSO Service 7 - Phase 1 + 2
Synthetic data generation (pandapower, IEEE 33-bus feeder) +
baseline topology identification (Pearson correlation + Maximum Spanning Tree)
"""

import numpy as np
import pandas as pd
import networkx as nx
import pandapower as pp
import pandapower.networks as pn
from sklearn.metrics import precision_score, recall_score

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Phase 1: synthetic data
# ---------------------------------------------------------------------------

def load_feeder():
    """IEEE 33-bus radial distribution test feeder."""
    net = pn.case33bw()
    return net


def ground_truth_graph(net):
    """Build ground-truth topology graph from the in-service lines/switches."""
    G = nx.Graph()
    G.add_nodes_from(net.bus.index.tolist())
    for _, row in net.line.iterrows():
        if row["in_service"]:
            G.add_edge(int(row["from_bus"]), int(row["to_bus"]))
    # also respect switches (case33bw has a few normally-open tie switches)
    if len(net.switch) > 0:
        for _, sw in net.switch.iterrows():
            if sw["et"] == "l":  # line switch
                line = net.line.loc[sw["element"]]
                edge = (int(line["from_bus"]), int(line["to_bus"]))
                if not sw["closed"] and G.has_edge(*edge):
                    G.remove_edge(*edge)
    return G


def simulate_timeseries(net, n_steps=500, load_std=0.25, seed=42):
    """
    Run n_steps power-flow snapshots with randomized loads around the base
    case values, return a DataFrame of per-bus voltage magnitudes (p.u.).
    """
    rng = np.random.default_rng(seed)
    base_p = net.load["p_mw"].copy()
    base_q = net.load["q_mvar"].copy()

    voltage_records = []
    angle_records = []

    for t in range(n_steps):
        # correlated daily-ish load scaling shared across all loads + per-load noise
        common_factor = 1.0 + 0.15 * np.sin(2 * np.pi * t / 96) + rng.normal(0, 0.05)
        noise = rng.normal(1.0, load_std, size=len(net.load))
        net.load["p_mw"] = np.clip(base_p * common_factor * noise, 0, None)
        net.load["q_mvar"] = np.clip(base_q * common_factor * noise, 0, None)

        try:
            pp.runpp(net)
        except Exception:
            continue

        voltage_records.append(net.res_bus["vm_pu"].copy())
        angle_records.append(net.res_bus["va_degree"].copy())

    net.load["p_mw"] = base_p
    net.load["q_mvar"] = base_q

    voltage_df = pd.DataFrame(voltage_records).reset_index(drop=True)
    angle_df = pd.DataFrame(angle_records).reset_index(drop=True)
    return voltage_df, angle_df


# ---------------------------------------------------------------------------
# Phase 2: baseline topology identification (correlation + MST)
# ---------------------------------------------------------------------------

def correlation_mst(voltage_df):
    """
    Pearson correlation between all bus voltage time-series -> distance
    matrix -> Maximum Spanning Tree (Kruskal) on the correlation graph.
    """
    corr = voltage_df.corr(method="pearson")
    # slack/reference bus has ~zero voltage variance -> NaN correlation; treat as weight 0
    corr = corr.fillna(0.0)

    G_full = nx.Graph()
    G_full.add_nodes_from(voltage_df.columns)
    n = len(voltage_df.columns)
    cols = voltage_df.columns.tolist()
    for i in range(n):
        for j in range(i + 1, n):
            w = corr.iloc[i, j]
            G_full.add_edge(cols[i], cols[j], weight=w)

    # Maximum spanning tree = minimum spanning tree on negated weights
    mst = nx.maximum_spanning_tree(G_full, weight="weight")
    return mst, corr


def evaluate(pred_graph, true_graph, all_nodes):
    """Edge-identification precision/recall against ground truth."""
    true_edges = {frozenset(e) for e in true_graph.edges()}
    pred_edges = {frozenset(e) for e in pred_graph.edges()}

    all_pairs = set()
    nodes = list(all_nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            all_pairs.add(frozenset((nodes[i], nodes[j])))

    y_true = [1 if pair in true_edges else 0 for pair in all_pairs]
    y_pred = [1 if pair in pred_edges else 0 for pair in all_pairs]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    correctly_found = true_edges & pred_edges
    graph_reconstruction_accuracy = len(correctly_found) / len(true_edges) if true_edges else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "reconstruction_accuracy": graph_reconstruction_accuracy,
        "n_true_edges": len(true_edges),
        "n_pred_edges": len(pred_edges),
        "n_correct_edges": len(correctly_found),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Phase 1: loading IEEE 33-bus feeder & simulating time-series ===")
    net = load_feeder()
    true_graph = ground_truth_graph(net)
    print(f"Ground truth: {true_graph.number_of_nodes()} buses, {true_graph.number_of_edges()} edges")

    voltage_df, angle_df = simulate_timeseries(net, n_steps=500)
    voltage_df.to_csv("data/voltage_timeseries.csv", index=False)
    print(f"Simulated {len(voltage_df)} valid power-flow snapshots")
    print(f"Voltage data shape: {voltage_df.shape}")

    print("\n=== Phase 2: correlation + maximum spanning tree baseline ===")
    mst, corr = correlation_mst(voltage_df)
    print(f"Predicted graph: {mst.number_of_nodes()} nodes, {mst.number_of_edges()} edges")

    metrics = evaluate(mst, true_graph, voltage_df.columns)
    print("\n=== Evaluation ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # save results
    results_df = pd.DataFrame([metrics])
    results_df.to_csv("results/phase1_2_baseline_results.csv", index=False)

    return net, true_graph, mst, voltage_df, metrics


if __name__ == "__main__":
    main()
