"""
DSO Service 7 - Phase 16
Does PMU-grade instrumentation (Class 0.1, sigma=0.0005 p.u.) enable much
better WHOLE-FEEDER reconstruction on the REAL SimBench feeders, compared
to typical Class 0.5 smart meters? Tests both baseline and the winning
GNN (Phase 14 Config B) across meter accuracy classes.
"""

import numpy as np
import pandas as pd
import torch

from phase11_full_gnn import TopologyGNN, DEVICE, predict_topology_gnn
from phase1_2_baseline import correlation_mst, evaluate as evaluate_graph
from phase5_simbench import ground_truth_graph, simulate_timeseries_simbench
import simbench as sb

FEEDERS = ["1-LV-rural1--0-sw", "1-LV-semiurb4--0-sw", "1-LV-urban6--0-sw"]

METER_CLASSES = {
    "Class 1 (older/cheaper)": 0.005,
    "Class 0.5 (typical smart meter)": 0.0025,
    "Class 0.1 (PMU-grade)": 0.0005,
    "No noise (theoretical ceiling)": 0.0,
}


def load_winning_model():
    model = TopologyGNN(edge_mlp_hidden=(64, 32)).to(DEVICE)
    model.load_state_dict(torch.load("results/phase14_best_model_configB.pt"))
    model.eval()
    return model


def add_noise(voltage_df, sigma, seed=2026):
    rng = np.random.default_rng(seed)
    if sigma == 0:
        return voltage_df.copy()
    return voltage_df + rng.normal(0, sigma, size=voltage_df.shape)


def main():
    model = load_winning_model()

    rows = []
    for code in FEEDERS:
        net = sb.get_simbench_net(code)
        true_graph = ground_truth_graph(net)
        week_start = 96 * 7 * 26
        voltage_df = simulate_timeseries_simbench(net, n_steps=672, start=week_start)

        print(f"\n{code} (buses={true_graph.number_of_nodes()}):")
        for label, sigma in METER_CLASSES.items():
            noisy_df = add_noise(voltage_df, sigma)

            mst_base, _ = correlation_mst(noisy_df)
            m_base = evaluate_graph(mst_base, true_graph, voltage_df.columns)

            mst_gnn = predict_topology_gnn(noisy_df, model)
            m_gnn = evaluate_graph(mst_gnn, true_graph, voltage_df.columns)

            print(f"  {label:35s} (sigma={sigma}): "
                  f"baseline={m_base['reconstruction_accuracy']:.3f}  "
                  f"GNN={m_gnn['reconstruction_accuracy']:.3f}")

            rows.append({"feeder": code, "n_buses": true_graph.number_of_nodes(),
                         "meter_class": label, "sigma": sigma,
                         "baseline_acc": m_base["reconstruction_accuracy"],
                         "gnn_acc": m_gnn["reconstruction_accuracy"]})

    results = pd.DataFrame(rows)
    results.to_csv("results/phase16_pmu_comparison.csv", index=False)
    print("\n" + results.pivot(index="feeder", columns="meter_class", values="gnn_acc").to_string())
    print("\nSaved results/phase16_pmu_comparison.csv")


if __name__ == "__main__":
    main()
