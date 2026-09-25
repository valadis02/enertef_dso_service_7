"""
DSO Service 7 - Phase 14
Tuned training on the BEST training data regime found so far (calibrated
synthetic, idio_noise_std=0.03, matching real correlation-spread stats
per phase12b). Three changes vs phase11's original run:
  1. More networks (100 -> 250) and more epochs (15 -> 40)
  2. Deeper edge MLP (1 hidden layer -> 2 hidden layers, with dropout)
  3. Proper class balancing: neg_sample_ratio swept (was fixed at 5)
"""

import pandas as pd
import torch

from phase10a_synthetic_generator import generate_training_set
from phase11_full_gnn import train_gnn, predict_topology_gnn
from phase1_2_baseline import correlation_mst, evaluate as evaluate_graph
from phase5_simbench import ground_truth_graph, simulate_timeseries_simbench, add_realistic_meter_noise
import simbench as sb

FEEDERS = ["1-LV-rural1--0-sw", "1-LV-semiurb4--0-sw", "1-LV-urban6--0-sw"]


def evaluate_on_test_feeders(model, label):
    rows = []
    for code in FEEDERS:
        net = sb.get_simbench_net(code)
        true_graph = ground_truth_graph(net)
        week_start = 96 * 7 * 26
        voltage_df = simulate_timeseries_simbench(net, n_steps=672, start=week_start)
        noisy_df = add_realistic_meter_noise(voltage_df)

        mst_base, _ = correlation_mst(noisy_df)
        m_base = evaluate_graph(mst_base, true_graph, voltage_df.columns)

        mst_gnn = predict_topology_gnn(noisy_df, model)
        m_gnn = evaluate_graph(mst_gnn, true_graph, voltage_df.columns)

        rows.append({"config": label, "feeder": code, "n_buses": true_graph.number_of_nodes(),
                      "baseline": m_base["reconstruction_accuracy"],
                      "gnn": m_gnn["reconstruction_accuracy"]})
    return rows


def main():
    print("=== Generating calibrated synthetic training set (250 networks, idio_noise_std=0.03) ===")
    dataset = generate_training_set(n_networks=250, n_steps=300, seed=42,
                                     size_range=(10, 50), idio_noise_std=0.03)
    print(f"Generated {len(dataset)} networks")

    all_rows = []

    print("\n" + "="*60)
    print("Config A: baseline architecture, more epochs (40) and networks")
    print("="*60)
    model_a = train_gnn(dataset, n_epochs=40, neg_sample_ratio=5)
    rows_a = evaluate_on_test_feeders(model_a, "A_more_epochs_networks")
    for r in rows_a:
        print(f"  {r['feeder']}: baseline={r['baseline']:.3f} gnn={r['gnn']:.3f}")
    all_rows.extend(rows_a)

    print("\n" + "="*60)
    print("Config B: + deeper edge MLP (2 hidden layers w/ dropout)")
    print("="*60)
    model_b = train_gnn(dataset, n_epochs=40, neg_sample_ratio=5, edge_mlp_hidden=(64, 32))
    rows_b = evaluate_on_test_feeders(model_b, "B_deeper_mlp")
    for r in rows_b:
        print(f"  {r['feeder']}: baseline={r['baseline']:.3f} gnn={r['gnn']:.3f}")
    all_rows.extend(rows_b)

    print("\n" + "="*60)
    print("Config C: + class balancing sweep (neg_sample_ratio=2, i.e. less imbalanced)")
    print("="*60)
    model_c = train_gnn(dataset, n_epochs=40, neg_sample_ratio=2, edge_mlp_hidden=(64, 32))
    rows_c = evaluate_on_test_feeders(model_c, "C_balanced_2to1")
    for r in rows_c:
        print(f"  {r['feeder']}: baseline={r['baseline']:.3f} gnn={r['gnn']:.3f}")
    all_rows.extend(rows_c)

    print("\n" + "="*60)
    print("Config D: neg_sample_ratio=10 (more imbalanced, closer to true ratio)")
    print("="*60)
    model_d = train_gnn(dataset, n_epochs=40, neg_sample_ratio=10, edge_mlp_hidden=(64, 32))
    rows_d = evaluate_on_test_feeders(model_d, "D_imbalanced_10to1")
    for r in rows_d:
        print(f"  {r['feeder']}: baseline={r['baseline']:.3f} gnn={r['gnn']:.3f}")
    all_rows.extend(rows_d)

    results = pd.DataFrame(all_rows)
    results.to_csv("results/phase14_tuning_results.csv", index=False)

    print("\n" + "="*60)
    print("SUMMARY (mean gnn recon_acc per config)")
    print("="*60)
    summary = results.groupby("config")["gnn"].mean().sort_values(ascending=False)
    print(summary)

    best_config = summary.idxmax()
    print(f"\nBest config: {best_config}")
    print("\nSaved results/phase14_tuning_results.csv")


if __name__ == "__main__":
    main()