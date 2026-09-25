"""
DSO Service 7 - Phase 9
Does network SIZE affect the noise-sensitivity of the correlation+MST
baseline? Tests the same method across several SimBench LV feeders of
increasing size.
"""

import pandas as pd
import simbench as sb

from phase1_2_baseline import correlation_mst, evaluate
from phase5_simbench import ground_truth_graph, simulate_timeseries_simbench, add_realistic_meter_noise

FEEDERS = [
    "1-LV-rural1--0-sw",    # ~15 buses
    "1-LV-semiurb4--0-sw",  # ~44 buses
    "1-LV-urban6--0-sw",    # ~58 buses
    "1-LV-rural2--0-sw",    # ~97 buses
]


def main():
    rows = []
    for code in FEEDERS:
        print(f"\n=== {code} ===")
        net = sb.get_simbench_net(code)
        true_graph = ground_truth_graph(net)
        n_buses = true_graph.number_of_nodes()
        n_edges = true_graph.number_of_edges()
        print(f"buses={n_buses} edges={n_edges}")

        week_start = 96 * 7 * 26
        voltage_df = simulate_timeseries_simbench(net, n_steps=672, start=week_start)
        if len(voltage_df) < 50:
            print("insufficient valid power-flow snapshots, skipping")
            continue

        mst_clean, _ = correlation_mst(voltage_df)
        m_clean = evaluate(mst_clean, true_graph, voltage_df.columns)

        noisy = add_realistic_meter_noise(voltage_df)
        mst_noisy, _ = correlation_mst(noisy)
        m_noisy = evaluate(mst_noisy, true_graph, voltage_df.columns)

        print(f"clean recon_acc={m_clean['reconstruction_accuracy']:.3f}  "
              f"noisy (Class 0.5) recon_acc={m_noisy['reconstruction_accuracy']:.3f}")

        rows.append({
            "feeder": code,
            "n_buses": n_buses,
            "n_edges": n_edges,
            "clean_reconstruction_accuracy": m_clean["reconstruction_accuracy"],
            "noisy_class0.5_reconstruction_accuracy": m_noisy["reconstruction_accuracy"],
        })

    results = pd.DataFrame(rows)
    results.to_csv("results/phase9_scale_comparison.csv", index=False)
    print("\n" + results.to_string(index=False))
    print("\nSaved results/phase9_scale_comparison.csv")


if __name__ == "__main__":
    main()