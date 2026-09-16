"""
DSO Service 7 - Phase 3
Robustness testing for the correlation + MST baseline topology identification:
  1. Gaussian noise on voltage measurements
  2. Missing data (random measurement dropout)
  3. Switching event (mid-stream topology change) detection
"""

import numpy as np
import pandas as pd
import networkx as nx
import pandapower as pp
import pandapower.networks as pn

from phase1_2_baseline import (
    load_feeder,
    ground_truth_graph,
    simulate_timeseries,
    correlation_mst,
    evaluate,
)

RESULTS_PATH = "results/phase3_robustness_results.csv"


# ---------------------------------------------------------------------------
# 1. Noise robustness
# ---------------------------------------------------------------------------

def test_noise_robustness(voltage_df, true_graph, noise_levels=(0.0, 0.001, 0.005, 0.01, 0.02, 0.05)):
    """Add Gaussian noise (in p.u.) to voltage measurements and re-evaluate."""
    rng = np.random.default_rng(123)
    rows = []
    for sigma in noise_levels:
        noisy = voltage_df + rng.normal(0, sigma, size=voltage_df.shape)
        mst, _ = correlation_mst(noisy)
        metrics = evaluate(mst, true_graph, voltage_df.columns)
        metrics["noise_sigma_pu"] = sigma
        rows.append(metrics)
        print(f"noise sigma={sigma:.3f} pu -> precision={metrics['precision']:.3f} "
              f"recall={metrics['recall']:.3f} recon_acc={metrics['reconstruction_accuracy']:.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Missing data robustness
# ---------------------------------------------------------------------------

def test_missing_data(voltage_df, true_graph, missing_fractions=(0.0, 0.05, 0.1, 0.2, 0.3)):
    """Randomly null out a fraction of (timestep, bus) entries, then
    reconstruct correlations using pairwise-complete observations
    (pandas .corr() already does this by default)."""
    rng = np.random.default_rng(456)
    rows = []
    for frac in missing_fractions:
        masked = voltage_df.copy()
        mask = rng.random(masked.shape) < frac
        masked = masked.mask(mask)
        mst, _ = correlation_mst(masked)
        metrics = evaluate(mst, true_graph, voltage_df.columns)
        metrics["missing_fraction"] = frac
        rows.append(metrics)
        print(f"missing={frac:.0%} -> precision={metrics['precision']:.3f} "
              f"recall={metrics['recall']:.3f} recon_acc={metrics['reconstruction_accuracy']:.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Switching event detection
# ---------------------------------------------------------------------------

def simulate_switching_event(net, n_steps_before=250, n_steps_after=250, seed=42):
    """
    Simulate a topology change mid-stream: open one normally-closed line
    (a real DSO reconfiguration/switching event) partway through the
    time-series, and return voltage data for the two segments plus both
    ground-truth graphs.
    """
    true_graph_before = ground_truth_graph(net)

    volt_before, _ = simulate_timeseries(net, n_steps=n_steps_before, seed=seed)

    # pick an in-service line to open (avoid disconnecting the feeder into >1
    # component if possible -> choose a line where both endpoints keep degree>=1
    # elsewhere; for a radial feeder any edge removal splits the tree, which is
    # realistic for a real switching operation isolating a downstream section)
    candidate_lines = net.line[net.line["in_service"]].index.tolist()
    line_to_open = candidate_lines[len(candidate_lines) // 2]  # a mid-feeder line
    opened_edge = (int(net.line.loc[line_to_open, "from_bus"]),
                   int(net.line.loc[line_to_open, "to_bus"]))

    net.line.loc[line_to_open, "in_service"] = False
    true_graph_after = ground_truth_graph(net)

    volt_after, _ = simulate_timeseries(net, n_steps=n_steps_after, seed=seed + 1)

    # restore for any later use
    net.line.loc[line_to_open, "in_service"] = True

    return volt_before, volt_after, true_graph_before, true_graph_after, opened_edge


def test_switching_detection(net, window=100):
    """
    Detect a topology change by comparing the predicted MST on a sliding
    window of recent measurements before vs after the switching event.
    """
    volt_before, volt_after, true_before, true_after, opened_edge = simulate_switching_event(net)

    print(f"\nSwitching event: line {opened_edge} opened mid-stream")

    mst_before, _ = correlation_mst(volt_before.tail(window))
    mst_after, _ = correlation_mst(volt_after.head(window))

    edges_before = {frozenset(e) for e in mst_before.edges()}
    edges_after = {frozenset(e) for e in mst_after.edges()}

    changed_edges = edges_before.symmetric_difference(edges_after)
    detected = len(changed_edges) > 0
    opened_edge_detected = frozenset(opened_edge) in edges_before and frozenset(opened_edge) not in edges_after

    metrics_before = evaluate(mst_before, true_before, volt_before.columns)
    metrics_after = evaluate(mst_after, true_after, volt_after.columns)

    print(f"Change detected (any edge diff): {detected}")
    print(f"Opened edge specifically flagged as removed: {opened_edge_detected}")
    print(f"Accuracy just before switch: {metrics_before['reconstruction_accuracy']:.3f}")
    print(f"Accuracy just after switch:  {metrics_after['reconstruction_accuracy']:.3f}")

    return {
        "opened_edge": str(opened_edge),
        "change_detected": detected,
        "opened_edge_correctly_flagged": opened_edge_detected,
        "n_edges_changed": len(changed_edges),
        "reconstruction_accuracy_before": metrics_before["reconstruction_accuracy"],
        "reconstruction_accuracy_after": metrics_after["reconstruction_accuracy"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    net = load_feeder()
    true_graph = ground_truth_graph(net)
    voltage_df, _ = simulate_timeseries(net, n_steps=500)

    print("=== 1. Noise robustness ===")
    noise_results = test_noise_robustness(voltage_df, true_graph)

    print("\n=== 2. Missing data robustness ===")
    missing_results = test_missing_data(voltage_df, true_graph)

    print("\n=== 3. Switching event detection ===")
    switching_result = test_switching_detection(net)

    noise_results.to_csv("results/phase3_noise_results.csv", index=False)
    missing_results.to_csv("results/phase3_missing_data_results.csv", index=False)
    pd.DataFrame([switching_result]).to_csv("results/phase3_switching_results.csv", index=False)

    print("\nSaved: results/phase3_noise_results.csv, "
          "results/phase3_missing_data_results.csv, "
          "results/phase3_switching_results.csv")


if __name__ == "__main__":
    main()

    