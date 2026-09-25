"""
DSO Service 7 - Phase 19
Voltage-difference variance method.

KEY IDEA. Every prior method struggled because of the common-mode term
in the linear grid model (V = RP + XQ + 1*1^T + N, per the tutorial
paper): all bus voltages move together, so raw correlations are all
~0.99 and the structural signal is a tiny residual (phase12 finding).

Instead of removing that common term statistically (phase7 -- failed),
CANCEL IT ALGEBRAICALLY by looking at voltage DIFFERENCES:

    V_i(t) - V_j(t)   -> the common term cancels EXACTLY

Under LinDistFlow, the variance of that difference is proportional to
the ELECTRICAL DISTANCE between i and j (the impedance of the path
between them): buses that are electrically adjacent have the smallest
Var(V_i - V_j). So the predicted topology is the MINIMUM spanning tree
over Var(V_i - V_j).

WHY IT SHOULD RESIST NOISE. With independent meter noise of variance
sigma^2 at every bus:

    Var(V_i - V_j + n_i - n_j) = Var(V_i - V_j) + 2*sigma^2

The noise adds the SAME CONSTANT (2*sigma^2) to every pair. A constant
offset does not change which pairs have the smallest values, so the MST
ranking is largely preserved -- fundamentally different from
correlation, where noise attenuates the signal multiplicatively and
destroys the ranking.
"""

import numpy as np
import pandas as pd
import networkx as nx

from phase1_2_baseline import correlation_mst, evaluate
from phase5_simbench import ground_truth_graph, simulate_timeseries_simbench, add_realistic_meter_noise
import simbench as sb

FEEDERS = ["1-LV-rural1--0-sw", "1-LV-semiurb4--0-sw", "1-LV-urban6--0-sw"]


def voltage_difference_variance_matrix(voltage_df):
    """Compute Var(V_i - V_j) for all pairs, vectorized.

    Var(Vi - Vj) = Var(Vi) + Var(Vj) - 2*Cov(Vi, Vj)
    """
    cov = voltage_df.cov().values
    var = np.diag(cov)
    D = var[:, None] + var[None, :] - 2 * cov
    np.fill_diagonal(D, 0.0)
    return D, voltage_df.columns.tolist()


def vdiff_topology(voltage_df):
    """MST over Var(V_i - V_j) -- minimum, since adjacent buses have the
    smallest voltage-difference variance."""
    D, nodes = voltage_difference_variance_matrix(voltage_df)
    n = len(nodes)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(nodes[i], nodes[j], weight=float(D[i, j]))
    mst = nx.minimum_spanning_tree(G, weight="weight")
    return mst


def main():
    print("=== Voltage-difference variance method vs baseline ===\n")
    rows = []
    for code in FEEDERS:
        net = sb.get_simbench_net(code)
        true_graph = ground_truth_graph(net)
        week_start = 96 * 7 * 26
        voltage_df = simulate_timeseries_simbench(net, n_steps=672, start=week_start)

        print(f"{code} (buses={true_graph.number_of_nodes()}):")
        for label, sigma in [("clean", 0.0),
                              ("Class 0.1 (PMU)", 0.0005),
                              ("Class 0.5 (typical)", 0.0025),
                              ("Class 1 (cheap)", 0.005)]:
            if sigma == 0:
                data = voltage_df
            else:
                rng = np.random.default_rng(2026)
                data = voltage_df + rng.normal(0, sigma, size=voltage_df.shape)

            mst_base, _ = correlation_mst(data)
            m_base = evaluate(mst_base, true_graph, voltage_df.columns)

            mst_vd = vdiff_topology(data)
            m_vd = evaluate(mst_vd, true_graph, voltage_df.columns)

            print(f"  {label:22s}: baseline={m_base['reconstruction_accuracy']:.3f}   "
                  f"V-diff variance={m_vd['reconstruction_accuracy']:.3f}")

            rows.append({"feeder": code, "n_buses": true_graph.number_of_nodes(),
                         "condition": label, "sigma": sigma,
                         "baseline_acc": m_base["reconstruction_accuracy"],
                         "vdiff_acc": m_vd["reconstruction_accuracy"]})
        print()

    results = pd.DataFrame(rows)
    results.to_csv("results/phase19_vdiff_results.csv", index=False)
    print("=== Mean across feeders ===")
    print(results.groupby("condition")[["baseline_acc", "vdiff_acc"]].mean())
    print("\nSaved results/phase19_vdiff_results.csv")


if __name__ == "__main__":
    main()
