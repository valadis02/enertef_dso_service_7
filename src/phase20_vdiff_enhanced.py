"""
DSO Service 7 - Phase 20
Two enhancements to the phase19 voltage-difference method:

  1. NOISE DEBIASING. We showed Var(Vi - Vj + noise) = Var(Vi - Vj) +
     2*sigma^2. If we can ESTIMATE sigma^2, we can subtract it and
     recover the clean electrical distance. sigma^2 is estimable from
     the data itself: meter noise is white (uncorrelated in time) while
     the true voltage signal is smooth/autocorrelated, so the
     high-frequency part of each bus's own time series is essentially
     pure noise. A standard estimator uses the lag-1 difference:
         Var(v[t] - v[t-1]) ~= 2*sigma^2  (for a smooth signal + white noise)

  2. TEMPORAL SMOOTHING. Since the true voltage signal is smooth in time
     and the noise is white, a short rolling mean over w samples cuts
     noise variance by ~w while leaving the structural signal largely
     intact -- directly improving the signal-to-noise ratio BEFORE the
     variance computation.
"""

import numpy as np
import pandas as pd
import networkx as nx

from phase19_voltage_difference import voltage_difference_variance_matrix, vdiff_topology
from phase1_2_baseline import correlation_mst, evaluate
from phase5_simbench import ground_truth_graph, simulate_timeseries_simbench
import simbench as sb

FEEDERS = ["1-LV-rural1--0-sw", "1-LV-semiurb4--0-sw", "1-LV-urban6--0-sw"]


def estimate_noise_variance(voltage_df):
    """Estimate per-bus white-noise variance from the lag-1 difference.
    For a smooth signal s plus white noise n:
        Var(x[t]-x[t-1]) = Var(s[t]-s[t-1]) + 2*sigma^2 ~= 2*sigma^2
    when the smooth part changes slowly between consecutive samples."""
    diffs = voltage_df.diff().dropna()
    sigma2 = diffs.var() / 2.0
    return sigma2


def vdiff_topology_debiased(voltage_df, debias=True, smooth_window=0):
    """Voltage-difference MST with optional noise debiasing and temporal
    smoothing."""
    df = voltage_df
    if smooth_window and smooth_window > 1:
        df = df.rolling(window=smooth_window, center=True, min_periods=1).mean()

    D, nodes = voltage_difference_variance_matrix(df)

    if debias:
        sigma2 = estimate_noise_variance(df)
        s2 = sigma2[nodes].values
        # subtract the additive noise contribution: sigma_i^2 + sigma_j^2
        offset = s2[:, None] + s2[None, :]
        D = D - offset
        np.fill_diagonal(D, 0.0)
        # distances must stay non-negative to be meaningful
        D = np.clip(D, 0.0, None)

    n = len(nodes)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(nodes[i], nodes[j], weight=float(D[i, j]))
    return nx.minimum_spanning_tree(G, weight="weight")


def main():
    configs = [
        ("V-diff (phase19)", dict(debias=False, smooth_window=0)),
        ("+ debias", dict(debias=True, smooth_window=0)),
        ("+ smoothing(w=5)", dict(debias=False, smooth_window=5)),
        ("+ smoothing(w=15)", dict(debias=False, smooth_window=15)),
        ("+ both (w=5)", dict(debias=True, smooth_window=5)),
        ("+ both (w=15)", dict(debias=True, smooth_window=15)),
    ]

    rows = []
    for code in FEEDERS:
        net = sb.get_simbench_net(code)
        true_graph = ground_truth_graph(net)
        week_start = 96 * 7 * 26
        voltage_df = simulate_timeseries_simbench(net, n_steps=672, start=week_start)

        for label, sigma in [("Class 0.1 (PMU)", 0.0005),
                              ("Class 0.5 (typical)", 0.0025),
                              ("Class 1 (cheap)", 0.005)]:
            rng = np.random.default_rng(2026)
            data = voltage_df + rng.normal(0, sigma, size=voltage_df.shape)

            mst_base, _ = correlation_mst(data)
            m_base = evaluate(mst_base, true_graph, voltage_df.columns)
            row = {"feeder": code, "condition": label, "baseline": m_base["reconstruction_accuracy"]}

            for cfg_label, kwargs in configs:
                mst = vdiff_topology_debiased(data, **kwargs)
                m = evaluate(mst, true_graph, voltage_df.columns)
                row[cfg_label] = m["reconstruction_accuracy"]

            rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv("results/phase20_vdiff_enhanced.csv", index=False)

    print("=== Mean reconstruction accuracy across the 3 real feeders ===\n")
    cols = ["baseline"] + [c for c, _ in configs]
    summary = results.groupby("condition")[cols].mean()
    print(summary.to_string())
    print("\nSaved results/phase20_vdiff_enhanced.csv")


if __name__ == "__main__":
    main()
