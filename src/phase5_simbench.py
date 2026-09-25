"""
DSO Service 7 - Phase 5
Transition from synthetic IEEE test feeders to SimBench: a real, published
German LV distribution network topology with real annual load/generation
time-series (15-min resolution, one year -> 35,136 timesteps), including
PV generation. This is the intermediate step before real EnerTEF data
(per the implementation plan).

SimBench code used: 1-LV-rural1--0-sw (rural LV feeder, 15 buses)
Reference: https://simbench.de
"""

import numpy as np
import pandas as pd
import networkx as nx
import pandapower as pp
import simbench as sb

from phase1_2_baseline import correlation_mst, evaluate

SIMBENCH_CODE = "1-LV-rural1--0-sw"


def load_simbench_feeder(code=SIMBENCH_CODE):
    net = sb.get_simbench_net(code)
    return net


def ground_truth_graph(net):
    """LV-side topology only (buses connected via lines, excluding the
    MV-side transformer bus)."""
    lv_buses = set(net.load["bus"]) | set(net.sgen["bus"]) | set(net.line["from_bus"]) | set(net.line["to_bus"])
    G = nx.Graph()
    G.add_nodes_from(lv_buses)
    for _, row in net.line.iterrows():
        if row["in_service"] and row["from_bus"] in lv_buses and row["to_bus"] in lv_buses:
            G.add_edge(int(row["from_bus"]), int(row["to_bus"]))
    return G


def simulate_timeseries_simbench(net, n_steps=1000, start=0, seed=42):
    """
    Run power-flow snapshots using SimBench's real annual load/generation
    profiles (subsampled to n_steps consecutive 15-min intervals starting
    at `start`, e.g. one week ~ 672 steps, one month ~ 2880 steps).
    """
    abs_vals = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    load_p = abs_vals[("load", "p_mw")]
    load_q = abs_vals[("load", "q_mvar")]
    sgen_p = abs_vals.get(("sgen", "p_mw"))

    end = min(start + n_steps, len(load_p))
    voltage_records = []

    for t in range(start, end):
        net.load["p_mw"] = load_p.loc[t].values
        net.load["q_mvar"] = load_q.loc[t].values
        if sgen_p is not None and len(net.sgen) > 0:
            net.sgen["p_mw"] = sgen_p.loc[t].values

        try:
            pp.runpp(net)
        except Exception:
            continue

        voltage_records.append(net.res_bus["vm_pu"].copy())

    voltage_df = pd.DataFrame(voltage_records).reset_index(drop=True)
    return voltage_df


def add_realistic_meter_noise(voltage_df, meter_class_sigma=0.0025, seed=2026):
    """Default: ANSI C12.20 Class 0.5 typical utility smart meter (see
    phase6_calibrated_noise.py for the derivation: sigma = 0.5%FS / 2)."""
    rng = np.random.default_rng(seed)
    return voltage_df + rng.normal(0, meter_class_sigma, size=voltage_df.shape)


def main():
    print(f"=== Loading SimBench feeder: {SIMBENCH_CODE} ===")
    net = load_simbench_feeder()
    true_graph = ground_truth_graph(net)
    print(f"LV buses: {true_graph.number_of_nodes()}, edges: {true_graph.number_of_edges()}")
    print(f"Loads: {len(net.load)}, PV (sgen): {len(net.sgen)}")

    # one week of real 15-min data (summer, e.g. week ~26 to catch PV effects)
    week_start = 96 * 7 * 26  # ~week 26 of the year
    print(f"\n=== Simulating 1 week (672 timesteps) starting at index {week_start} ===")
    voltage_df = simulate_timeseries_simbench(net, n_steps=672, start=week_start)
    print(f"Simulated {len(voltage_df)} valid power-flow snapshots")

    print("\n=== Clean (no added instrument noise) ===")
    mst_clean, _ = correlation_mst(voltage_df)
    metrics_clean = evaluate(mst_clean, true_graph, voltage_df.columns)
    print(f"precision={metrics_clean['precision']:.3f} recall={metrics_clean['recall']:.3f} "
          f"reconstruction_accuracy={metrics_clean['reconstruction_accuracy']:.3f}")

    print("\n=== With realistic Class 0.5 smart-meter noise (sigma=0.0025 p.u.) ===")
    noisy_df = add_realistic_meter_noise(voltage_df, meter_class_sigma=0.0025)
    mst_noisy, _ = correlation_mst(noisy_df)
    metrics_noisy = evaluate(mst_noisy, true_graph, voltage_df.columns)
    print(f"precision={metrics_noisy['precision']:.3f} recall={metrics_noisy['recall']:.3f} "
          f"reconstruction_accuracy={metrics_noisy['reconstruction_accuracy']:.3f}")

    results = pd.DataFrame([
        {**metrics_clean, "condition": "clean_simbench"},
        {**metrics_noisy, "condition": "class0.5_noise_simbench"},
    ])
    results.to_csv("results/phase5_simbench_results.csv", index=False)
    voltage_df.to_csv("data/simbench_voltage_timeseries.csv", index=False)
    print("\nSaved results/phase5_simbench_results.csv, data/simbench_voltage_timeseries.csv")


if __name__ == "__main__":
    main()
    