"""
DSO Service 7 - Event-based sampling
Instead of computing correlation over ALL timesteps uniformly, select only
the timesteps with the LARGEST load changes (biggest voltage swings).
Rationale (Luan et al. 2024, IET Energy Systems Integration): most
correlation-based errors come from "low difference" between electrically
close buses -- the structural signal is weak relative to noise during
calm periods. Focusing on high-swing ("event") timesteps raises the
signal-to-noise ratio before the noise ever gets a chance to blur it.
"""

import numpy as np
import pandas as pd
import networkx as nx
import pandapower as pp

from phase1_2_baseline import load_feeder, ground_truth_graph, correlation_mst, evaluate


def simulate_timeseries_with_events(net, n_steps=2000, seed=42):
    """
    Similar to simulate_timeseries, but with MORE pronounced load swings
    (bigger common_factor amplitude + occasional large step events) so
    that there is a meaningful subset of high-SNR timesteps to select
    from. Returns voltage_df and a companion 'load_change' series used
    to rank timesteps by event magnitude.
    """
    rng = np.random.default_rng(seed)
    base_p = net.load["p_mw"].copy()
    base_q = net.load["q_mvar"].copy()

    voltage_records = []
    total_load_records = []

    prev_total = None
    for t in range(n_steps):
        # base daily-ish pattern + occasional large random step "events"
        # (e.g. EV charger turning on, large motor starting)
        common_factor = 1.0 + 0.15 * np.sin(2 * np.pi * t / 96) + rng.normal(0, 0.05)
        if rng.random() < 0.08:  # ~8% of timesteps are "events"
            common_factor += rng.choice([-1, 1]) * rng.uniform(0.3, 0.6)
        common_factor = max(common_factor, 0.05)

        noise = rng.normal(1.0, 0.25, size=len(net.load))
        net.load["p_mw"] = np.clip(base_p * common_factor * noise, 0, None)
        net.load["q_mvar"] = np.clip(base_q * common_factor * noise, 0, None)

        try:
            pp.runpp(net)
        except Exception:
            continue

        voltage_records.append(net.res_bus["vm_pu"].copy())
        total = net.load["p_mw"].sum()
        total_load_records.append(total)

    net.load["p_mw"] = base_p
    net.load["q_mvar"] = base_q

    voltage_df = pd.DataFrame(voltage_records).reset_index(drop=True)
    total_load = pd.Series(total_load_records).reset_index(drop=True)
    # "event magnitude" = |change in total load from previous timestep|
    load_change = total_load.diff().abs().fillna(0)

    return voltage_df, load_change


def select_event_timesteps(voltage_df, load_change, top_fraction=0.3):
    """Keep only the top_fraction of timesteps with the largest load change."""
    threshold = load_change.quantile(1 - top_fraction)
    idx = load_change[load_change >= threshold].index
    return voltage_df.loc[idx].reset_index(drop=True)


def compare_uniform_vs_event(voltage_df, load_change, true_graph, noise_sigma, seed=999,
                              top_fraction=0.3):
    rng = np.random.default_rng(seed)
    noisy_full = voltage_df + rng.normal(0, noise_sigma, size=voltage_df.shape)

    # uniform: use all timesteps (same total budget would be unfair, so
    # also compare against a matching-size RANDOM subset for a fair test)
    mst_uniform_all, _ = correlation_mst(noisy_full)
    metrics_uniform_all = evaluate(mst_uniform_all, true_graph, voltage_df.columns)

    n_event = int(len(voltage_df) * top_fraction)
    random_idx = rng.choice(voltage_df.index, size=n_event, replace=False)
    mst_random_subset, _ = correlation_mst(noisy_full.loc[random_idx])
    metrics_random_subset = evaluate(mst_random_subset, true_graph, voltage_df.columns)

    # event-based: same noisy data, but only the high-load-change timesteps
    event_voltage = select_event_timesteps(noisy_full, load_change, top_fraction)
    mst_event, _ = correlation_mst(event_voltage)
    metrics_event = evaluate(mst_event, true_graph, voltage_df.columns)

    print(f"noise sigma={noise_sigma} p.u.  (event subset size={n_event}/{len(voltage_df)})")
    print(f"  All timesteps (n={len(voltage_df)})      : recon_acc={metrics_uniform_all['reconstruction_accuracy']:.3f}")
    print(f"  Random subset (n={n_event}, same size)    : recon_acc={metrics_random_subset['reconstruction_accuracy']:.3f}")
    print(f"  Event-based subset (n={n_event}, top-{top_fraction:.0%} load change): recon_acc={metrics_event['reconstruction_accuracy']:.3f}")

    return metrics_uniform_all, metrics_random_subset, metrics_event


def main():
    net = load_feeder()
    true_graph = ground_truth_graph(net)

    print("Simulating time-series with pronounced load events...")
    voltage_df, load_change = simulate_timeseries_with_events(net, n_steps=2000)
    print(f"Simulated {len(voltage_df)} snapshots\n")

    rows = []
    for sigma in (0.0, 0.001, 0.005, 0.01, 0.02):
        m_all, m_rand, m_event = compare_uniform_vs_event(voltage_df, load_change, true_graph, sigma)
        for label, m in [("all_timesteps", m_all), ("random_subset", m_rand), ("event_based", m_event)]:
            m["method"] = label
            m["noise_sigma_pu"] = sigma
            rows.append(m)
        print()

    results = pd.DataFrame(rows)
    results.to_csv("results/phase5_event_based_results.csv", index=False)
    print("Saved results/phase5_event_based_results.csv")


if __name__ == "__main__":
    main()