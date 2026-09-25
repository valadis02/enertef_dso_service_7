"""
DSO Service 7 - Phase 10a
Fast synthetic radial feeder generator for GNN training data.

Running full pandapower AC power-flow for hundreds of random topologies
would be slow. Instead, use the LinDistFlow (linearized voltage-drop)
approximation, which is fast (pure linear algebra) and is exactly the
kind of approximate physics model that Bolognani et al. use as the basis
for their identification method:

    v_i(t) ~= 1 - sum_{e in path(slack, i)} r_e * P_downstream_e(t)

where P_downstream_e(t) is the total real power flowing through edge e
at time t (sum of all loads in the subtree beyond e).
"""

import numpy as np
import pandas as pd
import networkx as nx


def random_radial_tree(n_buses, rng, max_children=3):
    """Build a random radial (tree) feeder: bus 0 = slack/substation.
    Each new bus attaches to a random existing bus that still has
    capacity for more children (keeps a feeder-like branching factor)."""
    G = nx.Graph()
    G.add_node(0)
    degree_cap = {0: max_children + 1}  # slack can have a few more
    for i in range(1, n_buses):
        candidates = [n for n in G.nodes if G.degree[n] < degree_cap.get(n, max_children)]
        parent = rng.choice(candidates)
        G.add_edge(parent, i)
        degree_cap[i] = max_children
    return G


def assign_random_impedances(G, rng, r_low=0.01, r_high=0.05):
    for u, v in G.edges():
        G[u][v]["r"] = rng.uniform(r_low, r_high)
    return G


def simulate_lindistflow(G, n_steps, rng, meter_noise_sigma=0.0025, correlated_load_noise=True,
                          idio_noise_std=0.3, common_factor_amplitude=0.2):
    """
    Simulate n_steps of approximate voltage measurements using LinDistFlow.
    Every non-slack bus gets a load with a shared daily pattern (so loads
    are naturally correlated, like real households) plus idiosyncratic
    noise, matching what we observed with real SimBench data.

    idio_noise_std controls how much each load deviates independently
    from the shared/common pattern -- lower values -> higher baseline
    pairwise correlation between ALL buses (shared signal dominates),
    matching real household behavior (see phase12 domain-gap finding:
    real SimBench correlations average ~0.35, much higher than a naively
    high idio_noise_std produces).
    """
    nodes = list(G.nodes())
    n = len(nodes)
    slack = 0

    # base load per bus (random, roughly realistic small residential loads in "p.u. power" units)
    base_load = {b: rng.uniform(0.5, 2.0) for b in nodes if b != slack}

    # precompute path from slack to every bus, and the tree structure
    paths = nx.single_source_dijkstra_path(G, slack, weight=None)

    # precompute, for every edge, the set of downstream buses
    # (root the tree at slack, then each edge (parent, child) downstream = subtree at child)
    T = nx.bfs_tree(G, slack)
    downstream = {}
    for edge in G.edges():
        u, v = edge
        # determine which is the parent (closer to slack)
        if len(paths[u]) < len(paths[v]):
            parent, child = u, v
        else:
            parent, child = v, u
        downstream[(parent, child)] = nx.descendants(T, child) | {child}

    voltage_records = []
    for t in range(n_steps):
        common_factor = 1.0 + common_factor_amplitude * np.sin(2 * np.pi * t / 96) + rng.normal(0, 0.05)
        common_factor = max(common_factor, 0.05)

        if correlated_load_noise:
            idio_noise = rng.normal(1.0, idio_noise_std, size=n)
        else:
            idio_noise = np.ones(n)

        p_load = {}
        for i, b in enumerate(nodes):
            if b == slack:
                continue
            p_load[b] = max(base_load[b] * common_factor * idio_noise[i], 0)

        # voltage at each bus = 1 - sum of r_e * P_downstream_e along its path
        v = {slack: 1.0}
        for b in nodes:
            if b == slack:
                continue
            path = paths[b]
            drop = 0.0
            for k in range(len(path) - 1):
                u, w = path[k], path[k + 1]
                edge_key = (u, w) if (u, w) in downstream else (w, u)
                r = G[u][w]["r"]
                p_down = sum(p_load.get(x, 0) for x in downstream[edge_key])
                drop += r * p_down
            v[b] = 1.0 - 0.01 * drop  # scale factor to keep drops small/realistic

        voltage_records.append([v[b] for b in nodes])

    voltage_df = pd.DataFrame(voltage_records, columns=nodes)

    if meter_noise_sigma > 0:
        voltage_df = voltage_df + rng.normal(0, meter_noise_sigma, size=voltage_df.shape)

    return voltage_df


def generate_training_set(n_networks=60, n_steps=300, seed=42, size_range=(10, 50), idio_noise_std=0.03):
    """Generate a list of (voltage_df, true_graph) pairs for training/eval.

    idio_noise_std default changed from 0.3 -> 0.03 following the phase12
    domain-gap diagnosis: real SimBench data has a MUCH tighter clean
    correlation spread (std~0.01) than a naive high-idiosyncratic-noise
    synthetic model produces (std~0.26 at idio_noise_std=0.3). 0.03
    empirically matches the real spread closely (see phase12b sweep)."""
    rng = np.random.default_rng(seed)
    dataset = []
    for i in range(n_networks):
        n_buses = int(rng.integers(size_range[0], size_range[1] + 1))
        G = random_radial_tree(n_buses, rng)
        G = assign_random_impedances(G, rng)
        voltage_df = simulate_lindistflow(G, n_steps, rng, idio_noise_std=idio_noise_std)
        dataset.append((voltage_df, G))
    return dataset


if __name__ == "__main__":
    # quick smoke test
    rng = np.random.default_rng(0)
    G = random_radial_tree(20, rng)
    G = assign_random_impedances(G, rng)
    voltage_df = simulate_lindistflow(G, 300, rng)
    print(f"Generated network: {G.number_of_nodes()} buses, {G.number_of_edges()} edges")
    print(f"Voltage range: {voltage_df.values.min():.4f} - {voltage_df.values.max():.4f}")
    print(voltage_df.head())