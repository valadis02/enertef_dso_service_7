"""Networks and simulated measurements.

Three sources:
  - SimBench German LV feeder models, with published topology and annual
    load/PV profiles at 15-minute resolution. These are network *models*;
    the power flow and the meter noise here are simulated, so nothing in
    this package constitutes field validation.
  - The IEEE 33-bus test feeder.
  - Randomly generated radial feeders, for distribution-level statements
    rather than single-network results.

Meter noise follows the accuracy classes of ANSI C12.20 / IEC 61557-12,
where the class is a percentage of full scale at 95% confidence (~2 sigma).
"""

import numpy as np
import pandas as pd
import networkx as nx
import pandapower as pp

# sigma in p.u. for voltage, degrees for angle
METER_CLASSES = {
    "class_1":   {"voltage": 0.005,  "angle": None},
    "class_0.5": {"voltage": 0.0025, "angle": None},
    "class_0.5+angle": {"voltage": 0.0025, "angle": 0.01},
    "good+angle":      {"voltage": 0.0005, "angle": 0.002},
    "clean":     {"voltage": 0.0,    "angle": 0.0},
}

SIMBENCH_FEEDERS = ["1-LV-rural1--0-sw", "1-LV-semiurb4--0-sw",
                    "1-LV-urban6--0-sw", "1-LV-rural2--0-sw",
                    "1-LV-rural3--0-sw", "1-LV-semiurb5--0-sw"]


def topology_of(net):
    """In-service line topology, excluding the MV-side transformer bus."""
    buses = set(net.line["from_bus"]) | set(net.line["to_bus"])
    G = nx.Graph()
    G.add_nodes_from(buses)
    for _, r in net.line.iterrows():
        if r["in_service"] and r["from_bus"] in buses and r["to_bus"] in buses:
            G.add_edge(int(r["from_bus"]), int(r["to_bus"]))
    return G


def load_simbench(code):
    import simbench as sb
    return sb.get_simbench_net(code)


def load_ieee33():
    import pandapower.networks as pn
    return pn.case33bw()


def simulate_simbench(net, n_steps=672, start=96 * 7 * 26):
    """Run power flow over real SimBench profiles. Returns (voltage, angle)."""
    import simbench as sb
    av = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    lp, lq = av[("load", "p_mw")], av[("load", "q_mvar")]
    sp = av.get(("sgen", "p_mw"))

    V, A = [], []
    for t in range(start, start + n_steps):
        net.load["p_mw"] = lp.loc[t].values
        net.load["q_mvar"] = lq.loc[t].values
        if sp is not None and len(net.sgen) > 0:
            net.sgen["p_mw"] = sp.loc[t].values
        try:
            pp.runpp(net)
        except Exception:
            continue
        V.append(net.res_bus["vm_pu"].copy())
        A.append(net.res_bus["va_degree"].copy())
    return (pd.DataFrame(V).reset_index(drop=True),
            pd.DataFrame(A).reset_index(drop=True))


def simulate_ieee(net, n_steps=500, seed=42):
    """Randomised loads around the base case. Returns (voltage, angle)."""
    rng = np.random.default_rng(seed)
    bp, bq = net.load["p_mw"].copy(), net.load["q_mvar"].copy()
    V, A = [], []
    for t in range(n_steps):
        f = 1 + 0.15 * np.sin(2 * np.pi * t / 96) + rng.normal(0, 0.05)
        ns = rng.normal(1, 0.25, size=len(net.load))
        net.load["p_mw"] = np.clip(bp * f * ns, 0, None)
        net.load["q_mvar"] = np.clip(bq * f * ns, 0, None)
        try:
            pp.runpp(net)
        except Exception:
            continue
        V.append(net.res_bus["vm_pu"].copy())
        A.append(net.res_bus["va_degree"].copy())
    net.load["p_mw"], net.load["q_mvar"] = bp, bq
    return (pd.DataFrame(V).reset_index(drop=True),
            pd.DataFrame(A).reset_index(drop=True))


def add_meter_noise(voltage, angle=None, meter_class="class_0.5", seed=7):
    """Apply the noise of a given meter accuracy class."""
    spec = METER_CLASSES[meter_class]
    rng = np.random.default_rng(seed)
    vn = voltage if not spec["voltage"] else voltage + rng.normal(
        0, spec["voltage"], size=voltage.shape)
    an = None
    if angle is not None and spec["angle"] is not None:
        an = angle if not spec["angle"] else angle + rng.normal(
            0, spec["angle"], size=angle.shape)
    return vn, an


# --- random radial feeders -------------------------------------------------

def random_radial_network(n_buses, rng, max_children=3):
    G = nx.Graph()
    G.add_node(0)
    cap = {0: max_children + 1}
    for i in range(1, n_buses):
        parent = rng.choice([n for n in G.nodes if G.degree[n] < cap.get(n, max_children)])
        G.add_edge(parent, i)
        cap[i] = max_children
    for u, v in G.edges():
        G[u][v]["r"] = rng.uniform(0.01, 0.05)
    return G


def simulate_radial(G, n_steps, rng, idio_noise=0.03, meter_sigma=0.0):
    """LinDistFlow approximation: fast enough to generate many networks.

    idio_noise controls how much each load deviates from the shared daily
    pattern. 0.03 reproduces the tight correlation spread seen in real
    SimBench data; larger values make the problem artificially easy.
    """
    nodes = list(G.nodes())
    slack = 0
    base = {b: rng.uniform(0.5, 2.0) for b in nodes if b != slack}
    paths = nx.single_source_dijkstra_path(G, slack, weight=None)
    T = nx.bfs_tree(G, slack)

    downstream = {}
    for u, v in G.edges():
        parent, child = (u, v) if len(paths[u]) < len(paths[v]) else (v, u)
        downstream[(parent, child)] = nx.descendants(T, child) | {child}

    rows = []
    for t in range(n_steps):
        f = max(1 + 0.2 * np.sin(2 * np.pi * t / 96) + rng.normal(0, 0.05), 0.05)
        noise = rng.normal(1.0, idio_noise, size=len(nodes))
        p = {b: max(base[b] * f * noise[i], 0)
             for i, b in enumerate(nodes) if b != slack}

        v = {slack: 1.0}
        for b in nodes:
            if b == slack:
                continue
            drop = 0.0
            path = paths[b]
            for k in range(len(path) - 1):
                u, w = path[k], path[k + 1]
                key = (u, w) if (u, w) in downstream else (w, u)
                drop += G[u][w]["r"] * sum(p.get(x, 0) for x in downstream[key])
            v[b] = 1.0 - 0.01 * drop
        rows.append([v[b] for b in nodes])

    df = pd.DataFrame(rows, columns=nodes)
    if meter_sigma:
        df = df + rng.normal(0, meter_sigma, size=df.shape)
    return df
