"""Simulated switching events, for evaluating the monitoring mode.

A realistic switching operation in a radial LV feeder is a load transfer:
open one line and close a tie to another branch, so that every bus stays
supplied and the network stays radial. Opening a line without a tie would
leave a de-energised section with no measurements, which is a different
and much easier problem.
"""

import copy

import numpy as np
import networkx as nx
import pandas as pd
import pandapower as pp

from .data import topology_of


def plan_transfer(net, seed=0, max_hops=3):
    """Choose a line to open and a tie to close.

    Returns (open_line_index, (tie_from, tie_to)). The tie connects the
    orphaned side to a bus outside it, within a few hops, so the feeder
    stays connected and radial.
    """
    rng = np.random.default_rng(seed)
    G = topology_of(net)
    root = int(net.trafo["lv_bus"].iloc[0]) if len(net.trafo) else list(G.nodes())[0]

    lines = list(net.line.index)
    rng.shuffle(lines)
    for li in lines:
        u, v = int(net.line.at[li, "from_bus"]), int(net.line.at[li, "to_bus"])
        H = G.copy()
        H.remove_edge(u, v)
        comps = list(nx.connected_components(H))
        if len(comps) != 2:
            continue
        orphan = next(c for c in comps if root not in c) if root in G else min(comps, key=len)
        if len(orphan) < 2:
            continue
        # a tie from somewhere in the orphaned section to a nearby bus outside it
        for x in rng.permutation(list(orphan)):
            near = [y for y, d in nx.single_source_shortest_path_length(G, int(x), cutoff=max_hops).items()
                    if y not in orphan and d >= 2 and not G.has_edge(int(x), y)]
            if near:
                return li, (int(x), int(rng.choice(near)))
    raise RuntimeError("no feasible load transfer found")


def simulate_with_event(net, event_step, open_line, tie, n_steps=480,
                        start=96 * 7 * 26):
    """Run SimBench profiles with a load transfer at `event_step`.

    Returns (voltage, angle, topology_before, topology_after).
    """
    import simbench as sb

    net = copy.deepcopy(net)
    av = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    lp, lq = av[("load", "p_mw")], av[("load", "q_mvar")]
    sp = av.get(("sgen", "p_mw"))

    before = topology_of(net)
    template = net.line.loc[open_line]
    tie_line = pp.create_line_from_parameters(
        net, tie[0], tie[1], length_km=float(template["length_km"]),
        r_ohm_per_km=float(template["r_ohm_per_km"]),
        x_ohm_per_km=float(template["x_ohm_per_km"]),
        c_nf_per_km=float(template["c_nf_per_km"]),
        max_i_ka=float(template["max_i_ka"]), in_service=False)

    V, A = [], []
    for k, t in enumerate(range(start, start + n_steps)):
        if k == event_step:
            net.line.at[open_line, "in_service"] = False
            net.line.at[tie_line, "in_service"] = True
        net.load["p_mw"] = lp.loc[t].values
        net.load["q_mvar"] = lq.loc[t].values
        if sp is not None and len(net.sgen) > 0:
            net.sgen["p_mw"] = sp.loc[t].values
        try:
            pp.runpp(net)
        except Exception:
            V.append(V[-1] if V else net.res_bus["vm_pu"].copy())
            A.append(A[-1] if A else net.res_bus["va_degree"].copy())
            continue
        V.append(net.res_bus["vm_pu"].copy())
        A.append(net.res_bus["va_degree"].copy())

    after = topology_of(net)
    return (pd.DataFrame(V).reset_index(drop=True),
            pd.DataFrame(A).reset_index(drop=True), before, after)
