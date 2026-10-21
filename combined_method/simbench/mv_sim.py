"""MV switch validation data: SimBench MV open-ring networks.

Each MV line has load-break switches; the normally open points (one per ring)
make the network radial. A truth configuration moves m open points: the
documented open line is closed and another line on the same ring is opened
(network stays radial and connected). Documentation = normal open points.

Cache per (network, config): one week of 15-min power flow,
V, angle, net P, Q per MV bus (p.u. on 1 MVA), the line status vector.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import json
import os
import sys
import time

import numpy as np
import networkx as nx
import pandapower as pp
import simbench as sb

MV_NETS = ["1-MV-rural--0-sw", "1-MV-semiurb--0-sw", "1-MV-urban--0-sw", "1-MV-comm--0-sw"]
OUT = (f"{SB_WORK}/mv_cache85" if os.environ.get("MVSET") == "85" else f"{SB_WORK}/mv_cache") + ("_month" if os.environ.get("MONTH") == "1" else "")
START, STEPS = 96 * 7 * 26, (2880 if os.environ.get("MONTH") == "1" else 672)
N_CONF = 16


def mv_model(net):
    """MV buses, line list, root bus set, normal line status (1 = closed)."""
    mv = set(net.bus.index[(net.bus.vn_kv > 1) & (net.bus.vn_kv < 60)])
    lines = [i for i in net.line.index if net.line.from_bus[i] in mv and net.line.to_bus[i] in mv]
    sw = net.switch[net.switch.et == "l"]
    open_l = set(sw.element[~sw.closed])
    status = np.array([0 if i in open_l else 1 for i in lines])
    roots = sorted(set(net.trafo.lv_bus) & mv)
    bb = [(int(r.bus), int(r.element)) for r in net.switch[(net.switch.et == "b") & net.switch.closed].itertuples()
          if r.bus in mv and r.element in mv]
    H = nx.Graph(bb); H.add_nodes_from(roots)
    bb += [(roots[0], r) for r in roots[1:] if not nx.has_path(H, roots[0], r)]   # joined through the HV grid
    return sorted(mv), lines, roots, status, bb


def graph(net, lines, status, bb):
    G = nx.Graph()
    G.add_nodes_from(set(net.line.from_bus[lines]) | set(net.line.to_bus[lines]))
    G.add_edges_from(bb)
    for k, i in enumerate(lines):
        if status[k]:
            G.add_edge(int(net.line.from_bus[i]), int(net.line.to_bus[i]), line=k)
    return G


def configs(net, n_conf=N_CONF, seed=0):
    mvb, lines, roots, normal, bb = mv_model(net)
    rng = np.random.default_rng(seed)
    S85 = os.environ.get("MVSET") == "85"
    out = [] if S85 else [normal.copy()]
    m85 = max(1, round(float(os.environ.get('DOCERR', '0.20')) * int((normal == 0).sum())))   # DOCERR of the open points moved
    no = list(np.where(normal == 0)[0])
    while len(out) < n_conf:
        m = m85 if S85 else 1 + (len(out) - 1) % 3        # 1, 2, 3 moved open points
        st = normal.copy()
        for k in rng.choice(no, size=min(m, len(no)), replace=False):
            st[k] = 1                                    # close the documented open point
            G = graph(net, lines, st, bb)
            try:
                cyc = nx.find_cycle(G, int(net.line.from_bus[lines[k]]))
            except nx.NetworkXNoCycle:                   # tie leads to an isolated part: leave it
                st[k] = 0; continue
            ring = [G.edges[e]["line"] for e in cyc if "line" in G.edges[e] and G.edges[e]["line"] != k]
            if not ring:
                st[k] = 0; continue
            st[rng.choice(ring)] = 0                     # open another line on the same ring
        G = graph(net, lines, st, bb)
        if S85 and int(((normal == 0) & (st == 1)).sum()) != m85:
            continue
        if nx.is_tree(G) and not any((st == o).all() for o in out):
            out.append(st)
    return out


def simulate(code, c, st, net, lines, av):
    net.line.loc[lines, "in_service"] = st.astype(bool)
    sw = net.switch.et == "l"
    net.switch.loc[sw, "closed"] = True                  # line status carried by in_service
    lp, lq = av[("load", "p_mw")], av[("load", "q_mvar")]
    sp = av.get(("sgen", "p_mw"))
    if os.environ.get("INDIV") == "1":                   # personalized profiles (indiv_profiles.py)
        import pandas as pd
        from indiv_profiles import individualise
        IP, IQ = individualise(lp, lq, net.load.profile.values, START, STEPS, sum(map(ord, code)), noise=0.1)
        idx = range(START, START + STEPS)
        lp = pd.DataFrame(IP, index=idx, columns=lp.columns); lq = pd.DataFrame(IQ, index=idx, columns=lq.columns)
    buses = sorted(set(net.line.from_bus[lines]) | set(net.line.to_bus[lines]))
    V, A, P, Q = [], [], [], []
    for t in range(START, START + STEPS):
        net.load["p_mw"] = lp.loc[t].values; net.load["q_mvar"] = lq.loc[t].values
        if sp is not None and len(net.sgen):
            net.sgen["p_mw"] = sp.loc[t].values
        pp.runpp(net, numba=True, init="results" if t > START else "auto")
        rb = net.res_bus.loc[buses]
        V.append(rb.vm_pu.values); A.append(rb.va_degree.values)
        P.append(-rb.p_mw.values); Q.append(-rb.q_mvar.values)
    np.savez_compressed(f"{OUT}/{code}__{c:02d}.npz", buses=np.array(buses), V=np.array(V), A=np.array(A),
                        P=np.array(P), Q=np.array(Q), status=st)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 100
    t0 = time.time()
    nets = [MV_NETS[int(i)] for i in os.environ.get('NETS', '0123')]
    for code in nets:
        net = sb.get_simbench_net(code)
        mvb, lines, roots, normal, bb = mv_model(net)
        cf = configs(net)
        json.dump(dict(lines=[int(i) for i in lines], roots=[int(r) for r in roots], bb=bb,
                       configs=[c.tolist() for c in cf]), open(f"{OUT}/{code}.json", "w"))
        av = None
        for c, st in enumerate(cf):
            if os.path.exists(f"{OUT}/{code}__{c:02d}.npz"):
                continue
            if time.time() - t0 > budget:
                print("budget"); sys.exit(0)
            av = av or sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
            simulate(code, c, st, net, lines, av)
            print(code, c, round(time.time() - t0), flush=True)
    print("ALL DONE")
