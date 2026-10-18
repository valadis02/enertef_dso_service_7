"""MV switch validation (D2.2 use case) with the same pipeline as LV / Far West.

Documented configuration = normal open points. Truth = 0-3 open points moved
(16 configurations per network, 4 SimBench MV networks, 3 noise draws each).
Candidates = every switchable line that lies on a ring (no knowledge of where
the errors are, unlike the decoy protocol of section 4.2).
Meters at every MV bus (secondary substations): voltage Class 0.5, optionally
angle 0.01 deg; P, Q injections with 1% error.

 1. features: variance distance, validate_topology result, residual of the
    linearised AC model when the line's status is flipped, documented status;
 2. ML: gradient boosting (or random forest, as in Liu et al.),
    (a) leave-one-network-out, (b) same network, other configurations (GroupKFold);
 3. decoding: spanning tree (radial) on the probabilities, non-ring lines fixed.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import json
import sys
import zlib

import numpy as np
import pandas as pd
import networkx as nx
import simbench as sb
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, SB_CODE)
sys.path.insert(0, TOPO)
import exp_lv_pipeline as L  # noqa: E402
from mv_sim import MV_NETS, OUT, mv_model  # noqa: E402
from topology import validate_topology  # noqa: E402
from topology.data import add_meter_noise  # noqa: E402
from topology.distance import electrical_distance  # noqa: E402

Z_BASE = 20.0 ** 2 / 1.0
import os as _os
L.TDM = _os.environ.get("TDM") == "1"
FEATS = (f"{SB_WORK}/mv_feats85" + ("_month" if _os.environ.get("MONTH") == "1" else "") if _os.environ.get("MVSET") == "85" else
         f"{SB_WORK}/mv_feats_tdm" if L.TDM else f"{SB_WORK}/mv_feats")
N_DRAW = 1 if _os.environ.get("MVSET") == "85" else 3


HV_ID = -1


def augment(info, nodes, V, A, P, Q):
    """Add the HV slack bus (known voltage, angle 0) when it is the reference."""
    if info.get("hv") is None:
        return nodes, V, A, P, Q
    T = len(V); z = np.zeros((T, 1))
    return (nodes + [HV_ID], np.hstack([V, np.full((T, 1), info["vhv"])]),
            None if A is None else np.hstack([A, z]), np.hstack([P, z]), np.hstack([Q, z]))


def net_info(code):
    net = sb.get_simbench_net(code)
    mvb, lines, roots, normal, bb = mv_model(net)
    js = json.load(open(f"{OUT}/{code}.json"))
    edges = [frozenset((int(net.line.from_bus[i]), int(net.line.to_bus[i]))) for i in lines]
    zb = float(net.bus.vn_kv[net.line.from_bus[lines[0]]]) ** 2 / 1.0      # 20 kV or 10 kV (urban)
    par = {e: (net.line.r_ohm_per_km[i] * net.line.length_km[i] / zb,
               net.line.x_ohm_per_km[i] * net.line.length_km[i] / zb) for e, i in zip(edges, lines)}
    nodes = {a for e in edges for a in e}
    sw = net.switch
    real = {frozenset((int(r.bus), int(r.element))) for r in sw[(sw.et == "b") & sw.closed].itertuples()}
    virtual = [e for e in bb if frozenset(e) not in real]  # trafo LV buses joined only through the HV grid
    Hb = nx.Graph([tuple(e) for e in bb if frozenset(e) in real]); Hb.add_nodes_from(roots)
    bb2, rep = [], {}
    for comp in nx.connected_components(Hb):           # contract busbar couplings onto metered buses
        inset = sorted(comp & nodes)
        bb2 += list(zip(inset[:-1], inset[1:]))
        for b in comp:
            rep[b] = inset[0]
    for a, b in bb2:
        par[frozenset((a, b))] = (1e-5, 1e-5)
    ztr = {}
    for t in net.trafo.itertuples():                   # trafo impedance on 1 MVA
        z = t.vk_percent / 100 / t.sn_mva; r = t.vkr_percent / 100 / t.sn_mva
        ztr[int(t.lv_bus)] = (r, np.sqrt(z ** 2 - r ** 2))
    hv = None
    if virtual:                                        # separate busbar sections: reference = HV slack bus
        hv = HV_ID
        for r in {x for e in virtual for x in e}:
            e = frozenset((rep[r], hv)); bb2.append((rep[r], hv)); par[e] = ztr.get(r, (1e-5, 1e-5))
        roots = [hv]
    else:
        roots = [rep[roots[0]]]
    bb = bb2
    med = (np.median([p[0] for p in par.values()]), np.median([p[1] for p in par.values()]))
    Gall = nx.Graph(); Gall.add_edges_from(map(tuple, edges)); Gall.add_edges_from(bb)
    ring = set()
    for cyc in nx.cycle_basis(Gall):
        cyc_e = {frozenset((cyc[i], cyc[(i + 1) % len(cyc)])) for i in range(len(cyc))}
        ring |= cyc_e
    cand = [k for k, e in enumerate(edges) if e in ring]
    return dict(hv=hv, vhv=float(net.ext_grid.vm_pu.iloc[0]), edges=edges, par=par, med=med, bb=[frozenset(e) for e in bb], roots=roots,
                normal=normal, cand=cand, configs=[np.array(c) for c in js["configs"]])


def graph_of(nodes, info, status):
    G = nx.Graph(); G.add_nodes_from(nodes)
    G.add_edges_from(tuple(e) for e in info["bb"])
    G.add_edges_from(tuple(info["edges"][k]) for k in np.where(status == 1)[0])
    return G


def channel_sigmas(nodes, info, status, P, Q, Vm, Va):
    """Weights for the residual: per channel, max(meter sigma, rms model residual of
    the starting configuration). Keeps linearisation error from dominating."""
    Z = L.zred(nodes, {frozenset(e) for e in graph_of(nodes, info, status).edges()}, info["par"], info["med"], info["roots"][0])
    W = (P / Vm - 1j * Q / Vm) @ Z.T
    out = []
    for meas, pred, s in ((Vm, W.real, 0.0025), (Va, W.imag, np.deg2rad(0.01))):
        if meas is None:
            out.append(s); continue
        r = meas - pred; r = r - r.mean(1, keepdims=True)
        if L.TDM:
            r = r - r.mean(0, keepdims=True)
        out.append(max(s, np.sqrt((r ** 2).mean())))
    return out


def swap_gains(nodes, info, status, P, Q, Vm, Va, sv=0.0025, sa=np.deg2rad(0.01)):
    """Branch exchange: for each open line o, close it and open each line k on the
    loop it forms. Returns J of the current config, per-line best relative gain,
    and the list of (gain, o, k)."""
    E = info["edges"]; root = info["roots"][0]
    cur = graph_of(nodes, info, status)
    cur_e = {frozenset(e) for e in cur.edges()}
    J = lambda es: L.resid_J(L.zred(nodes, es, info["par"], info["med"], root), P, Q, Vm, Va, sv, sa)
    J0 = J(cur_e)
    idx = {e: k for k, e in enumerate(E)}
    best = np.full(len(E), -1.0); swaps = []
    for o in np.where(status == 0)[0]:
        a, b = tuple(E[o])
        if not nx.has_path(cur, a, b):
            continue
        path = nx.shortest_path(cur, a, b)
        for u, v in zip(path[:-1], path[1:]):
            k = idx.get(frozenset((u, v)))
            if k is None:
                continue
            g = (J0 - J((cur_e | {E[o]}) - {E[k]})) / J0
            swaps.append((g, o, k)); best[o] = max(best[o], g); best[k] = max(best[k], g)
    return J0, best, swaps


def greedy_exchange(nodes, info, status, P, Q, Vm, Va, min_gain=0.01, max_iter=6, adaptive=False):
    st = status.copy()
    sv, sa = channel_sigmas(nodes, info, st, P, Q, Vm, Va) if adaptive else (0.0025, np.deg2rad(0.01))
    for _ in range(max_iter):
        _, _, sw = swap_gains(nodes, info, st, P, Q, Vm, Va, sv, sa)
        if not sw:
            break
        g, o, k = max(sw)
        if g < min_gain:
            break
        st[o], st[k] = 1, 0
    return st


def case(code, info, c, j, meters):
    d = dict(np.load(f"{OUT}/{code}__{c:02d}.npz")); nodes = [int(b) for b in d["buses"]]
    truth = d["status"]; doc = info["normal"]
    seed = zlib.crc32(f"{code}#{c}#{j}".encode())
    Vn, An = add_meter_noise(pd.DataFrame(d["V"], columns=nodes), pd.DataFrame(d["A"], columns=nodes), meters, seed=seed)
    rng = np.random.default_rng(seed + 1)
    P = d["P"] * (1 + rng.normal(0, 0.01, d["P"].shape)); Q = d["Q"] * (1 + rng.normal(0, 0.01, d["Q"].shape))
    Vm = Vn.values; Va = np.deg2rad(An.values) if An is not None else None
    nodes_m = nodes
    nodes, Vm, Va, P, Q = augment(info, nodes, Vm, Va, P, Q)
    E = info["edges"]; cand = info["cand"]
    docG = graph_of(nodes, info, doc)
    base = validate_topology(Vn, graph_of(nodes_m, info, doc), nodes_m, An, possible_connections=[E[k] for k in cand])
    base_e = {frozenset(e) for e in base.topology.edges()}
    Dm, dn = electrical_distance((An if An is not None else Vn)[nodes_m], 15)
    ix = {n: k for k, n in enumerate(dn)}
    dist = np.array([Dm[ix[a], ix[b]] for a, b in (tuple(E[k]) for k in cand)])
    fixed = [e for e in {frozenset(e) for e in docG.edges()} if e not in {E[k] for k in cand}]
    dfix = np.median([Dm[ix[a], ix[b]] for a, b in map(tuple, fixed) if a in ix and b in ix])
    doc_e = {frozenset(e) for e in docG.edges()}
    root = info["roots"][0]
    _, gain, _ = swap_gains(nodes, info, doc, P, Q, Vm, Va)
    rows = []
    for n, k in enumerate(cand):
        rows.append(dict(k=k, log_dist=np.log10(dist[n] / dfix), dist_rank=(dist < dist[n]).mean(),
                         kruskal_in=int(E[k] in base_e), doc=int(doc[k]), dJ=gain[k], label=int(truth[k])))
    X = pd.DataFrame(rows); X["net"] = code; X["cfg"] = f"{code}#{c}"; X["case"] = f"{code}#{c}#{j}"
    base_status = np.array([int(e in base_e) for e in E])
    greedy = greedy_exchange(nodes, info, doc, P, Q, Vm, Va)
    return X, dict(nodes=nodes, truth=truth, doc=doc, base=base_status, greedy=greedy)


def decode(info, nodes, p, cand):
    G = nx.Graph(); G.add_nodes_from(nodes)
    candset = set(cand)
    fixed = [tuple(e) for e in info["bb"]] + [tuple(info["edges"][k]) for k in range(len(info["edges"]))
                                              if k not in candset and info["normal"][k]]
    status = np.array(info["normal"]).copy(); status[cand] = 0
    uf = nx.utils.UnionFind(nodes)
    for a, b in fixed:
        uf.union(a, b)
    for n in np.argsort(-p):
        a, b = tuple(info["edges"][cand[n]])
        if uf[a] != uf[b]:
            uf.union(a, b); status[cand[n]] = 1
    return status


def metrics(st, truth, doc):
    moved = (doc == 0) & (truth == 1)                 # documented open point that is actually closed
    newopen = (doc == 1) & (truth == 0)               # line actually open, documented closed
    return dict(exact=int((st == truth).all()),
                found=int(((st == 0) & newopen).sum()), total=int(newopen.sum()),
                closed_ok=int(((st == 1) & moved).sum()),
                false_open=int(((st == 0) & (truth == 1) & (doc == 1)).sum()),
                elem_acc=(st == truth).mean())


def predict(X, cols, model, split):
    p = np.zeros(len(X))
    if split == "lono":
        folds = [((X.net != n).values, (X.net == n).values) for n in X.net.unique()]
    else:
        folds = []
        for n in X.net.unique():
            sub = np.where(X.net == n)[0]
            for tr, te in GroupKFold(4).split(sub, groups=X.cfg.iloc[sub]):
                m_tr = np.zeros(len(X), bool); m_te = np.zeros(len(X), bool)
                m_tr[sub[tr]] = True; m_te[sub[te]] = True
                folds.append((m_tr, m_te))
    for tr, te in folds:
        clf = (HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0) if model == "gb"
               else RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1))
        clf.fit(X.loc[tr, cols], X.loc[tr, "label"])
        p[te] = clf.predict_proba(X.loc[te, cols])[:, 1]
    return p


def summarize(name, stats):
    S = pd.DataFrame(stats)
    return dict(method=name, exact_config=round(100 * S.exact.mean(), 1),
                moved_open_points_found=f"{S.found.sum()}/{S.total.sum()}",
                false_openings=int(S.false_open.sum()), element_acc=round(100 * S.elem_acc.mean(), 2))


if __name__ == "__main__":
    import pickle
    meters = sys.argv[1]
    if len(sys.argv) > 2:                                  # features for one network
        code = sys.argv[2]; info = net_info(code); Xs, ctx = [], {}
        lo, hi = map(int, sys.argv[3].split("-"))
        for c in range(lo, hi):
            for j in range(N_DRAW):
                X, cx = case(code, info, c, j, meters)
                Xs.append(X); ctx[X.case.iloc[0]] = cx
        pickle.dump((pd.concat(Xs, ignore_index=True), ctx), open(f"{FEATS}/{meters}_{code}_{lo:02d}.pkl", "wb"))
        print(code, "features", flush=True); sys.exit(0)
    infos = {c: net_info(c) for c in MV_NETS}
    Xs, ctx = [], {}
    import glob
    for f in sorted(glob.glob(f"{FEATS}/{meters}_*.pkl")):
        x, cx = pickle.load(open(f, "rb"))
        Xs.append(x); ctx.update(cx)
    X = pd.concat(Xs, ignore_index=True)
    out = []
    out.append(summarize("documented (normal open points)",
                         [metrics(c["doc"], c["truth"], c["doc"]) for c in ctx.values()]))
    out.append(summarize("variance distance + Kruskal (validate_topology)",
                         [metrics(c["base"], c["truth"], c["doc"]) for c in ctx.values()]))
    out.append(summarize("physics only: greedy branch exchange on residual",
                         [metrics(c["greedy"], c["truth"], c["doc"]) for c in ctx.values()]))
    F = ["log_dist", "dist_rank", "kruskal_in", "doc", "dJ"]
    for name, cols, model, split in (("ML residual only, other networks", ["dJ", "doc"], "gb", "lono"),
                                     ("ML all, other networks (GB)", F, "gb", "lono"),
                                     ("ML all, other networks (RF)", F, "rf", "lono"),
                                     ("ML all, same network other configs (GB)", F, "gb", "cfg"),
                                     ("ML all, same network other configs (RF, Liu)", F, "rf", "cfg")):
        p = predict(X, cols, model, split)
        st = []
        for cs, ix in X.groupby("case").indices.items():
            code = cs.split("#")[0]; c = ctx[cs]
            s = decode(infos[code], c["nodes"], p[ix], list(X.k.iloc[ix]))
            st.append(metrics(s, c["truth"], c["doc"]))
        out.append(summarize(name, st))
    T = pd.DataFrame(out); T.insert(0, "meters", meters)
    print(T.to_string(index=False))
    T.to_csv(f"{SB_WORK}/mv_results_{meters}.csv", index=False)
