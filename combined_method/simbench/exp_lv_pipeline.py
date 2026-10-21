"""Same pipeline as Far West (steps 1-4), on the six SimBench LV feeders,
with the section 4.2 protocol: documented topology with 15% mixed errors,
candidate set = real discrepancies + as many correctly documented decoys.

 1. Features per candidate edge: variance distance (as variance_distance_spanning_tree),
    whether variance_distance_spanning_tree keeps it, residual improvement of the
    linearised AC model (needs P, Q), documented status.
 2. ML: gradient boosting, leave-one-feeder-out (a feeder is never seen in training).
 3. Decoding: radial Kruskal on the probabilities, trusted edges fixed.
 4. Line length scale per edge on the decoded tree (LinDistFlow least squares).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import sys
import zlib

import numpy as np
import pandas as pd
import networkx as nx
import simbench as sb
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, TOPO)
from topology import errors, evaluate, variance_distance_spanning_tree  # noqa: E402
from topology.data import topology_of, add_meter_noise, SIMBENCH_FEEDERS  # noqa: E402
from topology.distance import electrical_distance  # noqa: E402

Z_BASE = 0.4 ** 2 / 1.0
import os as _os
SUFM = "_month" if _os.environ.get("MONTH") == "1" else ""
CACHE = f"{SB_WORK}/sb_cache" + SUFM
SEEDS = range(12)                       # seed 2 is the one used in section 4.2


def line_params(net):
    par = {}
    for _, r in net.line.iterrows():
        par[frozenset((int(r.from_bus), int(r.to_bus)))] = (
            r.r_ohm_per_km * r.length_km / Z_BASE, r.x_ohm_per_km * r.length_km / Z_BASE)
    med = (np.median([p[0] for p in par.values()]), np.median([p[1] for p in par.values()]))
    return par, med


def zred(nodes, edges, par, med, ref):
    idx = {n: k for k, n in enumerate(nodes)}
    Y = np.zeros((len(nodes), len(nodes)), complex)
    for e in edges:
        a, b = tuple(e)
        r, x = par.get(frozenset(e), med)
        y = 1 / complex(max(r, 1e-6), max(x, 1e-6))
        i, j = idx[a], idx[b]
        Y[i, i] += y; Y[j, j] += y; Y[i, j] -= y; Y[j, i] -= y
    Y += np.eye(len(nodes)) * 1e-6        # keeps Y invertible if a hypothesis is disconnected
    keep = [k for k in range(len(nodes)) if nodes[k] != ref]
    Z = np.zeros_like(Y)
    Z[np.ix_(keep, keep)] = np.linalg.inv(Y[np.ix_(keep, keep)])
    return Z


DOCERR = float(__import__('os').environ.get('DOCERR', '0.20'))   # share of wrong documentation (0.20 = 80% correct)
TDM = False          # True: residual of the fluctuations (per-meter time mean removed too)


def resid_J(Z, P, Q, Vm, Va, sv, sa):
    W = (P / Vm - 1j * Q / Vm) @ Z.T
    J = 0.0
    for meas, pred, s in ((Vm, W.real, sv), (Va, W.imag, sa)):
        if meas is None:
            continue
        r = meas - pred
        r = r - r.mean(1, keepdims=True)
        if TDM:
            r = r - r.mean(0, keepdims=True)
        J += ((r / s) ** 2).sum()
    return J


CANDS = __import__('os').environ.get('CANDS', 'list')
REG = __import__('os').environ.get('REG', 'ideal')
SPARE, LEN_SHARE, LEN_ERR = 0.20, 0.20, (0.10, 0.30)


def spare_cables(tg, existing, n, seed):
    """Unused cables in the register: pairs of nodes 2-3 steps apart in the real network."""
    rng = np.random.default_rng(seed)
    pool = sorted({frozenset((a, b)) for a in tg.nodes for b, d in nx.single_source_shortest_path_length(tg, a, cutoff=3).items()
                   if d >= 2 and a != b} - set(existing), key=lambda e: tuple(sorted(e)))
    pick = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return {pool[i] for i in pick}


def wrong_lengths(par, share, err, seed):
    """The register's length is wrong by 10-30% (either sign) on `share` of the cables."""
    rng = np.random.default_rng(seed)
    keys = sorted(par, key=lambda e: tuple(sorted(e))); out = dict(par)
    for i in rng.choice(len(keys), size=round(share * len(keys)), replace=False):
        f = 1 + rng.choice([-1, 1]) * rng.uniform(*err)
        out[keys[i]] = (par[keys[i]][0] * f, par[keys[i]][1] * f)
    return out


def case_features(code, net, tg, d, seed, meters, par0=None):
    nodes = list(d["buses"])
    root = int(net.trafo.lv_bus.iloc[0])
    par, med = line_params(net) if par0 is None else par0
    n_err = max(1, round(tg.number_of_edges() * DOCERR))
    doc = errors.mixed(tg, n_err, seed=seed)
    if CANDS == "all":   # nothing trusted: every documented line and every physically existing cable is in doubt
        cands = {frozenset(e) for e in doc.edges()} | {frozenset(e) for e in tg.edges()}
        if REG == "real":   # realistic asset register: spare cables (20% of the list) and wrong lengths
            cands |= spare_cables(tg, cands, round(SPARE / (1 - SPARE) * tg.number_of_edges()), seed + 200)
            if par0 is None:   # synthetic training cases already use the register's lengths
                par = wrong_lengths(par, LEN_SHARE, LEN_ERR, seed + 300)
    else:                # errors + as many correctly documented decoys
        cands = errors.possible_connections(tg, doc, seed=seed + 100)
    Vdf = pd.DataFrame(d["V"], columns=nodes); Adf = pd.DataFrame(d["A"], columns=nodes)
    Vn, An = add_meter_noise(Vdf, Adf, meters, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    P = d["P"] * (1 + rng.normal(0, 0.01, d["P"].shape))
    Q = d["Q"] * (1 + rng.normal(0, 0.01, d["Q"].shape))
    Vm = Vn.values
    Va = np.deg2rad(An.values) if An is not None else None
    sv, sa = (0.0025, np.deg2rad(0.01))

    base = variance_distance_spanning_tree(Vn, doc, nodes, An, possible_connections=cands)
    Dm, dn = electrical_distance((An if An is not None else Vn)[nodes], 15)
    idx = {n: k for k, n in enumerate(dn)}
    doc_e = {frozenset(e) for e in doc.edges()}
    fixed = [e for e in doc_e if e not in cands]
    dfix = np.median([Dm[idx[a], idx[b]] for a, b in map(tuple, fixed or cands)])
    base_e = {frozenset(e) for e in base.topology.edges()}
    true_e = {frozenset(e) for e in tg.edges()}

    J0 = resid_J(zred(nodes, doc_e, par, med, root), P, Q, Vm, Va, sv, sa)
    cl = sorted(cands, key=lambda e: tuple(sorted(e)))
    dist = np.array([Dm[idx[a], idx[b]] for a, b in map(tuple, cl)])
    rows = []
    for k, e in enumerate(cl):
        flipped = doc_e ^ {e}
        J1 = resid_J(zred(nodes, flipped, par, med, root), P, Q, Vm, Va, sv, sa)
        rows.append(dict(edge=e, log_dist=np.log10(dist[k] / dfix), dist_rank=(dist < dist[k]).mean(),
                         kruskal_in=int(e in base_e), doc=int(e in doc_e), dJ=(J0 - J1) / J0,
                         label=int(e in true_e)))
    X = pd.DataFrame(rows)
    X["feeder"] = code; X["case"] = f"{code}#{seed}"
    ctx = dict(nodes=nodes, fixed=fixed, tg=tg, doc=doc, base=base.topology, cands=cands,
               V=Vm, P=P, Q=Q, par=par, med=med, root=root)
    return X, ctx


def decode(nodes, fixed, edges, p):
    G = nx.Graph(); G.add_nodes_from(nodes); G.add_edges_from(map(tuple, fixed))
    uf = nx.utils.UnionFind(nodes)
    for a, b in map(tuple, fixed):
        uf.union(a, b)
    for k in np.argsort(-p):
        a, b = tuple(edges[k])
        if uf[a] != uf[b]:
            uf.union(a, b); G.add_edge(a, b)
    return G


def length_scale(G, ctx):
    """Per edge on the decoded tree: fit V_parent - V_child = s * (r P_down + x Q_down)."""
    T = nx.bfs_tree(G, ctx["root"]); idx = {n: k for k, n in enumerate(ctx["nodes"])}
    load_p, load_q = -ctx["P"], -ctx["Q"]
    out = []
    for p, c in T.edges():
        down = [idx[n] for n in nx.descendants(T, c) | {c}]
        r, x = ctx["par"].get(frozenset((p, c)), ctx["med"])
        if r < 1e-3:                          # busbar-like connectors carry no voltage drop
            continue
        g = r * load_p[:, down].sum(1) + x * load_q[:, down].sum(1)
        dv = ctx["V"][:, idx[p]] - ctx["V"][:, idx[c]]
        s = (g @ dv) / (g @ g)
        out.append(dict(edge=frozenset((p, c)), true_edge=frozenset((p, c)) in {frozenset(e) for e in ctx["tg"].edges()}, scale=s))
    return pd.DataFrame(out)


def main(meters):
    X_all, ctxs = [], {}
    for code in SIMBENCH_FEEDERS:
        net = sb.get_simbench_net(code); tg = topology_of(net)
        d = dict(np.load(f"{CACHE}/{code}.npz"))
        for seed in SEEDS:
            X, ctx = case_features(code, net, tg, d, seed, meters)
            X_all.append(X); ctxs[X.case.iloc[0]] = ctx
        print(meters, code, "features done", flush=True)
    return pd.concat(X_all, ignore_index=True), ctxs


def evaluate_all(X, ctxs, feats, passes=1):
    X = X.copy(); X["cur_in"] = X["doc"]; X["dJ_cur"] = X["dJ"]
    hist = []
    for k in range(passes):
        cols = feats + (["cur_in", "dJ_cur"] if k > 0 else [])
        p = np.zeros(len(X))
        for f in X.feeder.unique():                     # leave one feeder out
            tr, te = X.feeder != f, X.feeder == f
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
            clf.fit(X.loc[tr, cols], X.loc[tr, "label"])
            p[te.values] = clf.predict_proba(X.loc[te, cols])[:, 1]
        res = {}
        for case, ix in X.groupby("case").indices.items():
            c = ctxs[case]
            res[case] = decode(c["nodes"], c["fixed"], list(X.edge.iloc[ix]), p[ix])
        hist.append(res)
        if k + 1 < passes:                              # refresh residual feature around the decoded topology
            for case, ix in X.groupby("case").indices.items():
                c = ctxs[case]; cur = {frozenset(e) for e in res[case].edges()}
                Va = None
                J0 = resid_J(zred(c["nodes"], cur, c["par"], c["med"], c["root"]), c["P"], c["Q"], c["V"], Va, 0.0025, 1)
                for i in ix:
                    e = X.edge.iloc[i]
                    J1 = resid_J(zred(c["nodes"], cur ^ {e}, c["par"], c["med"], c["root"]), c["P"], c["Q"], c["V"], Va, 0.0025, 1)
                    X.iat[i, X.columns.get_loc("dJ_cur")] = (J0 - J1) / J0
                    X.iat[i, X.columns.get_loc("cur_in")] = int(e in cur)
    return hist


def score(results, ctxs, X):
    rows = []
    for case, G in results.items():
        c = ctxs[case]
        rows.append(dict(case=case, feeder=case.split("#")[0], seed=int(case.split("#")[1]),
                         acc=evaluate(G, c["tg"])["reconstruction_accuracy"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import os
    import pickle
    meters = sys.argv[1]
    if os.environ.get("TDM") == "1":                    # fluctuation residual, one feeder per call
        TDM = True
        os.makedirs(f"{SB_WORK}/lv_tdm", exist_ok=True)
        if sys.argv[2] == "combine":
            Xs, cs = [], {}
            for i in range(len(SIMBENCH_FEEDERS)):
                x, c = pickle.load(open(f"{SB_WORK}/lv_tdm/{meters}{SUFM}_{i}.pkl", "rb")); Xs.append(x); cs.update(c)
            pd.concat(Xs, ignore_index=True).to_pickle(f"{SB_WORK}/lv_feats_{meters}_tdm{SUFM}.pkl")
            pickle.dump(cs, open(f"{SB_WORK}/lv_ctx_{meters}_tdm{SUFM}.pkl", "wb")); sys.exit(0)
        code = SIMBENCH_FEEDERS[int(sys.argv[2])]
        net = sb.get_simbench_net(code); tg = topology_of(net); d = dict(np.load(f"{CACHE}/{code}.npz"))
        Xs, cs = [], {}
        for seed in SEEDS:
            X, ctx = case_features(code, net, tg, d, seed, meters); Xs.append(X); cs[X.case.iloc[0]] = ctx
        pickle.dump((pd.concat(Xs, ignore_index=True), cs), open(f"{SB_WORK}/lv_tdm/{meters}{SUFM}_{sys.argv[2]}.pkl", "wb"))
        print(code, "done", flush=True); sys.exit(0)
    X, ctxs = main(meters)
    X.to_pickle(f"{SB_WORK}/lv_feats_{meters}.pkl")
    pickle.dump(ctxs, open(f"{SB_WORK}/lv_ctx_{meters}.pkl", "wb"))
