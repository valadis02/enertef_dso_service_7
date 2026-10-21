"""Use case 1 on SimBench LV, training only from the operator's own register.

For each evaluated case the random forest never sees a verified topology. Its
training examples are made from the case's own register: the documented topology
(completed to a radial tree) is taken as if it were true, voltages are predicted on
it from the measured P, Q of the customers with the register's line data, the
register is corrupted again with synthetic errors (same error model), and the
labels are those synthetic errors. The forest is then applied to the real
measurements. Same decoding and refinement as exp_final_lv.py.
usage: CANDS=all REG=real TDM=1 MONTH=1 exp_lv_selftrain.py <feeder index> | summary
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK  # noqa: E402
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
import networkx as nx
import simbench as sb

sys.path.insert(0, SB_CODE)
sys.path.insert(0, TOPO)
import exp_final_lv as FL  # noqa: E402
import exp_liu_features as LF  # noqa: E402
from topology.data import SIMBENCH_FEEDERS  # noqa: E402

L = FL.L
K = int(os.environ.get("NSYN", "8"))          # synthetic registers per evaluated case
OUT = f"{FL.OUT}_selftrain"
os.makedirs(OUT, exist_ok=True)


def log_corr(V, nodes, edges):
    idx = {n: k for k, n in enumerate(nodes)}
    _, Cc = LF.pair_stats(LF.smooth(V), [(idx[a], idx[b]) for a, b in map(tuple, edges)])
    return Cc


def synthetic_truth(c, rng):
    """The register's topology completed to a radial tree (documented lines first)."""
    edges = sorted(c["cands"], key=lambda e: tuple(sorted(e)))
    doc_e = {frozenset(e) for e in c["doc"].edges()}
    p = np.array([1.0 if e in doc_e else 0.0 for e in edges]) + rng.uniform(0, 0.01, len(edges))
    return L.decode(c["nodes"], [], edges, p)


def main(arg):
    X, C = LF.add_features(FL.METERS)
    code = SIMBENCH_FEEDERS[int(arg)]
    net = sb.get_simbench_net(code)
    rows = []
    for case, ix in X.groupby("case").indices.items():
        if not case.startswith(code + "#"):
            continue
        c = C[case]; seed = int(case.split("#")[1]); rng = np.random.default_rng(9000 + seed)
        T = synthetic_truth(c, rng)
        idx = {n: k for k, n in enumerate(c["nodes"])}
        Z = L.zred(c["nodes"], {frozenset(e) for e in T.edges()}, c["par"], c["med"], c["root"])
        W = (c["P"] / c["V"] - 1j * c["Q"] / c["V"]) @ Z.T
        Vsyn = c["V"][:, [idx[c["root"]]]] + W.real               # voltages the register's network would give
        d = dict(buses=np.array(c["nodes"]), V=Vsyn, A=np.zeros_like(Vsyn), P=c["P"], Q=c["Q"])
        Xs = []
        for k in range(K):
            x, cx = L.case_features(code, net, T, d, 5000 + 97 * seed + k, FL.METERS, par0=(c["par"], c["med"]))
            x["log_corr"] = log_corr(cx["V"], cx["nodes"], x.edge)
            Xs.append(x)
        Xt = pd.concat(Xs, ignore_index=True); xr = X.iloc[ix]
        P = {}
        for name, cols in (("final", FL.FINAL), ("liu", FL.LIU)):
            clf = LF.RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(Xt[cols], Xt["label"]); P[name] = clf.predict_proba(xr[cols])[:, 1]
        edges = list(xr.edge); cands = set(edges)
        r = dict(case=case, documented=FL.share(c["doc"], c["tg"]))
        G = nx.Graph(); G.add_nodes_from(c["nodes"]); G.add_edges_from(map(tuple, c["fixed"]))
        G.add_edges_from(tuple(e) for e, q in zip(edges, P["liu"]) if q > 0.5)
        r["liu_only"] = FL.share(G, c["tg"])
        r["variance_distance_spanning_tree"] = FL.share(c["base"], c["tg"])
        Gd = L.decode(c["nodes"], c["fixed"], edges, P["final"])
        r["combined_steps_1_3"] = FL.share(Gd, c["tg"])
        r["combined"] = FL.share(FL.refine(Gd, cands, c, p=dict(zip(edges, P["final"]))), c["tg"])
        rows.append(r)
        print(case, {k: round(v, 3) for k, v in r.items() if k != "case"}, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/res_{arg}.csv", index=False)


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/res_*.csv"))])
        print(len(R), "cases"); print((100 * R.drop(columns=["case"]).mean()).round(2).to_string()); sys.exit(0)
    main(sys.argv[1])
