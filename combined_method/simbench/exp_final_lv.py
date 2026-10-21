"""Final method + ablations on SimBench LV, use case 1 (topology), voltage only.
usage: TDM=1 exp_final_lv.py <feeder index> | summary
Metric: share of correct lines = min(precision, recall) of the line set
(= recall for a tree; penalises extra lines of an unconstrained answer)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
import networkx as nx

os.environ["TDM"] = "1"
sys.path.insert(0, SB_CODE)
sys.path.insert(0, TOPO)
import exp_lv_pipeline as L  # noqa: E402
import exp_liu_features as LF  # noqa: E402
from topology import evaluate  # noqa: E402
from topology.data import SIMBENCH_FEEDERS  # noqa: E402

L.TDM = True
METERS = "class_0.5"
OUT = f"{SB_WORK}/final_lv" + os.environ.get("SUFFIX", "")
os.makedirs(OUT, exist_ok=True)
FINAL = ["log_dist", "dist_rank", "kruskal_in", "dJ", "log_corr", "doc"]
LIU = ["log_corr", "doc"]


def share(G, tg):
    m = evaluate(G, tg)
    return min(m["precision"], m["recall"])


def probs():
    path = f"{OUT}/probs.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    X, C = LF.add_features(METERS)
    P = {}
    for name, cols in (("final", FINAL), ("liu", LIU)):
        p = np.zeros(len(X))
        for f in X.feeder.unique():
            tr, te = X.feeder != f, X.feeder == f
            clf = LF.RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X.loc[tr, cols], X.loc[tr, "label"]); p[te.values] = clf.predict_proba(X.loc[te, cols])[:, 1]
        P[name] = p
    out = (X[["case", "feeder", "edge"]].copy(), P)
    pickle.dump(out, open(path, "wb"))
    return out


def sigmas(c, edges, Va=None):
    Z = L.zred(c["nodes"], edges, c["par"], c["med"], c["root"])
    W = (c["P"] / c["V"] - 1j * c["Q"] / c["V"]) @ Z.T
    r = c["V"] - W.real; r = r - r.mean(1, keepdims=True); r = r - r.mean(0, keepdims=True)
    return max(0.0025, np.sqrt((r ** 2).mean()))


GATE = (float(os.environ.get("GLO", 0.2)), float(os.environ.get("GHI", 0.8)))


def refine(G, cands, c, p=None, min_gain=0.01, max_iter=30):
    """Branch exchange, pure additions and pure removals of candidate lines,
    accepted while the residual drops > min_gain; with p, only 0.2 < p < 0.8 may move."""
    nodes, root = c["nodes"], c["root"]
    free = set(cands) if p is None else {e for e, q in p.items() if GATE[0] < q < GATE[1]}
    cur = {frozenset(e) for e in G.edges()}
    sv = sigmas(c, cur)
    J = lambda es: L.resid_J(L.zred(nodes, es, c["par"], c["med"], root), c["P"], c["Q"], c["V"], None, sv, 1.0)
    J0 = J(cur)
    for _ in range(max_iter):
        T = nx.Graph(); T.add_nodes_from(nodes); T.add_edges_from(map(tuple, cur))
        ncomp = nx.number_connected_components(T)
        best = (min_gain, None)
        for e in cands:
            a, b = tuple(e)
            if e in cur:
                if e not in free:
                    continue
                T.remove_edge(a, b)                               # pure removal if it keeps connectivity
                if nx.number_connected_components(T) == ncomp:
                    g = (J0 - J(cur - {e})) / J0
                    if g > best[0]:
                        best = (g, (None, e))
                T.add_edge(a, b)
                continue
            if e not in free:
                continue
            if not nx.has_path(T, a, b):
                g = (J0 - J(cur | {e})) / J0
                if g > best[0]:
                    best = (g, (e, None))
                continue
            path = nx.shortest_path(T, a, b)
            for u, v in zip(path[:-1], path[1:]):
                f = frozenset((u, v))
                if f not in cands or (p is not None and f not in free):
                    continue
                g = (J0 - J((cur | {e}) - {f})) / J0
                if g > best[0]:
                    best = (g, (e, f))
        if best[1] is None:
            break
        e, f = best[1]
        cur = (cur | ({e} if e else set())) - ({f} if f else set())
        J0 = J(cur)
    H = nx.Graph(); H.add_nodes_from(nodes); H.add_edges_from(map(tuple, cur))
    return H


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "summary":
        R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/res_*.csv"))])
        print(len(R), "cases"); print((100 * R.drop(columns=["case"]).mean()).round(2).to_string())
        for c in R.columns[2:]:
            print(c, "worse than documented:", int((R[c] < R.documented).sum()))
        sys.exit(0)
    Xe, P = probs()
    C = pickle.load(open(f"{SB_WORK}/lv_ctx_{METERS}_tdm{L.SUFM}.pkl", "rb"))
    code = SIMBENCH_FEEDERS[int(arg)]
    rows = []
    sub = Xe[Xe.feeder == code]
    for case, g in sub.groupby("case"):
        ix = g.index.values; c = C[case]
        edges = list(g.edge); cands = set(edges)
        r = dict(case=case, documented=share(c["doc"], c["tg"]))
        # Liu only: forest on correlations + documentation, threshold 0.5, no constraint
        G = nx.Graph(); G.add_nodes_from(c["nodes"]); G.add_edges_from(map(tuple, c["fixed"]))
        G.add_edges_from(tuple(e) for e, q in zip(edges, P["liu"][ix]) if q > 0.5)
        r["liu_only"] = share(G, c["tg"])
        # ours only: steps 1, 3, 4 without the forest: variance_distance_spanning_tree (Kruskal on the
        # variance distance) -> physics refinement
        r["variance_distance_spanning_tree"] = share(c["base"], c["tg"])
        r["ours_only"] = share(refine(c["base"], cands, c), c["tg"])
        # combined
        Gd = L.decode(c["nodes"], c["fixed"], edges, P["final"][ix])
        r["combined_steps_1_3"] = share(Gd, c["tg"])
        r["combined"] = share(refine(Gd, cands, c, p=dict(zip(edges, P["final"][ix]))), c["tg"])
        rows.append(r)
    pd.DataFrame(rows).to_csv(f"{OUT}/res_{arg}.csv", index=False)
    print(code, "done", flush=True)
