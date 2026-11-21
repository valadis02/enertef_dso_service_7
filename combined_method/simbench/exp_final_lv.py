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


def opens(G, tg, cands):
    """Out-of-service (open) candidate lines: found, false alarms, missed."""
    on_t = {frozenset(e) for e in tg.edges()}; on_p = {frozenset(e) for e in G.edges()}
    cs = {frozenset(e) for e in cands}
    to, po = cs - on_t, cs - on_p
    return len(to & po), len(po - to), len(to - po)


def share(G, tg):
    m = evaluate(G, tg)
    return min(m["precision"], m["recall"])


CLF = os.environ.get("CLF", "rf")            # classifier of the combination: rf (Liu's forest), rf300, rfbal, hgb, logreg, mlp
PASS2 = os.environ.get("PASS2") == "1"        # second pass of the forest around the first-pass topology


def make_clf(name):
    """Liu's random forest always stays a random forest; CLF only swaps the classifier of the combination."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "liu" or CLF == "rf":
        return LF.RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    import clf_extra
    if CLF in clf_extra.NAMES:
        return clf_extra.make(CLF)
    if False:
        return LF.RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    if CLF == "rf300":
        return LF.RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    if CLF == "rfbal":
        return LF.RandomForestClassifier(n_estimators=100, class_weight="balanced", min_samples_leaf=3, random_state=42, n_jobs=-1)
    if CLF == "hgb":
        return HistGradientBoostingClassifier(random_state=42)
    if CLF == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if CLF == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, early_stopping=True, random_state=42))
    raise ValueError(CLF)


def probs():
    path = f"{OUT}/probs.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    X, C = LF.add_features(METERS)
    import clf_extra
    clf_extra.GROUPS = X["feeder"]                     # inner tuning folds by feeder
    import fe
    X, EXTRA = fe.add(X, os.environ.get("FE", ""))      # feature engineering for the combination's classifier
    P = {}
    if os.environ.get("P1_FROM"):                   # reuse a cached first pass (same classifier and features)
        P = dict(pickle.load(open(os.environ["P1_FROM"], "rb"))[1])
    for name, cols in (("final", FINAL + EXTRA), ("liu", LIU)):
        if name in P:
            continue
        p = np.zeros(len(X))
        for f in X.feeder.unique():
            tr, te = X.feeder != f, X.feeder == f
            clf = make_clf(name)
            clf.fit(X.loc[tr, cols], X.loc[tr, "label"]); p[te.values] = clf.predict_proba(X.loc[te, cols])[:, 1]
        P[name] = p
    if PASS2:                                    # second pass: residual and state around the first-pass tree
        X = X.copy(); X["dJ2"] = 0.0; X["dec_in"] = 0
        for case, ix in X.groupby("case").indices.items():
            c = C[case]; edges = list(X.edge.iloc[ix])
            cur = {frozenset(e) for e in L.decode(c["nodes"], c["fixed"], edges, P["final"][ix]).edges()}
            Jf = lambda es: L.resid_J(L.zred(c["nodes"], es, c["par"], c["med"], c["root"]), c["P"], c["Q"], c["V"], None, 0.0025, 1.0)
            J0 = Jf(cur)
            for i, e in zip(ix, edges):
                e = frozenset(e)
                X.iat[i, X.columns.get_loc("dJ2")] = (J0 - Jf(cur ^ {e})) / J0
                X.iat[i, X.columns.get_loc("dec_in")] = int(e in cur)
        cols = FINAL + EXTRA + ["dJ2", "dec_in"]; p = np.zeros(len(X))
        for f in X.feeder.unique():                      # first-pass p are out-of-feeder, so no leakage
            tr, te = X.feeder != f, X.feeder == f
            clf = make_clf("final"); clf.fit(X.loc[tr, cols], X.loc[tr, "label"]); p[te.values] = clf.predict_proba(X.loc[te, cols])[:, 1]
        P["final"] = p
    out = (X[["case", "feeder", "edge"]].copy(), P)
    pickle.dump(out, open(path, "wb"))
    return out


def sigmas(c, edges, Va=None):
    Z = L.zred(c["nodes"], edges, c["par"], c["med"], c["root"])
    W = (c["P"] / c["V"] - 1j * c["Q"] / c["V"]) @ Z.T
    r = c["V"] - W.real; r = r - r.mean(1, keepdims=True); r = r - r.mean(0, keepdims=True)
    return max(0.0025, np.sqrt((r ** 2).mean()))


GATE = (float(os.environ.get("GLO", 0.2)), float(os.environ.get("GHI", 0.8)))


def refine(G, cands, c, p=None, min_gain=float(os.environ.get("MINGAIN", "0.01")), max_iter=30):
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
        print(len(R), "cases"); print((100 * R[[c for c in R.columns if c != "case" and "__" not in c]].mean()).round(2).to_string())
        for c in [c for c in R.columns[2:] if "__" not in c]:
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
        res = {"documented": c["doc"]}
        # Liu only: forest on correlations + documentation, threshold 0.5, no constraint
        G = nx.Graph(); G.add_nodes_from(c["nodes"]); G.add_edges_from(map(tuple, c["fixed"]))
        G.add_edges_from(tuple(e) for e, q in zip(edges, P["liu"][ix]) if q > 0.5)
        r["liu_only"] = share(G, c["tg"]); res["liu_only"] = G
        # ours only: steps 1, 3, 4 without the forest: variance_distance_spanning_tree (Kruskal on the
        # variance distance) -> physics refinement
        r["variance_distance_spanning_tree"] = share(c["base"], c["tg"])
        r["ours_only"] = share(refine(c["base"], cands, c), c["tg"])
        # combined
        Gd = L.decode(c["nodes"], c["fixed"], edges, P["final"][ix])
        r["combined_steps_1_3"] = share(Gd, c["tg"])
        Gc = refine(Gd, cands, c, p=dict(zip(edges, P["final"][ix])))
        r["combined"] = share(Gc, c["tg"]); res["combined_steps_1_3"] = Gd; res["combined"] = Gc
        for m, Gm in res.items():                       # second metric: open lines found / false alarms / missed
            r[f"{m}__open_tp"], r[f"{m}__open_fp"], r[f"{m}__open_fn"] = opens(Gm, c["tg"], cands)
        rows.append(r)
    pd.DataFrame(rows).to_csv(f"{OUT}/res_{arg}.csv", index=False)
    print(code, "done", flush=True)
