"""EXTENSION 4: graph-aware classifier in step 2 (message passing over the candidate graph). Original files untouched.

The forest classifies every candidate line alone. Here every line also sees its neighbours:
the buses are the nodes, the candidate lines are the edges of one graph per case.

  node feature   h0_i  = mean of the 6 edge features of the lines touching bus i, plus the number of candidate lines at i
  propagation    h1_i  = mean_{j in N(i)} h0_j ,   h2_i = mean_{j in N(i)} h1_j        (K = 2 rounds, no trainable weights)
  edge feature   [own 6 features] + [h0, h1, h2 of the two end buses: mean and |difference|]
                 + competition: rank of dJ and of log_dist among the lines at each end bus (min and max over the two ends),
                   number of competing lines at each end bus (min and max)
This is the "precomputed message passing" form of a GNN (SIGN / SGC): the graph is used, but the weights are learned by
the classifier on top (MODEL below). No torch is needed.

MODEL (env GNN_CLF):  gb (default, HistGradientBoosting) | mlp (sklearn MLP, scaled) | rf
GNN_K=2 rounds of propagation.  GNN_GRAPH=0 switches the graph features off (control: same classifier, only the 6 own features).
Then steps 3 and 4 as in the original: Kruskal decode + physics refinement (gate GLO/GHI from the environment).

Columns written to res_<feeder>.csv:  gnn_steps_1_3 (after decode)   gnn (after decode + refinement)
Results go to  final_lv<SUFFIX>_gnn_<clf>  next to the baseline folder.
usage: exp_gnn_lv_ext.py <feeder index> | summary
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import glob
import os
import sys as _s0
_s0.argv = _s0.argv[:2]        # run_methods_ext.py passes an extra 'tag' argument that other modules must not see
import pickle
import sys

import numpy as np
import pandas as pd
import networkx as nx

import exp_final_lv_ext as F
from exp_final_lv_ext import L, LF, METERS, SIMBENCH_FEEDERS, SB_WORK, FINAL  # noqa: E402

CLF = os.environ.get("GNN_CLF", "gb")
K = int(os.environ.get("GNN_K", 2))
USE_GRAPH = os.environ.get("GNN_GRAPH", "1") == "1"
OUT = F.OUT + "_gnn_" + CLF + ("" if USE_GRAPH else "_nograph")
os.makedirs(OUT, exist_ok=True)
BASEF = ["log_dist", "dist_rank", "kruskal_in", "dJ", "log_corr", "doc"]


def graph_features(Xc):
    """Xc: rows of ONE case (columns edge + BASEF). Returns a DataFrame with the same index and the graph features."""
    Fe = Xc[BASEF].copy()
    Fe["dJ"] = F._slog(Fe["dJ"].values)                                 # dJ can be +-1e14 for a disconnecting flip
    M = Fe.values.astype(float)                                          # (m, 6)
    ends = [tuple(e) for e in Xc["edge"]]
    nodes = sorted({n for e in ends for n in e}, key=str)
    pos = {n: k for k, n in enumerate(nodes)}
    n, m = len(nodes), len(ends)
    inc = np.zeros((n, m))                                               # bus x line incidence
    for k, (a, b) in enumerate(ends):
        inc[pos[a], k] = inc[pos[b], k] = 1
    deg = inc.sum(1)
    H0 = (inc @ M) / np.maximum(deg, 1)[:, None]
    A = (inc @ inc.T > 0).astype(float); np.fill_diagonal(A, 0)         # bus adjacency in the candidate graph
    An = A / np.maximum(A.sum(1), 1)[:, None]
    Hs = [H0]
    for _ in range(K):
        Hs.append(An @ Hs[-1])
    ia = np.array([pos[a] for a, _ in ends]); ib = np.array([pos[b] for _, b in ends])
    cols = {}
    for r, H in enumerate(Hs):
        for j, name in enumerate(BASEF):
            cols[f"h{r}_{name}_mean"] = (H[ia, j] + H[ib, j]) / 2
            cols[f"h{r}_{name}_absdiff"] = np.abs(H[ia, j] - H[ib, j])
    # competition at each end bus: rank of this line's dJ / log_dist among the lines that touch that bus
    def rank_at(col):
        v = Fe[col].values
        out = np.zeros((m, 2))
        for s, ix in enumerate((ia, ib)):
            for k in range(m):
                mates = np.where(inc[ix[k]] > 0)[0]
                out[k, s] = (v[mates] > v[k]).sum() / max(len(mates) - 1, 1)     # share of rivals with a larger value
        return out
    rd, rl = rank_at("dJ"), rank_at("log_dist")
    cols["rank_dJ_min"], cols["rank_dJ_max"] = rd.min(1), rd.max(1)
    cols["rank_dist_min"], cols["rank_dist_max"] = rl.min(1), rl.max(1)
    cols["deg_min"], cols["deg_max"] = np.minimum(deg[ia], deg[ib]), np.maximum(deg[ia], deg[ib])
    return pd.DataFrame(cols, index=Xc.index)


def make_clf():
    if CLF == "gb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
    if CLF == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    if CLF == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), MLPClassifier((64, 32), alpha=1e-3, max_iter=400, early_stopping=True, random_state=0))
    raise ValueError(f"GNN_CLF must be gb, rf or mlp, got {CLF!r}")


def probs():
    path = f"{OUT}/probs.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    X, _ = LF.add_features(METERS)
    X = X.reset_index(drop=True)
    X["dJ_raw"] = X["dJ"]
    if USE_GRAPH:
        G = pd.concat([graph_features(g) for _, g in X.groupby("case")]).loc[X.index]
    else:
        G = pd.DataFrame(index=X.index)
    Z = pd.concat([X[BASEF].assign(dJ=F._slog(X["dJ"].values)), G], axis=1)
    p = np.zeros(len(X))
    for f in X.feeder.unique():
        tr, te = (X.feeder != f).values, (X.feeder == f).values
        clf = make_clf().fit(Z[tr], X.label[tr])
        p[te] = clf.predict_proba(Z[te])[:, 1]
    out = (X[["case", "feeder", "edge"]].copy(), p)
    pickle.dump(out, open(path, "wb"))
    return out


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "summary":
        R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/res_*.csv"))])
        base = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{F.OUT}/res_*.csv"))])
        R = base.merge(R, on="case", suffixes=("", "_gnn"))
        print(len(R), "cases (percent correct lines)")
        print((100 * R.drop(columns=["case"]).mean()).round(2).to_string())
        sys.exit(0)
    Xe, p = probs()
    C = pickle.load(open(f"{SB_WORK}/lv_ctx_{METERS}_tdm{L.SUFM}.pkl", "rb"))
    code = SIMBENCH_FEEDERS[int(arg)]
    rows = []
    sub = Xe[Xe.feeder == code]
    for case, g in sub.groupby("case"):
        ix = g.index.values; c = C[case]
        edges = list(g.edge); cands = set(edges)
        Gd = L.decode(c["nodes"], c["fixed"], edges, p[ix])
        r = dict(case=case, gnn_steps_1_3=F.share(Gd, c["tg"]),
                 gnn=F.share(F.refine(Gd, cands, c, p=dict(zip(edges, p[ix]))), c["tg"]))
        rows.append(r)
    pd.DataFrame(rows).to_csv(f"{OUT}/res_{arg}.csv", index=False)
    print(code, "done", flush=True)
