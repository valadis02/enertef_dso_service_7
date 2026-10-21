"""Add three ideas from Liu et al., 'A data-driven approach for topology correction
in LV networks with PVs' (arXiv 2506.20238), to the SimBench pipeline:

  a) log-transformed Pearson correlation of voltages as an edge feature,
  b) variance distance / correlation recomputed on time steps where PV < load
     (their dynamic data selection; here: net feeder injection < 0),
  c) random forest classifier (their switch-state classifier) vs gradient boosting.

Same cases, same noise realisations, same leave-one-feeder-out protocol as
exp_lv_pipeline.py.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

sys.path.insert(0, SB_CODE)
sys.path.insert(0, TOPO)
import exp_lv_pipeline as L  # noqa: E402
from topology.data import add_meter_noise  # noqa: E402

BASE = ["log_dist", "dist_rank", "kruskal_in", "doc", "dJ"]
NEW = ["log_corr", "log_dist_noPV", "log_corr_noPV"]


def smooth(a, w=15):
    return pd.DataFrame(a).rolling(w, center=True, min_periods=1).mean().values


def pair_stats(M, pairs):
    C = np.corrcoef(M.T)
    var = M.var(0)
    cov = np.cov(M.T)
    D = var[:, None] + var[None, :] - 2 * cov
    return (np.array([D[i, j] for i, j in pairs]),
            np.array([np.log10(max(1 - C[i, j], 1e-12)) for i, j in pairs]))


def add_features(meters):
    import os
    suf = ("_tdm" if os.environ.get("TDM") == "1" else "") + ("_month" if os.environ.get("MONTH") == "1" else "")
    X = pd.read_pickle(f"{SB_WORK}/lv_feats_{meters}{suf}.pkl")
    C = pickle.load(open(f"{SB_WORK}/lv_ctx_{meters}{suf}.pkl", "rb"))
    cache = {}
    new = {k: np.zeros(len(X)) for k in NEW}
    for case, ix in X.groupby("case").indices.items():
        code, seed = case.split("#"); seed = int(seed)
        if code not in cache:
            cache[code] = dict(np.load(f"{L.CACHE}/{code}.npz"))
        d = cache[code]; nodes = list(d["buses"]); idx = {n: k for k, n in enumerate(nodes)}
        Vn, An = add_meter_noise(pd.DataFrame(d["V"], columns=nodes), pd.DataFrame(d["A"], columns=nodes),
                                 meters, seed=seed)                    # identical noise to the main run
        M = smooth((An if An is not None else Vn).values)
        mask = d["P"].sum(1) < 0                                       # PV below load
        c = C[case]
        fixed = [tuple(e) for e in c["fixed"]]
        cand = [tuple(e) for e in X.edge.iloc[ix]]
        pf = [(idx[a], idx[b]) for a, b in fixed]; pc = [(idx[a], idx[b]) for a, b in cand]
        for tag, Mx in (("", M), ("_noPV", M[mask])):
            Df, _ = pair_stats(Mx, pf); Dc, Cc = pair_stats(Mx, pc)
            if tag:
                new["log_dist_noPV"][ix] = np.log10(Dc / np.median(Df))
                new["log_corr_noPV"][ix] = Cc
            else:
                new["log_corr"][ix] = Cc
    for k, v in new.items():
        X[k] = v
    return X, C


def lofo(X, cols, model):
    p = np.zeros(len(X))
    for f in X.feeder.unique():
        tr, te = X.feeder != f, X.feeder == f
        clf = (HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
               if model == "gb" else
               RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        clf.fit(X.loc[tr, cols], X.loc[tr, "label"])
        p[te.values] = clf.predict_proba(X.loc[te, cols])[:, 1]
    res = {}
    for case, ix in X.groupby("case").indices.items():
        c = C_GLOBAL[case]
        res[case] = L.decode(c["nodes"], c["fixed"], list(X.edge.iloc[ix]), p[ix])
    return L.score(res, C_GLOBAL, X)


if __name__ == "__main__":
    rows = []
    for meters in ("class_0.5", "class_0.5+angle"):
        X, C_GLOBAL = add_features(meters)
        base = L.score({k: c["base"] for k, c in C_GLOBAL.items()}, C_GLOBAL, X).set_index("case").acc
        for lab, cols, model in (("previous best (GB, all features)", BASE, "gb"),
                                 ("+ log-correlation", BASE + ["log_corr"], "gb"),
                                 ("+ PV-filtered features", BASE + ["log_dist_noPV", "log_corr_noPV"], "gb"),
                                 ("+ all three new features", BASE + NEW, "gb"),
                                 ("+ all three, random forest", BASE + NEW, "rf"),
                                 ("previous features, random forest", BASE, "rf")):
            s = lofo(X, cols, model).set_index("case").acc
            d = s - base
            rows.append(dict(meters=meters, variant=lab, mean=round(100 * s.mean(), 1),
                             worst_feeder=round(100 * s.groupby(s.index.str.split("#").str[0]).mean().min(), 1),
                             better_than_our_method=int((d > 0).sum()), worse=int((d < 0).sum())))
        print(meters, "done", flush=True)
    T = pd.DataFrame(rows)
    print(T.to_string(index=False))
    T.to_csv(f"{SB_WORK}/lv_liu_features.csv", index=False)
