"""Final method + ablations on Far West, voltage only (|V| Class 0.5, P, Q 1%).
usage: exp_final_fw.py uc1|uc4 run <lo> <hi>   |   exp_final_fw.py uc1|uc4 summary
One noise draw per truth configuration (92 cases per protocol)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, FW_CODE)
import fw_ac as F  # noqa: E402
import pipeline as PL  # noqa: E402

GATE = (0.05, 0.95)
FINAL = ["log_dist", "dist_rank", "kruskal_in", "dJ", "log_corr", "doc", "base_in"]
LIU = ["log_corr", "doc"]
OUT = f"{FW_WORK}/final_fw" + ("_month" if os.environ.get("MONTH") == "1" else "")
os.makedirs(OUT, exist_ok=True)


def smooth(a, w=4):                                  # ~4 hours, as 15 x 15 min in LV
    return pd.DataFrame(a).rolling(w, center=True, min_periods=1).mean().values


def extra_features(Vm):
    Vs = smooth(Vm)
    d = Vs[:, F.FI] - Vs[:, F.TI]; D = d.var(0)
    C = np.array([np.corrcoef(Vs[:, i], Vs[:, j])[0, 1] for i, j in zip(F.FI, F.TI)])
    uf = list(range(F.N))
    def find(a):
        while uf[a] != a:
            uf[a] = uf[uf[a]]; a = uf[a]
        return a
    kin = np.zeros(F.E, int)
    for k in np.argsort(D):                           # Kruskal on the variance distance
        a, b = find(F.FI[k]), find(F.TI[k])
        if a != b:
            uf[a] = b; kin[k] = 1
    return dict(log_dist=np.log10(D / np.median(D) + 1e-12), dist_rank=D.argsort().argsort() / F.E,
                kruskal_in=kin, log_corr=np.log10(np.clip(1 - C, 1e-12, None)))


def load(proto):
    idx = [i for i, (t, j) in enumerate(F.case_list()) if j == 0]
    C = []
    for i in idx:
        if not os.path.exists(f"{F.CASES}/{proto}_{i:03d}.pkl"):    # fast mode builds every other case
            continue
        c = pickle.load(open(f"{F.CASES}/{proto}_{i:03d}.pkl", "rb"))
        for k, v in extra_features(c["Vm"]).items():
            c["X"][k] = v
        c["X"]["dJ"] = np.clip(np.nan_to_num(c["X"]["dJ"], nan=0.0, posinf=1.0, neginf=-10.0), -10, 1)
        c["i"] = i
        C.append(c)
    return C


def probs(proto, C):
    path = f"{OUT}/probs_{proto}.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    X = pd.concat([c["X"] for c in C], ignore_index=True)
    P = {}
    for name, cols in (("final", FINAL), ("liu", LIU)):
        p = np.zeros(len(X))
        for tr, te in GroupKFold(5).split(X, groups=X.group):
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X.iloc[tr][cols], X.iloc[tr].truth); p[te] = clf.predict_proba(X.iloc[te][cols])[:, 1]
        P[name] = p.reshape(len(C), F.E)
    pickle.dump(P, open(path, "wb"))
    return P


def run(proto, lo, hi):
    C = load(proto); P = probs(proto, C)
    for n, c in enumerate(C):
        if not lo <= n < hi:
            continue
        path = f"{OUT}/{proto}_{n:03d}.json"
        if os.path.exists(path):
            continue
        base = c["X"]["base_in"].values.astype(int)
        liu = (P["liu"][n] > 0.5).astype(int)
        dec = PL.decode(P["final"][n])
        sub = slice(None, None, 3)                      # search on every third hour (speed)
        comb = F.refine(dec, c["S"][sub], c["Vm"][sub], p=P["final"][n], gate=GATE, top=0)
        ours = F.refine(base, c["S"][sub], c["Vm"][sub], max_iter=6, top=0)
        json.dump(dict(case=c["X"].case.iloc[0], truth=c["truth"].tolist(), doc=c["doc"].tolist(),
                       liu_only=liu.tolist(), ours_only=ours.tolist(), combined_steps_1_3=dec.tolist(),
                       variance_distance_spanning_tree=c["X"]["kruskal_in"].values.astype(int).tolist(),
                       combined=comb.tolist()), open(path, "w"))


def summary(proto):
    rows = []
    for f in sorted(glob.glob(f"{OUT}/{proto}_*.json")):
        r = json.load(open(f)); t = np.array(r["truth"]); d = np.array(r["doc"])
        for m in ("doc", "variance_distance_spanning_tree", "liu_only", "ours_only", "combined_steps_1_3", "combined"):
            s = np.array(r[m])
            rows.append(dict(method=m, acc=(s == t).mean(), exact=(s == t).all(),
                             out_found=((s == 0) & (t == 0)).sum(), out_total=(t == 0).sum(),
                             false_out=((s == 0) & (t == 1)).sum()))
    R = pd.DataFrame(rows)
    g = R.groupby("method", sort=False).agg(n=("acc", "size"), acc=("acc", "mean"), exact=("exact", "mean"),
                                            out_found=("out_found", "sum"), out_total=("out_total", "sum"),
                                            false_out=("false_out", "sum"))
    g["acc"] *= 100; g["exact"] *= 100
    print(proto); print(g.round(2).to_string())
    return g


if __name__ == "__main__":
    proto, cmd = sys.argv[1], sys.argv[2]
    if cmd == "run":
        run(proto, int(sys.argv[3]), int(sys.argv[4]))
    else:
        summary(proto)
