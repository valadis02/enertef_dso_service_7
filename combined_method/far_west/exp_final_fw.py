"""Final method + ablations on Far West (|V| Class 0.5, P, Q 1%; PMU angles with ANGLE=<sigma deg>).
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

GATE = (float(os.environ.get("FWGLO", "0.05")), float(os.environ.get("FWGHI", "0.95")))
MINGAIN = float(os.environ.get("MINGAIN", "0.01"))   # acceptance: minimum relative residual gain
FINAL = ["log_dist", "dist_rank", "kruskal_in", "dJ", "log_corr", "doc", "base_in"]
if F.ANG:                                             # angle variant: variance distance on the angle differences too
    FINAL = FINAL + ["log_dist_a", "dist_rank_a", "kruskal_a"]
LIU = ["log_corr", "doc"]
OUT = f"{FW_WORK}/final_fw" + ("_month" if os.environ.get("MONTH") == "1" else "") + os.environ.get("OUTSFX", "")
CLF = os.environ.get("CLF", "rf")                    # classifier of the combination (Liu's forest never changes)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simbench"))


def make_clf(name, X=None):
    if name == "liu" or CLF == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    import clf_extra
    if X is not None:
        clf_extra.GROUPS = X["group"]
    return clf_extra.make(CLF)


def engineer(X):
    """Feature engineering (FE env, e.g. rel,loc): end-points of every branch for the local features."""
    import fe
    X = X.copy(); X["case"] = np.arange(len(X)) // F.E          # one block of F.E branches per case
    X["edge"] = [(int(a), int(b)) for a, b in zip(np.tile(F.FI, len(X) // F.E), np.tile(F.TI, len(X) // F.E))]
    return fe.add(X, os.environ.get("FE", ""))
os.makedirs(OUT, exist_ok=True)
PASS2 = os.environ.get("PASS2") == "1"       # second pass: residual recomputed around the first-pass topology
OUT1 = OUT                                   # first pass
OUT2 = OUT + "_pass2"
if PASS2:
    OUT = OUT2; os.makedirs(OUT, exist_ok=True)
P2COLS = ["dJ2", "cg2", "dec_in"]


def pass2_features(proto, lo, hi):
    """Residual features around the decoded first-pass topology (first-pass p are out-of-fold)."""
    os.makedirs(OUT2, exist_ok=True)
    C = load(proto); P = pickle.load(open(f"{OUT1}/probs_{proto}.pkl", "rb"))
    for n, c in enumerate(C):
        path = f"{OUT2}/feat_{proto}_{n:03d}.pkl"
        if not lo <= n < hi or os.path.exists(path):
            continue
        dec = PL.decode(P["final"][n])
        X2 = F.features(dec, c["Vm"], c["S"], c.get("Va"))
        pd.DataFrame(dict(dJ2=np.clip(np.nan_to_num(X2["dJ"], nan=0.0, posinf=1.0, neginf=-10.0), -10, 1),
                          cg2=X2["conv_gain"].values, dec_in=dec)).to_pickle(path)


def smooth(a, w=4):                                  # ~4 hours, as 15 x 15 min in LV
    return pd.DataFrame(a).rolling(w, center=True, min_periods=1).mean().values


def _kruskal(D):
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
    return kin


def extra_features(Vm, Va=None):
    Vs = smooth(Vm)
    d = Vs[:, F.FI] - Vs[:, F.TI]; D = d.var(0)
    C = np.array([np.corrcoef(Vs[:, i], Vs[:, j])[0, 1] for i, j in zip(F.FI, F.TI)])
    out = dict(log_dist=np.log10(D / np.median(D) + 1e-12), dist_rank=D.argsort().argsort() / F.E,
               kruskal_in=_kruskal(D), log_corr=np.log10(np.clip(1 - C, 1e-12, None)))
    if Va is not None:                                # Variance-Distance Spanning Tree on the angles
        As = smooth(Va); Da = (As[:, F.FI] - As[:, F.TI]).var(0)
        out.update(log_dist_a=np.log10(Da / np.median(Da) + 1e-12), dist_rank_a=Da.argsort().argsort() / F.E,
                   kruskal_a=_kruskal(Da))
    return out


def load(proto):
    idx = [i for i, (t, j) in enumerate(F.case_list()) if j == 0]
    C = []
    for i in idx:
        if not os.path.exists(f"{F.CASES}/{proto}_{i:03d}.pkl"):    # fast mode builds every other case
            continue
        c = pickle.load(open(f"{F.CASES}/{proto}_{i:03d}.pkl", "rb"))
        for k, v in extra_features(c["Vm"], c.get("Va")).items():
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
    X, extra = engineer(X)
    P = {}
    for name, cols in (("final", FINAL + extra + (P2COLS if PASS2 else [])), ("liu", LIU)):
        p = np.zeros(len(X))
        for tr, te in GroupKFold(5).split(X, groups=X.group):
            clf = make_clf(name, X)
            clf.fit(X.iloc[tr][cols], X.iloc[tr].truth); p[te] = clf.predict_proba(X.iloc[te][cols])[:, 1]
        P[name] = p.reshape(len(C), F.E)
    pickle.dump(P, open(path, "wb"))
    return P


def run(proto, lo, hi):
    C = load(proto)
    if PASS2:
        for n, c in enumerate(C):
            ext = pd.read_pickle(f"{OUT2}/feat_{proto}_{n:03d}.pkl")
            for k in ext.columns:
                c["X"][k] = ext[k].values
    P = probs(proto, C)
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
        va = None if c.get("Va") is None else c["Va"][sub]
        comb = F.refine(dec, c["S"][sub], c["Vm"][sub], p=P["final"][n], gate=GATE, top=0, Va=va, min_gain=MINGAIN)
        ours = F.refine(base, c["S"][sub], c["Vm"][sub], max_iter=6, top=0, Va=va)
        vdst = c["X"]["kruskal_a" if F.ANG else "kruskal_in"].values.astype(int)
        json.dump(dict(case=c["X"].case.iloc[0], truth=c["truth"].tolist(), doc=c["doc"].tolist(),
                       liu_only=liu.tolist(), ours_only=ours.tolist(), combined_steps_1_3=dec.tolist(),
                       variance_distance_spanning_tree=vdst.tolist(),
                       vdst_voltage=c["X"]["kruskal_in"].values.astype(int).tolist(),
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
    if cmd == "pass2":
        pass2_features(proto, int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "run":
        run(proto, int(sys.argv[3]), int(sys.argv[4]))
    else:
        summary(proto)
