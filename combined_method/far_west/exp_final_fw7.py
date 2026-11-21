"""Use case 4 on Far West: truth = FWK branches out (default 5);
documentation = all but one right, the last reported on a wrong branch
(4 of 5 = 80% with FWK = 5).
Metric: share of the truly open (out-of-service) branches found, plus false outages.
usage: exp_final_fw7.py build <lo> <hi> | run <lo> <hi> | summary"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import glob
import json
import os
import pickle
import sys
import zlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, FW_CODE)
import fw_ac as F  # noqa: E402
import pipeline as PL  # noqa: E402
import exp_final_fw as EF  # noqa: E402

SRC = f"{FW_WORK}/scen_fw7" + ("_month" if os.environ.get("MONTH") == "1" else "")
OUT = f"{FW_WORK}/final_fw7" + ("_month" if os.environ.get("MONTH") == "1" else "")
SRC_OUT = OUT                                         # cases are built once and shared by every variant
OUT = OUT + os.environ.get("OUTSFX", "")
os.makedirs(OUT, exist_ok=True)
PASS2 = os.environ.get("PASS2") == "1"       # second pass: residual recomputed around the first-pass topology
OUT2 = OUT + "_pass2"
NDRAW = 1 if os.environ.get("MONTH") == "1" else 2


def cases():
    sets = json.load(open(f"{SRC}/sets.json"))
    out = []
    for f in sorted(glob.glob(f"{SRC}/fw7_*.npz")):
        i = int(f[-7:-4]); d = dict(np.load(f))
        if d["ok"].mean() < 140 / 168:                  # keep scenarios with >= 83% solvable hours
            continue
        st = np.ones(F.E, int); st[PL.CAND[sets[i]]] = 0
        for j in range(NDRAW):
            out.append((f"fw7_{i:03d}", j, st, f))
    return out


def build(lo, hi):
    for n, (tid, j, truth, f) in enumerate(cases()):
        path = f"{SRC_OUT}/case_{n:03d}.pkl"
        if not lo <= n < hi or os.path.exists(path) or (F.FWHALF and n % 2 == 1):
            continue
        d = dict(np.load(f)); ok = d["ok"].astype(bool)
        d = {k: (v[ok] if k in ("V", "A", "Pinj", "Ftie", "Qinj", "Qtie") else v) for k, v in d.items()}
        rng = np.random.default_rng(zlib.crc32(f"fw7{tid}#{j}".encode()))
        Vm, S, Va = F.measurements(d, rng)
        doc = truth.copy()
        outs = np.where(truth == 0)[0]
        while True:                                       # one outage reported on a wrong branch
            hide = rng.choice(outs); wrong = rng.choice(PL.CAND)
            if truth[wrong] == 1:
                t = doc.copy(); t[hide] = 1; t[wrong] = 0
                if PL.connected_ok(t):
                    doc = t; break
        X = F.features(doc, Vm, S, Va)
        for k, v in EF.extra_features(Vm, Va).items():
            X[k] = v
        X["dJ"] = np.clip(np.nan_to_num(X["dJ"], nan=0.0, posinf=1.0, neginf=-10.0), -10, 1)
        X["truth"] = truth; X["group"] = tid; X["case"] = f"{tid}#{j}"
        pickle.dump(dict(X=X, Vm=Vm, S=S, Va=Va, truth=truth, doc=doc), open(path, "wb"))


def pass2_features(lo, hi):
    """Residual features around the decoded first-pass topology (first-pass p are out-of-fold)."""
    os.makedirs(OUT2, exist_ok=True)
    P = pickle.load(open(f"{OUT}/probs.pkl", "rb"))
    for n, f in enumerate(sorted(glob.glob(f"{SRC_OUT}/case_*.pkl"))):
        path = f"{OUT2}/feat_{n:03d}.pkl"
        if not lo <= n < hi or os.path.exists(path):
            continue
        c = pickle.load(open(f, "rb"))
        dec = PL.decode(P["final"][n])
        X2 = F.features(dec, c["Vm"], c["S"], c.get("Va"))
        pd.DataFrame(dict(dJ2=np.clip(np.nan_to_num(X2["dJ"], nan=0.0, posinf=1.0, neginf=-10.0), -10, 1),
                          cg2=X2["conv_gain"].values, dec_in=dec)).to_pickle(path)


def run(lo, hi):
    files = sorted(glob.glob(f"{SRC_OUT}/case_*.pkl")); C = [pickle.load(open(f, "rb")) for f in files]
    out = OUT2 if PASS2 else OUT
    if PASS2:
        for n, c in enumerate(C):
            ext = pd.read_pickle(f"{OUT2}/feat_{n:03d}.pkl")
            for k in ext.columns:
                c["X"][k] = ext[k].values
    pp_ = f"{out}/probs.pkl"
    if os.path.exists(pp_):
        P = pickle.load(open(pp_, "rb"))
    else:
        X = pd.concat([c["X"] for c in C], ignore_index=True); P = {}
        X, extra = EF.engineer(X)
        for name, cols in (("final", EF.FINAL + ["conv_gain"] + extra + (["dJ2", "cg2", "dec_in"] if PASS2 else [])), ("liu", EF.LIU)):
            p = np.zeros(len(X))
            for tr, te in GroupKFold(5).split(X, groups=X.group):
                clf = EF.make_clf(name, X)
                clf.fit(X.iloc[tr][cols], X.iloc[tr].truth); p[te] = clf.predict_proba(X.iloc[te][cols])[:, 1]
            P[name] = p.reshape(len(C), F.E)
        pickle.dump(P, open(pp_, "wb"))
    for n, c in enumerate(C):
        path = f"{out}/res_{n:03d}.json"
        if not lo <= n < hi or os.path.exists(path):
            continue
        liu = (P["liu"][n] > 0.5).astype(int)
        dec = PL.decode(P["final"][n])
        sub = slice(None, None, 3)                      # search on every third hour (speed)
        va = None if c.get("Va") is None else c["Va"][sub]
        comb = F.refine(dec, c["S"][sub], c["Vm"][sub], p=P["final"][n], gate=EF.GATE, top=0, max_iter=6, Va=va, min_gain=EF.MINGAIN)
        ours = F.refine(c["doc"], c["S"][sub], c["Vm"][sub], top=0, max_iter=6, Va=va)
        json.dump(dict(truth=c["truth"].tolist(), doc=c["doc"].tolist(), liu_only=liu.tolist(),
                       variance_distance_spanning_tree=c["X"]["kruskal_a" if F.ANG else "kruskal_in"].values.astype(int).tolist(),
                       vdst_voltage=c["X"]["kruskal_in"].values.astype(int).tolist(),
                       ours_only=ours.tolist(), combined_steps_1_3=dec.tolist(), combined=comb.tolist()),
                  open(path, "w"))


def summary():
    rows = []
    for f in sorted(glob.glob(f"{OUT}/res_*.json")):
        r = json.load(open(f)); t = np.array(r["truth"])
        for m in ("doc", "variance_distance_spanning_tree", "liu_only", "ours_only", "combined_steps_1_3", "combined"):
            s = np.array(r[m]); to = t == 0
            rows.append(dict(method=m, acc=(s == t).mean(), open_found=((s == 0) & to).sum() / to.sum(),
                             false_out=((s == 0) & ~to).sum(), exact=(s == t).all()))
    R = pd.DataFrame(rows)
    g = R.groupby("method", sort=False)[["acc", "open_found", "exact"]].mean().mul(100).round(2)
    g["false_out"] = R.groupby("method", sort=False).false_out.sum()
    print(len(R) // 6, "cases"); print(g.to_string())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "pass2":
        pass2_features(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "run":
        run(int(sys.argv[2]), int(sys.argv[3]))
    else:
        summary()
