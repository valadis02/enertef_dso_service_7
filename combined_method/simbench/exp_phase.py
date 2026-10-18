"""Phase identification on the six SimBench LV feeders (D2.2 use case).

Truth: every customer (bus with load and/or PV) on a random phase.
Documented phases 85% correct (15% of customers labelled with a wrong phase).
Measurements: customer smart meter |V| of its phase (Class 0.5) and P, Q (1%);
substation LV busbar |V| of the three phases (Class 0.5). Topology known.

Methods
  - documented as-is
  - correlation with the substation phase voltages (classic, no documentation)
  - Liu-style: customer-customer voltage correlation + hierarchical clustering
    into 3 groups, groups named by the documented majority
  - kNN correction of the documented labels (customer-customer correlation)
  - physics only: linearised 3-phase model, others at documented phases,
    residual of each customer for each phase hypothesis
  - pipeline: all the above as features, ML (GB / RF) leave-one-feeder-out,
    argmax per customer; optional second pass with features refreshed on the
    first-pass labels.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import sys
import zlib

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

sys.path.insert(0, SB_CODE)
sys.path.insert(0, TOPO)
from phase_sim import Feeder, PH  # noqa: E402
from topology.data import SIMBENCH_FEEDERS  # noqa: E402

SEEDS = range(12)
SV = 0.0025


def common_path_Z(f, cust):
    """Zc[i, j] = sum of positive-sequence line impedance on the shared root path."""
    N = len(f.order); edges_of = []
    for b in cust:
        k = f.idx[b]; path = []
        while k > 0:
            path.append(k); k = f.parent[k]
        edges_of.append(set(path))
    Zc = np.zeros((len(cust), len(cust)), complex)
    for i in range(len(cust)):
        for j in range(i, len(cust)):
            Zc[i, j] = Zc[j, i] = f.z1[list(edges_of[i] & edges_of[j])].sum()
    return Zc


def make_case(f, d, cust, seed):
    rng = np.random.default_rng(zlib.crc32(f"{seed}".encode()) + len(cust))
    pos = {b: k for k, b in enumerate(d["buses"])}
    truth = rng.integers(0, 3, len(cust))
    S = np.zeros((len(d["V"]), len(f.order), 3), complex)
    Sc = np.zeros((len(d["V"]), len(cust)), complex)
    for i, b in enumerate(cust):
        s = 3 * (d["P"][:, pos[b]] + 1j * d["Q"][:, pos[b]])     # whole bus on one phase
        S[:, f.idx[b], truth[i]] = s; Sc[:, i] = s
    V = np.abs(f.solve(S))
    Vc = V[:, [f.idx[b] for b in cust], truth] + rng.normal(0, SV, (len(V), len(cust)))
    Vs = V[:, 0, :] + rng.normal(0, SV, (len(V), 3))
    Sm = Sc.real * (1 + rng.normal(0, 0.01, Sc.shape)) + 1j * Sc.imag * (1 + rng.normal(0, 0.01, Sc.shape))
    doc = truth.copy()
    flip = rng.choice(len(cust), size=max(1, round(0.15 * len(cust))), replace=False)
    for i in flip:
        doc[i] = (truth[i] + rng.integers(1, 3)) % 3
    return dict(truth=truth, doc=doc, Vc=Vc, Vs=Vs, S=Sm)


def corr(a, b):
    a = (a - a.mean(0)) / (a.std(0) + 1e-12); b = (b - b.mean(0)) / (b.std(0) + 1e-12)
    return a.T @ b / len(a)


def physics_resid(c, Zc, lab):
    """rms residual of each customer for each phase, others at labels `lab`."""
    I = -np.conj(c["S"]) * PH[lab][None, :]                       # T x N currents at nominal angle
    out = np.zeros((len(lab), 3))
    own = np.diag(Zc)
    for ph in range(3):
        w = 1 + (lab == ph)                                       # zs = 2 z1 same phase, zm = z1 otherwise
        W = (I * w) @ Zc.T                                        # T x N
        Iown = -np.conj(c["S"]) * PH[ph]
        W += own * (2 * Iown - w * I)                             # replace own term by hypothesis ph
        pred = c["Vs"][:, [ph]] - (W * np.conj(PH[ph])).real
        r = c["Vc"] - pred
        out[:, ph] = np.sqrt(((r - r.mean(0)) ** 2).mean(0))
    return out


def features(c, Zc, lab):
    dV = np.diff(c["Vc"], axis=0); dS = np.diff(c["Vs"], axis=0)
    Cs = corr(dV, dS)                                             # customer x phase
    Cc = corr(dV, dV); np.fill_diagonal(Cc, np.nan)
    R = physics_resid(c, Zc, lab)
    rows = []
    for i in range(len(lab)):
        for ph in range(3):
            peers = np.where((lab == ph) & (np.arange(len(lab)) != i))[0]
            cc = np.sort(Cc[i, peers])[::-1] if len(peers) else np.array([0.0])
            rows.append(dict(i=i, ph=ph, sub_corr=Cs[i, ph], sub_rank=(Cs[i] > Cs[i, ph]).sum(),
                             peer_top3=np.nanmean(cc[:3]), peer_mean=np.nanmean(cc),
                             resid=np.log10(R[i, ph] / R[i].min()), resid_rank=(R[i] < R[i, ph]).sum(),
                             doc=int(lab[i] == ph)))
    return pd.DataFrame(rows), Cs, Cc, R


def liu_cluster(Cc, doc):
    D = np.log10(np.clip(1 - np.nan_to_num(Cc, nan=1.0), 1e-9, None)); D -= D.min(); np.fill_diagonal(D, 0)
    cl = fcluster(linkage(squareform((D + D.T) / 2, checks=False), "average"), 3, "maxclust") - 1
    lab = doc.copy()
    for k in np.unique(cl):
        m = cl == k; lab[m] = np.bincount(doc[m], minlength=3).argmax()
    return lab


def knn(Cc, doc, k=5):
    lab = doc.copy()
    for i in range(len(doc)):
        nn = np.argsort(-np.nan_to_num(Cc[i], nan=-9))[:k]
        lab[i] = np.bincount(doc[nn], minlength=3).argmax()
    return lab


def main():
    rows, base = [], []
    for code in SIMBENCH_FEEDERS:
        f = Feeder(code); d = dict(np.load(f"{SB_WORK}/sb_cache" + ("_month" if __import__("os").environ.get("MONTH") == "1" else "") + f"/{code}.npz"))
        net = f.net
        cust = sorted((set(net.load.bus) | set(net.sgen.bus)) & set(f.idx) - {f.root})
        Zc = common_path_Z(f, cust)
        for seed in SEEDS:
            c = make_case(f, d, cust, seed)
            X, Cs, Cc, R = features(c, Zc, c["doc"])
            case = f"{code}#{seed}"
            X["case"] = case; X["feeder"] = code; X["label"] = (c["truth"][X.i] == X.ph).astype(int).values
            rows.append(X)
            t = c["truth"]
            base.append(dict(case=case, feeder=code, n=len(cust),
                             documented=(c["doc"] == t).mean(),
                             substation_corr=(Cs.argmax(1) == t).mean(),
                             liu_cluster=(liu_cluster(Cc, c["doc"]) == t).mean(),
                             knn=(knn(Cc, c["doc"]) == t).mean(),
                             physics=(R.argmin(1) == t).mean()))
            CASES[case] = (c, Zc)
        print(code, len(cust), "customers", flush=True)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(base)


CASES = {}


def ml(X, cols, model):
    p = np.zeros(len(X))
    for fd in X.feeder.unique():
        tr, te = (X.feeder != fd).values, (X.feeder == fd).values
        clf = (HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0) if model == "gb"
               else RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1))
        clf.fit(X.loc[tr, cols], X.loc[tr, "label"])
        p[te] = clf.predict_proba(X.loc[te, cols])[:, 1]
    X = X.assign(p=p)
    lab = {cs: g.sort_values(["i", "ph"]).p.values.reshape(-1, 3).argmax(1) for cs, g in X.groupby("case")}
    return lab


if __name__ == "__main__":
    X, B = main()
    F = ["sub_corr", "sub_rank", "peer_top3", "peer_mean", "resid", "resid_rank", "doc"]
    res = {}
    for name, cols, model in (("ML: correlations only (GB)", ["sub_corr", "sub_rank", "peer_top3", "peer_mean", "doc"], "gb"),
                              ("ML: all features (GB)", F, "gb"),
                              ("ML: all features (RF)", F, "rf")):
        lab = ml(X, cols, model)
        res[name] = lab
        B[name] = [(lab[cs] == CASES[cs][0]["truth"]).mean() for cs in B.case]
    # second pass: features recomputed around first-pass labels
    lab1 = res["ML: all features (GB)"]
    X2 = []
    for cs in B.case:
        c, Zc = CASES[cs]
        x, *_ = features(c, Zc, lab1[cs])
        x["case"] = cs; x["feeder"] = cs.split("#")[0]; x["label"] = (c["truth"][x.i] == x.ph).astype(int).values
        x["doc"] = X[X.case == cs].doc.values                     # documented flag kept from the asset DB
        x["prev"] = (lab1[cs][x.i] == x.ph).astype(int).values
        X2.append(x)
    X2 = pd.concat(X2, ignore_index=True)
    lab2 = ml(X2, F + ["prev"], "gb")
    B["ML: all features, 2 passes (GB)"] = [(lab2[cs] == CASES[cs][0]["truth"]).mean() for cs in B.case]
    B.to_csv(f"{SB_WORK}/phase_results.csv", index=False)
    cols = [c for c in B.columns if c not in ("case", "feeder", "n")]
    print((100 * B[cols].mean()).round(1).to_string())
    print((100 * B.groupby("feeder")[cols].mean()).round(1).T.to_string())
