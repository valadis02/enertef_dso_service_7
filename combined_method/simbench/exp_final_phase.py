"""Use case 3 (phase identification) with the combined method, voltage magnitude only.

Cases from phase_cases.py (random phases, documentation 1 - DOCERR correct).
Features per (customer, phase): correlation with the substation phase voltages,
correlation with the customers documented on the phase, three-phase residual
(linearised, Z0 = 4 Z1), documented label. Random forest leave-one-feeder-out,
highest probability, then residual refinement for customers with p < 0.95.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK  # noqa: E402
import os
import zlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, SB_CODE)
sys.path.insert(0, TOPO)
from phase_cases import Feeder, PH, SEEDS, make_case  # noqa: E402
from topology.data import SIMBENCH_FEEDERS  # noqa: E402
import networkx as nx  # noqa: E402
import liu_cluster as LC  # noqa: E402

GATE_HI = 0.95
CASES = {}
LIU_LAB = {}


REG = os.environ.get("REG", "ideal")


def wrong_z1(z1, seed, share=0.20, err=(0.10, 0.30)):
    """Register lengths wrong by 10-30% (either sign) on `share` of the lines (index 0 is the root)."""
    rng = np.random.default_rng(seed + 300); z = z1.copy()
    for k in rng.choice(np.arange(1, len(z)), size=round(share * (len(z) - 1)), replace=False):
        z[k] *= 1 + rng.choice([-1, 1]) * rng.uniform(*err)
    return z


def common_path_Z(f, cust, z1=None):
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
            Zc[i, j] = Zc[j, i] = (f.z1 if z1 is None else z1)[list(edges_of[i] & edges_of[j])].sum()
    return Zc


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




def cases():
    rows, base = [], []
    for code in SIMBENCH_FEEDERS:
        f = Feeder(code)
        d = dict(np.load(f"{SB_WORK}/sb_cache" + ("_month" if os.environ.get("MONTH") == "1" else "") + f"/{code}.npz"))
        cust = sorted((set(f.net.load.bus) | set(f.net.sgen.bus)) & set(f.idx) - {f.root})
        Zc = common_path_Z(f, cust)
        H = f.G.copy(); H.remove_node(f.root)
        comp = {b: k for k, cc in enumerate(nx.connected_components(H)) for b in cc}
        groups = np.array([comp[b] for b in cust]); keep = d["P"].sum(1) < 0
        for seed in SEEDS:
            c = make_case(f, d, cust, seed)
            if REG == "real":
                Zc = common_path_Z(f, cust, wrong_z1(f.z1, zlib.crc32(f"{code}#{seed}".encode())))
            X, Cs, Cc, R = features(c, Zc, c["doc"])
            case = f"{code}#{seed}"
            X["case"] = case; X["feeder"] = code; X["label"] = (c["truth"][X.i] == X.ph).astype(int).values
            rows.append(X); base.append(dict(case=case)); CASES[case] = (c, Zc)
            LIU_LAB[case] = LC.phase(c["Vc"], c["doc"], groups, keep)
        print(code, len(cust), "customers", flush=True)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(base)


def vt_assign(V, doc, k=5):
    """variance_distance_spanning_tree principle for assignments: variance distance of smoothed
    voltages; each customer takes the majority documented label of its k
    electrically closest customers."""
    Vs = pd.DataFrame(V).rolling(15, center=True, min_periods=1).mean().values
    var = Vs.var(0); D = var[:, None] + var[None, :] - 2 * np.cov(Vs.T); np.fill_diagonal(D, np.inf)
    lab = doc.copy()
    for i in range(len(doc)):
        nn = np.argsort(D[i])[:k]
        lab[i] = np.bincount(doc[nn], minlength=doc.max() + 1).argmax()
    return lab


def rf_lofo(X, cols, group, K):
    p = np.zeros(len(X))
    for g in X[group].unique():
        tr, te = (X[group] != g).values, (X[group] == g).values
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X.loc[tr, cols], X.loc[tr, "label"]); p[te] = clf.predict_proba(X.loc[te, cols])[:, 1]
    X = X.assign(p=p)
    return {cs: g.sort_values(["i", g.columns[1]]).p.values.reshape(-1, K(cs)) for cs, g in X.groupby("case")}


def phase():
    X, B = cases()
    LIU = ["sub_corr", "sub_rank", "peer_top3", "peer_mean", "doc"]
    FIN = ["sub_corr", "sub_rank", "peer_top3", "peer_mean", "resid", "resid_rank", "doc"]
    X = X[["i", "ph"] + [c for c in X.columns if c not in ("i", "ph")]]

    def refine(c, Zc, lab, free, margin=0.05, rounds=6):
        lab = lab.copy()
        for _ in range(rounds):
            R = physics_resid(c, Zc, lab)
            cur = R[np.arange(len(lab)), lab]
            move = free & (cur > (1 + margin) * R.min(1))
            if not move.any():
                break
            lab[move] = R.argmin(1)[move]
        return lab

    PL = rf_lofo(X, LIU, "feeder", lambda cs: 3); PF = rf_lofo(X, FIN, "feeder", lambda cs: 3)
    rows = []
    for cs in B.case:
        c, Zc = CASES[cs]; t = c["truth"]
        allfree = np.ones(len(t), bool)
        dec = PF[cs].argmax(1)
        rows.append(dict(case=cs, documented=(c["doc"] == t).mean(),
                         variance_distance_spanning_tree=(vt_assign(c["Vc"], c["doc"]) == t).mean(),
                         liu_only=(PL[cs].argmax(1) == t).mean(), liu_cluster=(LIU_LAB[cs] == t).mean(),
                         ours_only=(refine(c, Zc, c["doc"], allfree) == t).mean(),
                         combined_steps_1_3=(dec == t).mean(),
                         combined=(refine(c, Zc, dec, PF[cs].max(1) < GATE_HI) == t).mean()))
    return pd.DataFrame(rows)




if __name__ == "__main__":
    R = phase()
    R.to_csv(f"{SB_WORK}/final_phase" + ("_month" if os.environ.get("MONTH") == "1" else "") + ".csv", index=False)
    cols = [c for c in R.columns if c != "case"]
    print("phase", len(R), "cases"); print((100 * R[cols].mean()).round(2).to_string())
