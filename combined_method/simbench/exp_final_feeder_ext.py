"""EXTENSION (direction 4 for use case 2): graph-aware classifier. Original exp_final_feeder.py untouched.
Per (customer i, feeder f) the forest sees i alone. Added here:
  competition  rank of i's feature for f among the K feeders, and the gap to the best other feeder
  neighbours   k = 5 customers with the most correlated voltage (graph = kNN on the same correlation as the
               features); h1 = mean of the neighbours' (j, f) features, h2 = one more round of the same averaging
Classifier: gradient boosting on own + graph features ("graph") and, as control, on own features only ("nograph").
Then the same power-balance refinement as the combination. Original columns are kept unchanged.
Results: final_feeder_ext<...>.csv

Original description:
Use case 2 (customer-feeder assignment): documented, Variance-Distance Spanning
Tree, random forest (Liu) and the combination.
Liu only: forest on correlation features + documentation, highest probability.
Combined: forest on all features -> highest probability -> power-balance
refinement for customers with highest p < ASSIGN_GATE (0.95; 1.01 = every customer)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, SB_CODE)
sys.path.insert(0, TOPO)
DOCERR = float(__import__('os').environ.get('DOCERR', '0.20'))   # share of wrong documentation (0.20 = 80% correct)
MISSING = __import__('os').environ.get('DOCMODE') == 'missing'   # labels unknown (-1) instead of wrong
NOHEAD = __import__('os').environ.get('NOHEAD') == '1'   # no per-feeder measurement at the LV substation
GATE_HI = float(__import__('os').environ.get('ASSIGN_GATE', '0.95'))   # 1.01 = refine every customer


def vt_assign(V, doc, k=5, known=None):
    """variance_distance_spanning_tree principle for assignments: variance distance of smoothed
    voltages; each customer takes the majority documented label of its k
    electrically closest customers."""
    Vs = pd.DataFrame(V).rolling(15, center=True, min_periods=1).mean().values
    var = Vs.var(0); D = var[:, None] + var[None, :] - 2 * np.cov(Vs.T); np.fill_diagonal(D, np.inf)
    lab = doc.copy()
    if known is not None:                                 # missing labels: only unknown customers, known neighbours
        D[:, ~known] = np.inf
    for i in range(len(doc)):
        if known is not None and known[i]:
            continue
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


FEATS = ["peer_top3", "peer_mean", "power_gain", "doc"]


def graph_feats(X, Cs, k=5):
    """Competition among the K feeders of each customer + 2 rounds of neighbour averaging on the kNN voltage graph."""
    out = []
    for cs, g in X.groupby("case", sort=False):
        n, K = g.i.max() + 1, g.f.max() + 1
        C = np.nan_to_num(Cs[cs], nan=-np.inf)
        nn = np.argsort(-C, axis=1)[:, :min(k, n - 1)]
        cols = {}
        for name in FEATS:
            M = np.full((n, K), np.nan); M[g.i.values, g.f.values] = g[name].values
            Mz = np.nan_to_num(M, nan=0.0)
            rank = (Mz[:, None, :] > Mz[:, :, None]).sum(2) / max(K - 1, 1)      # share of other feeders with a larger value
            other = np.array([[np.max(np.delete(Mz[i], f)) if K > 1 else 0.0 for f in range(K)] for i in range(n)])
            h1 = Mz[nn].mean(1); h2 = h1[nn].mean(1)
            for key, A in (("rank", rank), ("gap", Mz - other), ("h1", h1), ("h2", h2)):
                cols[f"{key}_{name}"] = A[g.i.values, g.f.values]
        out.append(pd.DataFrame(cols, index=g.index))
    return pd.concat(out).loc[X.index]


def gb_lofo(X, cols, group, K):
    from sklearn.ensemble import HistGradientBoostingClassifier
    p = np.zeros(len(X))
    for g in X[group].unique():
        tr, te = (X[group] != g).values, (X[group] == g).values
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
        clf.fit(X.loc[tr, cols], X.loc[tr, "label"]); p[te] = clf.predict_proba(X.loc[te, cols])[:, 1]
    X = X.assign(p=p)
    return {cs: g.sort_values(["i", g.columns[1]]).p.values.reshape(-1, K(cs)) for cs, g in X.groupby("case")}


def feeder():
    import simbench as sb
    import exp_feeder as FD
    import liu_cluster as LC
    from topology.data import SIMBENCH_FEEDERS

    def corr(a):
        a = (a - a.mean(0)) / (a.std(0) + 1e-12); C = a.T @ a / len(a); np.fill_diagonal(C, np.nan); return C

    def correct(Pc, head, lab, K, free, max_moves=200):
        lab = lab.copy()
        R = head - np.stack([Pc[:, lab == f].sum(1) for f in range(K)], 1)
        for _ in range(max_moves):
            best, bi, bf = 0.0, None, None
            for i in np.where(free)[0]:
                a = lab[i]; p = Pc[:, i]; base = (R[:, a] ** 2).sum()
                for f in range(K):
                    if f == a:
                        continue
                    g = base + (R[:, f] ** 2).sum() - ((R[:, a] + p) ** 2).sum() - ((R[:, f] - p) ** 2).sum()
                    if g > best:
                        best, bi, bf = g, i, f
            if bi is None:
                break
            R[:, lab[bi]] += Pc[:, bi]; R[:, bf] -= Pc[:, bi]; lab[bi] = bf
        return lab

    rows, cases, Cs, VV = [], {}, {}, {}
    for code in SIMBENCH_FEEDERS:
        net = sb.get_simbench_net(code); labmap = FD.feeders(net); root = int(net.trafo.lv_bus.iloc[0])
        d = dict(np.load(f"{FD.CACHE}/{code}.npz")); buses = list(d["buses"]); keep = d["P"].sum(1) < 0
        cust = [b for b in buses if b in labmap]; ci = [buses.index(b) for b in cust]; ri = buses.index(root)
        truth = np.array([labmap[b] for b in cust]); K = truth.max() + 1; Pt = -d["P"][:, ci]
        for seed in range(12):
            rng = np.random.default_rng(seed)
            loss = rng.uniform(0.01, 0.03, K)
            head = np.stack([Pt[:, truth == f].sum(1) for f in range(K)], 1) * (1 + loss) * (1 + rng.normal(0, 0.01, (len(Pt), K)))
            Pc = Pt * (1 + rng.normal(0, 0.01, Pt.shape))
            V = d["V"][:, ci] + rng.normal(0, 0.0025, (len(Pt), len(ci)))
            Vs = d["V"][:, [ri]] + rng.normal(0, 0.0025, (len(Pt), 1))
            VV[f"{code}#{seed}"] = (V.astype(np.float32), Vs.astype(np.float32))
            doc = truth.copy(); flip = rng.choice(len(doc), size=max(1, round(DOCERR * len(doc))), replace=False)
            for i in flip:
                doc[i] = -1 if MISSING else rng.choice([f for f in range(K) if f != truth[i]])
            C = corr(np.diff(pd.DataFrame(V - Vs).rolling(15, center=True, min_periods=1).mean().values, axis=0))
            Cs[f"{code}#{seed}"] = C
            R = head - np.stack([Pc[:, doc == f].sum(1) for f in range(K)], 1); J = (R ** 2).sum()
            for i in range(len(cust)):
                a = doc[i]; p = Pc[:, i]
                for f in range(K):
                    peers = np.where((doc == f) & (np.arange(len(doc)) != i))[0]
                    cc = np.sort(C[i, peers])[::-1] if len(peers) else np.array([0.0])
                    if a < 0:                                          # unknown: gain of adding i to f
                        g = ((R[:, f] ** 2).sum() - ((R[:, f] - p) ** 2).sum()) / J
                    else:
                        g = 0.0 if f == a else ((R[:, a] ** 2).sum() + (R[:, f] ** 2).sum()
                                                - ((R[:, a] + p) ** 2).sum() - ((R[:, f] - p) ** 2).sum()) / J
                    rows.append(dict(i=i, f=f, case=f"{code}#{seed}", net=code, peer_top3=np.nanmean(cc[:3]),
                                     peer_mean=np.nanmean(cc), power_gain=g, doc=(-1 if a < 0 else int(a == f)), label=int(truth[i] == f)))
            cases[f"{code}#{seed}"] = (truth, doc, Pc, head, K, V - Vs,
                                         LC.user_feeder_knn(V, doc, K, keep) if MISSING else LC.user_feeder(V, doc, K, keep))
        print(code, flush=True)
    X = pd.DataFrame(rows)
    Kof = lambda cs: cases[cs][4]
    PL = rf_lofo(X, ["peer_top3", "peer_mean", "doc"], "net", Kof)
    own = ["peer_top3", "peer_mean", "doc"] + ([] if NOHEAD else ["power_gain"])
    PF = rf_lofo(X, own, "net", Kof)
    GF = graph_feats(X, Cs)
    if NOHEAD:
        GF = GF[[c for c in GF.columns if not c.endswith("power_gain")]]
    XG = pd.concat([X, GF], axis=1)
    PG = gb_lofo(XG, own + list(GF.columns), "net", Kof)                  # graph-aware
    PN = gb_lofo(X, own, "net", Kof)                                       # control: same classifier, no graph features
    if not MISSING and not NOHEAD:                                         # cache for the multi-task model (FM step 2-3)
        import pickle
        sfx = "_month" if __import__("os").environ.get("MONTH") == "1" else ""
        pickle.dump(dict(X=X, Cs=Cs, VV=VV,
                         cases={cs: (v[0], v[1], v[2].astype(np.float32), v[3].astype(np.float32), v[4]) for cs, v in cases.items()}),
                    open(f"{SB_WORK}/uc2_cache{sfx}.pkl", "wb"))
    out = []
    for cs, (truth, doc, Pc, head, K, Vrel, lab_liu) in cases.items():
        dec = PF[cs].argmax(1)
        known = doc >= 0
        if NOHEAD:                                             # no feeder-head balance: forest decision only
            vt = vt_assign(Vrel, doc) if not MISSING else vt_assign(Vrel, doc, known=known)
            if MISSING:
                dec = np.where(known, doc, dec)
            ours = np.full(len(doc), -9); comb = dec
        elif MISSING:                                          # known labels are correct: keep them, free the rest
            dec = np.where(known, doc, dec); free = ~known
            vt = vt_assign(Vrel, doc, known=known); ours = correct(Pc, head, np.where(known, doc, 0), K, free)
            comb = correct(Pc, head, dec, K, free & (PF[cs].max(1) < GATE_HI))
        else:
            vt = vt_assign(Vrel, doc); ours = correct(Pc, head, doc, K, np.ones(len(doc), bool))
            comb = correct(Pc, head, dec, K, PF[cs].max(1) < GATE_HI)
        ext = {}
        for tag, PP in (("graph", PG), ("nograph", PN)):
            d2 = PP[cs].argmax(1)
            if MISSING:
                d2 = np.where(known, doc, d2)
            if NOHEAD:
                c2 = d2
            elif MISSING:
                c2 = correct(Pc, head, d2, K, free & (PP[cs].max(1) < GATE_HI))
            else:
                c2 = correct(Pc, head, d2, K, PP[cs].max(1) < GATE_HI)
            ext[f"{tag}_steps_1_3"] = (d2 == truth).mean(); ext[tag] = (c2 == truth).mean()
        out.append(dict(**ext, case=cs, documented=(doc == truth).mean(),
                        variance_distance_spanning_tree=(vt == truth).mean(), liu_only=(PL[cs].argmax(1) == truth).mean(), liu_cluster=(lab_liu == truth).mean(),
                        ours_only=(ours == truth).mean(), combined_steps_1_3=(dec == truth).mean(), combined=(comb == truth).mean(),
                        liu_cluster_unknown=(lab_liu == truth)[~known].mean() if MISSING else np.nan,
                        combined_unknown=(comb == truth)[~known].mean() if MISSING else np.nan))
    return pd.DataFrame(out)


if __name__ == "__main__":
    which = "feeder"
    R = feeder()
    R = R[["case"] + [c for c in R.columns if c != "case"]]
    R.to_csv(f"{SB_WORK}/final_{which}_ext" + ("_missing" if MISSING else "") + ("_nohead" if NOHEAD else "") + ("_month" if __import__("os").environ.get("MONTH") == "1" else "") + ".csv", index=False)
    cols = [c for c in R.columns if c != "case"]
    print(which, len(R), "cases"); print((100 * R[cols].mean()).round(2).to_string())
    for c in cols[1:]:
        print(c, "worse than documented:", int((R[c] < R.documented).sum()))
