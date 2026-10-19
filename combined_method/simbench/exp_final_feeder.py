"""Use case 2 (customer-feeder assignment): documented, Variance-Distance Spanning
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
GATE_HI = float(__import__('os').environ.get('ASSIGN_GATE', '0.95'))   # 1.01 = refine every customer


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
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1)
        clf.fit(X.loc[tr, cols], X.loc[tr, "label"]); p[te] = clf.predict_proba(X.loc[te, cols])[:, 1]
    X = X.assign(p=p)
    return {cs: g.sort_values(["i", g.columns[1]]).p.values.reshape(-1, K(cs)) for cs, g in X.groupby("case")}


def feeder():
    import simbench as sb
    import exp_feeder as FD
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

    rows, cases = [], {}
    for code in SIMBENCH_FEEDERS:
        net = sb.get_simbench_net(code); labmap = FD.feeders(net); root = int(net.trafo.lv_bus.iloc[0])
        d = dict(np.load(f"{FD.CACHE}/{code}.npz")); buses = list(d["buses"])
        cust = [b for b in buses if b in labmap]; ci = [buses.index(b) for b in cust]; ri = buses.index(root)
        truth = np.array([labmap[b] for b in cust]); K = truth.max() + 1; Pt = -d["P"][:, ci]
        for seed in range(12):
            rng = np.random.default_rng(seed)
            loss = rng.uniform(0.01, 0.03, K)
            head = np.stack([Pt[:, truth == f].sum(1) for f in range(K)], 1) * (1 + loss) * (1 + rng.normal(0, 0.01, (len(Pt), K)))
            Pc = Pt * (1 + rng.normal(0, 0.01, Pt.shape))
            V = d["V"][:, ci] + rng.normal(0, 0.0025, (len(Pt), len(ci)))
            Vs = d["V"][:, [ri]] + rng.normal(0, 0.0025, (len(Pt), 1))
            doc = truth.copy(); flip = rng.choice(len(doc), size=max(1, round(.15 * len(doc))), replace=False)
            for i in flip:
                doc[i] = rng.choice([f for f in range(K) if f != truth[i]])
            C = corr(np.diff(pd.DataFrame(V - Vs).rolling(15, center=True, min_periods=1).mean().values, axis=0))
            R = head - np.stack([Pc[:, doc == f].sum(1) for f in range(K)], 1); J = (R ** 2).sum()
            for i in range(len(cust)):
                a = doc[i]; p = Pc[:, i]
                for f in range(K):
                    peers = np.where((doc == f) & (np.arange(len(doc)) != i))[0]
                    cc = np.sort(C[i, peers])[::-1] if len(peers) else np.array([0.0])
                    g = 0.0 if f == a else ((R[:, a] ** 2).sum() + (R[:, f] ** 2).sum()
                                            - ((R[:, a] + p) ** 2).sum() - ((R[:, f] - p) ** 2).sum()) / J
                    rows.append(dict(i=i, f=f, case=f"{code}#{seed}", net=code, peer_top3=np.nanmean(cc[:3]),
                                     peer_mean=np.nanmean(cc), power_gain=g, doc=int(a == f), label=int(truth[i] == f)))
            cases[f"{code}#{seed}"] = (truth, doc, Pc, head, K, V - Vs)
        print(code, flush=True)
    X = pd.DataFrame(rows)
    Kof = lambda cs: cases[cs][4]
    PL = rf_lofo(X, ["peer_top3", "peer_mean", "doc"], "net", Kof)
    PF = rf_lofo(X, ["peer_top3", "peer_mean", "power_gain", "doc"], "net", Kof)
    out = []
    for cs, (truth, doc, Pc, head, K, Vrel) in cases.items():
        dec = PF[cs].argmax(1)
        out.append(dict(case=cs, documented=(doc == truth).mean(),
                        variance_distance_spanning_tree=(vt_assign(Vrel, doc) == truth).mean(), liu_only=(PL[cs].argmax(1) == truth).mean(),
                        ours_only=(correct(Pc, head, doc, K, np.ones(len(doc), bool)) == truth).mean(),
                        combined_steps_1_3=(dec == truth).mean(),
                        combined=(correct(Pc, head, dec, K, PF[cs].max(1) < GATE_HI) == truth).mean()))
    return pd.DataFrame(out)


if __name__ == "__main__":
    which = "feeder"
    R = feeder()
    R.to_csv(f"{SB_WORK}/final_{which}" + ("_month" if __import__("os").environ.get("MONTH") == "1" else "") + ".csv", index=False)
    cols = [c for c in R.columns if c != "case"]
    print(which, len(R), "cases"); print((100 * R[cols].mean()).round(2).to_string())
    for c in cols[1:]:
        print(c, "worse than documented:", int((R[c] < R.documented).sum()))
