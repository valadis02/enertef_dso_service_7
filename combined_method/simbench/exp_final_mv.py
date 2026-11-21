"""Final method + ablations on SimBench MV (use case 4), voltage only, one noise draw.
usage: TDM=1 exp_final_mv.py <network index> [cfg range lo-hi] | summary
Metric for switches: share of the true open points found open (documented: the
open points that were not moved), plus false openings."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import glob
import os
import pickle
import sys
import zlib

import numpy as np
import pandas as pd

os.environ["TDM"] = "1"
sys.path.insert(0, SB_CODE)
import exp_mv as M  # noqa: E402
import exp_lv_pipeline as LL  # noqa: E402
from topology.data import add_meter_noise  # noqa: E402

LL.TDM = True
METERS = "class_0.5"
VAR = os.environ.get("VAR", "")           # variant of the combination: hgb, logreg, mlp, nof3, gate01 (default: as published)
GATE = (0.01, 0.99) if VAR == "gate01" else (0.05, 0.95)
if os.environ.get("GLO"):                              # gate override (refinement settings)
    GATE = (float(os.environ["GLO"]), float(os.environ["GHI"]))
BASE_OUT = f"{SB_WORK}/final_mv" + ("85" if os.environ.get("MVSET") == "85" else "") + ("_month" if os.environ.get("MONTH") == "1" else "")
OUT = BASE_OUT + (f"_{VAR}" if VAR else "") + os.environ.get("OUTTAG", "")
os.makedirs(OUT, exist_ok=True)
FINAL = ["log_dist", "dist_rank", "kruskal_in", "dJ", "log_corr", "doc"]
if VAR == "nof3":                                  # ablation: no power-flow residual
    FINAL = [c for c in FINAL if c != "dJ"]
SWAPF = ["dJ_close", "dJ_open", "dJ_second", "n_swaps"]   # idea 1: exchange residual split by role, runner-up, count
PASS2 = ["dJ2", "dec_in"]                                 # idea 5: residual recomputed around the first-pass topology
EXT = {"swapfeat": SWAPF, "pass2": PASS2, "both": SWAPF + PASS2}.get(VAR, [])


def ext_features(net_index):
    """Extra per-line features for the swap and second-pass variants, one network at a time."""
    X, ctx, P = pickle.load(open(f"{BASE_OUT}/data.pkl", "rb"))
    if os.environ.get("P1SRC"):                          # first pass of another classifier (e.g. XGB): decode from it
        P = pickle.load(open(os.environ["P1SRC"], "rb"))[2]
    infos = {c: M.net_info(c) for c in M.MV_NETS}
    code = M.MV_NETS[net_index]; info = infos[code]
    rows = []
    for cs, ix in X.groupby("case").indices.items():
        if not cs.startswith(code + "#"):
            continue
        c_ = int(cs.split("#")[1]); cx = ctx[cs]
        nodes, Vn, Pm, Qm = meas(code, info, c_)
        nodes, Vm, _, Pm, Qm = M.augment(info, nodes, Vn.values, None, Pm, Qm)
        cand = list(X.k.iloc[ix])
        _, _, sw = M.swap_gains(nodes, info, cx["doc"], Pm, Qm, Vm, None)
        dec = M.decode(info, nodes, P["final"][ix], cand)          # out-of-fold first-pass probabilities
        _, best2, _ = M.swap_gains(nodes, info, dec, Pm, Qm, Vm, None)
        for i, k in zip(ix, cand):
            gc = [g for g, o, kk in sw if o == k]; go = [g for g, o, kk in sw if kk == k]
            allg = sorted(gc + go, reverse=True)
            rows.append(dict(row=i, dJ_close=max(gc, default=-1.0), dJ_open=max(go, default=-1.0),
                             dJ_second=allg[1] if len(allg) > 1 else -1.0, n_swaps=sum(g > 0.01 for g in allg),
                             dJ2=best2[k], dec_in=int(dec[k])))
    pd.DataFrame(rows).to_pickle(f"{OUT if os.environ.get('P1SRC') else BASE_OUT}/ext_{net_index}.pkl")
    print(code, "ext done", flush=True)
LIU = ["log_corr", "doc"]


RFC = M.RandomForestClassifier


def predict_var(X, cols):
    """Same folds as M.predict; only the classifier of the combination changes."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    import clf_extra
    clf_extra.GROUPS = X["cfg"]                          # inner tuning folds by configuration
    if VAR in clf_extra.NAMES:
        orig = M.RandomForestClassifier
        M.RandomForestClassifier = lambda **kw: clf_extra.make(VAR)
        try:
            return M.predict(X, cols, "rf", "cfg")
        finally:
            M.RandomForestClassifier = orig
    make = {"hgb": lambda: HistGradientBoostingClassifier(random_state=42),
            "rf300": lambda: RFC(n_estimators=300, random_state=42, n_jobs=-1),
            "rfbal": lambda: RFC(n_estimators=100, class_weight="balanced", min_samples_leaf=3, random_state=42, n_jobs=-1),
            "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            "mlp": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000,
                                                                         early_stopping=True, random_state=42))}[VAR]
    orig = M.RandomForestClassifier
    M.RandomForestClassifier = lambda **kw: make()
    try:
        return M.predict(X, cols, "rf", "cfg")
    finally:
        M.RandomForestClassifier = orig


def meas(code, info, c_):
    d = dict(np.load(f"{M.OUT}/{code}__{c_:02d}.npz")); nodes = [int(b) for b in d["buses"]]
    seed = zlib.crc32(f"{code}#{c_}#0".encode())
    Vn, An = add_meter_noise(pd.DataFrame(d["V"], columns=nodes), pd.DataFrame(d["A"], columns=nodes), METERS, seed=seed)
    rng = np.random.default_rng(seed + 1)
    P = d["P"] * (1 + rng.normal(0, 0.01, d["P"].shape)); Q = d["Q"] * (1 + rng.normal(0, 0.01, d["Q"].shape))
    Vm, P, Q = M.partial(code, info, nodes, Vn.values, P, Q, seed)
    return nodes, pd.DataFrame(Vm, columns=nodes), P, Q


def data():
    path = f"{OUT}/data.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    if VAR and os.path.exists(f"{BASE_OUT}/data.pkl"):    # variants reuse the features; only the probabilities change
        X, ctx, P = pickle.load(open(f"{BASE_OUT}/data.pkl", "rb"))
        if VAR in ("hgb", "logreg", "mlp", "rf300", "rfbal", "xgb_tuned", "xgb_tp", "tabfm"):
            import fe
            Xf, extra = fe.add(X, os.environ.get("FE", ""))   # feature engineering
            if os.environ.get("MVPASS2") == "1":           # second pass: residual around the first-pass topology
                E = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(f"{OUT}/ext_*.pkl"))]).set_index("row").sort_index()
                assert len(E) == len(Xf)
                for col in PASS2:
                    Xf[col] = E[col].values
                extra = extra + PASS2
            P = dict(P, final=predict_var(Xf, FINAL + extra))
        elif VAR == "nof3":
            P = dict(P, final=M.predict(X, FINAL, "rf", "cfg"))
        elif EXT:
            E = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(f"{BASE_OUT}/ext_*.pkl"))]).set_index("row").sort_index()
            assert len(E) == len(X)
            X = X.copy()
            for col in EXT:
                X[col] = E[col].values
            P = dict(P, final=M.predict(X, FINAL + EXT, "rf", "cfg"))
        pickle.dump((X, ctx, P), open(path, "wb"))
        return X, ctx, P
    infos = {c: M.net_info(c) for c in M.MV_NETS}
    Xs, ctx = [], {}
    for f in sorted(glob.glob(f"{M.FEATS}/{METERS}_*.pkl")):
        x, cx = pickle.load(open(f, "rb"))
        x = x[x.case.str.endswith("#0")].copy()
        for cs, g in x.groupby("case"):
            code, c_, _ = cs.split("#"); info = infos[code]
            nodes, Vn, P, Q = meas(code, info, int(c_))
            Vs = pd.DataFrame(Vn.values).rolling(15, center=True, min_periods=1).mean().values
            ix = {n: k for k, n in enumerate(nodes)}
            lc = []
            for k in g.k:
                a, b = tuple(info["edges"][k])
                rho = np.corrcoef(Vs[:, ix[a]], Vs[:, ix[b]])[0, 1] if a in ix and b in ix else 0.0
                lc.append(np.log10(max(1 - rho, 1e-12)))
            x.loc[g.index, "log_corr"] = lc
            ctx[cs] = cx[cs]
        Xs.append(x)
    X = pd.concat(Xs, ignore_index=True)
    P = {}
    for name, cols in (("final", FINAL), ("liu", LIU)):
        P[name] = M.predict(X, cols, "rf", "cfg")
    out = (X, ctx, P)
    pickle.dump(out, open(path, "wb"))
    return out


def metrics(s, truth, doc):
    to = truth == 0
    return dict(acc=(s == truth).mean(), open_found=((s == 0) & to).sum() / to.sum(),
                false_open=int(((s == 0) & ~to).sum()), exact=int((s == truth).all()))


def gated_exchange(nodes, info, st, P, Q, Vm, p, cand):
    """greedy_exchange restricted to lines the forest is unsure about."""
    unsure = {k for k, q in zip(cand, p) if GATE[0] < q < GATE[1]}
    orig = M.swap_gains
    def swaps(*a, **kw):
        J0, best, sw = orig(*a, **kw)
        return J0, best, [x for x in sw if x[1] in unsure and x[2] in unsure]
    M.swap_gains = swaps
    try:
        return M.greedy_exchange(nodes, info, st, P, Q, Vm, None, adaptive=True,
                                 min_gain=float(os.environ.get("MINGAIN", "0.01")))   # acceptance threshold
    finally:
        M.swap_gains = orig


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "ext":
        ext_features(int(sys.argv[2])); sys.exit(0)
    if arg == "summary":
        R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/res_*.csv"))])
        g = R.groupby("method", sort=False)[["acc", "open_found", "exact"]].mean().mul(100).round(2)
        g["false_open"] = R.groupby("method", sort=False).false_open.sum()
        print(len(R) // R.method.nunique(), "cases"); print(g.to_string()); sys.exit(0)
    X, ctx, P = data()
    infos = {c: M.net_info(c) for c in M.MV_NETS}
    code = M.MV_NETS[int(arg)]; info = infos[code]
    lo, hi = map(int, sys.argv[2].split("-")) if len(sys.argv) > 2 else (0, 99)
    rows = []
    for cs, ix in X.groupby("case").indices.items():
        if not cs.startswith(code + "#"):
            continue
        c_ = int(cs.split("#")[1])
        if not lo <= c_ < hi:
            continue
        cx = ctx[cs]
        nodes, Vn, Pm, Qm = meas(code, info, c_)
        nodes, Vm, _, Pm, Qm = M.augment(info, nodes, Vn.values, None, Pm, Qm)
        cand = list(X.k.iloc[ix])
        liu = cx["doc"].copy(); liu[cand] = (P["liu"][ix] > 0.5).astype(int)
        ours = M.greedy_exchange(nodes, info, cx["base"], Pm, Qm, Vm, None, adaptive=True)
        dec = M.decode(info, nodes, P["final"][ix], cand)
        comb = gated_exchange(nodes, info, dec, Pm, Qm, Vm, P["final"][ix], cand)
        for name, s in (("documented", cx["doc"]), ("liu_only", liu), ("variance_distance_spanning_tree", cx["base"]),
                        ("ours_only", ours), ("combined_steps_1_3", dec), ("combined", comb)):
            rows.append(dict(case=cs, method=name, **metrics(s, cx["truth"], cx["doc"])))
    tag = f"{arg}_{lo}" if len(sys.argv) > 2 else arg
    pd.DataFrame(rows).to_csv(f"{OUT}/res_{tag}.csv", index=False)
    print(code, "done", flush=True)
