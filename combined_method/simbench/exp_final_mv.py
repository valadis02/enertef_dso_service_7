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
GATE = (0.05, 0.95)
OUT = f"{SB_WORK}/final_mv" + ("85" if os.environ.get("MVSET") == "85" else "") + ("_month" if os.environ.get("MONTH") == "1" else "")
os.makedirs(OUT, exist_ok=True)
FINAL = ["log_dist", "dist_rank", "kruskal_in", "dJ", "log_corr", "doc"]
LIU = ["log_corr", "doc"]


def meas(code, info, c_):
    d = dict(np.load(f"{M.OUT}/{code}__{c_:02d}.npz")); nodes = [int(b) for b in d["buses"]]
    seed = zlib.crc32(f"{code}#{c_}#0".encode())
    Vn, An = add_meter_noise(pd.DataFrame(d["V"], columns=nodes), pd.DataFrame(d["A"], columns=nodes), METERS, seed=seed)
    rng = np.random.default_rng(seed + 1)
    P = d["P"] * (1 + rng.normal(0, 0.01, d["P"].shape)); Q = d["Q"] * (1 + rng.normal(0, 0.01, d["Q"].shape))
    return nodes, Vn, P, Q


def data():
    path = f"{OUT}/data.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
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
        return M.greedy_exchange(nodes, info, st, P, Q, Vm, None, adaptive=True)
    finally:
        M.swap_gains = orig


if __name__ == "__main__":
    arg = sys.argv[1]
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
