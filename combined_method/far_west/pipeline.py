"""Far West topology validation: documented topology ~85% correct + measurements.

Steps
 1. Physics features per physically existing element e (the 108 in-area
    lines/transformers = candidate set):
      - var_dist   : std_t(theta_i - theta_j)            (variance distance, as in the LV method)
      - flow_std   : std_t((theta_i - theta_j) / x_e)    (implied flow; uses asset-DB reactance)
      - dJ         : relative drop of the DC power-flow residual if e's documented
                     status is flipped (needs injections P and tie flows)
      - doc        : documented status
 2. ML: gradient-boosted classifier -> probability that e is in service.
    Trained/tested with GroupKFold over truth topologies (no truth seen in training).
 3. Constrained decoding: Kruskal on the probabilities so every bus stays
    connected (radial constraint is the LV special case), then keep every other
    element with p > 0.5.
 4. Line parameters (Liu-style, our implementation): on the decoded topology,
    least-squares estimate of 1/x per corridor from angles and injections.

Baselines: documented as-is; physics-only greedy residual descent (no ML).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import glob
import json
import zlib
import sys

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, FW_CODE)
import detect  # noqa: E402

INSIDE, B, M, TIE_AT, CAND = detect.network_model()
E = len(B)
FW = detect.FW


# ---------------------------------------------------------------- data
def truths():
    """Yield (truth_id, status vector (1 = in service), measurements dict)."""
    for f in sorted(glob.glob(f"{FW_WORK}/scen/scen_*.npz")):
        s = int(f[-7:-4]); d = dict(np.load(f))
        if not d["ok"].all():
            continue
        st = np.ones(E, int)
        if s > 0:
            st[CAND[s - 1]] = 0
        yield f"single_{s:03d}", st, d
    sets = json.load(open(f"{FW_WORK}/scen_multi/sets.json"))
    for f in sorted(glob.glob(f"{FW_WORK}/scen_multi/multi_*.npz")):
        i = int(f[-7:-4]); d = dict(np.load(f))
        if not d["ok"].all():
            continue
        st = np.ones(E, int); st[CAND[sets[i]]] = 0
        yield f"multi_{i:03d}", st, d


def documented(truth, rate, rng):
    doc = truth.copy()
    k = int(round(rate * E))
    flip = rng.choice(E, size=k, replace=False)
    doc[flip] = 1 - doc[flip]
    return doc


# ---------------------------------------------------------------- step 1
def residual_parts(A, P, F):
    theta = np.deg2rad(A)
    flows = (theta @ M) * B                       # T x E, flow if element in service
    inj = (P - F @ TIE_AT.T) / 100.0              # T x N
    return flows, inj


def residual(status, flows, inj):
    return inj - (flows * status) @ M.T           # T x N


def flip_gain(status, flows, inj):
    """Relative residual drop for flipping each element individually."""
    r = residual(status, flows, inj)
    J = (r ** 2).sum()
    rM = r @ M                                    # T x E : r_i - r_j
    sign = np.where(status == 1, 1.0, -1.0)       # remove: r + f m ; add: r - f m
    dJ = 2 * sign * (flows * rM).sum(0) + 2 * (flows ** 2).sum(0)
    return -dJ / J                                # > 0 means flipping helps


def features(A, P, F, doc):
    flows, inj = residual_parts(A, P, F)
    dtheta = np.deg2rad(A) @ M
    return pd.DataFrame({
        "log_var_dist": np.log10(dtheta.std(0) + 1e-12),
        "log_flow_std": np.log10(flows.std(0) + 1e-12),
        "dJ": flip_gain(doc, flows, inj),
        "doc": doc,
    }), flows, inj


# ---------------------------------------------------------------- step 3
def connected_ok(status):
    G = nx.Graph(); G.add_nodes_from(FW + ["OUT"])
    for k, r in enumerate(INSIDE.itertuples()):
        if status[k]:
            G.add_edge(r.f, r.t)
    for r in pd.read_csv(detect.DATA + "branches.csv").query("tie").itertuples():
        G.add_edge(r.f if r.f in FW else r.t, "OUT")
    return nx.is_connected(G)


_TIES = None


def decode(p):
    """Kruskal on probabilities (connectivity), then add every element with p > 0.5."""
    global _TIES
    if _TIES is None:
        _TIES = [r.f if r.f in FW else r.t for r in pd.read_csv(detect.DATA + "branches.csv").query("tie").itertuples()]
    uf = {n: n for n in FW + ["OUT"]}
    def find(a):
        while uf[a] != a:
            uf[a] = uf[uf[a]]; a = uf[a]
        return a
    for b in _TIES:
        uf[find(b)] = find("OUT")
    status = np.zeros(E, int)
    for k in np.argsort(-p):
        r = INSIDE.iloc[k]; a, b = find(r.f), find(r.t)
        if a != b:
            uf[a] = b; status[k] = 1
    status[p > 0.5] = 1
    return status


# ---------------------------------------------------------------- baseline
def greedy(doc, flows, inj, min_gain=0.01, max_iter=60):
    s = doc.copy()
    for _ in range(max_iter):
        g = flip_gain(s, flows, inj)
        for k in np.argsort(-g):
            if g[k] < min_gain:
                return s
            t = s.copy(); t[k] = 1 - t[k]
            if t[k] == 1 or connected_ok(t):
                s = t
                break
        else:
            return s
    return s


# ---------------------------------------------------------------- step 4
def estimate_b(status, A, P, F):
    """Least squares for 1/x per corridor (parallel elements merged) on the decoded topology."""
    flows, inj = residual_parts(A, P, F)
    dtheta = np.deg2rad(A) @ M
    on = np.where(status == 1)[0]
    corr = {}
    for k in on:
        r = INSIDE.iloc[k]; corr.setdefault(frozenset((r.f, r.t)), []).append(k)
    keys = list(corr)
    cols = np.array([corr[c][0] for c in keys])
    X = np.concatenate([M[:, [c]].T[None] * dtheta[:, [c]][:, :, None] for c in cols], axis=1)  # T x C x N
    X = X.transpose(0, 2, 1).reshape(-1, len(cols))
    b_hat, *_ = np.linalg.lstsq(X, inj.reshape(-1), rcond=None)
    b_true = np.array([B[corr[c]].sum() for c in keys])
    return keys, b_hat, b_true


# ---------------------------------------------------------------- run
def build_rows(rate=0.15, sigma_deg=0.01, p_rel=0.01, n_doc=3, hours=168):
    rows, ctx = [], {}
    for tid, truth, d in truths():
        for j in range(n_doc):
            rng = np.random.default_rng(zlib.crc32(f"{tid}#{j}".encode()))
            part = {k: d[k][:hours] for k in ("A", "Pinj", "Ftie")}
            A, P, F = detect.add_noise(part, rng, sigma_deg=sigma_deg, p_rel=p_rel, p_floor=0.1 if p_rel else 0)
            doc = documented(truth, rate, rng)
            X, flows, inj = features(A, P, F, doc)
            X["truth"] = truth; X["group"] = tid; X["case"] = f"{tid}#{j}"
            rows.append(X); ctx[f"{tid}#{j}"] = (truth, doc, flows, inj, A, P, F)
    return pd.concat(rows, ignore_index=True), ctx


def run(rate=0.15, sigma_deg=0.01, p_rel=0.01, hours=168, feature_sets=None):
    df, ctx = build_rows(rate, sigma_deg, p_rel, hours=hours)
    feature_sets = feature_sets or {
        "ML: all features": ["log_var_dist", "log_flow_std", "dJ", "doc"],
        "ML: no injections (var_dist, flow_std)": ["log_var_dist", "log_flow_std", "doc"],
    }
    out = []
    probs = {name: np.zeros(len(df)) for name in feature_sets}
    for name, cols in feature_sets.items():
        for tr, te in GroupKFold(5).split(df, groups=df.group):
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
            clf.fit(df.iloc[tr][cols], df.iloc[tr].truth)
            probs[name][te] = clf.predict_proba(df.iloc[te][cols])[:, 1]
    for case, idx in df.groupby("case").indices.items():
        truth, doc, flows, inj, A, P, F = ctx[case]
        res = {"case": case, "documented": (doc == truth).mean()}
        g = greedy(doc, flows, inj)
        res["physics only (greedy residual)"] = (g == truth).mean()
        for name in feature_sets:
            st = decode(probs[name][idx])
            res[name] = (st == truth).mean()
            if name == "ML: all features":
                wrong = doc != truth; flagged = st != doc
                res["prec"] = (flagged & wrong).sum() / max(flagged.sum(), 1)
                res["rec"] = (flagged & wrong).sum() / max(wrong.sum(), 1)
                res["_status"] = st
        out.append(res)
    return pd.DataFrame(out), ctx


# ---------------------------------------------------------------- two-pass version
def _cv_probs(df, cols, label="truth"):
    p = np.zeros(len(df))
    for tr, te in GroupKFold(5).split(df, groups=df.group):
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
        clf.fit(df.iloc[tr][cols], df.iloc[tr][label])
        p[te] = clf.predict_proba(df.iloc[te][cols])[:, 1]
    return p


def run_two_pass(rate=0.15, sigma_deg=0.01, p_rel=0.01, hours=168, passes=3):
    """Pass 1 as run(); each later pass recomputes the physics features on the
    previous pass's decoded topology (not on the documented one) and retrains."""
    df, ctx = build_rows(rate, sigma_deg, p_rel, hours=hours)
    cols = ["log_var_dist", "log_flow_std", "dJ", "doc"]
    df["cur"] = df["doc"]
    hist = []
    for k in range(passes):
        use = cols + (["cur", "dJ_cur"] if k > 0 else [])
        p = _cv_probs(df, use)
        dec = {}
        for case, idx in df.groupby("case").indices.items():
            dec[case] = decode(p[idx])
        hist.append(dec)
        # recompute residual feature around the decoded topology for the next pass
        dJc = np.zeros(len(df)); cur = np.zeros(len(df), int)
        for case, idx in df.groupby("case").indices.items():
            truth, doc, flows, inj, A, P, F = ctx[case]
            dJc[idx] = flip_gain(dec[case], flows, inj); cur[idx] = dec[case]
        df["dJ_cur"] = dJc; df["cur"] = cur
    return hist, ctx


def outage_metrics(dec, ctx):
    rows = []
    for case, (truth, doc, *_ ) in ctx.items():
        s = dec[case]
        rows.append(dict(acc=(s == truth).mean(),
                         hidden_found=((s == 0) & (truth == 0) & (doc == 1)).sum(),
                         hidden_total=((truth == 0) & (doc == 1)).sum(),
                         false_out=((s == 0) & (truth == 1)).sum(),
                         stale_fixed=((s == 1) & (truth == 1) & (doc == 0)).sum(),
                         stale_total=((truth == 1) & (doc == 0)).sum()))
    r = pd.DataFrame(rows).sum()
    return dict(accuracy=round(100 * pd.DataFrame(rows).acc.mean(), 1),
                hidden_outages=f"{int(r.hidden_found)}/{int(r.hidden_total)}",
                hidden_recall=round(100 * r.hidden_found / r.hidden_total, 1),
                false_outages=int(r.false_out),
                stale_fixed=f"{int(r.stale_fixed)}/{int(r.stale_total)}")
