"""Liu's original switch-state classifier on SimBench MV (use case 4).

As in his public code (Switch state identification): each sample is one time step,
the features are the voltage magnitudes of the meters of the "switch bar", the label
is which switch is open, StandardScaler + RandomForest(100 trees), equal number of
samples per class, only times when PV < load. Here each normal open point defines a
switch bar: the ring it closes, whose lines are the possible open positions.
Training on the other switching configurations of the same network (4-fold grouped
by configuration, as for the combined method); one decision per ring = majority vote
over the time steps of the evaluated month.

usage: CM_WORK=... DOCERR=... MVSET=85 MONTH=1 TDM=1 exp_mv_liu_switchbar.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import SB_CODE  # noqa: E402
import sys

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, SB_CODE)
import exp_final_mv as E  # noqa: E402
import exp_mv as M  # noqa: E402


def rings_of(info):
    n = info["normal"]
    G = nx.Graph(); G.add_edges_from(tuple(info["edges"][k]) for k in range(len(n)) if n[k] == 1)
    G.add_edges_from(tuple(e) for e in info["bb"]); G.add_edges_from(("S", r) for r in info["roots"] + info["trafo_lv"])
    eid = {frozenset(e): k for k, e in enumerate(map(tuple, info["edges"]))}
    R = {}
    for o in np.where(n == 0)[0]:
        a, b = tuple(info["edges"][o]); p = nx.shortest_path(G, a, b)
        R[int(o)] = [eid[frozenset(e)] for e in zip(p[:-1], p[1:]) if frozenset(e) in eid] + [int(o)]
    return R


def ring_labels(truth, normal, R):
    """Position of each ring's open point in this configuration."""
    lab = {o: o for o in R if truth[o] == 0}
    moved = [o for o in R if truth[o] != 0]
    new = [k for k in np.where(truth == 0)[0] if normal[k] == 1]
    if moved:
        C = np.array([[0 if k in R[o] else 1 for k in new] for o in moved])
        r, c = linear_sum_assignment(C)
        for i, j in zip(r, c):
            lab[moved[i]] = int(new[j])
    return lab


def main():
    X, ctx, _ = E.data()
    rows = []
    for code in M.MV_NETS:
        info = M.net_info(code); R = rings_of(info); normal = info["normal"]
        cases = sorted([c for c in ctx if c.startswith(code + "#")], key=lambda s: int(s.split("#")[1]))
        data = {}
        for cs in cases:
            c_ = int(cs.split("#")[1])
            nodes, Vn, Pm, _ = E.meas(code, info, c_)
            load = Pm.sum(1); keep = load * np.sign(load.mean()) > 0         # PV < load
            data[cs] = (nodes, Vn.values[keep], ring_labels(ctx[cs]["truth"], normal, R))
        cfg = np.arange(len(cases))
        pred = {cs: {} for cs in cases}
        for tr, te in GroupKFold(4).split(cfg, groups=cfg):
            for o, ring in R.items():
                ends = sorted({b for k in ring for b in info["edges"][k]})
                Xs, ys = [], []
                for i in tr:
                    nodes, V, lab = data[cases[i]]; cols = [nodes.index(b) for b in ends if b in nodes]
                    Xs.append(V[:, cols]); ys.append(np.full(len(V), lab[o]))
                Xa = np.vstack(Xs); y = np.concatenate(ys)
                cls, cnt = np.unique(y, return_counts=True); rng = np.random.default_rng(0)
                sel = np.concatenate([rng.choice(np.where(y == c)[0], cnt.min(), replace=False) for c in cls])
                sc = StandardScaler().fit(Xa[sel])
                clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(sc.transform(Xa[sel]), y[sel])
                for i in te:
                    nodes, V, _ = data[cases[i]]; cols = [nodes.index(b) for b in ends if b in nodes]
                    p = clf.predict(sc.transform(V[:, cols]))
                    v, c = np.unique(p, return_counts=True)
                    pred[cases[i]][o] = int(v[c.argmax()])
        for cs in cases:
            s = np.ones(len(normal), int)
            for k in pred[cs].values():
                s[k] = 0
            rows.append(dict(case=cs, method="liu_switchbar", **E.metrics(s, ctx[cs]["truth"], ctx[cs]["doc"])))
        print(code, "done", flush=True)
    out = pd.DataFrame(rows); out.to_csv(f"{E.OUT}/liu_switchbar.csv", index=False)
    print(len(out), "cases: open found", round(100 * out.open_found.mean(), 1), "false openings", int(out.false_open.sum()))


if __name__ == "__main__":
    main()
