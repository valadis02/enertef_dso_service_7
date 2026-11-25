"""EXTENSION 7: joint estimation of topology AND line parameters (step 4 replaced). Original files untouched.

Before: J(tree) uses the register's r, x of every line. With wrong lengths in the register, J is wrong too.
Now:    every line of a tree gets a length factor s_e (r and x scaled together, as a length error does):
            V~_i(t) = sum_{e on path root->i} s_e * g_e(t),
            g_e(t)  = r_e * sum_{k downstream of e} P_k/V_k  +  x_e * sum_{k downstream of e} Q_k/V_k
        (with s = 1 this is exactly the model of the original J). The residual is linear in s, so for every tree
            J*(tree) = min_s  ||R(tree, s)||^2 / sigma^2  +  ||s - 1||^2 / TAU^2
        has a closed form (ridge regression; the prior says the register is mostly right).
        Same two mean removals as J (per time over buses, per meter over time).
The search over trees is branch exchange (add a candidate line, remove one on the cycle it closes), so every
hypothesis is a radial tree; a move is accepted while J* drops by more than 1%, as in the original refinement.

Columns written to res_<feeder>.csv (each "fixed" column is the control: same search, s fixed to 1):
  phys_fixed / phys_joint          start = variance_distance_spanning_tree, every candidate may move (no forest)
  comb_fixed / comb_joint          start = Kruskal on the forest p (baseline probs), only GLO < p < GHI may move
  gnn_fixed  / gnn_joint           start = Kruskal on the graph-aware p (if exp_gnn_lv_ext.py was run), same gate
  scale_dev                        mean |s_e - 1| on the final phys_joint tree (diagnostic)

Environment variables:  TAU (prior width of the length factor, default 0.3)  TSTRIDE (every k-th time step, default 2)
usage: exp_joint_lv_ext.py <feeder index> | summary
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import glob
import os
import sys as _s0
_s0.argv = _s0.argv[:2]        # run_methods_ext.py passes an extra 'tag' argument that other modules must not see
import pickle
import sys

import numpy as np
import pandas as pd
import networkx as nx

import exp_final_lv_ext as F
from exp_final_lv_ext import L, METERS, SIMBENCH_FEEDERS, SB_WORK  # noqa: E402

TAU = float(os.environ.get("TAU", 0.3))
TSTRIDE = int(os.environ.get("TSTRIDE", 2))
OUT = F.OUT + "_joint"
os.makedirs(OUT, exist_ok=True)


class JointJ:
    """J*(tree) with per-line length factors (fit=True) or the register's lengths (fit=False)."""

    def __init__(self, c, sv, fit=True, tau=TAU, stride=TSTRIDE):
        sl = slice(None, None, stride)
        V, P, Q = c["V"][sl], c["P"][sl], c["Q"][sl]
        self.nodes, self.root = c["nodes"], c["root"]
        self.idx = {n: k for k, n in enumerate(self.nodes)}
        self.par, self.med = c["par"], c["med"]
        self.Pa, self.Qa = P / V, Q / V                                   # (T, N)
        Vc = V - V.mean(1, keepdims=True); Vc = Vc - Vc.mean(0, keepdims=True)
        self.Vc, self.nV = Vc, float((Vc ** 2).sum())
        self.sv2, self.lam, self.fit = sv ** 2, 1.0 / tau ** 2, fit

    def path_matrix(self, edges):
        """Pm[k, e] = 1 if line e is on the path root -> k (= k is downstream of e). None if not a spanning tree."""
        T = nx.Graph(); T.add_nodes_from(self.nodes); T.add_edges_from(map(tuple, edges))
        if not nx.is_tree(T):
            return None, None
        B = nx.bfs_tree(T, self.root)
        E = [frozenset(e) for e in B.edges()]
        Pm = np.zeros((len(self.nodes), len(E)))
        for j, (u, v) in enumerate(B.edges()):
            for k in nx.descendants(B, v) | {v}:
                Pm[self.idx[k], j] = 1.0
        return Pm, E

    def __call__(self, edges, return_s=False):
        Pm, E = self.path_matrix(edges)
        if Pm is None:
            return (np.inf, None, None) if return_s else np.inf
        r = np.array([self.par.get(e, self.med)[0] for e in E]); x = np.array([self.par.get(e, self.med)[1] for e in E])
        G = (self.Pa @ Pm) * r + (self.Qa @ Pm) * x                       # g_e(t), (T, E)
        Gc = G - G.mean(0, keepdims=True)                                 # per-meter time mean (on the model side)
        Qm = Pm - Pm.mean(0, keepdims=True)                               # per-time bus mean
        A = (Gc.T @ Gc) * (Qm.T @ Qm)                                     # ||Gc diag(s) Qm^T||^2 = s^T A s
        b = ((self.Vc @ Qm) * Gc).sum(0)                                  # <Vc, Gc diag(s) Qm^T> = b^T s
        if self.fit:
            s = np.linalg.solve(A / self.sv2 + self.lam * np.eye(len(E)), b / self.sv2 + self.lam)
        else:
            s = np.ones(len(E))
        J = (self.nV - 2 * b @ s + s @ A @ s) / self.sv2 + (self.lam * ((s - 1) ** 2).sum() if self.fit else 0.0)
        return (J, s, E) if return_s else J


def refine_bx(G0, cands, c, Jf, p=None, min_gain=0.01, max_iter=30):
    """Branch exchange on radial trees with the objective Jf; with p only GLO < p < GHI may move (as in the original)."""
    nodes = c["nodes"]
    free = set(cands) if p is None else {e for e, q in p.items() if F.GATE[0] < q < F.GATE[1]}
    cur = {frozenset(e) for e in G0.edges()}
    J0 = Jf(cur)
    for _ in range(max_iter):
        T = nx.Graph(); T.add_nodes_from(nodes); T.add_edges_from(map(tuple, cur))
        best = (min_gain, None)
        for e in cands:
            if e in cur or e not in free:
                continue
            a, b = tuple(e)
            for u, v in zip(*(lambda pth: (pth[:-1], pth[1:]))(nx.shortest_path(T, a, b))):
                f = frozenset((u, v))
                if f not in cands or f not in free:
                    continue
                g = (J0 - Jf((cur | {e}) - {f})) / J0
                if g > best[0]:
                    best = (g, (e, f))
        if best[1] is None:
            break
        e, f = best[1]
        cur = (cur | {e}) - {f}
        J0 = Jf(cur)
    H = nx.Graph(); H.add_nodes_from(nodes); H.add_edges_from(map(tuple, cur))
    return H


def gnn_probs():
    path = F.OUT + "_gnn_gb/probs.pkl"
    return pickle.load(open(path, "rb")) if os.path.exists(path) else None


def solve_case(c, edges, pf, pg):
    cands = set(edges)
    sv = F.sigmas(c, {frozenset(e) for e in c["doc"].edges()})
    Jfix, Jjoint = JointJ(c, sv, fit=False), JointJ(c, sv, fit=True)
    res = {}
    for name, Jf in (("fixed", Jfix), ("joint", Jjoint)):
        res[f"phys_{name}"] = refine_bx(c["base"], cands, c, Jf)
        Gd = L.decode(c["nodes"], c["fixed"], edges, pf)
        res[f"comb_{name}"] = refine_bx(Gd, cands, c, Jf, p=dict(zip(edges, pf)))
        if pg is not None:
            Gg = L.decode(c["nodes"], c["fixed"], edges, pg)
            res[f"gnn_{name}"] = refine_bx(Gg, cands, c, Jf, p=dict(zip(edges, pg)))
    out = {k: F.share(v, c["tg"]) for k, v in res.items()}
    _, s, _ = Jjoint({frozenset(e) for e in res["phys_joint"].edges()}, return_s=True)
    out["scale_dev"] = float(np.mean(np.abs(s - 1))) if s is not None else np.nan
    return out


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "summary":
        R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/res_*.csv"))])
        base = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{F.OUT}/res_*.csv"))])
        R = base.merge(R, on="case")
        print(len(R), "cases (percent correct lines)")
        print((100 * R.drop(columns=["case"]).mean()).round(2).to_string())
        sys.exit(0)
    Xe, P = F.probs()
    G = gnn_probs()
    C = pickle.load(open(f"{SB_WORK}/lv_ctx_{METERS}_tdm{L.SUFM}.pkl", "rb"))
    code = SIMBENCH_FEEDERS[int(arg)]
    rows = []
    sub = Xe[Xe.feeder == code]
    if G is not None:
        gmap = dict(zip(zip(G[0].case, G[0].edge), G[1]))
    for case, g in sub.groupby("case"):
        ix = g.index.values; c = C[case]; edges = list(g.edge)
        pg = np.array([gmap[(case, e)] for e in edges]) if G is not None else None
        r = dict(case=case); r.update(solve_case(c, edges, P["final"][ix], pg))
        rows.append(r)
        print(code, case, {k: round(v, 3) for k, v in r.items() if k != "case"}, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/res_{arg}.csv", index=False)
    print(code, "done", flush=True)
