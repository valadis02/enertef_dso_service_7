"""EXTENSION 6: single-shot MILP. Replaces steps 2-4 (no forest, no training, no greedy search). Original files untouched.

Binary z_e = 1 if candidate line e exists (documented lines that are not in doubt are fixed to 1).
Physics = exactly the linear model behind J:   v_i = v_root + sum_k Z_ik I_k ,  I_k = (P_k - jQ_k)/V_k = a_k + j b_k .
On a tree this is the same as: for every line e=(a->b):  v_b - v_a = r_e * fa_e - x_e * fb_e ,
where fa_e, fb_e are the sums of a_k, b_k over all buses downstream of the line (a flow of "current").
Constraints (for every sampled time step t):
  flow conservation        inflow_k - outflow_k = a_k(t)  (and = b_k(t) for fb), all non-root buses k
  line switched off        |fa_e|, |fb_e| <= M z_e
  voltage drop (big-M)     |v_b - v_a - (r fa - x fb)| <= Mv (1 - z_e)
  radial                   sum_e z_e = N - 1,   plus a static unit-flow g so that every bus is connected to the root
Objective (L1, linear):    min sum_t sum_i | Vmeas_i(t) - v_i(t) - d_i |      d_i = per-meter offset, sum d_i = 0
(v_root(t) is free, d_i removes the per-meter time mean: the same two mean removals as in J.)
Columns written:  milp         MILP solution as it is (Kruskal on z if it is not a tree)
                  milp_gated   same, but lines with forest p > GHI fixed to 1 and p < GLO fixed to 0 (hybrid, needs baseline probs)
                  milp_gap     relative MIP gap reported by HiGHS for `milp` (diagnostic; 0 = proven optimal)

Environment variables:  OBJ (l1 | pwl: objective, default l1)  TSEL (linspace | extreme: which time steps, default linspace)  KT (number of sampled time steps, default 24)  TLIM (seconds per solve, default 120)
                        GAP (relative gap, default 1e-3)
usage: exp_milp_lv_ext.py <feeder index> | summary
Needs scipy >= 1.9 (scipy.optimize.milp, HiGHS). No other solver.
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
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

import exp_final_lv_ext as F
from exp_final_lv_ext import L, METERS, SIMBENCH_FEEDERS, SB_WORK  # noqa: E402

KT = int(os.environ.get("KT", 24))
TLIM = float(os.environ.get("TLIM", 120))
GAP = float(os.environ.get("GAP", 1e-3))
OBJ = os.environ.get("OBJ", "l1")                         # l1 | pwl (piecewise-linear approximation of the squared residual, Huber-like)
S0 = 0.0025                                               # sigma of the voltage meters
PWL = (0.0, 1.0, -1.0, 2.5, -2.5, 5.0, -5.0)               # tangent points in units of sigma
TSEL = os.environ.get("TSEL", "linspace")                 # linspace | extreme
OUT = F.OUT + "_milp"
os.makedirs(OUT, exist_ok=True)


class Rows:
    """Sparse constraint builder: rows of  lo <= sum coef*x <= hi."""
    def __init__(self):
        self.r, self.c, self.v, self.lo, self.hi, self.n = [], [], [], [], [], 0

    def add(self, cols, vals, lo, hi):
        self.r += [self.n] * len(cols); self.c += list(cols); self.v += list(vals)
        self.lo.append(lo); self.hi.append(hi); self.n += 1

    def matrix(self, nvar):
        return coo_matrix((self.v, (self.r, self.c)), shape=(self.n, nvar)).tocsr()


def solve_milp(c, edges, fixed, force=None):
    """edges: list of candidate frozensets. fixed: lines forced to 1. force: {edge: 0/1} extra fixings.
    Returns (z as dict edge->float, status, gap)."""
    nodes, root = c["nodes"], c["root"]
    N = len(nodes); idx = {n: k for k, n in enumerate(nodes)}
    E = [e for e in sorted(set(edges) | set(fixed), key=lambda e: tuple(sorted(e)))]
    nE = len(E)
    ab = [tuple(sorted(e)) for e in E]
    r = np.array([c["par"].get(e, c["med"])[0] for e in E]); x = np.array([c["par"].get(e, c["med"])[1] for e in E])
    if TSEL == "extreme":      # time steps with the largest voltage drops: half with the lowest, half with the highest net current
        I_all = (c["P"] / c["V"]).sum(1)
        order = np.argsort(I_all)
        h = KT // 2
        T = np.sort(np.concatenate([order[:h], order[-(KT - h):]]))
    else:                      # evenly spaced
        T = np.linspace(0, c["V"].shape[0] - 1, KT).astype(int)
    Vm = c["V"][T]
    I = (c["P"][T] / c["V"][T]) - 1j * (c["Q"][T] / c["V"][T])
    A_, B_ = I.real, I.imag                                                # (KT, N): a_k, b_k
    Mf_t = 1.05 * np.maximum(np.abs(A_).sum(1), np.abs(B_).sum(1)) + 1e-9      # big-M per time step (tighter than one global value)
    # variable layout
    nz = nE; off_z = 0
    off_fa = off_z + nz; off_fb = off_fa + KT * nE; off_v = off_fb + KT * nE
    off_u = off_v + KT * N; off_d = off_u + KT * N; off_g = off_d + N; nvar = off_g + nE
    fa = lambda e, t: off_fa + t * nE + e
    fb = lambda e, t: off_fb + t * nE + e
    v = lambda i, t: off_v + t * N + i
    u = lambda i, t: off_u + t * N + i
    d = lambda i: off_d + i
    g = lambda e: off_g + e
    lb = np.full(nvar, -np.inf); ub = np.full(nvar, np.inf)
    lb[off_z:off_z + nz] = 0; ub[off_z:off_z + nz] = 1
    fx = {frozenset(e) for e in fixed}
    force = force or {}
    for k, e in enumerate(E):
        if e in fx or force.get(e) == 1:
            lb[k] = 1
        if force.get(e) == 0 and e not in fx:
            ub[k] = 0
    for t in range(KT):
        for o in (off_fa, off_fb):
            lb[o + t * nE:o + (t + 1) * nE] = -Mf_t[t]; ub[o + t * nE:o + (t + 1) * nE] = Mf_t[t]
    lb[off_v:off_u] = 0.8; ub[off_v:off_u] = 1.2
    lb[off_u:off_d] = 0; ub[off_u:off_d] = np.inf
    lb[off_d:off_g] = -0.2; ub[off_d:off_g] = 0.2
    lb[off_g:] = -(N - 1); ub[off_g:] = N - 1
    R = Rows()
    ri = idx[root]
    for t in range(KT):
        for k, (a, b) in enumerate(ab):
            ia, ib = idx[a], idx[b]
            for f_ in (fa, fb):                                            # |f| <= Mf z
                R.add([f_(k, t), k], [1, -Mf_t[t]], -np.inf, 0)
                R.add([f_(k, t), k], [-1, -Mf_t[t]], -np.inf, 0)
            Mv = 0.4 + (r[k] + x[k]) * Mf_t[t]                             # |v_b - v_a| <= 0.4 plus the largest possible drop
            # v_b - v_a - r fa + x fb  in [-Mv(1-z), Mv(1-z)]
            R.add([v(ib, t), v(ia, t), fa(k, t), fb(k, t), k], [1, -1, -r[k], x[k], Mv], -np.inf, Mv)
            R.add([v(ib, t), v(ia, t), fa(k, t), fb(k, t), k], [-1, 1, r[k], -x[k], Mv], -np.inf, Mv)
        for i, n in enumerate(nodes):
            if n == root:
                continue
            for f_, inj in ((fa, A_[t, i]), (fb, B_[t, i])):               # inflow - outflow = injection
                cols, vals = [], []
                for k, (a, b) in enumerate(ab):
                    if idx[b] == i:
                        cols.append(f_(k, t)); vals.append(1)
                    elif idx[a] == i:
                        cols.append(f_(k, t)); vals.append(-1)
                R.add(cols, vals, inj, inj)
        for i in range(N):
            if OBJ == "l1":                                                # u >= |Vm - v - d|
                R.add([v(i, t), d(i), u(i, t)], [-1, -1, -1], -np.inf, -Vm[t, i])
                R.add([v(i, t), d(i), u(i, t)], [1, 1, -1], -np.inf, Vm[t, i])
            else:                                                          # u >= 2c*rho - c^2, rho = (Vm - v - d)/S0 in units of sigma
                for cc in PWL:
                    R.add([v(i, t), d(i), u(i, t)], [-2 * cc / S0, -2 * cc / S0, -1], -np.inf, cc * cc - 2 * cc * Vm[t, i] / S0)
    R.add(list(range(off_z, off_z + nz)), [1] * nz, N - 1, N - 1)          # radial: N-1 lines
    R.add([d(i) for i in range(N)], [1] * N, 0, 0)
    for k in range(nE):                                                    # connectivity flow
        R.add([g(k), k], [1, -(N - 1)], -np.inf, 0)
        R.add([g(k), k], [-1, -(N - 1)], -np.inf, 0)
    for i, n in enumerate(nodes):
        if n == root:
            continue
        cols, vals = [], []
        for k, (a, b) in enumerate(ab):
            if idx[b] == i:
                cols.append(g(k)); vals.append(1)
            elif idx[a] == i:
                cols.append(g(k)); vals.append(-1)
        R.add(cols, vals, 1, 1)
    obj = np.zeros(nvar); obj[off_u:off_d] = 1.0
    integ = np.zeros(nvar); integ[off_z:off_z + nz] = 1
    res = milp(obj, constraints=LinearConstraint(R.matrix(nvar), R.lo, R.hi), integrality=integ,
               bounds=Bounds(lb, ub), options=dict(time_limit=TLIM, mip_rel_gap=GAP, disp=False))
    if res.x is None:
        return None, res.status, np.nan
    z = {E[k]: float(res.x[k]) for k in range(nE)}
    gap = getattr(res, "mip_gap", np.nan)
    return z, res.status, gap


def to_graph(c, z):
    """Kruskal on z (descending) with the fixed lines: guarantees a radial tree even if the solver stopped early."""
    edges = list(z)
    return L.decode(c["nodes"], c["fixed"], edges, np.array([z[e] for e in edges]))


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "summary":
        R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/res_*.csv"))])
        base = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{F.OUT}/res_*.csv"))])
        R = base.merge(R, on="case", suffixes=("", "_milp"))
        print(len(R), "cases (percent correct lines)")
        print((100 * R.drop(columns=["case"]).mean()).round(2).to_string())
        sys.exit(0)
    Xe, P = F.probs()
    C = pickle.load(open(f"{SB_WORK}/lv_ctx_{METERS}_tdm{L.SUFM}.pkl", "rb"))
    code = SIMBENCH_FEEDERS[int(arg)]
    rows = []
    sub = Xe[Xe.feeder == code]
    for case, g_ in sub.groupby("case"):
        ix = g_.index.values; c = C[case]
        edges = list(g_.edge); fixed = [frozenset(e) for e in c["fixed"]]
        r_ = dict(case=case)
        z, st, gap = solve_milp(c, edges, fixed)
        r_["milp"] = F.share(to_graph(c, z), c["tg"]) if z else np.nan
        r_["milp_gap"] = gap
        pf = P["final"][ix]
        force = {e: (1 if q > F.GATE[1] else 0) for e, q in zip(edges, pf) if q > F.GATE[1] or q < F.GATE[0]}
        z2, _, _ = solve_milp(c, edges, fixed, force)
        r_["milp_gated"] = F.share(to_graph(c, z2), c["tg"]) if z2 else np.nan
        rows.append(r_)
        print(code, case, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r_.items() if k != "case"}, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/res_{arg}.csv", index=False)
    print(code, "done", flush=True)
