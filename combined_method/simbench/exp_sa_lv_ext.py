"""EXTENSION 5: step 4 (greedy refinement) replaced by simulated annealing / MCMC with J as likelihood.
Original files are untouched.

State   = a spanning tree over  fixed lines + candidate lines.
Move    = add one candidate line e that is not in the tree, remove one candidate line f on the cycle
          that e closes (branch exchange, the tree stays radial). Hastings ratio is included.
Energy  E(tree) = J(tree)/2  +  LAM * sum_e  -log( p_e if e in tree else 1-p_e )     (second term only with prior)
          J = squared voltage residual of the linearised AC model (same J as the original code),
          exp(-J/2) is the Gaussian likelihood, exp(-prior) the forest as prior.
Chain   = Metropolis at temperature T. Phase 1 (annealing): T falls geometrically from T0 to 1.
          Phase 2 (sampling): T = 1, count how often every line is in the tree -> posterior marginals.

Columns written to res_<feeder>.csv
  sa_physics   annealing, start = variance_distance_spanning_tree, no prior   (pure physics, no forest)
  sa_prior     annealing, start = Kruskal on forest p (step 3), forest as prior (best tree seen)
  post_prior   Kruskal on the posterior marginals of the sampling phase (sa_prior chain)
  post_cert    mean posterior marginal of the lines of the true tree (diagnostic, not a topology score)

Environment variables:  STEPS (anneal steps, default 3000)  SAMPLES (default 1500)  T0FRAC (T0 = T0FRAC*E_start, default 0.02)
                        LAM (prior weight, default 1.0)  TSTRIDE (use every k-th time step inside the chain, default 4)
                        SEED (default 0)
usage: exp_sa_lv_ext.py <feeder index> | summary
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

import exp_final_lv_ext as F                       # same folder: probs(), share(), sigmas()
from exp_final_lv_ext import L, METERS, SIMBENCH_FEEDERS, SB_WORK  # noqa: E402

STEPS = int(os.environ.get("STEPS", 3000))
SAMPLES = int(os.environ.get("SAMPLES", 1500))
T0FRAC = float(os.environ.get("T0FRAC", 0.02))
LAM = float(os.environ.get("LAM", 1.0))
TSTRIDE = int(os.environ.get("TSTRIDE", 4))
SEED = int(os.environ.get("SEED", 0))
OUT = F.OUT + "_sa"
os.makedirs(OUT, exist_ok=True)


def run_chain(c, cands, start, p, lam, steps, samples, t0, rng, sv):
    """Metropolis on spanning trees. Returns (best tree edge set, marginals dict, E_start, E_best)."""
    nodes, root = c["nodes"], c["root"]
    sl = slice(None, None, TSTRIDE)
    P_, Q_, V_ = c["P"][sl], c["Q"][sl], c["V"][sl]
    fixed = {frozenset(e) for e in c["fixed"]}
    cand_list = sorted(cands, key=lambda e: tuple(sorted(e)))
    if p is not None:
        lp1 = {e: np.log(np.clip(p[e], 1e-4, 1 - 1e-4)) for e in cand_list}
        lp0 = {e: np.log(1 - np.clip(p[e], 1e-4, 1 - 1e-4)) for e in cand_list}

    def prior(e_in):                                    # -log prior contribution of line e being in / out
        return 0.0 if p is None else -lam * (lp1[e_in])

    def prior_out(e_out):
        return 0.0 if p is None else -lam * (lp0[e_out])

    def J(es):
        Z = L.zred(nodes, es, c["par"], c["med"], root)
        return L.resid_J(Z, P_, Q_, V_, None, sv, 1.0)

    cur = set(start)
    Ecur = 0.5 * J(cur)
    if p is not None:
        Ecur += sum(prior(e) if e in cur else prior_out(e) for e in cand_list)
    E_start, best, Ebest = Ecur, set(cur), Ecur
    T = nx.Graph(); T.add_nodes_from(nodes); T.add_edges_from(map(tuple, cur))
    counts = {e: 0 for e in cand_list}
    total = steps + samples
    cool = (1.0 / max(t0, 1.0)) ** (1.0 / max(steps, 1))
    temp = max(t0, 1.0)
    for it in range(total):
        out = [e for e in cand_list if e not in cur]
        if out:
            e = out[rng.integers(len(out))]
            a, b = tuple(e)
            path = nx.shortest_path(T, a, b)                      # the tree path a..b = the cycle that e closes
            rem = [frozenset(x) for x in zip(path[:-1], path[1:]) if frozenset(x) in cands]
            if rem:
                f = rem[rng.integers(len(rem))]
                new = (cur | {e}) - {f}
                # reverse move: pick f out of the new tree's outside edges (same count), then e among the cycle's candidate edges
                T2 = nx.Graph(); T2.add_nodes_from(nodes); T2.add_edges_from(map(tuple, new))
                fa, fb = tuple(f)
                path2 = nx.shortest_path(T2, fa, fb)
                rem2 = [frozenset(x) for x in zip(path2[:-1], path2[1:]) if frozenset(x) in cands]
                log_q = np.log(len(rem)) - np.log(max(len(rem2), 1))     # q(rev)/q(fwd) = (1/rem2)/(1/rem)
                Enew = 0.5 * J(new)
                if p is not None:
                    Enew += sum(prior(x) if x in new else prior_out(x) for x in cand_list)
                logA = -(Enew - Ecur) / temp + log_q
                if np.log(rng.random() + 1e-300) < logA:
                    cur, Ecur = new, Enew
                    T = T2
                    if Ecur < Ebest:
                        best, Ebest = set(cur), Ecur
        if it >= steps:
            for e in cur:
                if e in counts:
                    counts[e] += 1
        else:
            temp = max(1.0, temp * cool)
    marg = {e: counts[e] / max(samples, 1) for e in cand_list}
    return best, marg, E_start, Ebest


def tree_graph(nodes, es):
    H = nx.Graph(); H.add_nodes_from(nodes); H.add_edges_from(map(tuple, es))
    return H


def solve_case(c, cands, edges, pf, seed):
    nodes = c["nodes"]
    rng = np.random.default_rng(seed)
    fixed = {frozenset(e) for e in c["fixed"]}
    sv = F.sigmas(c, {frozenset(e) for e in c["doc"].edges()})
    res = {}
    # (a) pure physics: start from the variance distance tree, no prior
    start = {frozenset(e) for e in c["base"].edges()}
    E0 = 0.5 * L.resid_J(L.zred(nodes, start, c["par"], c["med"], c["root"]), c["P"][::TSTRIDE], c["Q"][::TSTRIDE], c["V"][::TSTRIDE], None, sv, 1.0)
    best, _, _, _ = run_chain(c, cands, start, None, 0.0, STEPS, 0, T0FRAC * abs(E0), rng, sv)
    res["sa_physics"] = F.share(tree_graph(nodes, best), c["tg"])
    # (b) forest as prior, start = decoded tree (step 3), with sampling phase
    p = dict(zip(edges, pf))
    Gd = L.decode(nodes, c["fixed"], edges, pf)
    start = {frozenset(e) for e in Gd.edges()}
    best, marg, _, _ = run_chain(c, cands, start, p, LAM, STEPS, SAMPLES, T0FRAC * abs(E0), rng, sv)
    res["sa_prior"] = F.share(tree_graph(nodes, best), c["tg"])
    pm = np.array([marg[e] for e in edges])
    res["post_prior"] = F.share(L.decode(nodes, c["fixed"], edges, pm), c["tg"])
    tt = {frozenset(e) for e in c["tg"].edges()}
    res["post_cert"] = float(np.mean([marg[e] for e in edges if e in tt])) if any(e in tt for e in edges) else np.nan
    return res


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "summary":
        R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/res_*.csv"))])
        base = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{F.OUT}/res_*.csv"))])
        R = base.merge(R, on="case", suffixes=("", "_sa"))
        print(len(R), "cases (percent correct lines)")
        print((100 * R.drop(columns=["case"]).mean()).round(2).to_string())
        sys.exit(0)
    Xe, P = F.probs()                                     # reads the baseline probs.pkl, no retraining
    C = pickle.load(open(f"{SB_WORK}/lv_ctx_{METERS}_tdm{L.SUFM}.pkl", "rb"))
    code = SIMBENCH_FEEDERS[int(arg)]
    rows = []
    sub = Xe[Xe.feeder == code]
    for k, (case, g) in enumerate(sub.groupby("case")):
        ix = g.index.values; c = C[case]
        edges = list(g.edge)
        r = dict(case=case)
        r.update(solve_case(c, set(edges), edges, P["final"][ix], SEED * 100003 + k))
        rows.append(r)
        print(code, case, {k_: round(v, 3) for k_, v in r.items() if k_ != "case"}, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/res_{arg}.csv", index=False)
    print(code, "done", flush=True)
