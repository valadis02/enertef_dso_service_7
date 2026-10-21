"""Detect an undocumented branch outage in Far West from simulated measurements.

Documented topology: all in-area branches in service. Truth: scenario k has
one candidate branch out. Two detectors:

  angle_only  — needs angles (PMU) + documented reactances. Scores each
                documented branch by std over time of the flow it implies,
                (theta_i - theta_j) / x. No injections used.
  dc_residual — needs angles + net injections at each bus + tie-line flows
                (what a TSO has from SCADA/PMU). Fits the DC flow model
                P = M diag(b) M^T theta under each hypothesis "branch e is out"
                and picks the hypothesis with the smallest residual.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import numpy as np
import pandas as pd
import pandapower as pp

DATA = FW_DATA + "/"
FW = list(range(1001, 1092))
POS = {b: i for i, b in enumerate(FW)}


def network_model():
    net = pp.from_json(DATA + "activsg2000_pp.json")
    br = pd.read_csv(DATA + "branches.csv")
    inside = br[br.in_fw].reset_index(drop=True)
    ties = br[br.tie].reset_index(drop=True)
    b = []
    for r in inside.itertuples():
        if r.et == "line":
            ln = net.line.loc[r.idx]
            zbase = net.bus.vn_kv[ln.from_bus] ** 2 / 100.0
            x = ln.x_ohm_per_km * ln.length_km / zbase / ln.parallel
        else:
            im = net.impedance.loc[r.idx]
            x = im.xft_pu * 100.0 / im.sn_mva
        b.append(1.0 / x)
    M = np.zeros((len(FW), len(inside)))
    for k, r in enumerate(inside.itertuples()):
        M[POS[r.f], k] = 1.0
        M[POS[r.t], k] = -1.0
    tie_at = np.zeros((len(FW), len(ties)))
    for k, r in enumerate(ties.itertuples()):
        tie_at[POS[r.f] if r.f in POS else POS[r.t], k] = 1.0
    cands = pd.read_csv(DATA + "fw_outage_candidates.csv")
    key = {(r.et, r.idx): k for k, r in enumerate(inside.itertuples())}
    cand_cols = np.array([key[(r.et, r.idx)] for r in cands.itertuples()])
    return inside, np.array(b), M, tie_at, cand_cols


def add_noise(d, rng, sigma_deg=0.01, p_rel=0.01, p_floor=0.1):
    A = d["A"] + rng.normal(0, sigma_deg, d["A"].shape)
    P = d["Pinj"] * (1 + rng.normal(0, p_rel, d["Pinj"].shape)) + rng.normal(0, p_floor, d["Pinj"].shape)
    F = d["Ftie"] * (1 + rng.normal(0, p_rel, d["Ftie"].shape)) + rng.normal(0, p_floor, d["Ftie"].shape)
    return A, P, F


def angle_only(A, b, M, cand_cols):
    dtheta = np.deg2rad(A) @ M          # T x E
    s = (dtheta * b).std(axis=0)        # implied flow std, p.u.
    return s[cand_cols]                 # higher = more suspicious


def dc_residual(A, P, F, b, M, tie_at, cand_cols):
    theta = np.deg2rad(A)               # T x N
    dtheta = theta @ M                  # T x E
    flows = dtheta * b                  # p.u., T x E
    r = (P - F @ tie_at.T) / 100.0 - flows @ M.T   # T x N, documented model
    J0 = (r ** 2).sum()
    J = np.empty(len(cand_cols))
    for k, e in enumerate(cand_cols):
        re = r + np.outer(flows[:, e], M[:, e])     # remove branch e from the model
        J[k] = (re ** 2).sum()
    return J0, J                        # lower J = better fit
