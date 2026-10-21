"""Far West without angles: same unified method as SimBench.

Minimum requirements (same as SimBench): documented topology ~85% correct,
branch impedances from the asset DB, |V| Class 0.5 at every bus, P and Q
injections 1% at every bus, P and Q on the boundary (tie lines) 1%.

Physics model: area network equations with one reference bus (measured |V|,
angle 0); tie flows enter as injections. Bus voltages are predicted from the
measured injections by fixed-point iteration I = conj(S / V) on the hypothesis'
admittance matrix; residual = measured |V| - predicted |V| on the fluctuations
(time mean per meter and mean across buses removed), as in LV/MV.

usage: fw_ac.py uc1|uc4
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import glob
import os
import json
import sys
import zlib

import numpy as np
import pandas as pd
import pandapower as pp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, FW_CODE)
import pipeline as PL  # noqa: E402  (decode, connectivity, CAND, INSIDE)

DATA = FW_DATA + "/"
FW = list(range(1001, 1092)); POS = {b: i for i, b in enumerate(FW)}
N = len(FW); SV = 0.0025


def ac_model():
    net = pp.from_json(DATA + "activsg2000_pp.json")
    br = pd.read_csv(DATA + "branches.csv")
    inside = br[br.in_fw].reset_index(drop=True); ties = br[br.tie].reset_index(drop=True)
    ys, bh = [], []
    for r in inside.itertuples():
        if r.et == "line":
            ln = net.line.loc[r.idx]; zb = net.bus.vn_kv[ln.from_bus] ** 2 / 100.0
            z = complex(ln.r_ohm_per_km, ln.x_ohm_per_km) * ln.length_km / zb / ln.parallel
            b = 2 * np.pi * 60 * ln.c_nf_per_km * 1e-9 * ln.length_km * zb * ln.parallel
        else:
            im = net.impedance.loc[r.idx]; z = complex(im.rft_pu, im.xft_pu) * 100.0 / im.sn_mva; b = 0.0
        ys.append(1 / z); bh.append(b / 2)
    ysh = np.zeros(N, complex)                      # bus shunts
    for s in net.shunt.itertuples():
        if s.bus in POS and s.in_service:
            ysh[POS[s.bus]] += (s.p_mw + 1j * s.q_mvar) / 100.0 * s.step
    tie_bus = np.array([POS[r.f] if r.f in POS else POS[r.t] for r in ties.itertuples()])
    ref = POS[int(pd.Series([r.f for r in inside.itertuples()]).value_counts().index[0])]
    return inside, np.array(ys), np.array(bh), ysh, tie_bus, ref


INSIDE, YS, BH, YSH, TIEBUS, REF = ac_model()
if os.environ.get("REG", "ideal") == "real":       # register lengths wrong by 10-30% (either sign) on 20% of branches
    _rng = np.random.default_rng(4242)
    for _k in _rng.choice(len(YS), size=round(0.20 * len(YS)), replace=False):
        _f = 1 + _rng.choice([-1, 1]) * _rng.uniform(0.10, 0.30); YS[_k] /= _f; BH[_k] *= _f
FWSUB = int(os.environ.get("FWSUB", "1"))          # fast mode: keep every FWSUB-th hour of the month
FWHALF = os.environ.get("FWHALF") == "1"           # fast mode: every other case only
E = len(INSIDE)
_fi = np.array([POS[r.f] for r in INSIDE.itertuples()]); _ti = np.array([POS[r.t] for r in INSIDE.itertuples()])
# buses with no in-area branch (1044, 1091: tie lines only) carry no topology information -> left out
AREA = np.array(sorted(set(_fi) | set(_ti))); NEW = {b: q for q, b in enumerate(AREA)}
FI = np.array([NEW[i] for i in _fi]); TI = np.array([NEW[i] for i in _ti]); REF = NEW[REF]; N = len(AREA)


def ybus(status):
    Y = np.zeros((N, N), complex)              # bus shunts are inside the measured injections (res_bus)
    for k in np.where(status == 1)[0]:
        i, j, y, b = FI[k], TI[k], YS[k], BH[k]
        Y[i, i] += y + 1j * b; Y[j, j] += y + 1j * b; Y[i, j] -= y; Y[j, i] -= y
    return Y + np.eye(N) * 1e-6


def refs_of(status):
    """REF plus one bus per internal island (island linked to the rest only by tie lines)."""
    import networkx as nx
    G = nx.Graph(); G.add_nodes_from(range(N)); G.add_edges_from((FI[k], TI[k]) for k in np.where(status == 1)[0])
    r = [REF]
    for comp in nx.connected_components(G):
        if REF not in comp:
            r.append(min(comp))
    return sorted(r)


def _dc_start(status, S, Vm):
    rf = refs_of(status); o = [k for k in range(N) if k not in rf]
    B = np.zeros((N, N))
    for k in np.where(status == 1)[0]:
        b = 1.0 / (1 / YS[k]).imag
        i, j = FI[k], TI[k]; B[i, i] += b; B[j, j] += b; B[i, j] -= b; B[j, i] -= b
    th = np.zeros_like(Vm)
    th[:, o] = S.real[:, o] @ np.linalg.inv(B[np.ix_(o, o)] + np.eye(len(o)) * 1e-6).T
    return th


TIE_AREA = {NEW[b] for b in TIEBUS if b in NEW}


def island_has_tie(status):
    import networkx as nx
    G = nx.Graph(); G.add_nodes_from(range(N)); G.add_edges_from((FI[k], TI[k]) for k in np.where(status == 1)[0])
    return all(REF in c or (c & TIE_AREA) for c in nx.connected_components(G))


def solve(status, S, Vm, iters=20, tol=1e-8, V0=None, inverse=True):
    """AC power flow for every hour at once: injections S at all buses but REF,
    REF at its measured |V| and angle 0. Returns complex V (T x N) and the
    inverse Jacobian (T x 2(N-1) x 2(N-1)) at the solution."""
    rf = refs_of(status)
    Y = ybus(status); o = np.array([k for k in range(N) if k not in rf]); n = len(o)
    if V0 is None:
        th = _dc_start(status, S, Vm); vm = Vm.astype(float).copy()       # start: measured |V|, DC angles
    else:
        th = np.angle(V0).copy(); vm = np.abs(V0).copy(); vm[:, rf] = Vm[:, rf]
    def mism(th_, vm_):
        V_ = vm_ * np.exp(1j * th_)
        return V_, V_ @ Y.T, (V_ * np.conj(V_ @ Y.T) - S)[:, o]
    V, I, F = mism(th, vm)
    for _ in range(iters):
        err = np.abs(F).max(1)
        if err.max() < tol:
            break
        J = _jac(Y, V, I, o)
        dx = np.linalg.solve(J, np.concatenate([F.real, F.imag], 1)[:, :, None])[:, :, 0]
        dx[~np.isfinite(dx)] = 0
        step = np.ones(len(V))
        for _ls in range(6):                              # backtracking per hour
            th2 = th.copy(); vm2 = vm.copy()
            th2[:, o] -= step[:, None] * dx[:, :n]; vm2[:, o] -= step[:, None] * dx[:, n:]
            V2, I2, F2 = mism(th2, vm2)
            worse = ~(np.abs(F2).max(1) < err)
            if not worse.any():
                break
            step[worse] *= 0.5
        ok = ~worse
        th[ok] = th2[ok]; vm[ok] = vm2[ok]
        V, I, F = mism(th, vm)
    conv = np.abs(F).max(1) < 1e-3                    # hours where the solve converged (0.1 MW / Mvar)
    return V, ((_jac_inv(Y, V, I, o), o) if inverse else None), conv


def _jac_inv(Y, V, I, o):
    return np.linalg.inv(_jac(Y, V, I, o))


def _jac(Y, V, I, o):
    dVa = 1j * V[:, :, None] * np.conj(I[:, :, None] * np.eye(N)[None] - Y[None] * V[:, None, :])
    Vn = V / np.abs(V)
    dVm = V[:, :, None] * np.conj(Y[None] * Vn[:, None, :]) + np.conj(I)[:, :, None] * np.eye(N)[None] * Vn[:, None, :]
    a = dVa[:, o][:, :, o]; m = dVm[:, o][:, :, o]
    return np.concatenate([np.concatenate([a.real, m.real], 2), np.concatenate([a.imag, m.imag], 2)], 1)


def line_flow(k, V):
    i, j, y, b = FI[k], TI[k], YS[k], BH[k]
    Si = V[:, i] * np.conj((V[:, i] - V[:, j]) * y + V[:, i] * 1j * b)
    Sj = V[:, j] * np.conj((V[:, j] - V[:, i]) * y + V[:, j] * 1j * b)
    return Si, Sj


def flip_vm(status, V, Jo, k, S, Vm, iters=4):
    """|V| after flipping element k: Newton steps whose Jacobian is the current
    one plus the element's own (low-rank) contribution, applied by Woodbury.
    If the flip changes the islands, an exact warm-started solve is used."""
    Jinv, o = Jo
    t = status.copy(); t[k] ^= 1
    if refs_of(t) != refs_of(status):
        if t[k] == 0 and not island_has_tie(t):
            return None                                  # buses left without supply: infeasible
        return np.abs(solve(t, S, Vm, iters=12, V0=None, inverse=False)[0])   # islands change: fresh DC start
    Y = ybus(t); n = len(o)
    i, j, y, b = FI[k], TI[k], YS[k], BH[k]
    sign = 1.0 if t[k] == 1 else -1.0
    V2 = V[:, [i, j]]; Y2 = np.array([[y + 1j * b, -y], [-y, y + 1j * b]])
    I2 = V2 @ Y2.T
    a = 1j * V2[:, :, None] * np.conj(I2[:, :, None] * np.eye(2)[None] - Y2[None] * V2[:, None, :])
    Vn = V2 / np.abs(V2)
    m = V2[:, :, None] * np.conj(Y2[None] * Vn[:, None, :]) + np.conj(I2)[:, :, None] * np.eye(2)[None] * Vn[:, None, :]
    J4 = sign * np.concatenate([np.concatenate([a.real, m.real], 2), np.concatenate([a.imag, m.imag], 2)], 1)
    pos = {bus: q for q, bus in enumerate(o)}
    keep = [q for q, bus in enumerate([i, j]) if bus in pos]          # drop the reference bus
    idx4 = [q for q in keep] + [2 + q for q in keep]
    R = np.array([pos[[i, j][q]] for q in keep] + [n + pos[[i, j][q]] for q in keep])
    D = J4[:, idx4][:, :, idx4]
    K = Jinv[:, :, R]
    Mr = np.eye(len(R))[None] + Jinv[:, R][:, :, R] @ D
    th = np.angle(V).copy(); vm = np.abs(V).copy()
    best = np.full(len(V), np.inf); out = vm.copy()
    for _ in range(iters + 1):
        Vc = vm * np.exp(1j * th)
        F = (Vc * np.conj(Vc @ Y.T) - S)[:, o]
        err = np.abs(F).max(1)
        better = err < best                               # keep, per hour, the best iterate (guards divergence)
        out[better] = vm[better]; best[better] = err[better]
        if _ == iters:
            break
        yv = (Jinv @ np.concatenate([F.real, F.imag], 1)[:, :, None])[:, :, 0]
        corr = np.linalg.solve(Mr, yv[:, R][:, :, None])
        dx = np.clip(yv - (K @ (D @ corr))[:, :, 0], -0.5, 0.5)
        dx[~np.isfinite(dx)] = 0
        th[:, o] -= dx[:, :n]; vm[:, o] -= dx[:, n:]
    return out


def resid(vm_pred, Vm, mask=None, s=SV):
    if mask is not None:
        vm_pred, Vm = vm_pred[mask], Vm[mask]
    r = Vm - vm_pred
    r = r - r.mean(1, keepdims=True); r = r - r.mean(0, keepdims=True)
    return ((r / s) ** 2).sum()


def connect(status):
    """Smallest repair so the documented model is solvable: add back elements
    (in index order) until every bus is connected. Uses no measurement."""
    st = status.copy()
    for k in range(E):
        if PL.connected_ok(st):
            break
        if st[k] == 0:
            t = st.copy(); t[k] = 1
            st = t
    return st


def measurements(d, rng):
    if FWSUB > 1:
        d = {k: (v[::FWSUB] if k in ("V", "A", "Pinj", "Ftie", "Qinj", "Qtie") else v) for k, v in d.items()}
    Vm = d["V"] + rng.normal(0, SV, d["V"].shape)
    noisy = lambda x: x * (1 + rng.normal(0, 0.01, x.shape)) + rng.normal(0, 0.1, x.shape)
    P, Q, Ft, Qt = noisy(d["Pinj"]), noisy(d["Qinj"]), noisy(d["Ftie"]), noisy(d["Qtie"])
    S = (P + 1j * Q) / 100.0
    np.add.at(S.T, TIEBUS, -(Ft + 1j * Qt).T / 100.0)          # flow leaving the area = load at boundary bus
    return Vm[:, AREA], S[:, AREA]


def features(doc, Vm, S):
    dv = Vm[:, FI] - Vm[:, TI]
    if MONTH:                                          # physics residual on every third hour of the month (speed)
        Vm, S = Vm[::3], S[::3]
    base = doc.copy()                                  # islands get their own reference bus
    V, Jinv, conv = solve(base, S, Vm)
    if conv.mean() < 0.5:                              # documented model not solvable: reference = all in service
        base = np.ones(E, int)
        V, Jinv, conv = solve(base, S, Vm)
    J0 = resid(np.abs(V), Vm, conv)
    cg = np.zeros(E)                                   # gain in solvable hours (exact solves only)
    def gain(k):
        if base[k] == 0:                                # few elements out: exact warm-started solve
            t = base.copy(); t[k] = 1
            Vt, _, ct = solve(t, S, Vm, V0=V, inverse=False)
            cg[k] = ct.mean() - conv.mean()
            return (J0 - resid(np.abs(Vt), Vm, conv & ct)) / J0
        vm = flip_vm(base, V, Jinv, k, S, Vm)
        return -1.0 if vm is None else (J0 - resid(vm, Vm, conv)) / J0
    dJ = np.array([gain(k) for k in range(E)])
    return pd.DataFrame({"log_var_dist": np.log10(dv.std(0) + 1e-12), "dJ": dJ, "doc": doc, "conv_gain": cg,
                         "base_in": base, "doc_solvable": int((base == doc).all())})


def refine(st, S, Vm, min_gain=0.01, max_iter=15, p=None, gate=(0.2, 0.8), top=0):
    """Physics refinement: single switch changes, and when none helps, exchanges
    (one element in, one out) among the top candidates of each kind, as in the
    MV branch exchange. Accepted while the exact residual drops by > min_gain.
    With p (forest probabilities), only elements with gate[0] < p < gate[1] move."""
    st = st.copy(); V = None
    free = np.ones(E, bool) if p is None else (p > gate[0]) & (p < gate[1])

    def exact_gain(t, J0, conv, V):
        Vt, _, ct = solve(t, S, Vm, V0=V, inverse=False)
        if ct[conv].mean() < 0.95:
            return -np.inf
        return (J0 - resid(np.abs(Vt), Vm, conv & ct)) / J0

    for _ in range(max_iter):
        V, Jinv, conv = solve(st, S, Vm, V0=V); J0 = resid(np.abs(V), Vm, conv)
        g = np.full(E, -np.inf)
        for k in np.where(free)[0]:
            t = st.copy(); t[k] ^= 1
            if t[k] == 0 and not PL.connected_ok(t):
                continue
            vm = flip_vm(st, V, Jinv, k, S, Vm)
            if vm is not None:
                g[k] = (J0 - resid(vm, Vm, conv)) / J0
        moved = False
        k = int(np.argmax(g))
        if g[k] > min_gain:
            t = st.copy(); t[k] ^= 1
            if exact_gain(t, J0, conv, V) > min_gain:
                st = t; moved = True
        if not moved:                                   # exchanges
            adds = [k for k in np.argsort(-g) if st[k] == 0 and free[k] and np.isfinite(g[k])][:top]
            rems = [k for k in np.argsort(-g) if st[k] == 1 and free[k] and np.isfinite(g[k])][:top]
            best, bt = min_gain, None
            for a_ in adds:
                for r_ in rems:
                    t = st.copy(); t[a_] = 1; t[r_] = 0
                    if not PL.connected_ok(t):
                        continue
                    gg = exact_gain(t, J0, conv, V)
                    if gg > best:
                        best, bt = gg, t
            if bt is None:
                break
            st = bt
    return st


MONTH = os.environ.get("MONTH") == "1"
SCEN = f"{FW_WORK}/scen_q" + ("_month" if MONTH else "")
SCEN_M = f"{FW_WORK}/scen_multi_q" + ("_month" if MONTH else "")


def _trim(d):
    """Keep the hours where the reference power flow converged (month data: >= 90% required)."""
    ok = d["ok"].astype(bool)
    return {k: (v[ok] if k in ("V", "A", "Pinj", "Ftie", "Qinj", "Qtie") else v) for k, v in d.items()}


MONTH = os.environ.get("MONTH") == "1"
SCEN = f"{FW_WORK}/scen_q" + ("_month" if MONTH else "")
SCEN_M = f"{FW_WORK}/scen_multi_q" + ("_month" if MONTH else "")


def _trim(d):
    """Keep the hours where the reference power flow converged."""
    ok = d["ok"].astype(bool)
    return {k: (v[ok] if k in ("V", "A", "Pinj", "Ftie", "Qinj", "Qtie") else v) for k, v in d.items()}


def truths():
    sets = json.load(open(f"{SCEN_M}/sets.json"))
    keep = (lambda d: d["ok"].all()) if not MONTH else (lambda d: d["ok"].mean() >= 0.9)
    for f in sorted(glob.glob(f"{SCEN}/scen_*.npz")):
        s = int(f[-7:-4]); d = dict(np.load(f))
        if not keep(d):
            continue
        st = np.ones(E, int)
        if s > 0:
            st[PL.CAND[s - 1]] = 0
        yield f"single_{s:03d}", st, _trim(d)
    for f in sorted(glob.glob(f"{SCEN_M}/multi_*.npz")):
        i = int(f[-7:-4]); d = dict(np.load(f))
        if not keep(d):
            continue
        st = np.ones(E, int); st[PL.CAND[sets[i]]] = 0
        yield f"multi_{i:03d}", st, _trim(d)


# ------------------------------------------------------------------ experiment
CASES = f"{FW_WORK}/fw_ac_cases" + ("_month" if os.environ.get("MONTH") == "1" else "")


def case_list():
    return [(tid, j) for tid, _, _ in truths() for j in range(3)]


def build(proto, lo, hi):
    import os, pickle
    os.makedirs(CASES, exist_ok=True)
    T = {tid: (st, d) for tid, st, d in truths()}
    import os as _o
    for n, (tid, j) in enumerate(case_list()):
        if not lo <= n < hi:
            continue
        if _o.environ.get("ONLYJ0") == "1" and j != 0:
            continue
        if FWHALF and (n // 3) % 2 == 1:
            continue
        path = f"{CASES}/{proto}_{n:03d}.pkl"
        if os.path.exists(path):
            continue
        truth, d = T[tid]
        rng = np.random.default_rng(zlib.crc32(f"ac{proto}{tid}#{j}".encode()))
        Vm, S = measurements(d, rng)
        if proto == "uc1":
            doc = truth.copy(); k = rng.choice(E, size=int(round(float(os.environ.get('DOCERR', '0.20')) * E)), replace=False); doc[k] = 1 - doc[k]
        else:
            doc = np.ones(E, int)
        X = features(doc, Vm, S); X["truth"] = truth; X["group"] = tid; X["case"] = f"{tid}#{j}"
        pickle.dump(dict(X=X, Vm=Vm, S=S, truth=truth, doc=doc), open(path, "wb"))


def evaluate_ml(proto, lo, hi):
    import os, pickle
    files = sorted(glob.glob(f"{CASES}/{proto}_*.pkl"))
    C = [pickle.load(open(f, "rb")) for f in files]
    X = pd.concat([c["X"] for c in C], ignore_index=True)
    X["dJ"] = np.clip(np.nan_to_num(X["dJ"], nan=0.0, posinf=1.0, neginf=-10.0), -10, 1)
    pp_ = f"{CASES}/probs_{proto}.npy"
    if os.path.exists(pp_):
        p = np.load(pp_)
    else:
        cols = ["log_var_dist", "dJ", "doc", "base_in"]; p = np.zeros(len(X))
        for tr, te in GroupKFold(5).split(X, groups=X.group):
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X.iloc[tr][cols], X.iloc[tr].truth); p[te] = clf.predict_proba(X.iloc[te][cols])[:, 1]
        np.save(pp_, p)
    out = []
    for n, (f, c) in enumerate(zip(files, C)):
        if not lo <= n < hi:
            continue
        res = f.replace(".pkl", ".res.json")
        if os.path.exists(res):
            continue
        ix = np.where(X.case == c["X"].case.iloc[0])[0]
        ml = PL.decode(p[ix]); rf = refine(ml, c["S"], c["Vm"], p=p[ix])
        json.dump(dict(case=c["X"].case.iloc[0], truth=c["truth"].tolist(), doc=c["doc"].tolist(),
                       ml=ml.tolist(), refined=rf.tolist()), open(res, "w"))


def summary(proto):
    rows = []
    for f in sorted(glob.glob(f"{CASES}/{proto}_*.res.json")):
        r = json.load(open(f)); t = np.array(r["truth"]); d = np.array(r["doc"])
        for name in ("doc", "ml", "refined"):
            s = np.array(r[name])
            rows.append(dict(method=name, acc=(s == t).mean(), exact=(s == t).all(),
                             found=((s == 0) & (t == 0) & (d == 1)).sum(), total=((t == 0) & (d == 1)).sum(),
                             false=((s == 0) & (t == 1)).sum()))
    R = pd.DataFrame(rows)
    g = R.groupby("method").agg(n=("acc", "size"), acc=("acc", "mean"), exact=("exact", "mean"),
                                found=("found", "sum"), total=("total", "sum"), false=("false", "sum"))
    g["acc"] *= 100; g["exact"] *= 100
    print(proto); print(g.round(2).to_string())


if __name__ == "__main__":
    cmd, proto = sys.argv[1], sys.argv[2]
    if cmd == "build":
        build(proto, int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "eval":
        evaluate_ml(proto, int(sys.argv[3]), int(sys.argv[4]))
    else:
        summary(proto)
