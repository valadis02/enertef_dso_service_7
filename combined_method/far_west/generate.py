"""Simulate Far West (ACTIVSg2000 area 1) measurements under topology scenarios.

For each scenario (base, or one in-area branch out of service) runs hourly AC
power flow on the full 2000-bus network with Far West loads and renewables
taken from the TAMU 2016 time series. Stores, for the 91 Far West buses:
voltage magnitude, angle, net active injection; and active flow on every
tie branch (Far West side).

Usage: python generate.py START_HOUR N_HOURS SCENARIO_FROM SCENARIO_TO OUT_DIR
Scenario 0 is the base case, scenario k>0 is row k-1 of fw_outage_candidates.csv.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import re
import sys
import numpy as np
import pandas as pd
import pandapower as pp

DATA = FW_DATA + "/"
TS = FW_DATA + "/ts/"
FW = list(range(1001, 1092))


def by_bus(path, kind):
    df = pd.read_csv(path)
    groups = {}
    for c in df.columns[2:]:
        b = int(re.match(rf"{kind} (\d+)", c).group(1))
        groups.setdefault(b, []).append(c)
    return pd.DataFrame({b: df[c].sum(axis=1) for b, c in groups.items()})


def main(start, n, s_from, s_to, out):
    P = by_bus(TS + "load_MW.csv", "Bus")
    Q = by_bus(TS + "load_MVAR.csv", "Bus")
    R = by_bus(TS + "ren_MW.csv", "Gen")
    cands = pd.read_csv(DATA + "fw_outage_candidates.csv")
    br = pd.read_csv(DATA + "branches.csv")
    ties = br[br.tie].reset_index(drop=True)

    import os
    for s in range(s_from, s_to):
        if os.path.exists(f"{out}/scen_{s:03d}.npz"):
            continue
        net = pp.from_json(DATA + "activsg2000_pp.json")
        if s > 0:
            c = cands.iloc[s - 1]
            getattr(net, c.et).at[c.idx, "in_service"] = False
        load_idx = net.load.groupby("bus").groups
        ren_idx = {b: [(et, i) for et in ("gen", "sgen")
                       for i in getattr(net, et).index[getattr(net, et).bus == b]]
                   for b in R.columns}
        V, A, Pinj, Ftie, ok = [], [], [], [], []
        Qinj, Qtie = [], []
        # vectorised element-level setpoints
        l_rows = [i for b in P.columns if b in load_idx for i in load_idx[b]]
        l_bus = net.load.bus.loc[l_rows].values
        l_cnt = pd.Series(l_bus).map(pd.Series(l_bus).value_counts()).values
        g_items = [(et, i, b, len(items)) for b, items in ren_idx.items() for et, i in items]
        tie_line = [(k, t.idx, t.f in FW) for k, t in enumerate(ties.itertuples()) if t.et == "line"]
        tie_imp = [(k, t.idx, t.f in FW) for k, t in enumerate(ties.itertuples()) if t.et == "impedance"]
        STEP = int(os.environ.get('FWSTEP', '1'))
        for h in range(start, start + n, STEP):
            net.load.loc[l_rows, "p_mw"] = P.loc[h, l_bus].values / l_cnt
            net.load.loc[l_rows, "q_mvar"] = Q.loc[h, l_bus].values / l_cnt
            for et, i, b, c in g_items:
                getattr(net, et).at[i, "p_mw"] = R.at[h, b] / c
            try:
                pp.runpp(net, init="results", enforce_q_lims=True, max_iteration=30, numba=True)
                good = True
            except Exception:
                try:
                    pp.runpp(net, init="dc", enforce_q_lims=True, max_iteration=50, numba=True)
                    good = True
                except Exception:
                    good = False
            ok.append(good)
            if not good:
                V.append(np.full(len(FW), np.nan)); A.append(np.full(len(FW), np.nan))
                Pinj.append(np.full(len(FW), np.nan)); Ftie.append(np.full(len(ties), np.nan))
                Qinj.append(np.full(len(FW), np.nan)); Qtie.append(np.full(len(ties), np.nan))
                continue
            rb = net.res_bus.loc[FW]
            V.append(rb.vm_pu.values)
            A.append(rb.va_degree.values)
            Pinj.append(-rb.p_mw.values)  # generation positive
            Qinj.append(-rb.q_mvar.values)
            f = np.zeros(len(ties)); fq = np.zeros(len(ties))
            for k, i, from_fw in tie_line:
                f[k] = net.res_line.at[i, "p_from_mw" if from_fw else "p_to_mw"]
                fq[k] = net.res_line.at[i, "q_from_mvar" if from_fw else "q_to_mvar"]
            for k, i, from_fw in tie_imp:
                f[k] = net.res_impedance.at[i, "p_from_mw" if from_fw else "p_to_mw"]
                fq[k] = net.res_impedance.at[i, "q_from_mvar" if from_fw else "q_to_mvar"]
            Ftie.append(f); Qtie.append(fq)  # flow leaving Far West
        np.savez_compressed(f"{out}/scen_{s:03d}.npz", V=np.array(V), A=np.array(A),
                            Pinj=np.array(Pinj), Ftie=np.array(Ftie), Qinj=np.array(Qinj), Qtie=np.array(Qtie), ok=np.array(ok),
                            hours=np.arange(start, start + n, int(os.environ.get('FWSTEP', '1'))))
        print(f"scenario {s}: {sum(ok)}/{n} converged", flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(int(a[1]), int(a[2]), int(a[3]), int(a[4]), a[5])
