"""Cache one week of SimBench power flow per feeder: V, angle, net P, Q per LV bus."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import os
import sys

import numpy as np
import pandapower as pp
import simbench as sb

sys.path.insert(0, TOPO)
from topology.data import topology_of, SIMBENCH_FEEDERS  # noqa: E402

MONTH = os.environ.get("MONTH") == "1"
INDIV = os.environ.get("INDIV") == "1"              # individual customer profiles; use with its own CM_WORK
OUT = f"{SB_WORK}/sb_cache_month" if MONTH else f"{SB_WORK}/sb_cache"
START, STEPS, S_BASE = 96 * 7 * 26, (2880 if MONTH else 672), 1.0


def run(code):
    path = f"{OUT}/{code}.npz"
    if os.path.exists(path):
        return
    net = sb.get_simbench_net(code)
    av = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    lp, lq = av[("load", "p_mw")], av[("load", "q_mvar")]
    sp = av.get(("sgen", "p_mw"))
    buses = sorted(topology_of(net).nodes())
    if INDIV:                                       # individual customer profiles (indiv_profiles.py)
        from indiv_profiles import individualise
        seed = sum(map(ord, code))
        IP, IQ = individualise(lp, lq, net.load.profile.values, START, STEPS, seed)
    V, A, P, Q = [], [], [], []
    for t in range(START, START + STEPS):
        net.load["p_mw"] = IP[t - START] if INDIV else lp.loc[t].values
        net.load["q_mvar"] = IQ[t - START] if INDIV else lq.loc[t].values
        if sp is not None and len(net.sgen):
            net.sgen["p_mw"] = sp.loc[t].values
        pp.runpp(net, numba=True)
        rb = net.res_bus.loc[buses]
        V.append(rb.vm_pu.values); A.append(rb.va_degree.values)
        P.append(-rb.p_mw.values / S_BASE); Q.append(-rb.q_mvar.values / S_BASE)
    np.savez_compressed(path, buses=np.array(buses), V=np.array(V), A=np.array(A),
                        P=np.array(P), Q=np.array(Q))
    print(code, "cached", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for c in sys.argv[1:]:
        run(c)
