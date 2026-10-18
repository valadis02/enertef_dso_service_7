"""Build data/activsg2000_pp.json and the Far West branch tables from the MATPOWER case.

Usage: python build_net.py   (expects data/case_ACTIVSg2000.m)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import os
import re

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.converter.pypower import from_ppc

DATA = FW_DATA
FW = set(range(1001, 1092))


def mat(txt, name):
    m = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", txt, re.S)
    rows = [l.split("%")[0].strip().rstrip(";") for l in m.group(1).splitlines()]
    return np.array([list(map(float, r.split())) for r in rows if r])


def main():
    txt = open(os.path.join(DATA, "case_ACTIVSg2000.m")).read()
    ppc = {"version": "2", "baseMVA": 100.0,
           "bus": mat(txt, "bus"), "gen": mat(txt, "gen"), "branch": mat(txt, "branch")}
    net = from_ppc(ppc, f_hz=60, validate_conversion=False)
    pp.runpp(net, init="dc", enforce_q_lims=True, max_iteration=50)
    pp.to_json(net, os.path.join(DATA, "activsg2000_pp.json"))

    rows = [(et, i, int(r.from_bus), int(r.to_bus), bool(r.in_service))
            for et in ("line", "impedance") for i, r in getattr(net, et).iterrows()]
    E = pd.DataFrame(rows, columns=["et", "idx", "f", "t", "on"])
    E["in_fw"] = E.f.isin(FW) & E.t.isin(FW)
    E["tie"] = E.f.isin(FW) ^ E.t.isin(FW)
    E.to_csv(os.path.join(DATA, "branches.csv"), index=False)

    G = nx.MultiGraph()
    G.add_nodes_from(net.bus.index)
    for r in E[E.on].itertuples():
        G.add_edge(r.f, r.t, key=(r.et, r.idx))
    cands = []
    for r in E[E.in_fw & E.on].itertuples():
        H = G.copy()
        H.remove_edge(r.f, r.t, key=(r.et, r.idx))
        if nx.is_connected(H):
            cands.append((r.et, r.idx, r.f, r.t))
    pd.DataFrame(cands, columns=["et", "idx", "f", "t"]).to_csv(
        os.path.join(DATA, "fw_outage_candidates.csv"), index=False)
    print(f"{len(net.bus)} buses; {E.in_fw.sum()} Far West branches; {len(cands)} outage candidates")


if __name__ == "__main__":
    main()
