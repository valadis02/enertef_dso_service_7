"""Minimal parser for a MATPOWER .m case (bus, gen, branch, genfuel, bus_name)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import re, numpy as np, pandas as pd

BUS = "bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin".split()
GEN = "bus Pg Qg Qmax Qmin Vg mBase status Pmax Pmin".split()
BR  = "fbus tbus r x b rateA rateB rateC ratio angle status angmin angmax".split()

def _matrix(txt, name):
    m = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", txt, re.S)
    rows = [l.split("%")[0].strip().rstrip(";") for l in m.group(1).splitlines()]
    return np.array([list(map(float, r.split())) for r in rows if r])

def _cell(txt, name):
    m = re.search(rf"mpc\.{name}\s*=\s*\{{(.*?)\}};", txt, re.S)
    return [s.strip().strip(";").strip().strip("'") for s in m.group(1).splitlines() if s.strip()]

def load(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    bus = pd.DataFrame(_matrix(txt, "bus")[:, :13], columns=BUS)
    gen = pd.DataFrame(_matrix(txt, "gen")[:, :10], columns=GEN)
    br  = pd.DataFrame(_matrix(txt, "branch")[:, :13], columns=BR)
    gen["fuel"] = _cell(txt, "genfuel")
    bus["name"] = _cell(txt, "bus_name")
    for d, cols in ((bus, ["bus_i", "type", "area", "zone"]), (gen, ["bus"]), (br, ["fbus", "tbus"])):
        d[cols] = d[cols].astype(int)
    return bus, gen, br
