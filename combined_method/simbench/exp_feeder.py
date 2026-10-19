"""Feeder labels of the SimBench LV customers (branches leaving the transformer LV bus)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import sys

import numpy as np
import networkx as nx
import simbench as sb

sys.path.insert(0, TOPO)
from topology.data import topology_of, SIMBENCH_FEEDERS  # noqa: E402

CACHE = f"{SB_WORK}/sb_cache" + ("_month" if __import__("os").environ.get("MONTH") == "1" else "")


def feeders(net):
    G = topology_of(net); root = int(net.trafo.lv_bus.iloc[0])
    H = G.copy(); H.remove_node(root)
    loads = set(net.load.bus.astype(int))
    lab = {}
    for k, comp in enumerate(nx.connected_components(H)):
        for b in comp & loads:
            lab[b] = k
    return lab
