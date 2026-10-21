"""Far West switching configurations for use case 4: FWK in-area branches out
of service (maintenance-type configuration, default 5), network stays connected.
Documentation later gets FWK - 1 of them right (one outage reported on a wrong
branch), i.e. 80% with FWK = 5.
usage: gen_fw7.py <from> <to>"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import json
import os
import sys

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate as G  # noqa: E402

DATA = FW_DATA + "/"
OUT = os.environ.get("FWOUT7", f"{FW_WORK}/scen_fw7")
BAD = {76, 47, 74}                     # candidates whose single outage does not converge


def make_sets(n_sets=80, k=int(os.environ.get('FWK', '5')), seed=7):
    br = pd.read_csv(DATA + "branches.csv"); cands = pd.read_csv(DATA + "fw_outage_candidates.csv")
    graph = nx.MultiGraph(); graph.add_nodes_from(set(br.f) | set(br.t))
    for r in br.itertuples():
        graph.add_edge(r.f, r.t, key=(r.et, r.idx))
    rng = np.random.default_rng(seed); sets = []
    while len(sets) < n_sets:
        pick = [int(i) for i in rng.choice(len(cands), size=k, replace=False)]
        if BAD & set(pick):
            continue
        H = graph.copy()
        for i in pick:
            c = cands.iloc[i]; H.remove_edge(c.f, c.t, key=(c.et, int(c.idx)))
        if nx.is_connected(H):
            sets.append(sorted(pick))
    return sets


def run(i_from, i_to):
    import pandapower as pp
    os.makedirs(OUT, exist_ok=True)
    sets = make_sets(); json.dump(sets, open(f"{OUT}/sets.json", "w"))
    cands = pd.read_csv(DATA + "fw_outage_candidates.csv"); orig = pp.from_json
    for i in range(i_from, i_to):
        if os.path.exists(f"{OUT}/fw7_{i:03d}.npz"):
            continue
        def patched(path, _set=sets[i]):
            net = orig(path)
            for j in _set:
                c = cands.iloc[j]; getattr(net, c.et).at[int(c.idx), "in_service"] = False
            return net
        G.pp.from_json = patched
        tmp = f"{OUT}/tmp_{i}"; os.makedirs(tmp, exist_ok=True)
        G.main(4000, int(os.environ.get("FWHOURS", "168")), 0, 1, tmp)
        os.rename(f"{tmp}/scen_000.npz", f"{OUT}/fw7_{i:03d}.npz"); os.rmdir(tmp)
        G.pp.from_json = orig


if __name__ == "__main__":
    run(int(sys.argv[1]), int(sys.argv[2]))
