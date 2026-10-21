"""Truth topologies with several simultaneous in-area outages (whole network stays connected)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import os, sys, json
import numpy as np, pandas as pd, networkx as nx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate as G

DATA = FW_DATA + "/"

def make_sets(n_sets=24, seed=0):
    br = pd.read_csv(DATA + "branches.csv")
    cands = pd.read_csv(DATA + "fw_outage_candidates.csv")
    graph = nx.MultiGraph()
    graph.add_nodes_from(set(br.f) | set(br.t))
    for r in br.itertuples():
        graph.add_edge(r.f, r.t, key=(r.et, r.idx))
    rng = np.random.default_rng(seed); sets = []
    while len(sets) < n_sets:
        k = int(rng.integers(2, 7))
        pick = rng.choice(len(cands), size=k, replace=False)
        H = graph.copy()
        for i in pick:
            c = cands.iloc[i]; H.remove_edge(c.f, c.t, key=(c.et, int(c.idx)))
        if nx.is_connected(H) and 76 not in pick:   # candidate 77 (1-based) does not converge
            sets.append(sorted(int(i) for i in pick))
    return sets

def run(i_from, i_to, out=os.environ.get("FWOUT_MULTI", f"{FW_WORK}/scen_multi_q")):
    import pandapower as pp
    os.makedirs(out, exist_ok=True)
    sets = make_sets()
    json.dump(sets, open(f"{out}/sets.json", "w"))
    import glob
    good = set(json.load(open(_os.path.join(FW_CODE, "configs.json")))["multi"])
    cands = pd.read_csv(DATA + "fw_outage_candidates.csv")
    orig = pp.from_json
    for i in range(i_from, i_to):
        if os.path.exists(f"{out}/multi_{i:03d}.npz") or i not in good:
            continue
        def patched(path, _set=sets[i]):
            net = orig(path)
            for j in _set:
                c = cands.iloc[j]; getattr(net, c.et).at[int(c.idx), "in_service"] = False
            return net
        G.pp.from_json = patched
        tmp = f"{out}/tmp_{i}"; os.makedirs(tmp, exist_ok=True)                            # per-scenario folder (parallel safe)
        G.main(4000, int(os.environ.get("FWHOURS", "168")), 0, 1, tmp)                      # scenario 0 = "as loaded"
        os.rename(f"{tmp}/scen_000.npz", f"{out}/multi_{i:03d}.npz"); os.rmdir(tmp)
        G.pp.from_json = orig

if __name__ == "__main__":
    run(int(sys.argv[1]), int(sys.argv[2]))
