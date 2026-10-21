"""Three-phase LV data for phase identification (D2.2 use case).

Each metered LV bus (customer, with its PV if any) is connected to one phase.
Unbalanced power flow by backward/forward sweep, vectorised over time:
  - line phase impedance from the positive sequence with Z0 = 4 Z1
    (typical of 4-core NAYY cables): Zs = (Z0 + 2Z1)/3, Zm = (Z0 - Z1)/3,
    neutral grounded at every customer (Kron-reduced 3x3),
  - Dyn transformer: Z0 = Z1, no mutual term,
  - MV source balanced at the SimBench ext_grid set point.
Per-phase p.u.: S_base 1/3 MVA per phase, so a single-phase customer of S MVA
is 3S p.u. on its phase. Validated against the balanced pandapower run.
Injections are the cached SimBench net bus injections (sb_cache).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "combined_method"))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import numpy as np
import networkx as nx
import simbench as sb

A = np.exp(2j * np.pi / 3)
PH = np.array([1, A ** -1, A ** -2])          # phase a, b, c rotation
Z_BASE = 0.4 ** 2 / 1.0


class Feeder:
    def __init__(self, code):
        net = sb.get_simbench_net(code)
        self.net = net
        tr = net.trafo.iloc[0]
        self.root = int(tr.lv_bus)
        z = tr.vk_percent / 100 / tr.sn_mva; r = tr.vkr_percent / 100 / tr.sn_mva
        self.zt = complex(r, np.sqrt(z ** 2 - r ** 2))       # p.u. on 1 MVA
        self.vsrc = float(net.ext_grid.vm_pu.iloc[0])
        G = nx.Graph()
        for _, l in net.line.iterrows():
            if l.in_service:
                G.add_edge(int(l.from_bus), int(l.to_bus),
                           z=complex(l.r_ohm_per_km, l.x_ohm_per_km) * l.length_km / Z_BASE)
        T = nx.bfs_tree(G, self.root)
        self.order = list(nx.topological_sort(T))            # root first
        self.idx = {b: k for k, b in enumerate(self.order)}
        self.parent = np.array([-1] + [self.idx[next(T.predecessors(b))] for b in self.order[1:]])
        z1 = np.array([0j] + [G.edges[next(T.predecessors(b)), b]["z"] for b in self.order[1:]])
        self.z1 = z1
        self.zs, self.zm = 2 * z1, 1 * z1                   # Z0 = 4 Z1
        self.G, self.T = G, T

    def solve(self, S, iters=12):
        """S: T x N x 3 complex injections (p.u. per phase, + = generation), bus order = self.order.
        Returns phase-to-neutral voltages T x N x 3 (complex)."""
        Tn, N, _ = S.shape
        V = np.broadcast_to(self.vsrc * PH, (Tn, N, 3)).astype(complex).copy()
        for _ in range(iters):
            I = -np.conj(S / V)                              # load current drawn from the bus
            Ib = I.copy()
            for k in range(N - 1, 0, -1):                    # backward: branch current into k
                Ib[:, self.parent[k]] += Ib[:, k]
            # source + transformer (Ib[:,0] = total current through the transformer)
            V[:, 0] = self.vsrc * PH - self.zt * Ib[:, 0]
            for k in range(1, N):                            # forward
                p = self.parent[k]; ib = Ib[:, k]
                drop = self.zs[k] * ib + self.zm[k] * (ib.sum(1, keepdims=True) - ib)
                V[:, k] = V[:, p] - drop
        return V


def validate(code="1-LV-rural1--0-sw"):
    d = dict(np.load(f"{SB_WORK}/sb_cache" + ("_month" if __import__("os").environ.get("MONTH") == "1" else "") + f"/{code}.npz"))
    f = Feeder(code)
    pos = {b: k for k, b in enumerate(d["buses"])}
    S = np.zeros((len(d["V"]), len(f.order), 3), complex)
    for b, k in f.idx.items():
        if b in pos:
            S[:, k, :] = (d["P"][:, pos[b]] + 1j * d["Q"][:, pos[b]])[:, None]   # balanced: S/3 per phase -> S p.u.
    V = f.solve(S)
    cols = [f.idx[b] for b in d["buses"]]
    err = np.abs(np.abs(V[:, cols, 0]) - d["V"])
    return err.max(), err.mean()


if __name__ == "__main__":
    import sys
    for c in sys.argv[1:]:
        print(c, validate(c))
