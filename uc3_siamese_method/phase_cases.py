"""Phase identification cases on the six SimBench LV feeders (use case 3).

Truth: every customer (bus with load and/or PV) on a random phase; documented
phases 85% correct. Measurements: customer smart meter |V| of its phase
(Class 0.5) and P, Q (1%); substation LV busbar |V| per phase (Class 0.5).
Used by siamese.py (siamese method of Dong Liu).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "combined_method"))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import sys
import zlib

import numpy as np

sys.path.insert(0, SB_CODE)
sys.path.insert(0, TOPO)
from phase_sim import Feeder, PH  # noqa: E402
from topology.data import SIMBENCH_FEEDERS  # noqa: E402

SEEDS = range(12)
SV = 0.0025


def make_case(f, d, cust, seed):
    rng = np.random.default_rng(zlib.crc32(f"{seed}".encode()) + len(cust))
    pos = {b: k for k, b in enumerate(d["buses"])}
    truth = rng.integers(0, 3, len(cust))
    S = np.zeros((len(d["V"]), len(f.order), 3), complex)
    Sc = np.zeros((len(d["V"]), len(cust)), complex)
    for i, b in enumerate(cust):
        s = 3 * (d["P"][:, pos[b]] + 1j * d["Q"][:, pos[b]])     # whole bus on one phase
        S[:, f.idx[b], truth[i]] = s; Sc[:, i] = s
    V = np.abs(f.solve(S))
    Vc = V[:, [f.idx[b] for b in cust], truth] + rng.normal(0, SV, (len(V), len(cust)))
    Vs = V[:, 0, :] + rng.normal(0, SV, (len(V), 3))
    Sm = Sc.real * (1 + rng.normal(0, 0.01, Sc.shape)) + 1j * Sc.imag * (1 + rng.normal(0, 0.01, Sc.shape))
    doc = truth.copy()
    flip = rng.choice(len(cust), size=max(1, round(0.15 * len(cust))), replace=False)
    for i in flip:
        doc[i] = (truth[i] + rng.integers(1, 3)) % 3
    return dict(truth=truth, doc=doc, Vc=Vc, Vs=Vs, S=Sm, S_true=Sc)
