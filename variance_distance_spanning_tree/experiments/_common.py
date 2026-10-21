"""Shared setup for the experiment scripts."""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)

SMALL = ["1-LV-rural1--0-sw", "1-LV-semiurb4--0-sw", "1-LV-urban6--0-sw"]
LARGE = ["1-LV-rural2--0-sw", "1-LV-rural3--0-sw", "1-LV-semiurb5--0-sw"]
ALL = SMALL + LARGE

ERROR_RATE = 0.15


def load(code, n_steps=672):
    """Network, ground-truth topology, voltage and angle time series."""
    from topology import data
    net = data.load_simbench(code)
    tg = data.topology_of(net)
    V, A = data.simulate_simbench(net, n_steps=n_steps)
    nodes = [c for c in V.columns if c in tg]
    return tg, V, A, nodes


def save(rows, name):
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, name), index=False)
    print(df.to_string(index=False))
    return df


def short(code):
    return code.split("--")[0]
