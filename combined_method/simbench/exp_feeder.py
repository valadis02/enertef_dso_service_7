"""Customer-feeder assignment on the six SimBench LV networks (D2.2 use case).

Feeders = branches leaving the transformer LV bus. Customers = buses with a
load (they have a smart meter); junction buses have no meter and are not used.
Measurements: voltage magnitude only, Class 0.5, one week at 15 min.

A) Unsupervised (Liu et al. 2025 style): distance between customers from the
   voltage correlation (log-transformed) or from our variance distance,
   hierarchical clustering into K = number of feeders; accuracy after the best
   one-to-one matching of clusters to feeders.
B) Correction of a documented assignment that is 85% correct (same protocol
   as the topology experiments): each customer takes the weighted majority
   label of its k nearest customers; two rounds.
Both with and without the PV-hour filter.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import sys

import numpy as np
import pandas as pd
import networkx as nx
import simbench as sb
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform

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


def distances(V, kind):
    Vs = pd.DataFrame(V).rolling(15, center=True, min_periods=1).mean().values
    if kind == "correlation":
        C = np.corrcoef(Vs.T)
        D = np.log10(np.clip(1 - C, 1e-9, None))          # log transform: amplifies small differences near rho = 1
        D = D - D.min()
    else:                                                  # our variance distance
        var = Vs.var(0); cov = np.cov(Vs.T)
        D = np.log10(np.clip(var[:, None] + var[None, :] - 2 * cov, 1e-14, None))
        D = D - D.min()
    np.fill_diagonal(D, 0)
    return (D + D.T) / 2


def cluster_acc(D, truth, K):
    Zl = linkage(squareform(D, checks=False), method="average")
    pred = fcluster(Zl, K, criterion="maxclust") - 1
    M = np.zeros((K, K))
    for p, t in zip(pred, truth):
        M[p, t] += 1
    r, c = linear_sum_assignment(-M)
    return M[r, c].sum() / len(truth)


def knn_correct(D, doc, k=5, rounds=2):
    lab = doc.copy()
    for _ in range(rounds):
        new = lab.copy()
        for i in range(len(lab)):
            nn = [j for j in np.argsort(D[i]) if j != i][:k]
            w = {}
            for j in nn:
                w[lab[j]] = w.get(lab[j], 0) + 1 / (D[i, j] + 1e-9)
            new[i] = max(w, key=w.get)
        lab = new
    return lab


def main(seeds=range(12), err=0.15):
    rows = []
    for code in SIMBENCH_FEEDERS:
        net = sb.get_simbench_net(code); lab = feeders(net)
        d = dict(np.load(f"{CACHE}/{code}.npz")); buses = list(d["buses"])
        cust = [b for b in buses if b in lab]
        ci = [buses.index(b) for b in cust]
        truth = np.array([lab[b] for b in cust]); K = truth.max() + 1
        pv_ok = d["P"].sum(1) < 0
        for seed in seeds:
            rng = np.random.default_rng(seed)
            V = d["V"][:, ci] + rng.normal(0, 0.0025, (len(d["V"]), len(ci)))
            doc = truth.copy()
            flip = rng.choice(len(doc), size=max(1, round(err * len(doc))), replace=False)
            for i in flip:
                doc[i] = rng.choice([f for f in range(K) if f != truth[i]])
            for filt, Vx in (("all hours", V), ("PV-filtered", V[pv_ok])):
                for kind in ("correlation", "variance distance"):
                    D = distances(Vx, kind)
                    rows.append(dict(network=code.split("-")[2], customers=len(cust), feeders=K, seed=seed,
                                     data=filt, distance=kind,
                                     unsupervised=cluster_acc(D, truth, K),
                                     documented=(doc == truth).mean(),
                                     corrected=(knn_correct(D, doc) == truth).mean()))
        print(code, "done", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = main()
    df.to_csv(f"{SB_WORK}/feeder_assignment.csv", index=False)
    g = df.groupby(["data", "distance"])[["unsupervised", "documented", "corrected"]].mean().mul(100).round(1)
    print(g.to_string())
    print(df[df.data == "PV-filtered"].groupby(["network", "distance"])[["unsupervised", "corrected"]].mean().mul(100).round(1).unstack().to_string())
