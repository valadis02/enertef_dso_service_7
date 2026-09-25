"""
DSO Service 7 - Phase 30
Final method + benchmark.

FINAL METHOD = hierarchy_auto (phase25) with an optional blind-clustering
wrapper for large networks (phase29b).

The wrapper: cluster buses by electrical distance (agglomerative, average
linkage on the Var(V_i - V_j) matrix -- no topology knowledge used),
reconstruct inside each cluster with hierarchy_auto, then link clusters
in descending mean-voltage order at their closest pair. Measured gain
grows with network size, so it is enabled only above a size threshold
(below ~50 buses it was neutral-to-slightly-worse).

BENCHMARK: 3 synthetic networks (clean) + 3 real SimBench feeders (with
realistic Class 0.5 meter noise).
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import AgglomerativeClustering

from phase19_voltage_difference import voltage_difference_variance_matrix
from phase25_hierarchy_v2 import hierarchy_auto, eigen_noise_sigma2
from phase1_2_baseline import evaluate, correlation_mst

CLUSTER_MIN_BUSES = 50      # below this, plain hierarchy_auto
CLUSTER_TARGET_SIZE = 12


ANGLE_MIN_BUSES = 20        # below this, voltage magnitude was still better
ANGLE_SNR_SKIP = 1e4        # if the magnitude signal is this clean, don't switch to angles


def _angle_distance(angle_df, nodes, use_smoothing=True, window=15):
    """Tree metric from phase angles.

    Under LinDistFlow the angle drop across a branch is
        d(theta) ~= (x*P - r*Q) / V^2
    a DIFFERENT linear combination of P and Q than the magnitude drop
    (r*P + x*Q)/V. So Var(theta_i - theta_j) is also a tree metric over
    the same tree, weighted by reactance instead of resistance -- genuinely
    independent information, not a copy of the magnitude signal.

    Measured: the angle carries the same SNR as the magnitude once the
    angle error is below ~0.053 deg, which modern micro-PMUs beat by a
    wide margin. Where angles are available the angle metric alone beats
    the magnitude metric on all but the smallest feeders, and beats it by
    more as the network grows.
    """
    df = angle_df[nodes]
    if use_smoothing and window > 1:
        df = df.rolling(window, center=True, min_periods=1).mean()
    D, dn = voltage_difference_variance_matrix(df)
    return D, dn


def _hierarchy_from_D(D, dn, voltage_df, nodes):
    """Hierarchy walk with an externally supplied distance matrix. The
    ORDER always comes from the raw mean VOLTAGES (that is what encodes
    depth); only the distances come from D."""
    idx = {c: i for i, c in enumerate(dn)}
    order = sorted(nodes, key=lambda c: -voltage_df[nodes].mean()[c])
    G = nx.Graph()
    G.add_nodes_from(nodes)
    placed = [order[0]]
    for n in order[1:]:
        p = min(placed, key=lambda x: D[idx[n], idx[x]])
        G.add_edge(n, p)
        placed.append(n)
    return G


def reconstruct(voltage_df, nodes=None, angle_df=None, use_smoothing=True,
                cluster_min=CLUSTER_MIN_BUSES, target=CLUSTER_TARGET_SIZE,
                angle_min=ANGLE_MIN_BUSES):
    """Final topology reconstruction entry point.

    angle_df: optional per-bus phase angles (degrees), same index/columns
    as voltage_df. When supplied -- and the feeder is not tiny -- the
    angle metric is used INSTEAD of the magnitude metric, not blended
    with it: measurements showed the blend is consistently worse than
    angles alone, because the magnitude is the noisier estimator and
    averaging it in adds variance.
    """
    nodes = list(voltage_df.columns) if nodes is None else list(nodes)

    # Angles help when the magnitude signal is buried in noise. When the
    # magnitude is already essentially noise-free the magnitude metric is
    # the better one, so gate on measured SNR rather than on size alone
    # (size is a proxy that misfires on clean synthetic data).
    use_angle = angle_df is not None and len(nodes) >= angle_min
    if use_angle:
        try:
            _s2 = eigen_noise_sigma2(voltage_df[nodes])
            _sig = voltage_df[nodes].var().median()
            if _s2 > 0 and _sig / _s2 > ANGLE_SNR_SKIP:
                use_angle = False
        except Exception:
            pass

    if len(nodes) < cluster_min:
        if use_angle:
            D, dn = _angle_distance(angle_df, nodes, use_smoothing)
            return _hierarchy_from_D(D, dn, voltage_df, nodes)
        return hierarchy_auto(voltage_df[nodes], use_smoothing=use_smoothing)[0]

    if use_angle:
        D, dn = _angle_distance(angle_df, nodes, use_smoothing)
    else:
        df = voltage_df[nodes]
        if use_smoothing:
            df = df.rolling(15, center=True, min_periods=1).mean()
        D, dn = voltage_difference_variance_matrix(df)
    idx = {c: i for i, c in enumerate(dn)}

    k = max(2, int(round(len(nodes) / target)))
    labels = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                      linkage="average").fit_predict(D)

    groups = {}
    for c, l in zip(dn, labels):
        groups.setdefault(l, []).append(c)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for g in groups.values():
        if len(g) > 1:
            if use_angle:
                Dg, dng = _angle_distance(angle_df, g, use_smoothing)
                G.add_edges_from(_hierarchy_from_D(Dg, dng, voltage_df, g).edges())
            else:
                G.add_edges_from(hierarchy_auto(voltage_df[g], use_smoothing=use_smoothing)[0].edges())

    means = voltage_df[nodes].mean()
    order = sorted(groups, key=lambda l: -means[groups[l]].mean())
    linked = [order[0]]
    for l in order[1:]:
        best = None
        for a in groups[l]:
            for lp in linked:
                for b in groups[lp]:
                    d = D[idx[a], idx[b]]
                    if best is None or d < best[0]:
                        best = (d, a, b)
        G.add_edge(best[1], best[2])
        linked.append(l)
    return G


# ---------------------------------------------------------------------------

def main():
    import simbench as sb
    from phase1_2_baseline import load_feeder, ground_truth_graph as gt_i, simulate_timeseries
    from phase5_simbench import ground_truth_graph as gt_s, simulate_timeseries_simbench
    from phase10a_synthetic_generator import (random_radial_tree, assign_random_impedances,
                                               simulate_lindistflow)

    rows = []

    print("=== SYNTHETIC (clean, no meter noise) ===")
    rng = np.random.default_rng(11)
    synth = []
    for n_buses in (15, 25):
        G = assign_random_impedances(random_radial_tree(n_buses, rng), rng)
        v = simulate_lindistflow(G, 500, rng, meter_noise_sigma=0.0, idio_noise_std=0.03)
        synth.append((f"random radial ({n_buses} buses)", G, v))
    net = load_feeder()
    synth.append(("IEEE 33-bus", gt_i(net), simulate_timeseries(net, n_steps=500)[0]))

    for name, tg, vdf in synth:
        nodes = [c for c in vdf.columns if c in tg]
        b = evaluate(correlation_mst(vdf)[0], tg, vdf.columns)["reconstruction_accuracy"]
        a = evaluate(reconstruct(vdf, nodes, use_smoothing=False), tg, vdf.columns)["reconstruction_accuracy"]
        print(f"  {name:28s} buses={tg.number_of_nodes():3d}  baseline={b:.3f}  FINAL={a:.3f}")
        rows.append({"set": "synthetic-clean", "network": name, "buses": tg.number_of_nodes(),
                      "baseline": round(b, 3), "final": round(a, 3)})

    print("\n=== REAL SimBench (Class 0.5 meter noise, sigma=0.0025) ===")
    for code in ["1-LV-rural1--0-sw", "1-LV-semiurb4--0-sw", "1-LV-urban6--0-sw"]:
        n2 = sb.get_simbench_net(code)
        tg = gt_s(n2)
        vdf = simulate_timeseries_simbench(n2, n_steps=672, start=96 * 7 * 26)
        nodes = [c for c in vdf.columns if c in tg]
        rng = np.random.default_rng(7)
        d = vdf + rng.normal(0, 0.0025, size=vdf.shape)
        b = evaluate(correlation_mst(d)[0], tg, vdf.columns)["reconstruction_accuracy"]
        a = evaluate(reconstruct(d, nodes, use_smoothing=True), tg, vdf.columns)["reconstruction_accuracy"]
        print(f"  {code:22s} buses={tg.number_of_nodes():3d}  baseline={b:.3f}  FINAL={a:.3f}")
        rows.append({"set": "real-class0.5", "network": code, "buses": tg.number_of_nodes(),
                      "baseline": round(b, 3), "final": round(a, 3)})

    res = pd.DataFrame(rows)
    res.to_csv("results/phase30_benchmark.csv", index=False)
    print("\nSaved results/phase30_benchmark.csv")


if __name__ == "__main__":
    main()
