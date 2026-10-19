"""Blind topology reconstruction: build the graph from measurements alone.

Evaluated and found measurement-limited under realistic meter noise (12-31%
with Class 0.5 meters and no phase angle). Retained as a capability and as
the benchmark against which the validation mode is compared, but not as the
operational mode -- see the report, section 3.

Method, in order:

  1. Order buses by mean voltage, descending. Voltage falls monotonically
     downstream in a radial feeder, so this ordering is the tree depth and
     a node's parent always comes earlier. The mean is an unbiased
     estimator with error ~sigma/sqrt(n), unlike every second-moment
     statistic, which carries an irreducible 2*sigma^2 bias.
  2. Score candidate parents by electrical distance.
  3. Use phase angles when supplied and the voltage signal is noisy;
     angles are a different projection of the same tree, and blending the
     two is consistently worse than using angles alone.
  4. Remove the common load factor first when the noise floor is well
     below the residual signal.
  5. Split networks above ~50 buses into clusters, since accuracy degrades
     with size.
"""

import numpy as np
import networkx as nx
from sklearn.cluster import AgglomerativeClustering

from .distance import electrical_distance, noise_variance, remove_common_factor

CLUSTER_MIN_BUSES = 50
CLUSTER_TARGET_SIZE = 12
ANGLE_MIN_BUSES = 20
ANGLE_SNR_SKIP = 1e4       # magnitude this clean -> angles are not needed
RESIDUAL_SNR = 5.0         # above this, removing the common factor pays off


def _hierarchy(D, dn, voltage, nodes):
    """Attach each bus, in descending mean-voltage order, to the nearest
    already-placed bus. Radiality holds by construction."""
    idx = {c: i for i, c in enumerate(dn)}
    order = sorted(nodes, key=lambda c: -voltage[nodes].mean()[c])
    G = nx.Graph()
    G.add_nodes_from(nodes)
    placed = [order[0]]
    for n in order[1:]:
        parent = min(placed, key=lambda p: D[idx[n], idx[p]])
        G.add_edge(n, parent)
        placed.append(n)
    return G


def _distance_for(voltage, angle, nodes, smooth):
    """Pick the measurement the distance is computed from, and whether to
    residualise it first."""
    use_angle = angle is not None and len(nodes) >= ANGLE_MIN_BUSES
    if use_angle:
        s2 = noise_variance(voltage[nodes])
        if s2 > 0 and voltage[nodes].var().median() / s2 > ANGLE_SNR_SKIP:
            use_angle = False          # magnitude is already clean enough

    if use_angle:
        return electrical_distance(angle[nodes], smooth)

    df = voltage[nodes]
    if smooth and smooth > 1:
        df = df.rolling(smooth, center=True, min_periods=1).mean()
    s2 = noise_variance(voltage[nodes])
    resid = remove_common_factor(df)
    snr = resid.var().median() / s2 if s2 > 1e-20 else np.inf
    return electrical_distance(resid if snr > RESIDUAL_SNR else df, 0)


def reconstruct(voltage, nodes=None, angle=None, smooth_window=15,
                cluster_min=CLUSTER_MIN_BUSES, cluster_size=CLUSTER_TARGET_SIZE):
    """Reconstruct the topology from measurements alone.

    voltage: DataFrame of per-bus voltage magnitudes (p.u.).
    angle:   optional DataFrame of per-bus phase angles (degrees).
    """
    nodes = list(voltage.columns) if nodes is None else list(nodes)

    if len(nodes) < cluster_min:
        D, dn = _distance_for(voltage, angle, nodes, smooth_window)
        return _hierarchy(D, dn, voltage, nodes)

    # large feeder: cluster by electrical distance, rebuild each cluster,
    # then link the clusters in descending mean-voltage order
    D, dn = _distance_for(voltage, angle, nodes, smooth_window)
    idx = {c: i for i, c in enumerate(dn)}
    k = max(2, round(len(nodes) / cluster_size))
    labels = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                      linkage="average").fit_predict(D)

    groups = {}
    for c, l in zip(dn, labels):
        groups.setdefault(l, []).append(c)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for g in groups.values():
        if len(g) > 1:
            Dg, dng = _distance_for(voltage, angle, g, smooth_window)
            G.add_edges_from(_hierarchy(Dg, dng, voltage, g).edges())

    means = voltage[nodes].mean()
    order = sorted(groups, key=lambda l: -means[groups[l]].mean())
    linked = [order[0]]
    for l in order[1:]:
        best = min(
            ((D[idx[a], idx[b]], a, b) for a in groups[l] for lp in linked for b in groups[lp]),
            default=None,
        )
        if best:
            G.add_edge(best[1], best[2])
        linked.append(l)
    return G
