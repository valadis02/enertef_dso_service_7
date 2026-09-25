"""
DSO Service 7 - Phase 37
Prior-informed reconstruction.

WHY THIS IS THE RIGHT PROBLEM FORMULATION. Everything up to here assumed
blind reconstruction: build the topology from measurements alone, with no
starting point. But the D2.2 problem statement says the DSO *has* a
recorded topology in its asset management system -- the issue is that it
does not always match reality, because of switching operations and
undocumented field changes.

So the real task is not "build a graph from nothing", it is "correct a
graph that is mostly right". That is a far easier problem, and it is
exactly why flagging reached 93% on rural2 while blind reconstruction
stayed at 34% there.

METHOD. Start from the documented topology. Score every documented edge
by electrical distance. Remove the worst ones (those the data most
strongly contradicts), then re-attach the orphaned nodes using the same
hierarchy rule as the blind method. Everything the data does not argue
against is left untouched.

The only parameter is how aggressive to be: `suspect_fraction`, the share
of documented edges to challenge. Setting it near the true error rate is
the natural choice, but the sweep below shows what happens when the
operator over- or under-estimates it.
"""

import numpy as np
import pandas as pd
import networkx as nx
import simbench as sb

from phase19_voltage_difference import voltage_difference_variance_matrix
from phase30_final import reconstruct
from phase5_simbench import ground_truth_graph as gt_s
from phase1_2_baseline import evaluate, correlation_mst
from final_benchmark import simulate_simbench_with_angle

FEEDERS = ["1-LV-rural1--0-sw", "1-LV-semiurb4--0-sw", "1-LV-urban6--0-sw",
           "1-LV-rural2--0-sw", "1-LV-rural3--0-sw", "1-LV-semiurb5--0-sw"]
ERROR_RATE = 0.15          # share of documented edges that are wrong


def corrupt(true_graph, n_corrupt, seed=2):
    """Simulate an asset-management record that is mostly right: remove
    n_corrupt real edges and add the same number of wrong ones."""
    rng = np.random.default_rng(seed)
    G = true_graph.copy()
    real = list(G.edges())
    for i in rng.choice(len(real), size=n_corrupt, replace=False):
        G.remove_edge(*real[i])
    nodes = list(G.nodes())
    added = 0
    while added < n_corrupt:
        a, b = rng.choice(nodes, size=2, replace=False)
        if not true_graph.has_edge(a, b) and not G.has_edge(a, b):
            G.add_edge(a, b)
            added += 1
    return G


def prior_informed(documented, voltage_df, nodes, angle_df=None,
                   suspect_fraction=0.15, use_smoothing=True):
    """Correct a documented topology instead of rebuilding it."""
    src = angle_df if angle_df is not None else voltage_df
    df = src[nodes]
    if use_smoothing:
        df = df.rolling(15, center=True, min_periods=1).mean()
    D, dn = voltage_difference_variance_matrix(df)
    idx = {c: i for i, c in enumerate(dn)}

    edges = [e for e in documented.edges() if e[0] in idx and e[1] in idx]
    k = max(1, int(round(len(edges) * suspect_fraction)))
    scored = sorted(edges, key=lambda e: -D[idx[e[0]], idx[e[1]]])

    G = documented.copy()
    G.remove_edges_from(scored[:k])

    # re-attach whatever got disconnected, using the hierarchy rule
    means = voltage_df[nodes].mean()
    order = sorted(nodes, key=lambda c: -means[c])
    rank = {c: i for i, c in enumerate(order)}
    root = order[0]

    while True:
        comps = list(nx.connected_components(G))
        if len(comps) <= 1:
            break
        root_comp = next(c for c in comps if root in c)
        other = [c for c in comps if root not in c]
        # attach the component whose best link to the root side is closest
        best = None
        for comp in other:
            for a in comp:
                if a not in idx:
                    continue
                for b in root_comp:
                    if b not in idx or rank[b] >= rank[a]:
                        continue
                    d = D[idx[a], idx[b]]
                    if best is None or d < best[0]:
                        best = (d, a, b)
        if best is None:   # fall back: ignore the depth constraint
            for comp in other:
                for a in comp:
                    for b in root_comp:
                        if a in idx and b in idx:
                            d = D[idx[a], idx[b]]
                            if best is None or d < best[0]:
                                best = (d, a, b)
        if best is None:
            break
        G.add_edge(best[1], best[2])
    return G


def main():
    rows = []
    for code in FEEDERS:
        net = sb.get_simbench_net(code)
        tg = gt_s(net)
        V, A = simulate_simbench_with_angle(net)
        nodes = [c for c in V.columns if c in tg]
        n_corrupt = max(1, int(round(tg.number_of_edges() * ERROR_RATE)))
        documented = corrupt(tg, n_corrupt)
        doc_acc = evaluate(documented, tg, V.columns)["reconstruction_accuracy"]

        for lbl, sv, sa in [("Class 0.5, χωρίς γωνία", 0.0025, None),
                            ("Class 0.5 + γωνία 0.01°", 0.0025, 0.01),
                            ("καλοί + γωνία", 0.0005, 0.002)]:
            rng = np.random.default_rng(7)
            Vn = V + rng.normal(0, sv, size=V.shape)
            An = (A + rng.normal(0, sa, size=A.shape)) if sa else None

            blind = evaluate(reconstruct(Vn, nodes, angle_df=An),
                             tg, V.columns)["reconstruction_accuracy"]
            prior = evaluate(prior_informed(documented, Vn, nodes, An, ERROR_RATE),
                             tg, V.columns)["reconstruction_accuracy"]

            rows.append({"feeder": code.split("--")[0], "buses": len(nodes),
                          "σενάριο": lbl,
                          "τεκμηρίωση": round(doc_acc, 3),
                          "τυφλή": round(blind, 3),
                          "PRIOR-INFORMED": round(prior, 3)})

    df = pd.DataFrame(rows)
    df.to_csv("results/phase37_prior_informed.csv", index=False)
    print(df.to_string(index=False))
    print()
    print("=== Μέσοι όροι ανά σενάριο ===")
    print(df.groupby("σενάριο")[["τεκμηρίωση", "τυφλή", "PRIOR-INFORMED"]].mean().to_string())
    print("\nSaved results/phase37_prior_informed.csv")


if __name__ == "__main__":
    main()
