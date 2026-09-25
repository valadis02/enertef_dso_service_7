"""
DSO Service 7 - Phase 38
Topology DETECTION with localised uncertainty.

WHY THIS IS A DIFFERENT PROBLEM. The literature separates two tasks:
  - topology IDENTIFICATION: rebuild the graph from measurements (what
    phases 1-36 did)
  - topology DETECTION / system configuration identification: the line
    infrastructure is known (impedances, which buses each line connects);
    what is unknown is which lines are currently energised. The tutorial
    paper calls this "a relatively simpler task".

Phase 37 got this wrong: it corrupted edges at random across the whole
feeder, so every one of the ~95 documented edges was equally suspect.
Reality is not like that. Uncertainty sits at known places -- switches,
recently worked-on sections -- so the operator knows WHICH connections
are doubtful. With s uncertain switches the search is over 2^s
configurations, not over all edge subsets. That is why library/projection
methods in the literature report near-100%.

THIS EXPERIMENT. Give the algorithm what a real DSO would have:
  - the documented topology,
  - a list of CANDIDATE uncertain edges (a superset of the real errors:
    the true errors plus some decoys, since the operator knows where the
    switches are but not their state).
Only those candidates may be switched on or off. Everything else is
trusted. We then pick the configuration that best fits the measured
electrical distances.

Compared against: doing nothing (the documented topology), and the
phase-37 version where all edges were equally suspect.
"""

import numpy as np
import pandas as pd
import networkx as nx
import simbench as sb

from phase19_voltage_difference import voltage_difference_variance_matrix
from phase37_prior_informed import corrupt, prior_informed, ERROR_RATE
from phase5_simbench import ground_truth_graph as gt_s
from phase1_2_baseline import evaluate
from final_benchmark import simulate_simbench_with_angle

FEEDERS = ["1-LV-rural1--0-sw", "1-LV-semiurb4--0-sw", "1-LV-urban6--0-sw",
           "1-LV-rural2--0-sw", "1-LV-rural3--0-sw", "1-LV-semiurb5--0-sw"]


def build_candidates(true_graph, documented, n_decoy_factor=1.0, seed=5):
    """What the operator knows: a set of edges whose status is uncertain.
    It contains the real discrepancies (wrong edges present in the record,
    and real edges missing from it) plus some decoys -- switches that are
    documented correctly but could have moved."""
    rng = np.random.default_rng(seed)
    true_e = {frozenset(e) for e in true_graph.edges()}
    doc_e = {frozenset(e) for e in documented.edges()}

    wrong_present = doc_e - true_e          # documented but not real
    missing = true_e - doc_e                # real but not documented
    real_uncertain = wrong_present | missing

    n_decoy = int(round(len(real_uncertain) * n_decoy_factor))
    correct_doc = list(doc_e & true_e)
    decoys = set()
    if correct_doc and n_decoy > 0:
        for i in rng.choice(len(correct_doc), size=min(n_decoy, len(correct_doc)),
                            replace=False):
            decoys.add(correct_doc[i])
    return real_uncertain | decoys


def detect_configuration(documented, candidates, voltage_df, nodes,
                         angle_df=None, use_smoothing=True):
    """
    Only the candidate edges may change state. Score each candidate by
    electrical distance; switch on the short ones, switch off the long
    ones, keeping the result a spanning tree.
    """
    src = angle_df if angle_df is not None else voltage_df
    df = src[nodes]
    if use_smoothing:
        df = df.rolling(15, center=True, min_periods=1).mean()
    D, dn = voltage_difference_variance_matrix(df)
    idx = {c: i for i, c in enumerate(dn)}

    def d(e):
        a, b = tuple(e)
        return D[idx[a], idx[b]] if a in idx and b in idx else np.inf

    # start from the trusted part: documented edges that are NOT candidates
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for e in documented.edges():
        if frozenset(e) not in candidates:
            G.add_edge(*e)

    # add candidates cheapest-first, Kruskal-style, while it stays a forest
    for e in sorted(candidates, key=d):
        a, b = tuple(e)
        if a not in idx or b not in idx:
            continue
        if not nx.has_path(G, a, b) if (a in G and b in G) else True:
            G.add_edge(a, b)
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
        candidates = build_candidates(tg, documented)

        for lbl, sv, sa in [("Class 0.5, χωρίς γωνία", 0.0025, None),
                            ("Class 0.5 + γωνία 0.01°", 0.0025, 0.01)]:
            rng = np.random.default_rng(7)
            Vn = V + rng.normal(0, sv, size=V.shape)
            An = (A + rng.normal(0, sa, size=A.shape)) if sa else None

            all_suspect = evaluate(prior_informed(documented, Vn, nodes, An, ERROR_RATE),
                                   tg, V.columns)["reconstruction_accuracy"]
            localised = evaluate(detect_configuration(documented, candidates, Vn, nodes, An),
                                 tg, V.columns)["reconstruction_accuracy"]

            rows.append({"feeder": code.split("--")[0], "buses": len(nodes),
                          "σενάριο": lbl,
                          "υποψήφιες": f"{len(candidates)}/{tg.number_of_edges()}",
                          "τεκμηρίωση": round(doc_acc, 3),
                          "όλες ύποπτες (φ37)": round(all_suspect, 3),
                          "ΕΝΤΟΠΙΣΜΕΝΗ": round(localised, 3)})

    df = pd.DataFrame(rows)
    df.to_csv("results/phase38_localised.csv", index=False)
    print(df.to_string(index=False))
    print()
    print("=== Μέσοι όροι ===")
    print(df.groupby("σενάριο")[["τεκμηρίωση", "όλες ύποπτες (φ37)", "ΕΝΤΟΠΙΣΜΕΝΗ"]].mean().to_string())
    print("\nSaved results/phase38_localised.csv")


if __name__ == "__main__":
    main()
