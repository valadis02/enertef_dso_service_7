"""
DSO Service 7 - Phase 22
Hierarchy-constrained reconstruction.

THE INSIGHT WE HADN'T USED. Every previous method scored candidate edges
with a SECOND-MOMENT statistic (correlation, covariance, variance of
voltage differences). Phase 21 showed exactly why those plateau: meter
noise adds an irreducible BIAS (2*sigma^2) to every second moment, which
does NOT shrink with more samples.

The FIRST moment behaves completely differently. The sample mean of a
noisy measurement is UNBIASED, and its error shrinks as sigma/sqrt(n).
With 672+ samples, mean voltage per bus is known far more accurately
than any covariance.

And in a RADIAL feeder the mean voltage carries strong structural
information: voltage drops monotonically as you move downstream from the
substation. So ordering buses by mean voltage approximates their DEPTH
in the tree. That gives a hard constraint:

    the parent of a bus must have a HIGHER mean voltage than the bus

This turns free-form MST search (where any of the ~N^2/2 pairs can be an
edge) into a constrained parent-assignment problem: for each bus, choose
its parent only among the buses above it in the voltage ordering, using
the electrical-distance metric to pick the nearest one. Roughly halves
the search space and enforces radiality by construction.
"""

import numpy as np
import pandas as pd
import networkx as nx

from phase19_voltage_difference import voltage_difference_variance_matrix
from phase1_2_baseline import correlation_mst, evaluate
from phase20_vdiff_enhanced import vdiff_topology_debiased


def hierarchy_constrained_topology(voltage_df, smooth_window=15, use_smoothing=True):
    """
    1. Order buses by mean voltage (descending) -> approximate depth order.
       The highest-mean bus is treated as the root (substation side).
    2. For each bus in that order, attach it to the already-placed bus
       with the smallest voltage-difference variance (electrical distance).
       Because we process in descending-voltage order, the chosen parent
       always has a higher mean voltage -- radiality is enforced.
    """
    df = voltage_df
    if use_smoothing and smooth_window > 1:
        df = df.rolling(window=smooth_window, center=True, min_periods=1).mean()

    nodes = df.columns.tolist()
    means = df.mean()
    # descending mean voltage = from substation outward
    order = sorted(nodes, key=lambda c: -means[c])

    D, d_nodes = voltage_difference_variance_matrix(df)
    idx = {c: i for i, c in enumerate(d_nodes)}

    G = nx.Graph()
    G.add_nodes_from(nodes)
    placed = [order[0]]
    for node in order[1:]:
        # parent = already-placed bus (higher mean voltage) that is
        # electrically closest
        best_parent = min(placed, key=lambda p: D[idx[node], idx[p]])
        G.add_edge(node, best_parent)
        placed.append(node)
    return G


def main():
    import simbench as sb
    from phase5_simbench import ground_truth_graph, simulate_timeseries_simbench
    from phase1_2_baseline import load_feeder, ground_truth_graph as gt_ieee, simulate_timeseries

    NOISE = [("clean", 0.0), ("PMU 0.0005", 0.0005),
             ("Class0.5 0.0025", 0.0025), ("Class1 0.005", 0.005)]

    cases = []
    net = load_feeder()
    cases.append(("IEEE 33-bus (synthetic)", gt_ieee(net),
                   simulate_timeseries(net, n_steps=500)[0], False))
    for code in ["1-LV-rural1--0-sw", "1-LV-semiurb4--0-sw", "1-LV-urban6--0-sw"]:
        n2 = sb.get_simbench_net(code)
        cases.append((code, gt_sb_wrap(n2), simulate_timeseries_simbench(n2, n_steps=672,
                                                                          start=96 * 7 * 26), True))

    rows = []
    for case_name, tg, vdf, is_real in cases:
        print(f"\n{case_name}  [{tg.number_of_nodes()} buses]")
        for label, sigma in NOISE:
            rng = np.random.default_rng(7)
            d = vdf if sigma == 0 else vdf + rng.normal(0, sigma, size=vdf.shape)

            mb = evaluate(correlation_mst(d)[0], tg, vdf.columns)["reconstruction_accuracy"]
            mv = evaluate(vdiff_topology_debiased(d, debias=False,
                                                   smooth_window=15 if is_real else 0),
                          tg, vdf.columns)["reconstruction_accuracy"]
            mh = evaluate(hierarchy_constrained_topology(d, use_smoothing=is_real),
                          tg, vdf.columns)["reconstruction_accuracy"]

            print(f"  {label:18s} baseline={mb:.3f}  Vdiff(best)={mv:.3f}  HIERARCHY={mh:.3f}")
            rows.append({"case": case_name, "noise": label, "baseline": mb,
                          "vdiff_best": mv, "hierarchy": mh})

    results = pd.DataFrame(rows)
    results.to_csv("results/phase22_hierarchy.csv", index=False)
    print("\n=== Mean per noise level ===")
    print(results.groupby("noise")[["baseline", "vdiff_best", "hierarchy"]].mean().to_string())
    print("\nSaved results/phase22_hierarchy.csv")


def gt_sb_wrap(net):
    from phase5_simbench import ground_truth_graph
    return ground_truth_graph(net)


if __name__ == "__main__":
    main()
