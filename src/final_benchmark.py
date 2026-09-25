"""
DSO Service 7 - FINAL BENCHMARK
Self-contained. Runs the best algorithm we arrived at, on:
  - 3 synthetic networks, clean (no meter noise)
  - 3 real SimBench LV feeders, with good meters + phase angles

BEST ALGORITHM = `reconstruct()` from phase30_final.py:

  1. Hierarchy from mean voltage. Buses sorted by descending mean
     voltage; in a radial feeder voltage falls monotonically downstream,
     so this ordering is the tree depth and a node's parent always comes
     earlier. The mean is an UNBIASED estimator (error ~ sigma/sqrt(n)),
     unlike every second-moment statistic, which carries an irreducible
     2*sigma^2 bias.

  2. Electrical distance. Var(V_i - V_j) -- the common-mode term cancels
     algebraically in the difference, and under LinDistFlow the variance
     is proportional to electrical distance. Meter noise adds the same
     constant to every pair, so the ranking survives.

  3. Phase angles when available. Var(theta_i - theta_j) is also a tree
     metric over the same tree, weighted by reactance instead of
     resistance -- independent information. Used INSTEAD of the
     magnitude metric (blending the two is consistently worse), on
     feeders of >= 20 buses.

  4. SNR gate. Removing the common load factor by per-bus regression
     gives a cleaner but much smaller signal; it pays off only when the
     noise floor is well below it. sigma^2 is estimated spectrally, from
     the median of the smallest covariance eigenvalues.

  5. Blind clustering above 50 buses. Buses are clustered by electrical
     distance (no topology knowledge), reconstructed per cluster, then
     the clusters are linked. Gain grows with network size.

Metric: reconstruction accuracy = fraction of true edges recovered.
"""

import numpy as np
import pandas as pd
import networkx as nx
import pandapower as pp
import simbench as sb

from phase30_final import reconstruct
from phase1_2_baseline import (load_feeder, ground_truth_graph as gt_ieee,
                                simulate_timeseries, correlation_mst, evaluate)
from phase5_simbench import ground_truth_graph as gt_sb
from phase10a_synthetic_generator import (random_radial_tree, assign_random_impedances,
                                           simulate_lindistflow)

# good meter + synchronised angle
SIGMA_V = 0.0005      # Class 0.1, PMU-grade magnitude
SIGMA_DEG = 0.002     # micro-PMU angle accuracy
REAL_FEEDERS = ["1-LV-rural1--0-sw", "1-LV-semiurb4--0-sw", "1-LV-urban6--0-sw"]


def simulate_ieee_with_angle(net, n_steps=500, seed=42):
    rng = np.random.default_rng(seed)
    bp, bq = net.load["p_mw"].copy(), net.load["q_mvar"].copy()
    V, A = [], []
    for t in range(n_steps):
        cf = 1 + 0.15 * np.sin(2 * np.pi * t / 96) + rng.normal(0, 0.05)
        ns = rng.normal(1, 0.25, size=len(net.load))
        net.load["p_mw"] = np.clip(bp * cf * ns, 0, None)
        net.load["q_mvar"] = np.clip(bq * cf * ns, 0, None)
        try:
            pp.runpp(net)
        except Exception:
            continue
        V.append(net.res_bus["vm_pu"].copy())
        A.append(net.res_bus["va_degree"].copy())
    net.load["p_mw"], net.load["q_mvar"] = bp, bq
    return pd.DataFrame(V).reset_index(drop=True), pd.DataFrame(A).reset_index(drop=True)


def simulate_simbench_with_angle(net, n_steps=672, start=96 * 7 * 26):
    av = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    lp, lq = av[("load", "p_mw")], av[("load", "q_mvar")]
    sp = av.get(("sgen", "p_mw"))
    V, A = [], []
    for t in range(start, start + n_steps):
        net.load["p_mw"] = lp.loc[t].values
        net.load["q_mvar"] = lq.loc[t].values
        if sp is not None and len(net.sgen) > 0:
            net.sgen["p_mw"] = sp.loc[t].values
        try:
            pp.runpp(net)
        except Exception:
            continue
        V.append(net.res_bus["vm_pu"].copy())
        A.append(net.res_bus["va_degree"].copy())
    return pd.DataFrame(V).reset_index(drop=True), pd.DataFrame(A).reset_index(drop=True)


def main():
    rows = []

    print("=" * 74)
    print("SYNTHETIC NETWORKS - clean (no meter noise)")
    print("=" * 74)
    print("Distribution over randomly generated radial feeders, 12 per size.")
    print("Reporting a distribution rather than a hand-picked network: a single")
    print("draw can land anywhere in the spread, so the spread is the result.\n")

    rng = np.random.default_rng(2026)
    for size in (15, 25, 40):
        accs, bases = [], []
        for _ in range(12):
            G = assign_random_impedances(random_radial_tree(size, rng), rng)
            V = simulate_lindistflow(G, 500, rng, meter_noise_sigma=0.0, idio_noise_std=0.03)
            nodes = [c for c in V.columns if c in G]
            bases.append(evaluate(correlation_mst(V)[0], G, V.columns)["reconstruction_accuracy"])
            accs.append(evaluate(reconstruct(V, nodes, use_smoothing=False),
                                 G, V.columns)["reconstruction_accuracy"])
        a, b = np.array(accs), np.array(bases)
        print(f"  random radial, {size:2d} buses (n=12): baseline={b.mean():.3f}   "
              f"BEST mean={a.mean():.3f} median={np.median(a):.3f} min={a.min():.3f}  "
              f"[{(a == 1).mean():.0%} at 100%, {(a >= 0.95).mean():.0%} at >=95%]")
        rows.append({"set": "synthetic (clean, n=12)", "network": f"random radial {size} buses",
                      "buses": size, "baseline": round(b.mean(), 3), "best": round(a.mean(), 3),
                      "best_median": round(float(np.median(a)), 3), "best_min": round(a.min(), 3)})

    # IEEE 33-bus kept as the recognisable literature benchmark
    net = load_feeder()
    tg = gt_ieee(net)
    V, A = simulate_ieee_with_angle(net)
    nodes = [c for c in V.columns if c in tg]
    base = evaluate(correlation_mst(V)[0], tg, V.columns)["reconstruction_accuracy"]
    acc = evaluate(reconstruct(V, nodes, angle_df=A, use_smoothing=False),
                   tg, V.columns)["reconstruction_accuracy"]
    print(f"  {'IEEE 33-bus':30s} baseline={base:.3f}   BEST={acc:.3f}")
    rows.append({"set": "synthetic (clean)", "network": "IEEE 33-bus", "buses": tg.number_of_nodes(),
                  "baseline": round(base, 3), "best": round(acc, 3)})

    print()
    print("=" * 74)
    print(f"REAL SimBench LV feeders - good meters (sigma_V={SIGMA_V} p.u.)")
    print(f"                           + phase angles (sigma_theta={SIGMA_DEG} deg)")
    print("=" * 74)
    for code in REAL_FEEDERS:
        net = sb.get_simbench_net(code)
        tg = gt_sb(net)
        V, A = simulate_simbench_with_angle(net)
        nodes = [c for c in V.columns if c in tg]
        rng2 = np.random.default_rng(7)
        Vn = V + rng2.normal(0, SIGMA_V, size=V.shape)
        An = A + rng2.normal(0, SIGMA_DEG, size=A.shape)
        base = evaluate(correlation_mst(Vn[nodes])[0], tg, V.columns)["reconstruction_accuracy"]
        acc = evaluate(reconstruct(Vn, nodes, angle_df=An), tg, V.columns)["reconstruction_accuracy"]
        print(f"  {code:30s} baseline={base:.3f}   BEST={acc:.3f}")
        rows.append({"set": "real (good meters + angles)", "network": code,
                      "buses": tg.number_of_nodes(), "baseline": round(base, 3), "best": round(acc, 3)})

    res = pd.DataFrame(rows)
    res.to_csv("results/final_benchmark.csv", index=False)
    print()
    print("=" * 74)
    print(res.to_string(index=False))
    print("\nSaved results/final_benchmark.csv")


if __name__ == "__main__":
    main()
