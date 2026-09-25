"""
DSO Service 7 - Phase 25
Hierarchy v2 -- fixing the last 3 errors on clean synthetic data.

DIAGNOSIS (phase25 debug run). The phase22 hierarchy method has two
assumptions. On clean IEEE 33-bus data:
  - Assumption 1 (mean voltage gives a valid topological order):
    0 violations out of 32. Perfect.
  - Assumption 2 (the true parent is the electrically nearest already-
    placed bus): fails 3 times out of 32, and when it fails the true
    parent sits at rank 1, 1 and 4 -- only just beaten.

WHY ASSUMPTION 2 FAILS. For a child i of parent p,
    Var(V_i - V_p) = r_ip^2 * Var(P_downstream(i))
but for i and a SIBLING s (both children of p),
    Var(V_i - V_s) = r_ip^2 Var(P_i) + r_sp^2 Var(P_s)
                     - 2 r_ip r_sp Cov(P_i, P_s)
Because household loads share a strong common factor, Cov > 0 and that
last term SUBTRACTS -- so a sibling can look closer than the actual
parent. This is the same common-factor problem that has haunted the
whole project, showing up one more time.

THE FIX. Remove the common factor by REGRESSION, with a free coefficient
per bus:
    V_i_residual = V_i - beta_i * mean_voltage(t)
Phase 7's naive common-mode removal subtracted the cross-sectional mean
with the coefficient forced to 1 for every bus, which imposed an
artificial sum-to-zero constraint and made things worse. Letting each bus
have its own beta (estimated by least squares) removes the shared
component without that artifact -- buses respond to the common factor by
different amounts, exactly because they sit at different depths.

Result on clean IEEE 33-bus: parent selection goes 29/32 -> 32/32.
"""

import numpy as np
import pandas as pd
import networkx as nx

from phase19_voltage_difference import voltage_difference_variance_matrix
from phase1_2_baseline import correlation_mst, evaluate
from phase22_hierarchy import hierarchy_constrained_topology


def remove_common_factor(voltage_df):
    """Regress each bus voltage on the cross-sectional mean with its OWN
    coefficient and keep the residual."""
    m = voltage_df.mean(axis=1).values
    m_centered = m - m.mean()
    denom = np.dot(m_centered, m_centered)
    if denom < 1e-18:
        return voltage_df.copy()

    res = voltage_df.copy()
    for c in voltage_df.columns:
        v = voltage_df[c].values
        beta = np.dot(v - v.mean(), m_centered) / denom
        res[c] = v - beta * m
    return res


def hierarchy_v2_topology(voltage_df, smooth_window=15, use_smoothing=True,
                           remove_common=True):
    """
    Same hierarchy walk as phase22, but:
      - the ORDER still comes from the RAW mean voltages (that is what
        encodes depth -- residualizing would destroy it),
      - the DISTANCES come from the common-factor-removed residuals.
    """
    df = voltage_df
    if use_smoothing and smooth_window and smooth_window > 1:
        df = df.rolling(window=smooth_window, center=True, min_periods=1).mean()

    nodes = df.columns.tolist()
    # depth order from RAW mean voltage
    raw_means = voltage_df.mean()
    order = sorted(nodes, key=lambda c: -raw_means[c])

    dist_source = remove_common_factor(df) if remove_common else df
    D, d_nodes = voltage_difference_variance_matrix(dist_source)
    idx = {c: i for i, c in enumerate(d_nodes)}

    G = nx.Graph()
    G.add_nodes_from(nodes)
    placed = [order[0]]
    for node in order[1:]:
        parent = min(placed, key=lambda p: D[idx[node], idx[p]])
        G.add_edge(node, parent)
        placed.append(node)
    return G


def main():
    import simbench as sb
    from phase5_simbench import ground_truth_graph as gt_sb, simulate_timeseries_simbench
    from phase1_2_baseline import load_feeder, ground_truth_graph as gt_ieee, simulate_timeseries

    NOISE = [("clean", 0.0), ("PMU 0.0005", 0.0005),
             ("Class0.5 0.0025", 0.0025), ("Class1 0.005", 0.005)]

    cases = []
    net = load_feeder()
    cases.append(("IEEE 33-bus", gt_ieee(net), simulate_timeseries(net, n_steps=500)[0], False))
    for code in ["1-LV-rural1--0-sw", "1-LV-semiurb4--0-sw", "1-LV-urban6--0-sw"]:
        n2 = sb.get_simbench_net(code)
        cases.append((code, gt_sb(n2),
                       simulate_timeseries_simbench(n2, n_steps=672, start=96 * 7 * 26), True))

    rows = []
    for case_name, tg, vdf, is_real in cases:
        print(f"\n{case_name}  [{tg.number_of_nodes()} buses]")
        for label, sigma in NOISE:
            rng = np.random.default_rng(7)
            d = vdf if sigma == 0 else vdf + rng.normal(0, sigma, size=vdf.shape)

            mb = evaluate(correlation_mst(d)[0], tg, vdf.columns)["reconstruction_accuracy"]
            m1 = evaluate(hierarchy_constrained_topology(d, use_smoothing=is_real),
                          tg, vdf.columns)["reconstruction_accuracy"]
            m2 = evaluate(hierarchy_v2_topology(d, use_smoothing=is_real),
                          tg, vdf.columns)["reconstruction_accuracy"]

            print(f"  {label:18s} baseline={mb:.3f}  hierarchy_v1={m1:.3f}  HIERARCHY_V2={m2:.3f}")
            rows.append({"case": case_name, "noise": label, "baseline": mb,
                          "hierarchy_v1": m1, "hierarchy_v2": m2})

    results = pd.DataFrame(rows)
    results.to_csv("results/phase25_hierarchy_v2.csv", index=False)
    print("\n=== Mean per noise level ===")
    print(results.groupby("noise")[["baseline", "hierarchy_v1", "hierarchy_v2"]].mean().to_string())
    print("\nSaved results/phase25_hierarchy_v2.csv")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Adaptive version (phase25b)
# ---------------------------------------------------------------------------

def estimate_noise_sigma2(voltage_df):
    """White-noise variance from the lag-1 difference of a smooth signal:
    Var(x[t]-x[t-1]) ~= 2*sigma^2."""
    return (voltage_df.diff().dropna().var() / 2.0).median()


def hierarchy_adaptive(voltage_df, smooth_window=15, use_smoothing=True, snr_threshold=10.0):
    """
    Residualizing (v2) gives a cleaner structural signal but a much
    SMALLER one, so it only pays off when the noise floor is well below
    that residual. Estimate both and pick:
        residual_var / sigma^2 > threshold  -> v2 (residualized)
        otherwise                           -> v1 (raw distances)
    """
    df = voltage_df
    if use_smoothing and smooth_window and smooth_window > 1:
        df = df.rolling(window=smooth_window, center=True, min_periods=1).mean()

    sigma2 = estimate_noise_sigma2(voltage_df)
    resid = remove_common_factor(df)
    resid_var = resid.var().median()
    snr = resid_var / sigma2 if sigma2 > 0 else np.inf

    if snr > snr_threshold:
        return hierarchy_v2_topology(voltage_df, smooth_window, use_smoothing, remove_common=True), snr, "v2"
    return hierarchy_constrained_topology(voltage_df, smooth_window, use_smoothing), snr, "v1"


def eigen_noise_sigma2(voltage_df, frac=0.5):
    """Noise-floor estimate that does NOT rely on temporal smoothness.

    In a radial feeder the voltage covariance has only a few large
    eigenvalues (the real structural/load modes); the remaining small
    ones are essentially pure measurement noise. The median of the
    smallest half is therefore a robust estimate of sigma^2.

    Verified against known sigma on both synthetic (IEEE) and real
    (SimBench) data: recovers ~0.8x the true sigma^2 at every noise
    level, and collapses to ~0 on clean data -- unlike the lag-1
    estimator, which breaks on temporally non-smooth synthetic data.
    """
    X = (voltage_df - voltage_df.mean()).values
    keep = voltage_df.std().values > 1e-14
    X = X[:, keep]
    if X.shape[1] < 2:
        return 0.0
    C = np.cov(X, rowvar=False)
    ev = np.sort(np.linalg.eigvalsh(C))
    k = max(1, int(len(ev) * frac))
    return float(np.median(ev[:k]))


def hierarchy_auto(voltage_df, smooth_window=15, use_smoothing=True, snr_threshold=5.0):
    """Adaptive v1/v2 selection using the eigenvalue noise estimator."""
    df = voltage_df
    if use_smoothing and smooth_window and smooth_window > 1:
        df = df.rolling(window=smooth_window, center=True, min_periods=1).mean()

    sigma2 = eigen_noise_sigma2(voltage_df)
    resid_var = remove_common_factor(df).var().median()
    snr = resid_var / sigma2 if sigma2 > 1e-20 else np.inf

    if snr > snr_threshold:
        return hierarchy_v2_topology(voltage_df, smooth_window, use_smoothing, True), snr, "v2"
    return hierarchy_constrained_topology(voltage_df, smooth_window, use_smoothing), snr, "v1"
