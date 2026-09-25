"""
DSO Service 7 - Phase 8
Two more principled alternatives to raw Pearson correlation + MST, tested
on real SimBench data (clean and with realistic Class 0.5 meter noise):

  1. Graphical Lasso with a manual alpha GRID (not CV-tuned, which gave a
     poor/inconsistent result in phase4) -> MST over |precision|.

  2. Chow-Liu tree: builds the MST using pairwise MUTUAL INFORMATION
     instead of Pearson correlation. This is the textbook-optimal method
     for recovering a tree-structured graphical model (Chow & Liu, 1968):
     if the true joint distribution over bus voltages factorizes as a
     tree (which a radial LV/MV feeder approximately does, per the
     Bolognani et al. Markov-random-field argument), the Chow-Liu tree
     converges to the TRUE topology as sample size grows, even under
     noise -- unlike Pearson correlation, which only captures LINEAR
     pairwise association and is provably confounded by indirect paths.
     We don't need to know line impedances in advance to use this (unlike
     a full physics/LinDistFlow model, which would need the very topology
     we're trying to identify).
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.covariance import GraphicalLasso
from sklearn.feature_selection import mutual_info_regression

from phase1_2_baseline import evaluate
from phase5_simbench import load_simbench_feeder, ground_truth_graph, simulate_timeseries_simbench, add_realistic_meter_noise


# ---------------------------------------------------------------------------
# 1. Graphical Lasso, manual alpha grid
# ---------------------------------------------------------------------------

def graphical_lasso_fixed_alpha(voltage_df, alpha, drop_slack=True):
    df = voltage_df.copy()
    if drop_slack:
        variances = df.var()
        df = df[variances[variances > 1e-12].index.tolist()]

    X = ((df - df.mean()) / df.std()).values
    model = GraphicalLasso(alpha=alpha, max_iter=1000)
    try:
        model.fit(X)
    except FloatingPointError:
        return None
    precision = model.precision_

    cols = df.columns.tolist()
    G_full = nx.Graph()
    G_full.add_nodes_from(cols)
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            G_full.add_edge(cols[i], cols[j], weight=abs(precision[i, j]))
    mst = nx.maximum_spanning_tree(G_full, weight="weight")
    mst.add_nodes_from(voltage_df.columns)
    return mst


def sweep_graphical_lasso(voltage_df, true_graph, alphas=(0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)):
    rows = []
    for alpha in alphas:
        mst = graphical_lasso_fixed_alpha(voltage_df, alpha)
        if mst is None:
            continue
        m = evaluate(mst, true_graph, voltage_df.columns)
        m["alpha"] = alpha
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Chow-Liu tree (mutual information MST)
# ---------------------------------------------------------------------------

def chow_liu_tree(voltage_df, n_neighbors=3):
    """Build MST using pairwise mutual information (k-NN estimator via
    sklearn) instead of Pearson correlation."""
    cols = voltage_df.columns.tolist()
    n = len(cols)
    X = voltage_df.values

    G = nx.Graph()
    G.add_nodes_from(cols)
    for i in range(n):
        # mutual_info_regression(X, y) gives MI between each column of X and y
        mi = mutual_info_regression(X, X[:, i], n_neighbors=n_neighbors, random_state=0)
        for j in range(n):
            if i == j:
                continue
            G.add_edge(cols[i], cols[j], weight=mi[j])

    mst = nx.maximum_spanning_tree(G, weight="weight")
    return mst


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    net = load_simbench_feeder()
    true_graph = ground_truth_graph(net)
    week_start = 96 * 7 * 26
    voltage_df = simulate_timeseries_simbench(net, n_steps=672, start=week_start)

    for condition, df in [("clean", voltage_df), ("noisy_class0.5", add_realistic_meter_noise(voltage_df))]:
        print(f"\n{'='*60}\nCondition: {condition}\n{'='*60}")

        print("\n--- Graphical Lasso alpha sweep ---")
        gl_results = sweep_graphical_lasso(df, true_graph)
        gl_results["condition"] = condition
        for _, row in gl_results.iterrows():
            print(f"  alpha={row['alpha']:.3f}: precision={row['precision']:.3f} "
                  f"recall={row['recall']:.3f} recon_acc={row['reconstruction_accuracy']:.3f} "
                  f"(n_edges={row['n_pred_edges']})")
        best_gl = gl_results.loc[gl_results["reconstruction_accuracy"].idxmax()]
        print(f"  BEST: alpha={best_gl['alpha']:.3f} -> recon_acc={best_gl['reconstruction_accuracy']:.3f}")
        gl_results.to_csv(f"results/phase8_glasso_sweep_{condition}.csv", index=False)

        print("\n--- Chow-Liu (mutual information) tree ---")
        cl_tree = chow_liu_tree(df)
        cl_metrics = evaluate(cl_tree, true_graph, df.columns)
        print(f"  precision={cl_metrics['precision']:.3f} recall={cl_metrics['recall']:.3f} "
              f"recon_acc={cl_metrics['reconstruction_accuracy']:.3f}")

        print("\n--- Baseline for comparison (Pearson correlation + MST) ---")
        from phase1_2_baseline import correlation_mst
        mst_baseline, _ = correlation_mst(df)
        base_metrics = evaluate(mst_baseline, true_graph, df.columns)
        print(f"  precision={base_metrics['precision']:.3f} recall={base_metrics['recall']:.3f} "
              f"recon_acc={base_metrics['reconstruction_accuracy']:.3f}")


if __name__ == "__main__":
    main()

    