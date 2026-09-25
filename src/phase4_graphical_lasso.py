"""
DSO Service 7 - Phase 4
Graphical Lasso: sparse inverse covariance estimation for topology
identification. Unlike raw Pearson correlation (which picks up indirect/
transitive correlations through intermediate buses), the precision matrix
(inverse covariance) captures *conditional* dependence -- entry (i,j) is
~0 if buses i and j are independent given all other buses. For a radial
grid, direct electrical neighbors should show up as the strongest nonzero
entries, which should be more robust to noise than raw correlation.
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso

from phase1_2_baseline import load_feeder, ground_truth_graph, simulate_timeseries, evaluate


def graphical_lasso_topology(voltage_df, alpha=None, drop_slack=True, use_mst=True):
    """
    Fit a Graphical Lasso model on standardized voltage measurements and
    build a predicted topology from the resulting precision matrix.

    If alpha is None, GraphicalLassoCV picks the sparsity penalty via
    cross-validation. drop_slack removes zero-variance columns (e.g. the
    slack bus) which would make the covariance matrix singular/undefined.

    use_mst=True takes the Maximum Spanning Tree over |precision| instead
    of a raw nonzero-threshold: this enforces the same radial/tree prior
    used for the correlation baseline, so the two methods are compared on
    equal footing (both forced to output exactly N-1 edges) rather than
    penalizing Graphical Lasso for outputting a denser graph.
    """
    df = voltage_df.copy()
    if drop_slack:
        # drop any column with ~zero variance (slack/reference bus)
        variances = df.var()
        keep_cols = variances[variances > 1e-12].index.tolist()
        df = df[keep_cols]

    # standardize (Graphical Lasso assumes comparable scales)
    X = (df - df.mean()) / df.std()
    X = X.values

    if alpha is None:
        model = GraphicalLassoCV(cv=5, max_iter=500)
    else:
        model = GraphicalLasso(alpha=alpha, max_iter=500)

    model.fit(X)
    precision = model.precision_

    cols = df.columns.tolist()
    n = len(cols)

    if use_mst:
        G_full = nx.Graph()
        G_full.add_nodes_from(cols)
        for i in range(n):
            for j in range(i + 1, n):
                G_full.add_edge(cols[i], cols[j], weight=abs(precision[i, j]))
        G = nx.maximum_spanning_tree(G_full, weight="weight")
        G.add_nodes_from(voltage_df.columns)  # keep dropped slack as isolated node
    else:
        G = nx.Graph()
        G.add_nodes_from(voltage_df.columns)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(precision[i, j]) > 1e-10:
                    G.add_edge(cols[i], cols[j], weight=abs(precision[i, j]))

    return G, model


def compare_methods(voltage_df, true_graph, noise_sigma, seed=999):
    from phase1_2_baseline import correlation_mst

    rng = np.random.default_rng(seed)
    noisy = voltage_df + rng.normal(0, noise_sigma, size=voltage_df.shape)

    mst, _ = correlation_mst(noisy)
    mst_metrics = evaluate(mst, true_graph, voltage_df.columns)

    glasso_graph, model = graphical_lasso_topology(noisy)
    glasso_metrics = evaluate(glasso_graph, true_graph, voltage_df.columns)

    print(f"noise sigma={noise_sigma} p.u.")
    print(f"  Correlation+MST : precision={mst_metrics['precision']:.3f} "
          f"recall={mst_metrics['recall']:.3f} recon_acc={mst_metrics['reconstruction_accuracy']:.3f} "
          f"(n_edges={mst_metrics['n_pred_edges']})")
    print(f"  Graphical Lasso : precision={glasso_metrics['precision']:.3f} "
          f"recall={glasso_metrics['recall']:.3f} recon_acc={glasso_metrics['reconstruction_accuracy']:.3f} "
          f"(n_edges={glasso_metrics['n_pred_edges']})")

    return mst_metrics, glasso_metrics


def main():
    net = load_feeder()
    true_graph = ground_truth_graph(net)
    voltage_df, _ = simulate_timeseries(net, n_steps=500)

    rows = []
    for sigma in (0.0, 0.001, 0.005, 0.01, 0.02, 0.05):
        mst_metrics, glasso_metrics = compare_methods(voltage_df, true_graph, sigma)
        mst_metrics["method"] = "correlation_mst"
        mst_metrics["noise_sigma_pu"] = sigma
        glasso_metrics["method"] = "graphical_lasso"
        glasso_metrics["noise_sigma_pu"] = sigma
        rows.append(mst_metrics)
        rows.append(glasso_metrics)
        print()

    results = pd.DataFrame(rows)
    results.to_csv("results/phase4_method_comparison.csv", index=False)
    print("Saved results/phase4_method_comparison.csv")


if __name__ == "__main__":
    main()

    