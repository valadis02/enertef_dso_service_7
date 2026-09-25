"""
DSO Service 7 - Phase 10b
Learned edge-scoring model ("GNN-lite"): one round of message passing
over the candidate (fully-connected) graph, followed by an MLP classifier
that predicts P(edge exists) for every node pair. Trained on many
synthetic feeders (phase10a), evaluated on real SimBench feeders.

Why "message passing" and not just raw correlation:
For each node, we aggregate (average) the raw pairwise-correlation
profile of its most-correlated neighbors, producing a node embedding
that reflects not just "how correlated is A with B" but "how similar
is A's whole neighborhood-correlation PROFILE to B's" -- closer in
spirit to how a GNN layer propagates structural information beyond
single pairwise statistics.
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from phase10a_synthetic_generator import generate_training_set
from phase1_2_baseline import correlation_mst, evaluate as evaluate_graph


# ---------------------------------------------------------------------------
# Feature engineering (message-passing-lite)
# ---------------------------------------------------------------------------

def node_embeddings_from_correlation(corr, top_k=5):
    """
    One 'message passing' round: each node's embedding = average
    correlation profile of its top_k most-correlated other nodes
    (weighted by correlation). This lets a pair (i, j) be scored not
    just by corr(i,j) but by how similar their broader correlation
    neighborhoods are.
    """
    nodes = corr.columns.tolist()
    n = len(nodes)
    corr_vals = corr.values
    embeddings = np.zeros((n, n))
    for i in range(n):
        row = corr_vals[i].copy()
        row[i] = -np.inf
        top_idx = np.argsort(row)[-top_k:]
        weights = np.clip(corr_vals[i, top_idx], 0, None)
        if weights.sum() == 0:
            embeddings[i] = corr_vals[i]
        else:
            embeddings[i] = np.average(corr_vals[top_idx], axis=0, weights=weights)
    return embeddings, nodes


def pairwise_features(voltage_df):
    """Build a feature vector for every node pair (i, j)."""
    from sklearn.covariance import GraphicalLasso
    from sklearn.feature_selection import mutual_info_regression

    corr = voltage_df.corr(method="pearson").fillna(0.0)
    embeddings, nodes = node_embeddings_from_correlation(corr)
    variances = voltage_df.var().fillna(0.0)
    n = len(nodes)

    # partial correlation proxy: precision matrix from a light Graphical Lasso
    X_std = ((voltage_df - voltage_df.mean()) / (voltage_df.std() + 1e-9)).values
    try:
        gl = GraphicalLasso(alpha=0.05, max_iter=200)
        gl.fit(X_std)
        precision = np.abs(gl.precision_)
        precision = precision / (precision.max() + 1e-9)
    except Exception:
        precision = np.zeros((n, n))

    # mutual information matrix (k-NN estimator), computed once per node
    mi_matrix = np.zeros((n, n))
    Xv = voltage_df.values
    for i in range(n):
        mi = mutual_info_regression(Xv, Xv[:, i], n_neighbors=3, random_state=0)
        mi_matrix[i] = mi
    mi_matrix = mi_matrix / (mi_matrix.max() + 1e-9)

    rows = []
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            emb_sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-9
            )
            feat = [
                corr.iloc[i, j],                                   # raw pairwise correlation
                emb_sim,                                           # message-passed embedding similarity
                abs(variances.iloc[i] - variances.iloc[j]),        # variance difference
                variances.iloc[i] + variances.iloc[j],             # combined variance
                precision[i, j],                                   # partial correlation proxy
                (mi_matrix[i, j] + mi_matrix[j, i]) / 2,            # mutual information (symmetrized)
            ]
            rows.append(feat)
            pairs.append((nodes[i], nodes[j]))

    return np.array(rows), pairs


def build_training_data(dataset):
    X_list, y_list = [], []
    for voltage_df, G in dataset:
        if voltage_df.shape[0] < 20:
            continue
        feats, pairs = pairwise_features(voltage_df)
        true_edges = {frozenset(e) for e in G.edges()}
        labels = np.array([1 if frozenset(p) in true_edges else 0 for p in pairs])
        X_list.append(feats)
        y_list.append(labels)
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


# ---------------------------------------------------------------------------
# Prediction -> topology
# ---------------------------------------------------------------------------

def predict_topology(voltage_df, model, scaler):
    feats, pairs = pairwise_features(voltage_df)
    feats_scaled = scaler.transform(feats)
    scores = model.predict_proba(feats_scaled)[:, 1]

    G_full = nx.Graph()
    G_full.add_nodes_from(voltage_df.columns)
    for (u, v), score in zip(pairs, scores):
        G_full.add_edge(u, v, weight=score)

    mst = nx.maximum_spanning_tree(G_full, weight="weight")
    return mst


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Generating synthetic training set (80 random radial feeders) ===")
    dataset = generate_training_set(n_networks=80, n_steps=300, seed=42, size_range=(10, 50))
    print(f"Generated {len(dataset)} networks, sizes: "
          f"{[G.number_of_nodes() for _, G in dataset[:10]]} ...")

    print("\n=== Building training features ===")
    X, y = build_training_data(dataset)
    print(f"Training examples: {len(y)} (positive edges: {y.sum()}, "
          f"negative pairs: {(y==0).sum()})")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n=== Training MLP edge classifier ===")
    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500,
                           class_weight=None, random_state=42) \
        if "class_weight" in MLPClassifier().get_params() else \
        MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    model.fit(X_scaled, y)
    print(f"Training accuracy: {model.score(X_scaled, y):.3f}")

    # --- Evaluate on real SimBench feeders (held out, never seen in training) ---
    print("\n=== Evaluating on real SimBench feeders (held out) ===")
    import simbench as sb
    from phase5_simbench import ground_truth_graph, simulate_timeseries_simbench, add_realistic_meter_noise

    feeders = ["1-LV-rural1--0-sw", "1-LV-semiurb4--0-sw", "1-LV-urban6--0-sw"]
    rows = []
    for code in feeders:
        net = sb.get_simbench_net(code)
        true_graph = ground_truth_graph(net)
        week_start = 96 * 7 * 26
        voltage_df = simulate_timeseries_simbench(net, n_steps=672, start=week_start)
        noisy_df = add_realistic_meter_noise(voltage_df)

        # baseline for comparison
        mst_base, _ = correlation_mst(noisy_df)
        m_base = evaluate_graph(mst_base, true_graph, voltage_df.columns)

        # GNN-lite
        mst_gnn = predict_topology(noisy_df, model, scaler)
        m_gnn = evaluate_graph(mst_gnn, true_graph, voltage_df.columns)

        print(f"\n{code} (buses={true_graph.number_of_nodes()}):")
        print(f"  Baseline (correlation+MST) : recon_acc={m_base['reconstruction_accuracy']:.3f}")
        print(f"  GNN-lite (learned scorer)  : recon_acc={m_gnn['reconstruction_accuracy']:.3f}")

        rows.append({"feeder": code, "n_buses": true_graph.number_of_nodes(),
                      "baseline_recon_acc": m_base["reconstruction_accuracy"],
                      "gnn_lite_recon_acc": m_gnn["reconstruction_accuracy"]})

    results = pd.DataFrame(rows)
    results.to_csv("results/phase10_gnn_lite_results.csv", index=False)
    print("\nSaved results/phase10_gnn_lite_results.csv")


if __name__ == "__main__":
    main()