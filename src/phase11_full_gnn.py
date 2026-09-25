"""
DSO Service 7 - Phase 11
Full multi-layer Graph Neural Network (PyTorch) for topology
identification -- the "real" GNN, as opposed to the phase10b "GNN-lite"
(single message-passing round + sklearn MLP).

Architecture:
  - For each candidate (fully-connected) graph, build an attention matrix
    from the pairwise Pearson correlation of voltage measurements
    (softmax-normalized per row) -- this plays the role of the "adjacency"
    a plain GCN would need, since the TRUE adjacency is exactly what we
    are trying to identify.
  - 3 rounds of graph-attention message passing (GATv2-style: attention
    recomputed at every layer from the CURRENT node embeddings, not fixed
    to the raw correlation after layer 1) update node embeddings.
  - An edge MLP scores every candidate pair from their final embeddings
    PLUS the raw pairwise statistics (correlation, variance diff.).
  - Trained end-to-end (backprop through the message-passing layers) with
    binary cross-entropy against the true adjacency, on many synthetic
    radial feeders (phase10a generator). Evaluated on held-out real
    SimBench feeders.
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase10a_synthetic_generator import generate_training_set
from phase1_2_baseline import correlation_mst, evaluate as evaluate_graph

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Graph featurization
# ---------------------------------------------------------------------------

def build_graph_tensors(voltage_df):
    """
    Turn a voltage time-series DataFrame into the tensors the GNN needs:
      - node_feat: (n, 3) initial per-node features (mean, std, skew-ish proxy)
      - corr: (n, n) Pearson correlation (used to seed attention)
    """
    nodes = voltage_df.columns.tolist()
    n = len(nodes)
    corr = voltage_df.corr(method="pearson").fillna(0.0).values
    np.fill_diagonal(corr, 0.0)

    mean = voltage_df.mean().values
    std = voltage_df.std().fillna(0.0).values
    node_feat = np.stack([mean, std, std ** 2], axis=1)  # (n, 3)
    # standardize node features
    node_feat = (node_feat - node_feat.mean(axis=0)) / (node_feat.std(axis=0) + 1e-9)

    return (
        torch.tensor(node_feat, dtype=torch.float32),
        torch.tensor(corr, dtype=torch.float32),
        nodes,
    )


# ---------------------------------------------------------------------------
# GNN model
# ---------------------------------------------------------------------------

class AttentionLayer(nn.Module):
    """One round of attention-weighted message passing. Attention logits
    combine a learned pairwise compatibility score with the (fixed) raw
    correlation prior, so the model can learn to deviate from raw
    correlation where useful, while still using it as a strong prior."""

    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.msg = nn.Linear(dim, dim)
        self.corr_gate = nn.Parameter(torch.tensor(1.0))
        self.out = nn.Linear(dim * 2, dim)

    def forward(self, h, corr):
        Q = self.q(h)
        K = self.k(h)
        learned_logits = Q @ K.T / (h.shape[1] ** 0.5)
        logits = learned_logits + self.corr_gate * corr
        n = h.shape[0]
        logits = logits.masked_fill(torch.eye(n, dtype=torch.bool), float("-inf"))
        attn = F.softmax(logits, dim=1)
        messages = attn @ self.msg(h)
        h_new = F.relu(self.out(torch.cat([h, messages], dim=1)))
        return h_new


class TopologyGNN(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=16, n_layers=3, edge_mlp_hidden=(32,)):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([AttentionLayer(hidden_dim) for _ in range(n_layers)])

        mlp_layers = []
        in_features = hidden_dim * 2 + 2
        for h in edge_mlp_hidden:
            mlp_layers.append(nn.Linear(in_features, h))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.1))
            in_features = h
        mlp_layers.append(nn.Linear(in_features, 1))
        self.edge_mlp = nn.Sequential(*mlp_layers)

    def forward(self, node_feat, corr):
        h = F.relu(self.input_proj(node_feat))
        for layer in self.layers:
            h = layer(h, corr)
        return h

    def score_edges(self, h, corr, var):
        n = h.shape[0]
        i_idx, j_idx = torch.triu_indices(n, n, offset=1)
        hi, hj = h[i_idx], h[j_idx]
        corr_ij = corr[i_idx, j_idx].unsqueeze(1)
        var_diff = (var[i_idx] - var[j_idx]).abs().unsqueeze(1)
        edge_input = torch.cat([hi, hj, corr_ij, var_diff], dim=1)
        logits = self.edge_mlp(edge_input).squeeze(1)
        return logits, i_idx, j_idx


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_gnn(dataset, n_epochs=15, lr=1e-3, neg_sample_ratio=5, seed=0,
              hidden_dim=16, n_layers=3, edge_mlp_hidden=(32,), weight_decay=0.0):
    torch.manual_seed(seed)
    model = TopologyGNN(hidden_dim=hidden_dim, n_layers=n_layers,
                         edge_mlp_hidden=edge_mlp_hidden).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    graphs = []
    for voltage_df, G in dataset:
        if voltage_df.shape[0] < 20:
            continue
        node_feat, corr, nodes = build_graph_tensors(voltage_df)
        var = torch.tensor(voltage_df.var().fillna(0.0).values, dtype=torch.float32)
        true_edges = {frozenset(e) for e in G.edges()}
        n = len(nodes)
        i_idx, j_idx = torch.triu_indices(n, n, offset=1)
        labels = torch.tensor(
            [1.0 if frozenset((nodes[a], nodes[b])) in true_edges else 0.0
             for a, b in zip(i_idx.tolist(), j_idx.tolist())],
            dtype=torch.float32,
        )
        graphs.append((node_feat, corr, var, labels))

    print(f"Training on {len(graphs)} graphs for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for node_feat, corr, var, labels in graphs:
            optimizer.zero_grad()
            h = model(node_feat, corr)
            logits, _, _ = model.score_edges(h, corr, var)

            pos_mask = labels == 1
            neg_mask = labels == 0
            n_pos = pos_mask.sum().item()
            n_neg_sample = min(int(n_pos * neg_sample_ratio), neg_mask.sum().item())
            neg_idx = torch.where(neg_mask)[0]
            if n_neg_sample > 0 and len(neg_idx) > 0:
                sampled_neg = neg_idx[torch.randperm(len(neg_idx))[:n_neg_sample]]
                pos_idx = torch.where(pos_mask)[0]
                sel = torch.cat([pos_idx, sampled_neg])
            else:
                sel = torch.arange(len(labels))

            loss = F.binary_cross_entropy_with_logits(logits[sel], labels[sel])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{n_epochs}  avg loss={total_loss/len(graphs):.4f}")

    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_topology_gnn(voltage_df, model):
    model.eval()
    with torch.no_grad():
        node_feat, corr, nodes = build_graph_tensors(voltage_df)
        var = torch.tensor(voltage_df.var().fillna(0.0).values, dtype=torch.float32)
        h = model(node_feat, corr)
        logits, i_idx, j_idx = model.score_edges(h, corr, var)
        scores = torch.sigmoid(logits).numpy()

    G_full = nx.Graph()
    G_full.add_nodes_from(nodes)
    for a, b, s in zip(i_idx.tolist(), j_idx.tolist(), scores):
        G_full.add_edge(nodes[a], nodes[b], weight=float(s))
    mst = nx.maximum_spanning_tree(G_full, weight="weight")
    return mst


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Generating synthetic training set (100 random radial feeders) ===")
    dataset = generate_training_set(n_networks=100, n_steps=300, seed=42, size_range=(10, 50))

    model = train_gnn(dataset, n_epochs=15)

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

        mst_base, _ = correlation_mst(noisy_df)
        m_base = evaluate_graph(mst_base, true_graph, voltage_df.columns)

        mst_gnn = predict_topology_gnn(noisy_df, model)
        m_gnn = evaluate_graph(mst_gnn, true_graph, voltage_df.columns)

        print(f"\n{code} (buses={true_graph.number_of_nodes()}):")
        print(f"  Baseline (correlation+MST)  : recon_acc={m_base['reconstruction_accuracy']:.3f}")
        print(f"  Full GNN (multi-layer, PyTorch): recon_acc={m_gnn['reconstruction_accuracy']:.3f}")

        rows.append({"feeder": code, "n_buses": true_graph.number_of_nodes(),
                      "baseline_recon_acc": m_base["reconstruction_accuracy"],
                      "full_gnn_recon_acc": m_gnn["reconstruction_accuracy"]})

    results = pd.DataFrame(rows)
    results.to_csv("results/phase11_full_gnn_results.csv", index=False)
    torch.save(model.state_dict(), "results/phase11_gnn_model.pt")
    print("\nSaved results/phase11_full_gnn_results.csv, results/phase11_gnn_model.pt")


if __name__ == "__main__":
    main()