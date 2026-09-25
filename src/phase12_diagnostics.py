"""
DSO Service 7 - Phase 12
Diagnostics: WHY does the full GNN plateau around 20-30% instead of
going higher? Four checks:

  1. Where do errors concentrate spatially? (near branch points, like
     the earlier raw-correlation failure mode, or different?)
  2. Score distribution: does the model separate true vs false edges
     cleanly, or is there heavy overlap (= genuine information limit)?
  3. Train/test domain gap: do synthetic (LinDistFlow) and real
     (SimBench) correlation matrices look statistically similar, or is
     the model trained on a different distribution than it's tested on?
  4. Degree/distance analysis: are errors concentrated on high-degree
     or "deep" (far from slack) nodes?
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["figure.dpi"] = 120

from phase11_full_gnn import TopologyGNN, build_graph_tensors, DEVICE
from phase10a_synthetic_generator import generate_training_set
from phase1_2_baseline import correlation_mst, evaluate as evaluate_graph
from phase5_simbench import ground_truth_graph, simulate_timeseries_simbench, add_realistic_meter_noise
import simbench as sb


def load_trained_model():
    model = TopologyGNN().to(DEVICE)
    model.load_state_dict(torch.load("results/phase11_gnn_model.pt"))
    model.eval()
    return model


def get_scores_and_truth(voltage_df, true_graph, model):
    with torch.no_grad():
        node_feat, corr, nodes = build_graph_tensors(voltage_df)
        var = torch.tensor(voltage_df.var().fillna(0.0).values, dtype=torch.float32)
        h = model(node_feat, corr)
        logits, i_idx, j_idx = model.score_edges(h, corr, var)
        scores = torch.sigmoid(logits).numpy()

    true_edges = {frozenset(e) for e in true_graph.edges()}
    pairs = [(nodes[a], nodes[b]) for a, b in zip(i_idx.tolist(), j_idx.tolist())]
    labels = np.array([1 if frozenset(p) in true_edges else 0 for p in pairs])
    return scores, labels, pairs, nodes


def check_score_separation(scores, labels, feeder_name):
    """Check 2: how well-separated are true vs false edge scores?"""
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    print(f"\n[{feeder_name}] Score distribution:")
    print(f"  True edges   : mean={pos_scores.mean():.3f} std={pos_scores.std():.3f} "
          f"min={pos_scores.min():.3f} max={pos_scores.max():.3f}")
    print(f"  False pairs  : mean={neg_scores.mean():.3f} std={neg_scores.std():.3f} "
          f"min={neg_scores.min():.3f} max={neg_scores.max():.3f}")

    # what fraction of false pairs score HIGHER than the median true edge?
    median_true = np.median(pos_scores)
    frac_false_above = (neg_scores > median_true).mean()
    print(f"  Fraction of FALSE pairs scoring above the median TRUE edge: {frac_false_above:.1%}")
    print(f"  (if this is high, there's heavy overlap = model can't cleanly separate)")

    return pos_scores, neg_scores


def check_error_location(mst_pred, true_graph, nodes):
    """Check 1 & 4: where (structurally) do errors happen?"""
    true_edges = {frozenset(e) for e in true_graph.edges()}
    pred_edges = {frozenset(e) for e in mst_pred.edges()}
    missed = true_edges - pred_edges
    spurious = pred_edges - true_edges

    # distance from an arbitrary root (bus with... just use first node as proxy root)
    root = list(true_graph.nodes())[0]
    try:
        dist_from_root = nx.single_source_shortest_path_length(true_graph, root)
    except Exception:
        dist_from_root = {n: 0 for n in true_graph.nodes()}
    degree = dict(true_graph.degree())

    def edge_stats(edges, label):
        if not edges:
            print(f"  {label}: none")
            return
        depths = [max(dist_from_root.get(list(e)[0], 0), dist_from_root.get(list(e)[1], 0)) for e in edges]
        degs = [max(degree.get(list(e)[0], 1), degree.get(list(e)[1], 1)) for e in edges]
        print(f"  {label}: n={len(edges)}, avg depth from root={np.mean(depths):.1f}, "
              f"avg max endpoint degree={np.mean(degs):.1f}")

    print(f"\nMissed/spurious edge structural profile:")
    all_depths = [dist_from_root.get(n, 0) for n in true_graph.nodes()]
    print(f"  (network avg depth={np.mean(all_depths):.1f}, avg degree={np.mean(list(degree.values())):.1f})")
    edge_stats(missed, "MISSED true edges")
    edge_stats(spurious, "SPURIOUS predicted edges")


def check_domain_gap():
    """Check 3: do synthetic and real correlation matrices look similar?"""
    print("\n=== Domain gap check: synthetic vs real correlation distributions ===")
    synth_dataset = generate_training_set(n_networks=10, n_steps=300, seed=999, size_range=(14, 14))
    synth_corrs = []
    for voltage_df, G in synth_dataset:
        c = voltage_df.corr().values
        synth_corrs.extend(c[np.triu_indices_from(c, k=1)])

    net = sb.get_simbench_net("1-LV-rural1--0-sw")
    week_start = 96 * 7 * 26
    voltage_df = simulate_timeseries_simbench(net, n_steps=672, start=week_start)
    noisy = add_realistic_meter_noise(voltage_df)
    c_real = noisy.corr().values
    real_corrs = c_real[np.triu_indices_from(c_real, k=1)]

    synth_corrs = np.array(synth_corrs)
    real_corrs = np.array(real_corrs)
    print(f"  Synthetic (LinDistFlow) pairwise correlations: mean={synth_corrs.mean():.3f} "
          f"std={synth_corrs.std():.3f} range=[{synth_corrs.min():.3f}, {synth_corrs.max():.3f}]")
    print(f"  Real (SimBench, noisy)  pairwise correlations: mean={real_corrs.mean():.3f} "
          f"std={real_corrs.std():.3f} range=[{real_corrs.min():.3f}, {real_corrs.max():.3f}]")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(synth_corrs, bins=40, alpha=0.5, label="Synthetic (training)", density=True)
    ax.hist(real_corrs, bins=40, alpha=0.5, label="Real SimBench (test)", density=True)
    ax.set_xlabel("Pairwise voltage correlation")
    ax.set_ylabel("Density")
    ax.set_title("Train vs test correlation distribution (domain gap check)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/phase12_domain_gap.png")
    print("  Saved results/phase12_domain_gap.png")


def main():
    model = load_trained_model()

    feeders = ["1-LV-rural1--0-sw", "1-LV-semiurb4--0-sw", "1-LV-urban6--0-sw"]
    for code in feeders:
        net = sb.get_simbench_net(code)
        true_graph = ground_truth_graph(net)
        week_start = 96 * 7 * 26
        voltage_df = simulate_timeseries_simbench(net, n_steps=672, start=week_start)
        noisy_df = add_realistic_meter_noise(voltage_df)

        scores, labels, pairs, nodes = get_scores_and_truth(noisy_df, true_graph, model)
        check_score_separation(scores, labels, code)

        # build predicted MST for structural error analysis
        G_full = nx.Graph()
        G_full.add_nodes_from(nodes)
        for (u, v), s in zip(pairs, scores):
            G_full.add_edge(u, v, weight=float(s))
        mst_pred = nx.maximum_spanning_tree(G_full, weight="weight")
        check_error_location(mst_pred, true_graph, nodes)

    check_domain_gap()


if __name__ == "__main__":
    main()
