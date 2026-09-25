"""
DSO Service 7 - Visualization
Plots for Phase 3 robustness results + a side-by-side view of the
true feeder topology vs. the predicted (correlation+MST) topology.

Run this AFTER phase1_2_baseline.py and phase3_robustness.py so that
results/*.csv exist.
"""

import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import pandas as pd

from phase1_2_baseline import load_feeder, ground_truth_graph, simulate_timeseries, correlation_mst

matplotlib.rcParams["figure.dpi"] = 120


def plot_noise_curve():
    df = pd.read_csv("results/phase3_noise_results.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["noise_sigma_pu"], df["precision"], marker="o", label="Precision")
    ax.plot(df["noise_sigma_pu"], df["recall"], marker="s", label="Recall", linestyle="--")
    ax.set_xlabel("Gaussian noise std (p.u.)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Topology reconstruction accuracy vs. measurement noise")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/plot_noise_robustness.png")
    print("Saved results/plot_noise_robustness.png")


def plot_missing_data_curve():
    df = pd.read_csv("results/phase3_missing_data_results.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["missing_fraction"] * 100, df["reconstruction_accuracy"], marker="o", color="green")
    ax.set_xlabel("Missing measurements (%)")
    ax.set_ylabel("Reconstruction accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Topology reconstruction accuracy vs. missing data")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/plot_missing_data_robustness.png")
    print("Saved results/plot_missing_data_robustness.png")


def plot_topology_comparison():
    net = load_feeder()
    true_graph = ground_truth_graph(net)
    voltage_df, _ = simulate_timeseries(net, n_steps=500)
    mst, _ = correlation_mst(voltage_df)

    pos = nx.kamada_kawai_layout(true_graph)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    nx.draw(true_graph, pos, ax=axes[0], with_labels=True, node_size=300,
            node_color="#4C72B0", font_size=7, font_color="white", edge_color="#333")
    axes[0].set_title("Ground truth topology (IEEE 33-bus)")

    nx.draw(mst, pos, ax=axes[1], with_labels=True, node_size=300,
            node_color="#DD8452", font_size=7, font_color="white", edge_color="#333")
    axes[1].set_title("Predicted topology (correlation + MST)")

    fig.tight_layout()
    fig.savefig("results/plot_topology_comparison.png")
    print("Saved results/plot_topology_comparison.png")


def plot_topology_comparison_noisy(noise_sigma=0.005, seed=999):
    """
    Same side-by-side comparison, but the prediction is built from noisy
    voltage measurements. Correct edges are drawn in gray, edges the model
    predicted that are WRONG (not in ground truth) in red (solid), and
    ground-truth edges the model MISSED are drawn in red (dashed) on the
    right panel so you can see exactly where reconstruction breaks down.
    """
    import numpy as np

    net = load_feeder()
    true_graph = ground_truth_graph(net)
    voltage_df, _ = simulate_timeseries(net, n_steps=500)

    rng = np.random.default_rng(seed)
    noisy_voltage_df = voltage_df + rng.normal(0, noise_sigma, size=voltage_df.shape)
    mst, _ = correlation_mst(noisy_voltage_df)

    true_edges = {frozenset(e) for e in true_graph.edges()}
    pred_edges = {frozenset(e) for e in mst.edges()}

    correct_edges = [tuple(e) for e in (true_edges & pred_edges)]
    false_positive_edges = [tuple(e) for e in (pred_edges - true_edges)]   # predicted but wrong
    missed_edges = [tuple(e) for e in (true_edges - pred_edges)]           # true but not predicted

    pos = nx.kamada_kawai_layout(true_graph)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # left panel: ground truth (highlight which edges get missed on the right)
    nx.draw_networkx_nodes(true_graph, pos, ax=axes[0], node_size=300, node_color="#4C72B0")
    nx.draw_networkx_labels(true_graph, pos, ax=axes[0], font_size=7, font_color="white")
    nx.draw_networkx_edges(true_graph, pos, ax=axes[0], edgelist=correct_edges, edge_color="#333")
    nx.draw_networkx_edges(true_graph, pos, ax=axes[0], edgelist=missed_edges,
                            edge_color="red", style="dashed", width=2)
    axes[0].set_title("Ground truth\n(red dashed = edges the model missed)")
    axes[0].axis("off")

    # right panel: predicted graph, wrong edges highlighted
    all_pred_nodes = list(voltage_df.columns)
    axes[1].scatter(*zip(*[pos[n] for n in all_pred_nodes]), s=300, c="#DD8452", zorder=2)
    for n in all_pred_nodes:
        x, y = pos[n]
        axes[1].text(x, y, str(n), ha="center", va="center", fontsize=7, color="white", zorder=3)
    for u, v in correct_edges:
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        axes[1].plot(x, y, color="#333", zorder=1)
    for u, v in false_positive_edges:
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        axes[1].plot(x, y, color="red", linewidth=2, zorder=1)
    axes[1].set_title(f"Predicted with noise (sigma={noise_sigma} p.u.)\n(red solid = wrong/spurious edges)")
    axes[1].axis("off")
    axes[1].set_aspect("equal")

    n_correct = len(correct_edges)
    n_total_true = len(true_edges)
    fig.suptitle(f"Correctly recovered: {n_correct}/{n_total_true} edges "
                 f"({n_correct/n_total_true:.0%})", fontsize=11, y=0.02)

    fig.tight_layout()
    fig.savefig(f"results/plot_topology_comparison_noisy_sigma{noise_sigma}.png")
    print(f"Saved results/plot_topology_comparison_noisy_sigma{noise_sigma}.png")


if __name__ == "__main__":
    plot_noise_curve()
    plot_missing_data_curve()
    plot_topology_comparison()
    plot_topology_comparison_noisy(noise_sigma=0.001)
    plot_topology_comparison_noisy(noise_sigma=0.005)

    