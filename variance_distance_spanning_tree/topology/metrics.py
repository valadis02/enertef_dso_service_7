"""Evaluation metrics required by D2.2: reconstruction accuracy plus
edge-identification precision and recall."""

import networkx as nx


def evaluate(predicted, truth, nodes=None):
    """Compare a predicted topology against ground truth.

    Returns reconstruction accuracy (share of true edges recovered) along
    with edge precision and recall, which D2.2 lists as required metrics.
    """
    true_edges = {frozenset(e) for e in truth.edges()}
    pred_edges = {frozenset(e) for e in predicted.edges()}
    correct = true_edges & pred_edges

    precision = len(correct) / len(pred_edges) if pred_edges else 0.0
    recall = len(correct) / len(true_edges) if true_edges else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "reconstruction_accuracy": recall,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_true_edges": len(true_edges),
        "n_pred_edges": len(pred_edges),
        "n_correct_edges": len(correct),
    }
