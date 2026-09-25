"""Simulated documentation errors, following the taxonomy in the literature.

Three kinds, which differ markedly in how observable they are:

  open_switch  A line recorded as closed is actually open. The commonest
               case in practice, and the easiest: it removes an edge
               without inventing a false one.
  mixed        Some real edges missing, some false ones present.
  reroute      A bus hangs off a neighbour of its true parent. The hardest
               case: two candidates that are electrically almost identical.
"""

import numpy as np
import networkx as nx


def open_switch(truth, n, seed=3):
    rng = np.random.default_rng(seed)
    G = truth.copy()
    edges = list(G.edges())
    for i in rng.choice(len(edges), size=n, replace=False):
        G.remove_edge(*edges[i])
    return G


def mixed(truth, n, seed=2):
    rng = np.random.default_rng(seed)
    G = truth.copy()
    edges = list(G.edges())
    for i in rng.choice(len(edges), size=n, replace=False):
        G.remove_edge(*edges[i])
    nodes = list(G.nodes())
    added = 0
    while added < n:
        a, b = rng.choice(nodes, size=2, replace=False)
        if not truth.has_edge(a, b) and not G.has_edge(a, b):
            G.add_edge(a, b)
            added += 1
    return G


def reroute(truth, n, seed=3):
    rng = np.random.default_rng(seed)
    G = truth.copy()
    root = list(truth.nodes())[0]
    T = nx.bfs_tree(truth, root)
    candidates = [x for x in truth.nodes() if list(T.predecessors(x))]
    rng.shuffle(candidates)
    done = 0
    for x in candidates:
        if done >= n:
            break
        p = list(T.predecessors(x))[0]
        near = [y for y in nx.single_source_shortest_path_length(truth, p, cutoff=2)
                if y not in (x, p) and not truth.has_edge(x, y)]
        if not near:
            continue
        G.remove_edge(x, p)
        G.add_edge(x, rng.choice(near))
        done += 1
    return G


def possible_connections(truth, documented, decoy_factor=1.0, recall=1.0, seed=5):
    """The set of physically existing lines whose state is uncertain.

    Contains the genuine discrepancies -- optionally only a `recall`
    fraction of them, to model an incomplete switch register -- plus
    `decoy_factor` times as many correctly documented edges, modelling
    switches the operator lists but that happen to be recorded correctly.
    """
    rng = np.random.default_rng(seed)
    te = {frozenset(e) for e in truth.edges()}
    de = {frozenset(e) for e in documented.edges()}
    real = list((de - te) | (te - de))

    k = int(round(len(real) * recall))
    kept = set(real[i] for i in rng.choice(len(real), size=k, replace=False)) if k else set()

    correct = list(de & te)
    n_decoy = int(round(len(real) * decoy_factor))
    decoys = set()
    if correct and n_decoy:
        for i in rng.choice(len(correct), size=min(n_decoy, len(correct)), replace=False):
            decoys.add(correct[i])
    return kept | decoys
