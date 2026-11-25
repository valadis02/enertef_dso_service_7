"""EXTENSION, FM roadmap step 1: a TRAINABLE graph neural network in step 2 (use case 1). Original files untouched.

Graph per case: nodes = buses, edges = candidate lines. Only torch is needed (no torch_geometric).
  node input  per bus, from the measurements: std of V, std of (V - V_root), std of P and Q, mean of P and Q,
              degree in the candidate graph, is-root flag                                   (8 numbers)
  edge input  the 6 features of the original method (log_dist, dist_rank, kruskal_in, dJ (signed log), log_corr, doc)
  layers      L rounds of edge-aware message passing:
                 m_ij  = MLP([h_i, h_j, e_ij])            message along every line, both directions
                 h_i  <- h_i + mean_j m_ij                bus update (residual)
                 e_ij <- e_ij + MLP([h_i + h_j, |h_i - h_j|, e_ij])   line update (residual, symmetric in i, j)
  output      p_ij = sigmoid(MLP([e_ij, h_i + h_j, |h_i - h_j|]))
Training: leave-one-feeder-out as everywhere else, binary cross-entropy, Adam; SEEDS models averaged.
Then steps 3 and 4 of the original (Kruskal decode + physics refinement, gate GLO/GHI).

Columns written to res_<feeder>.csv:  gnnt_steps_1_3 (after decode)   gnnt (after decode + refinement)
Results: final_lv<SUFFIX>_gnn_torch

Environment variables: EPOCHS (default 150)  HIDDEN (64)  LAYERS (3)  LR (1e-3)  SEEDS (3)  THREADS (2)  BATCH (8)
Progress of the training: train_log.txt in the result folder (one line per fold and seed).

usage:  python exp_gnn_torch_lv_ext.py smoke            quick self-test on synthetic graphs (seconds, no data needed)
        python exp_gnn_torch_lv_ext.py <feeder index>   (normally through run_methods_ext.py --method gnnt)
        python exp_gnn_torch_lv_ext.py summary
"""
import os
import sys

import numpy as np

EPOCHS = int(os.environ.get("EPOCHS", 150))
HIDDEN = int(os.environ.get("HIDDEN", 64))
LAYERS = int(os.environ.get("LAYERS", 3))
LR = float(os.environ.get("LR", 1e-3))
SEEDS = int(os.environ.get("SEEDS", 3))
THREADS = int(os.environ.get("THREADS", 2))
BATCH = int(os.environ.get("BATCH", 8))            # graphs merged into one update
EDGE_COLS = ["log_dist", "dist_rank", "kruskal_in", "dJ", "log_corr", "doc"]


def slog(a):
    a = np.asarray(a, dtype=float)
    return np.sign(a) * np.log10(1 + np.abs(a))


# ----------------------------------------------------------------------------- model (torch only)
def build_model(n_in, e_in):
    import torch
    import torch.nn as nn

    def mlp(i, o):
        return nn.Sequential(nn.Linear(i, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, o))

    class EdgeGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.n0, self.e0 = mlp(n_in, HIDDEN), mlp(e_in, HIDDEN)
            self.msg = nn.ModuleList([mlp(3 * HIDDEN, HIDDEN) for _ in range(LAYERS)])
            self.upd = nn.ModuleList([mlp(3 * HIDDEN, HIDDEN) for _ in range(LAYERS)])
            self.out = nn.Sequential(nn.Linear(3 * HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(0.1), nn.Linear(HIDDEN, 1))

        def forward(self, xn, xe, ei):
            """xn (N, n_in), xe (E, e_in), ei (2, E) long. Returns logits (E,)."""
            h, e = self.n0(xn), self.e0(xe)
            a, b = ei[0], ei[1]
            src = torch.cat([a, b]); dst = torch.cat([b, a])                     # both directions
            deg = torch.zeros(h.shape[0]).index_add_(0, dst, torch.ones(dst.shape[0])).clamp(min=1).unsqueeze(1)
            for msg, upd in zip(self.msg, self.upd):
                ee = torch.cat([e, e])
                m = msg(torch.cat([h[dst], h[src], ee], 1))
                h = h + torch.zeros_like(h).index_add_(0, dst, m) / deg
                e = e + upd(torch.cat([h[a] + h[b], (h[a] - h[b]).abs(), e], 1))
            return self.out(torch.cat([e, h[a] + h[b], (h[a] - h[b]).abs()], 1)).squeeze(1)

    return EdgeGNN()


def train_predict(train_graphs, test_graphs, seed):
    """graphs: list of dicts xn, xe (numpy), ei (2, E) int, y (E,) or None. Returns list of p arrays for test_graphs."""
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(THREADS)
    # standardise with training statistics
    XN = np.concatenate([g["xn"] for g in train_graphs]); XE = np.concatenate([g["xe"] for g in train_graphs])
    mn, sn = XN.mean(0), XN.std(0) + 1e-9; me, se = XE.mean(0), XE.std(0) + 1e-9
    T = lambda g: (torch.tensor((g["xn"] - mn) / sn, dtype=torch.float32), torch.tensor((g["xe"] - me) / se, dtype=torch.float32),
                   torch.tensor(g["ei"], dtype=torch.long))
    tr = [(T(g), torch.tensor(g["y"], dtype=torch.float32)) for g in train_graphs]

    def merge(items):
        """several graphs -> one disconnected graph (node indices shifted)."""
        xn, xe, ei, y, off = [], [], [], [], 0
        for (a, b, c), yy in items:
            xn.append(a); xe.append(b); ei.append(c + off); y.append(yy); off += a.shape[0]
        return torch.cat(xn), torch.cat(xe), torch.cat(ei, 1), torch.cat(y)
    model = build_model(XN.shape[1], XE.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    for ep in range(EPOCHS):
        model.train()
        order = rng.permutation(len(tr))
        for s0 in range(0, len(order), BATCH):
            xn, xe, ei, y = merge([tr[k] for k in order[s0:s0 + BATCH]])
            opt.zero_grad(); loss = lossf(model(xn, xe, ei), y); loss.backward(); opt.step()
    model.eval()
    out = []
    with torch.no_grad():
        for g in test_graphs:
            xn, xe, ei = T(g)
            out.append(torch.sigmoid(model(xn, xe, ei)).numpy())
    return out, float(loss)


# ----------------------------------------------------------------------------- graphs from the real data
def node_inputs(c, nodes, cand_edges):
    V, P, Q = c["V"], c["P"], c["Q"]
    idx = {n: k for k, n in enumerate(c["nodes"])}
    r = idx[c["root"]]
    deg = {n: 0 for n in nodes}
    for e in cand_edges:
        for n in e:
            deg[n] += 1
    rows = []
    for n in nodes:
        k = idx[n]
        rows.append([V[:, k].std(), (V[:, k] - V[:, r]).std(), P[:, k].std(), Q[:, k].std(),
                     P[:, k].mean(), Q[:, k].mean(), deg[n], float(n == c["root"])])
    A = np.array(rows, float)
    A[:, :6] = np.log10(np.abs(A[:, :6]) + 1e-9) * np.sign(A[:, :6] + 1e-30)       # the scales differ by decades
    return A


def make_graph(c, g):
    """g: the rows of one case (columns edge + EDGE_COLS + label). Edge order = row order of g."""
    edges = list(g.edge)
    nodes = list(c["nodes"]); pos = {n: k for k, n in enumerate(nodes)}
    ei = np.array([[pos[tuple(e)[0]] for e in edges], [pos[tuple(e)[1]] for e in edges]], int)
    xe = g[EDGE_COLS].to_numpy(float).copy(); xe[:, EDGE_COLS.index("dJ")] = slog(xe[:, EDGE_COLS.index("dJ")])
    return dict(xn=node_inputs(c, nodes, edges), xe=xe, ei=ei, y=g["label"].to_numpy(float))


# ----------------------------------------------------------------------------- smoke test
def smoke():
    import networkx as nx
    rng = np.random.default_rng(0)
    graphs = []
    for k in range(24):
        n = int(rng.integers(12, 30))
        tree = nx.random_labeled_tree(n, seed=int(rng.integers(1e9)))
        true = {frozenset(e) for e in tree.edges()}
        decoys = set()
        while len(decoys) < n // 2:
            a, b = rng.choice(n, 2, replace=False)
            if frozenset((a, b)) not in true:
                decoys.add(frozenset((int(a), int(b))))
        edges = list(true) + list(decoys)
        y = np.array([1.0] * len(true) + [0.0] * len(decoys))
        xe = rng.normal(0, 1, (len(edges), 6)) + 1.2 * y[:, None] * rng.uniform(0.3, 1, 6)
        xn = rng.normal(0, 1, (n, 8))
        ei = np.array([[tuple(e)[0] for e in edges], [tuple(e)[1] for e in edges]], int)
        graphs.append(dict(xn=xn, xe=xe, ei=ei, y=y))
    global EPOCHS
    EPOCHS = min(EPOCHS, 30)
    import time
    t = time.time()
    P, last = train_predict(graphs[:18], graphs[18:], seed=0)
    acc = np.mean(np.concatenate([(p > 0.5) == (g["y"] > 0.5) for p, g in zip(P, graphs[18:])]))
    import torch
    print(f"smoke test OK  torch {torch.__version__}  last training loss {last:.3f}  test accuracy {acc:.3f}  ({time.time() - t:.1f} s)")
    print("expected: accuracy clearly above 0.6 (the synthetic edge features are informative)")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "smoke":
    smoke(); sys.exit(0)

# ----------------------------------------------------------------------------- real run (imports the original code)
import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
_sys.argv = _sys.argv[:2]
import glob  # noqa: E402
import pickle  # noqa: E402
import pandas as pd  # noqa: E402
import exp_final_lv_ext as F  # noqa: E402
from exp_final_lv_ext import L, LF, METERS, SIMBENCH_FEEDERS, SB_WORK  # noqa: E402

OUT = F.OUT + "_gnn_torch"
os.makedirs(OUT, exist_ok=True)


def probs(C):
    path = f"{OUT}/probs.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    X, _ = LF.add_features(METERS)
    X = X.reset_index(drop=True)
    cases = list(X.groupby("case", sort=False))
    G = {cs: make_graph(C[cs], g) for cs, g in cases}
    feeder_of = {cs: g.feeder.iloc[0] for cs, g in cases}
    p = np.zeros(len(X))
    import time
    log = open(f"{OUT}/train_log.txt", "a"); t0 = time.time()
    log.write(f"start: {len(G)} graphs, EPOCHS={EPOCHS} SEEDS={SEEDS} BATCH={BATCH} THREADS={THREADS}\n"); log.flush()
    for f in X.feeder.unique():
        tr = [G[cs] for cs in G if feeder_of[cs] != f]
        te_cases = [cs for cs in G if feeder_of[cs] == f]
        acc = [np.zeros(len(G[cs]["y"])) for cs in te_cases]
        for s in range(SEEDS):
            P, last = train_predict(tr, [G[cs] for cs in te_cases], seed=s)
            acc = [a + q / SEEDS for a, q in zip(acc, P)]
            log.write(f"{time.time() - t0:7.0f} s  test feeder {f}  seed {s}  last loss {last:.3f}\n"); log.flush()
        for cs, a in zip(te_cases, acc):
            p[X.index[X.case == cs].values] = a
        print(f, "trained, last loss %.3f" % last, flush=True)
    out = (X[["case", "feeder", "edge"]].copy(), p)
    pickle.dump(out, open(path, "wb"))
    return out


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "summary":
        R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/res_*.csv"))])
        base = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{F.OUT}/res_*.csv"))])
        R = base.merge(R, on="case")
        print(len(R), "cases (percent correct lines)"); print((100 * R.drop(columns=["case"]).mean()).round(2).to_string())
        sys.exit(0)
    C = pickle.load(open(f"{SB_WORK}/lv_ctx_{METERS}_tdm{L.SUFM}.pkl", "rb"))
    Xe, p = probs(C)
    code = SIMBENCH_FEEDERS[int(arg)]
    rows = []
    for case, g in Xe[Xe.feeder == code].groupby("case"):
        ix = g.index.values; c = C[case]; edges = list(g.edge)
        Gd = L.decode(c["nodes"], c["fixed"], edges, p[ix])
        rows.append(dict(case=case, gnnt_steps_1_3=F.share(Gd, c["tg"]),
                         gnnt=F.share(F.refine(Gd, set(edges), c, p=dict(zip(edges, p[ix]))), c["tg"])))
    pd.DataFrame(rows).to_csv(f"{OUT}/res_{arg}.csv", index=False)
    print(code, "done", flush=True)
