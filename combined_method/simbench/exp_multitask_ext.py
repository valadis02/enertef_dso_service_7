"""FM roadmap steps 2-3: ONE graph format and ONE graph neural network for use case 1 (topology) and use case 2
(customer-feeder assignment). Original files untouched.

Unified graph (step 2). Every case is a graph with typed nodes and typed edges:
  node types  0 = bus (UC1)   1 = customer (UC2)   2 = feeder head, a virtual node (UC2)
  node input  the same 8 numbers for every node, from its own measurements:
              std V, std (V - V_substation), std P, std Q, mean P, mean Q (injection sign), degree, is-root
              (+ one-hot node type). A feeder node uses its head measurement as P; missing quantities are 0.
  edge types  0 = candidate line between two buses             (UC1, labelled: line exists)
              1 = customer -> feeder node, one per feeder        (UC2, labelled: customer is on this feeder)
              2 = customer - customer, 5 most correlated voltages (context only, no label)
  edge input  type 0: the 6 features of the original UC1 method; type 1: the 4 features of the original UC2 method;
              type 2: the voltage correlation. Padded to 6 numbers; every edge type has its own input encoder.
Model (step 3). Shared message passing (as exp_gnn_torch_lv_ext.py) over ALL edges, one output head per task.
  Loss = mean over the tasks present in a batch of the binary cross-entropy of that task's labelled edges.
Decoding as in the original methods:
  UC1  Kruskal on p -> physics refinement (gate GLO/GHI)          columns uc1_steps_1_3, uc1
  UC2  each customer takes the feeder with the highest p -> power-balance refinement (every customer, as run_all.py)
                                                                   columns uc2_steps_1_3, uc2
TASKS = uc1 | uc2 | both    (single-task controls and the multi-task model, same code)
Training leave-one-feeder-out: the test feeder is unseen for BOTH tasks.
Needs: lv_doc80 features of UC1 (run_all_ext.py --stage lv_uc1_80) and the UC2 cache (run_methods_ext.py --method uc2).

Environment variables: QUICK (1 = 40 epochs, 1 seed, folders *_quick)  TASKS (both)  EPOCHS (150)  HIDDEN (64)  LAYERS (3)  LR (1e-3)  SEEDS (3)  THREADS (2)  BATCH (8)
usage:  python exp_multitask_ext.py smoke            self-test on synthetic graphs of both tasks (seconds)
        python exp_multitask_ext.py <feeder index>   (normally:  run_methods_ext.py --method mt --set TASKS=both)
        python exp_multitask_ext.py summary
Results: <CM_WORK>/lv_doc80/simbench/multitask_<TASKS>/ (res_uc1_<i>.csv, res_uc2_<i>.csv, train_log.txt)
"""
import os
import sys
import time

import numpy as np

TASKS = os.environ.get("TASKS", "both")
QUICK = os.environ.get("QUICK") == "1"                 # feasibility check: 40 epochs, 1 seed, own result folder
EPOCHS = int(os.environ.get("EPOCHS", 40 if QUICK else 150))
HIDDEN = int(os.environ.get("HIDDEN", 64))
LAYERS = int(os.environ.get("LAYERS", 3))
LR = float(os.environ.get("LR", 1e-3))
SEEDS = int(os.environ.get("SEEDS", 1 if QUICK else 3))
THREADS = int(os.environ.get("THREADS", 2))
BATCH = int(os.environ.get("BATCH", 8))
NE, N_NODE_TYPES, N_EDGE_TYPES = 6, 3, 3
UC1_COLS = ["log_dist", "dist_rank", "kruskal_in", "dJ", "log_corr", "doc"]
UC2_COLS = ["peer_top3", "peer_mean", "power_gain", "doc"]


def slog(a):
    a = np.asarray(a, dtype=float)
    return np.sign(a) * np.log10(1 + np.abs(a))


def node_row(V=None, Vref=None, P=None, Q=None, deg=0, root=0.0):
    z = lambda: 0.0
    f = [V.std() if V is not None else z(), (V - Vref).std() if V is not None and Vref is not None else z(),
         P.std() if P is not None else z(), Q.std() if Q is not None else z(),
         P.mean() if P is not None else z(), Q.mean() if Q is not None else z()]
    f = [np.sign(x) * np.log10(abs(x) + 1e-9) if x != 0 else 0.0 for x in f]
    return f + [float(deg), float(root)]


def finish(nodes_feat, node_type, ei, et, xe, y):
    xn = np.concatenate([np.array(nodes_feat, float), np.eye(N_NODE_TYPES)[node_type]], 1)
    return dict(xn=xn, xe=np.asarray(xe, float), ei=np.asarray(ei, int), et=np.asarray(et, int), y=np.asarray(y, float))


def pad(x):
    x = np.asarray(x, float)
    return np.concatenate([x, np.zeros((x.shape[0], NE - x.shape[1]))], 1) if x.shape[1] < NE else x


# ----------------------------------------------------------------------------- graph builders (unified format)
def graph_uc1(c, g):
    edges = list(g.edge); nodes = list(c["nodes"]); pos = {n: k for k, n in enumerate(nodes)}
    idx = {n: k for k, n in enumerate(c["nodes"])}; r = idx[c["root"]]
    deg = {n: 0 for n in nodes}
    for e in edges:
        for n in e:
            deg[n] += 1
    feat = [node_row(c["V"][:, idx[n]], c["V"][:, r], c["P"][:, idx[n]], c["Q"][:, idx[n]], deg[n], n == c["root"]) for n in nodes]
    ei = [[pos[tuple(e)[0]] for e in edges], [pos[tuple(e)[1]] for e in edges]]
    xe = g[UC1_COLS].to_numpy(float).copy(); xe[:, 3] = slog(xe[:, 3])
    return finish(feat, [0] * len(nodes), ei, [0] * len(edges), pad(xe), g["label"].to_numpy(float))


def graph_uc2(g, C, V, Vs, Pc, head, K, k=5):
    """g: UC2 rows of one case (i, f, features, label). Customers 0..n-1, feeder nodes n..n+K-1."""
    g = g.sort_values(["i", "f"])
    n = int(g.i.max()) + 1
    Pinj, Hinj = -Pc, -head                                           # injection sign, as in UC1
    Cn = np.nan_to_num(C, nan=-np.inf); nn = np.argsort(-Cn, 1)[:, :min(k, n - 1)]
    ctx = {(min(i, j), max(i, j)) for i in range(n) for j in nn[i]}
    a1 = g.i.to_numpy(int); b1 = n + g.f.to_numpy(int)
    a2 = np.array([p[0] for p in ctx], int); b2 = np.array([p[1] for p in ctx], int)
    deg = np.bincount(np.concatenate([a1, b1, a2, b2]), minlength=n + K)
    feat = [node_row(V[:, i], Vs[:, 0], Pinj[:, i], None, deg[i]) for i in range(n)]
    feat += [node_row(None, None, Hinj[:, f], None, deg[n + f]) for f in range(K)]
    xe1 = g[UC2_COLS].to_numpy(float); xe1 = np.nan_to_num(xe1, nan=0.0)
    xe2 = np.array([[C[i, j]] for i, j in ctx]) if len(ctx) else np.zeros((0, 1))
    ei = np.concatenate([np.stack([a1, b1]), np.stack([a2, b2])], 1)
    et = np.concatenate([np.ones(len(a1), int), 2 * np.ones(len(a2), int)])
    xe = np.concatenate([pad(xe1), pad(xe2)])
    y = np.concatenate([g.label.to_numpy(float), np.zeros(len(a2))])
    return finish(feat, [1] * n + [2] * K, ei, et, xe, y)


# ----------------------------------------------------------------------------- model
def build_model(n_in):
    import torch
    import torch.nn as nn

    def mlp(i, o):
        return nn.Sequential(nn.Linear(i, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, o))

    class MultiTaskGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.n0 = mlp(n_in, HIDDEN)
            self.enc = nn.ModuleList([mlp(NE, HIDDEN) for _ in range(N_EDGE_TYPES)])   # one input encoder per edge type
            self.msg = nn.ModuleList([mlp(3 * HIDDEN, HIDDEN) for _ in range(LAYERS)])  # shared
            self.upd = nn.ModuleList([mlp(3 * HIDDEN, HIDDEN) for _ in range(LAYERS)])  # shared
            self.head = nn.ModuleList([nn.Sequential(nn.Linear(3 * HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(0.1),
                                                     nn.Linear(HIDDEN, 1)) for _ in range(2)])  # UC1, UC2

        def forward(self, xn, xe, ei, et):
            h = self.n0(xn)
            e = sum(self.enc[t](xe) * (et == t).float().unsqueeze(1) for t in range(N_EDGE_TYPES))
            a, b = ei[0], ei[1]
            src = torch.cat([a, b]); dst = torch.cat([b, a])
            deg = torch.zeros(h.shape[0]).index_add_(0, dst, torch.ones(dst.shape[0])).clamp(min=1).unsqueeze(1)
            for msg, upd in zip(self.msg, self.upd):
                m = msg(torch.cat([h[dst], h[src], torch.cat([e, e])], 1))
                h = h + torch.zeros_like(h).index_add_(0, dst, m) / deg
                e = e + upd(torch.cat([h[a] + h[b], (h[a] - h[b]).abs(), e], 1))
            z = torch.cat([e, h[a] + h[b], (h[a] - h[b]).abs()], 1)
            return torch.where(et == 0, self.head[0](z).squeeze(1), self.head[1](z).squeeze(1))

    return MultiTaskGNN()


def train_predict(train_graphs, test_graphs, seed):
    import torch
    torch.manual_seed(seed); torch.set_num_threads(THREADS)
    XN = np.concatenate([g["xn"] for g in train_graphs])
    mn, sn = XN.mean(0), XN.std(0) + 1e-9
    stats = {}
    for t in range(N_EDGE_TYPES):                                    # edge inputs standardised per edge type
        rows = [g["xe"][g["et"] == t] for g in train_graphs]
        A = np.concatenate(rows) if sum(len(r) for r in rows) else np.zeros((1, NE))
        stats[t] = (A.mean(0), A.std(0) + 1e-9)

    def T(g):
        xe = g["xe"].copy()
        for t in range(N_EDGE_TYPES):
            m = g["et"] == t
            xe[m] = (xe[m] - stats[t][0]) / stats[t][1]
        return (torch.tensor((g["xn"] - mn) / sn, dtype=torch.float32), torch.tensor(xe, dtype=torch.float32),
                torch.tensor(g["ei"], dtype=torch.long), torch.tensor(g["et"], dtype=torch.long),
                torch.tensor(g["y"], dtype=torch.float32))

    tr = [T(g) for g in train_graphs]

    def merge(items):
        out, off = [[], [], [], [], []], 0
        for xn, xe, ei, et, y in items:
            out[0].append(xn); out[1].append(xe); out[2].append(ei + off); out[3].append(et); out[4].append(y); off += xn.shape[0]
        return torch.cat(out[0]), torch.cat(out[1]), torch.cat(out[2], 1), torch.cat(out[3]), torch.cat(out[4])

    model = build_model(XN.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    rng = np.random.default_rng(seed)
    last = float("nan")
    for ep in range(EPOCHS):
        model.train()
        order = rng.permutation(len(tr))
        for s0 in range(0, len(order), BATCH):
            xn, xe, ei, et, y = merge([tr[k] for k in order[s0:s0 + BATCH]])
            out = model(xn, xe, ei, et)
            losses = [bce(out[et == t], y[et == t]) for t in (0, 1) if bool((et == t).any())]
            loss = sum(losses) / len(losses)
            opt.zero_grad(); loss.backward(); opt.step(); last = float(loss)
    model.eval()
    res = []
    with torch.no_grad():
        for g in test_graphs:
            xn, xe, ei, et, _ = T(g)
            res.append(torch.sigmoid(model(xn, xe, ei, et)).numpy())
    return res, last


# ----------------------------------------------------------------------------- smoke test
def smoke():
    import networkx as nx
    import pandas as pd
    rng = np.random.default_rng(0)
    graphs = []
    for k in range(12):                                              # UC1-like
        n = int(rng.integers(12, 25)); tree = nx.random_labeled_tree(n, seed=int(rng.integers(1e9)))
        true = [frozenset(e) for e in tree.edges()]
        dec = []
        while len(dec) < n // 2:
            a, b = rng.choice(n, 2, replace=False)
            if frozenset((int(a), int(b))) not in true and frozenset((int(a), int(b))) not in dec:
                dec.append(frozenset((int(a), int(b))))
        edges = true + dec; y = np.array([1.0] * len(true) + [0.0] * len(dec))
        T = 200; V = 1 + rng.normal(0, .01, (T, n)); P = rng.normal(0, 1, (T, n)); Q = rng.normal(0, 1, (T, n))
        c = dict(nodes=list(range(n)), root=0, V=V, P=P, Q=Q)
        g = pd.DataFrame(dict(edge=edges, label=y, **{col: rng.normal(0, 1, len(edges)) + 1.2 * y for col in UC1_COLS}))
        graphs.append(graph_uc1(c, g))
    for k in range(12):                                              # UC2-like
        n, K, T = 20, 3, 200
        truth = rng.integers(0, K, n); Z = rng.normal(size=(T, K)); V = 1 + 0.01 * (Z[:, truth] + rng.normal(0, 1, (T, n)))
        C = np.corrcoef(V.T); np.fill_diagonal(C, np.nan)
        rows = [dict(i=i, f=f, label=int(truth[i] == f), **{col: rng.normal() + 1.2 * (truth[i] == f) for col in UC2_COLS})
                for i in range(n) for f in range(K)]
        Pc = np.abs(rng.normal(1, .3, (T, n))); head = np.stack([Pc[:, truth == f].sum(1) for f in range(K)], 1)
        graphs.append(graph_uc2(pd.DataFrame(rows), C, V, V[:, :1], Pc, head, K))
    global EPOCHS
    EPOCHS = min(EPOCHS, 30)
    t = time.time()
    tr = graphs[:9] + graphs[12:21]; te = graphs[9:12] + graphs[21:24]
    P, last = train_predict(tr, te, 0)
    for name, sl in (("UC1", slice(0, 3)), ("UC2", slice(3, 6))):
        acc = np.mean(np.concatenate([((p > .5) == (g["y"] > .5))[g["et"] < 2] for p, g in zip(P[sl], te[sl])]))
        print(f"{name} test accuracy {acc:.3f}")
    import torch
    print(f"smoke test OK  torch {torch.__version__}  last loss {last:.3f}  ({time.time() - t:.1f} s)  expected: both accuracies clearly above 0.6")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "smoke":
    smoke(); sys.exit(0)

# ----------------------------------------------------------------------------- real data
import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
_sys.argv = _sys.argv[:2]
import glob  # noqa: E402
import pickle  # noqa: E402
import pandas as pd  # noqa: E402
import exp_final_lv_ext as F  # noqa: E402
from exp_final_lv_ext import L, LF, METERS, SIMBENCH_FEEDERS, SB_WORK  # noqa: E402

QS = "_quick" if QUICK else ""
OUT = f"{SB_WORK}/multitask_{TASKS}{QS}"
os.makedirs(OUT, exist_ok=True)
UC2_CACHE = f"{SB_WORK}/uc2_cache{L.SUFM}.pkl"


def correct(Pc, head, lab, K, free, max_moves=200):
    """Power-balance refinement of the original UC2 method (copied from exp_final_feeder.py)."""
    lab = lab.copy()
    R = head - np.stack([Pc[:, lab == f].sum(1) for f in range(K)], 1)
    for _ in range(max_moves):
        best, bi, bf = 0.0, None, None
        for i in np.where(free)[0]:
            a = lab[i]; p = Pc[:, i]; base = (R[:, a] ** 2).sum()
            for f in range(K):
                if f == a:
                    continue
                g = base + (R[:, f] ** 2).sum() - ((R[:, a] + p) ** 2).sum() - ((R[:, f] - p) ** 2).sum()
                if g > best:
                    best, bi, bf = g, i, f
        if bi is None:
            break
        R[:, lab[bi]] += Pc[:, bi]; R[:, bf] -= Pc[:, bi]; lab[bi] = bf
    return lab


def load_all():
    G = {}                                       # key (task, case) -> graph;  meta for decoding
    meta = {}
    if TASKS in ("uc1", "both"):
        C = pickle.load(open(f"{SB_WORK}/lv_ctx_{METERS}_tdm{L.SUFM}.pkl", "rb"))
        X, _ = LF.add_features(METERS); X = X.reset_index(drop=True)
        for cs, g in X.groupby("case", sort=False):
            G[("uc1", cs)] = graph_uc1(C[cs], g)
            meta[("uc1", cs)] = dict(feeder=g.feeder.iloc[0], edges=list(g.edge))
        del C
    if TASKS in ("uc2", "both"):
        if not os.path.exists(UC2_CACHE):
            sys.exit(f"missing {UC2_CACHE}: run  python run_methods_ext.py --method uc2  first (the new version writes it)")
        D = pickle.load(open(UC2_CACHE, "rb"))
        for cs, g in D["X"].groupby("case", sort=False):
            truth, doc, Pc, head, K = D["cases"][cs]; V, Vs = D["VV"][cs]
            G[("uc2", cs)] = graph_uc2(g, D["Cs"][cs], V, Vs, Pc, head, K)
            meta[("uc2", cs)] = dict(feeder=cs.split("#")[0], truth=truth, doc=doc, K=K)
    return G, meta


def probs():
    path = f"{OUT}/probs.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    G, meta = load_all()
    log = open(f"{OUT}/train_log.txt", "a"); t0 = time.time()
    log.write(f"start TASKS={TASKS}: {len(G)} graphs, EPOCHS={EPOCHS} SEEDS={SEEDS} BATCH={BATCH} THREADS={THREADS}\n"); log.flush()
    P = {}
    for f in SIMBENCH_FEEDERS:
        tr = [G[k] for k in G if meta[k]["feeder"] != f]
        te = [k for k in G if meta[k]["feeder"] == f]
        acc = {k: np.zeros(len(G[k]["y"])) for k in te}
        for s in range(SEEDS):
            out, last = train_predict(tr, [G[k] for k in te], s)
            for k, q in zip(te, out):
                acc[k] += q / SEEDS
            log.write(f"{time.time() - t0:7.0f} s  test feeder {f}  seed {s}  last loss {last:.3f}\n"); log.flush()
        P.update(acc)
    out = dict(P=P, meta={k: {kk: vv for kk, vv in m.items() if kk in ("feeder", "edges", "truth", "doc", "K")} for k, m in meta.items()},
               et={k: G[k]["et"] for k in G})
    pickle.dump(out, open(path, "wb"))
    return out


def run_feeder(i):
    code = SIMBENCH_FEEDERS[i]
    R = probs()
    P, meta, ET = R["P"], R["meta"], R["et"]
    if TASKS in ("uc1", "both"):
        C = pickle.load(open(f"{SB_WORK}/lv_ctx_{METERS}_tdm{L.SUFM}.pkl", "rb"))
        rows = []
        for (task, cs), m in meta.items():
            if task != "uc1" or m["feeder"] != code:
                continue
            c = C[cs]; edges = m["edges"]; p = P[(task, cs)]
            Gd = L.decode(c["nodes"], c["fixed"], edges, p)
            rows.append(dict(case=cs, uc1_steps_1_3=F.share(Gd, c["tg"]),
                             uc1=F.share(F.refine(Gd, set(edges), c, p=dict(zip(edges, p))), c["tg"])))
        pd.DataFrame(rows).to_csv(f"{OUT}/res_uc1_{i}.csv", index=False)
    if TASKS in ("uc2", "both"):
        D = pickle.load(open(UC2_CACHE, "rb"))
        rows = []
        for (task, cs), m in meta.items():
            if task != "uc2" or m["feeder"] != code:
                continue
            truth, doc, Pc, head, K = D["cases"][cs]
            p = P[(task, cs)][ET[(task, cs)] == 1]
            g = D["X"][D["X"].case == cs].sort_values(["i", "f"])
            M = np.zeros((int(g.i.max()) + 1, K)); M[g.i.values, g.f.values] = p
            dec = M.argmax(1)
            comb = correct(Pc.astype(float), head.astype(float), dec, K, np.ones(len(dec), bool))
            rows.append(dict(case=cs, uc2_steps_1_3=(dec == truth).mean(), uc2=(comb == truth).mean()))
        pd.DataFrame(rows).to_csv(f"{OUT}/res_uc2_{i}.csv", index=False)
    print(code, "done", flush=True)


def summary():
    pd.set_option("display.width", 200)

    def pair(M, a, b):
        d = 100 * (M[a] - M[b]); n = d.notna().sum()
        return f"{a:>16s} - {b:<16s} {d.mean():+6.2f} ± {1.96 * d.std(ddof=1) / np.sqrt(n):.2f}   better {int((d > 1e-9).sum())}  worse {int((d < -1e-9).sum())}"

    rd = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pat))]) if glob.glob(pat) else None
    for task in ("uc1", "uc2"):
        mine = {t: rd(f"{SB_WORK}/multitask_{t}{QS}/res_{task}_*.csv") for t in ("uc1", "uc2", "both")}
        mine = {t: d.rename(columns={task: f"{task}_{t}", f"{task}_steps_1_3": f"{task}_steps_1_3_{t}"}) for t, d in mine.items() if d is not None}
        if not mine:
            continue
        if task == "uc1":
            base = rd(f"{F.OUT}/res_*.csv")[["case", "combined"]]
            tor = rd(f"{F.OUT}_gnn_torch/res_*.csv")
            if tor is not None:
                base = base.merge(tor, on="case")
        else:
            base = pd.read_csv(f"{SB_WORK}/final_feeder_ext{L.SUFM}.csv")[["case", "combined", "graph", "nograph"]]
        M = base
        for d in mine.values():
            M = M.merge(d, on="case")
        print(f"\n===== {task.upper()}  ({len(M)} cases, percent correct) =====")
        print((100 * M.drop(columns="case").mean()).round(2).to_string())
        print("paired differences (95% CI treats cases as independent):")
        for t in mine:
            print(pair(M, f"{task}_{t}", "combined"))
        if "both" in mine and task in mine:
            print(pair(M, f"{task}_both", f"{task}_{task}"), "   <- multi-task vs single-task, same code")


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "summary":
        summary()
    else:
        run_feeder(int(arg))
