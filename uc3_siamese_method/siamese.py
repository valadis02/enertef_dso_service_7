"""Phase identification with the method of Liu, Giraldo, Palensky, Vergara,
"A siamese neural network model for phase identification in distribution
networks", IJEPES 169 (2025) 110718 -- our implementation from the paper text,
on the 72 SimBench LV phase cases (use case 3), run with personalized profiles.

Inputs (as in the paper): active power of every customer (smart meter, 1% error)
and active power of each transformer phase (1% error, 1-3% unmodelled losses).
No voltages, no documentation.

Method, following the paper:
  1. self-taught training: customers shuffled into three balanced groups; the sum
     of each group is a pseudo transformer phase; pairs [phase window, customer
     window] are labelled 1 for the customer's own group, 0 otherwise; positives
     duplicated; Gaussian noise added;
  2. features: each one-day window (w = 96) normalised, Haar wavelet with two
     levels, the components L2, H2, H1 reconstructed to length 96;
  3. siamese network with shared weights: recurrent layer (input 96, hidden 500)
     over the three components, fully connected layer of 500 with ELU; Euclidean
     distance; contrastive loss with margin 1; SGD, learning rate 0.005, 100 epochs;
  4. estimation with the real transformer phases: distances summed over windows,
     normalised over the three phases, smallest = phase;
  5. Kendall label modification: low-confidence customers (largest difference of
     their mean distances below the margin) are moved to the phase that most
     increases the Kendall tau between the first differences of the measured and
     the identified phase profiles;
  6. several independent runs; the run with the largest Kendall tau is kept
     (the paper averages the five best of 20 runs by the same criterion).

Not given in the paper, our choices: vanilla RNN (tanh) as the recurrent layer,
noise sigma 0.05 on the normalised data, momentum 0.9 and mini-batches of 1/8 of
the customers, 5 runs per case.

usage: siamese.py <feeder index> [runs]   (or run.py for everything)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "combined_method"))
from paths import TOPO, SB_CODE, SB_WORK  # noqa: E402
import os
import sys

import numpy as np
import pandas as pd
import pywt
import torch
from scipy.stats import kendalltau

sys.path.insert(0, SB_CODE)
sys.path.insert(0, TOPO)
import phase_cases as E  # noqa: E402
from topology.data import SIMBENCH_FEEDERS  # noqa: E402

torch.set_num_threads(1)
W, HID, MARGIN, LR, EPOCHS, NOISE = 96, 500, 1.0, 0.005, 100, 0.05


def windows(x):
    """x: T (or T x N) -> (n_win, 3, 96) per series: normalised, Haar level 2, L2/H2/H1 reconstructed."""
    x = np.atleast_2d(x.T).T
    n_win = len(x) // W
    out = np.zeros((x.shape[1], n_win, 3, W), np.float32)
    for j in range(x.shape[1]):
        for m in range(n_win):
            s = x[m * W:(m + 1) * W, j]
            s = (s - s.mean()) / (s.std() + 1e-9)
            cA2, cD2, cD1 = pywt.wavedec(s, "haar", level=2)
            z2, z1 = np.zeros_like(cD2), np.zeros_like(cD1)
            out[j, m, 0] = pywt.waverec([cA2, z2, z1], "haar")
            out[j, m, 1] = pywt.waverec([np.zeros_like(cA2), cD2, z1], "haar")
            out[j, m, 2] = pywt.waverec([np.zeros_like(cA2), z2, cD1], "haar")
    return out                                                    # series x windows x 3 x 96


class Branch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(W, HID, batch_first=True)
        self.fc = torch.nn.Linear(HID, HID)

    def forward(self, x):                                         # x: B x 3 x 96
        _, h = self.rnn(x)
        return torch.nn.functional.elu(self.fc(h[-1]))


def kendall(Pt, Pc, lab):
    dT = np.diff(Pt, axis=0)
    K = 0.0
    for k in range(3):
        dI = np.diff(Pc[:, lab == k].sum(1)) if (lab == k).any() else np.zeros(len(dT))
        K += kendalltau(dT[:, k], dI)[0] if dI.std() > 0 else -1.0
    return K / 3


def run_once(Pc, Pt, seed):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    N = Pc.shape[1]
    perm = rng.permutation(N); grp = np.zeros(N, int); grp[perm] = np.arange(N) % 3
    pseudo = np.stack([Pc[:, grp == k].sum(1) for k in range(3)], 1)
    Xc = torch.tensor(windows(Pc)); Xp = torch.tensor(windows(pseudo)); Xt = torch.tensor(windows(Pt))
    M = Xc.shape[1]
    net = Branch(); opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    lab_t = torch.tensor(grp)
    for ep in range(EPOCHS):
        order = rng.permutation(N)
        for b in np.array_split(order, 8):
            b = torch.tensor(b)
            xc = Xc[b].reshape(-1, 3, W); xc = xc + NOISE * torch.randn_like(xc)
            xp = Xp.reshape(-1, 3, W); xp = xp + NOISE * torch.randn_like(xp)
            ec = net(xc).reshape(len(b), M, HID); ep_ = net(xp).reshape(3, M, HID)
            D = torch.linalg.norm(ec[:, None] - ep_[None], dim=-1)           # cust x phase x window
            pos = (lab_t[b][:, None] == torch.arange(3)[None]).float()[..., None].expand_as(D)
            # positives duplicated (weight 2) to balance the 2:1 negative ratio
            loss = (2 * pos * D ** 2 + (1 - pos) * torch.clamp(MARGIN - D, min=0) ** 2).sum() / (4 * D.numel() / 3)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ec = net(Xc.reshape(-1, 3, W)).reshape(N, M, HID); et = net(Xt.reshape(-1, 3, W)).reshape(3, M, HID)
        Dm = torch.linalg.norm(ec[:, None] - et[None], dim=-1).mean(-1).numpy()   # cust x phase, mean over windows
    lab = (Dm / Dm.sum(1, keepdims=True)).argmin(1)
    # Kendall label modification
    K = kendall(Pt, Pc, lab)
    spread = Dm.max(1) - Dm.min(1)
    for i in np.argsort(spread):
        if spread[i] >= MARGIN:
            break
        best, bk = K, lab[i]
        for k in range(3):
            if k == lab[i]:
                continue
            l2 = lab.copy(); l2[i] = k; K2 = kendall(Pt, Pc, l2)
            if K2 > best:
                best, bk = K2, k
        lab[i], K = bk, best
    return lab, K, float(loss)


def main(fi, runs):
    code = SIMBENCH_FEEDERS[fi]
    f = E.Feeder(code)
    d = dict(np.load(f"{SB_WORK}/sb_cache" + ("_month" if os.environ.get("MONTH") == "1" else "") + f"/{code}.npz"))
    net = f.net
    cust = sorted((set(net.load.bus) | set(net.sgen.bus)) & set(f.idx) - {f.root})
    rows = []
    for seed in E.SEEDS:
        c = E.make_case(f, d, cust, seed)
        rng = np.random.default_rng(1000 + seed)
        t = c["truth"]; Ptrue = c["S_true"].real
        loss_k = rng.uniform(0.01, 0.03, 3)
        Pt = np.stack([Ptrue[:, t == k].sum(1) for k in range(3)], 1) * (1 + loss_k) * (1 + rng.normal(0, 0.01, (len(Ptrue), 3)))
        Pc = c["S"].real
        best = None
        for r in range(runs):
            lab, K, L = run_once(Pc, Pt, 100 * seed + r)
            if best is None or K > best[1]:
                best = (lab, K, L)
        rows.append(dict(case=f"{code}#{seed}", documented=(c["doc"] == t).mean(), liu_siamese=(best[0] == t).mean(),
                         kendall=best[1], final_loss=best[2]))
        print(code, seed, round(100 * rows[-1]["liu_siamese"], 1), "K", round(best[1], 3), flush=True)
    out = f"{SB_WORK}/phase_siamese" + ("_month" if os.environ.get("MONTH") == "1" else "")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{out}/res_{fi}.csv", index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 5)
