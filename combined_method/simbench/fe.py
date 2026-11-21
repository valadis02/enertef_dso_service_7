"""Feature engineering for the classifier of the combination (FE env: comma list of rel, loc, day).

rel : case-relative versions of the main features (z-score and percentile rank within the case):
      a branch is judged against the other candidates of the same network snapshot.
loc : local competition (LV, needs edge end-points): number of candidates at the two ends and the best /
      worst value of log_dist and dJ among the candidates that share an end-point (the alternative parents).
day : variance distance and correlation without the PV hours (already computed by the LV pipeline)."""
import numpy as np
import pandas as pd

BASE = ["log_dist", "dJ", "log_corr"]


def add(X, spec):
    parts = [s for s in (spec or "").split(",") if s]
    if not parts:
        return X, []
    X = X.copy(); extra = []
    if "rel" in parts:
        g = X.groupby("case")
        for f in BASE:
            if f not in X:
                continue
            m, s = g[f].transform("mean"), g[f].transform("std").replace(0, 1).fillna(1)
            X[f"{f}_z"] = (X[f] - m) / s
            X[f"{f}_pct"] = g[f].rank(pct=True)
            extra += [f"{f}_z", f"{f}_pct"]
    if "loc" in parts and "edge" in X:
        cols = {f: np.full(len(X), np.nan) for f in ("n_cand", "ld_nb_min", "ld_nb_max", "dJ_nb_min", "dJ_nb_max")}
        for case, ix in X.groupby("case").indices.items():
            ends = [tuple(e) for e in X.edge.iloc[ix]]
            at = {}
            for j, (a, b) in enumerate(ends):
                at.setdefault(a, []).append(j); at.setdefault(b, []).append(j)
            ld = X.log_dist.values[ix]; dj = X.dJ.values[ix]
            for j, (a, b) in enumerate(ends):
                nb = sorted(set(at[a] + at[b]) - {j}); i = ix[j]
                cols["n_cand"][i] = len(nb)
                if nb:
                    cols["ld_nb_min"][i] = ld[j] - ld[nb].min(); cols["ld_nb_max"][i] = ld[j] - ld[nb].max()
                    cols["dJ_nb_min"][i] = dj[j] - dj[nb].min(); cols["dJ_nb_max"][i] = dj[j] - dj[nb].max()
        for f, v in cols.items():
            X[f] = np.nan_to_num(v, nan=0.0); extra.append(f)
    if "day" in parts:
        extra += [c for c in ("log_dist_noPV", "log_corr_noPV") if c in X]
    for c in ("dJ_nb_min", "dJ_nb_max"):                # heavy-tailed: signed log so scale-sensitive models cope
        if c in X:
            X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))
    for c in extra:
        X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, extra
