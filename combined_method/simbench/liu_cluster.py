"""Steps 2 and 3 of Liu et al., "A data-driven approach for topology correction in
low voltage distribution networks with PVs" (arXiv 2506.20238), our implementation.

Step 2, user-feeder: Pearson correlation of the raw voltage magnitudes (Eq. 6),
Fisher-type transform FPCC = ln((1 + r) / (1 - r + alpha)) (Eq. 7), distance
MFP = 1 - (1/4) ln(1 + exp(a + 4 FPCC)) (Eq. 8), agglomerative clustering into
the number of feeders. Step 3, phase: per-time z-score across the customers of a
feeder (Eq. 12), clustering into 3 groups. Data selection (Eq. 14): only the
times when PV < load.

Settings from Liu's public code (github.com/distributionnetworksTUDelft/
Data-driven-approach-for-topology-correction-in-LVDNs-): alpha = 0.5, a = 1, the
clip of Eq. 8, complete linkage; phase: per-time standardisation, distance
1.1 - r, complete linkage; Algorithm 2: KNN (k = 5) on the voltage series of the
recorded customers. Our choice (not in the code): naming the clusters by one-to-one
matching with the documented labels (Hungarian), i.e. the documentation is used as
the recordings.
"""
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform

ALPHA, A = 0.5, 1.0


def _cluster(D, K):
    D = D - D.min(); np.fill_diagonal(D, 0); D = (D + D.T) / 2
    return fcluster(linkage(squareform(D, checks=False), "complete"), K, "maxclust") - 1


def _name(cl, doc, K):
    M = np.zeros((cl.max() + 1, K))
    for c, d in zip(cl, doc):
        M[c, d] += 1
    r, c = linear_sum_assignment(-M)
    m = dict(zip(r, c))
    return np.array([m.get(k, doc[i]) for i, k in enumerate(cl)])


def user_feeder(V, doc, K, keep=None):
    """V: T x N raw voltage magnitudes; doc: documented feeder per customer."""
    X = V if keep is None else V[keep]
    r = np.clip(np.corrcoef(X.T), -0.999999, 0.999999)
    F = np.log((1 + r) / (1 - r + ALPHA))
    D = 1 - np.minimum(0.25 * np.log1p(np.exp(A + 4 * F)), 1)
    return _name(_cluster(D, K), doc, K)


def phase(V, doc, groups, keep=None):
    """V: T x N phase voltage magnitude of each customer; groups: feeder of each customer."""
    X = V if keep is None else V[keep]
    lab = doc.copy()
    for g in np.unique(groups):
        m = np.where(groups == g)[0]
        if len(m) < 4:
            continue
        Z = X[:, m]; Z = (Z - Z.mean(1, keepdims=True)) / (Z.std(1, keepdims=True) + 1e-12)
        D = 1.1 - np.corrcoef(Z.T)
        lab[m] = _name(_cluster(D, 3), doc[m], 3)
    return lab


def user_feeder_knn(V, doc, K, keep=None, k=5):
    """Algorithm 2 (recorded connections), as in Liu's code: each unknown customer
    (doc = -1) takes the majority label of its k nearest recorded customers, KNN on
    the voltage series (Euclidean)."""
    from sklearn.neighbors import KNeighborsClassifier
    X = (V if keep is None else V[keep]).T
    known = doc >= 0
    lab = doc.copy()
    knn = KNeighborsClassifier(n_neighbors=k).fit(X[known], doc[known])
    lab[~known] = knn.predict(X[~known])
    return lab
