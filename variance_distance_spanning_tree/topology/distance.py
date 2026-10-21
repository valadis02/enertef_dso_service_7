"""Electrical distance between buses.

Under the LinDistFlow approximation the variance of the difference of two
bus quantities is proportional to the electrical distance between them.
Taking the difference cancels the common-mode term algebraically -- the
term that makes raw correlations sit near 0.99 for every pair -- and
independent meter noise adds the same constant 2*sigma^2 to every pair,
so it shifts all distances equally and leaves their ranking intact.

Works on voltage magnitudes or on phase angles. Angles carry a different
linear combination of P and Q (weighted by reactance rather than
resistance), so they are independent information rather than a copy.
"""

import numpy as np


def electrical_distance(measurements, smooth_window=15):
    """Pairwise Var(x_i - x_j) for every pair of columns.

    measurements: DataFrame, one column per bus.
    smooth_window: rolling mean applied first. Meter noise is white while
        the underlying signal is smooth in time, so this improves the
        signal-to-noise ratio. Set to 0 or 1 for data that is not
        temporally autocorrelated (e.g. synthetic loads drawn independently
        per timestep), where smoothing removes signal rather than noise.
    """
    df = measurements
    if smooth_window and smooth_window > 1:
        df = df.rolling(smooth_window, center=True, min_periods=1).mean()

    cov = df.cov().values
    var = np.diag(cov)
    D = var[:, None] + var[None, :] - 2 * cov
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0.0, None), list(df.columns)


def noise_variance(measurements, frac=0.5):
    """Estimate meter noise variance from the covariance spectrum.

    A radial feeder has only a few large eigenvalues -- the real load and
    structural modes. The remaining small ones are essentially pure noise,
    so the median of the smallest half estimates sigma^2. Unlike a lag-1
    estimator this does not assume the signal is smooth in time, so it
    works on synthetic and real data alike.
    """
    X = (measurements - measurements.mean()).values
    keep = measurements.std().values > 1e-14
    X = X[:, keep]
    if X.shape[1] < 2:
        return 0.0
    ev = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))
    k = max(1, int(len(ev) * frac))
    return float(np.median(ev[:k]))


def remove_common_factor(measurements):
    """Regress out the shared load factor, one coefficient per bus.

    Every household follows the same daily pattern, which dominates the
    measurements. Subtracting the cross-sectional mean with a coefficient
    fixed at 1 imposes an artificial sum-to-zero constraint and makes
    things worse; letting each bus have its own beta does not.
    """
    m = measurements.mean(axis=1).values
    mc = m - m.mean()
    denom = float(np.dot(mc, mc))
    if denom < 1e-18:
        return measurements.copy()

    out = measurements.copy()
    for c in measurements.columns:
        v = measurements[c].values
        beta = np.dot(v - v.mean(), mc) / denom
        out[c] = v - beta * m
    return out
