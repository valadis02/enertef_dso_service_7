"""Individual customer load profiles for SimBench LV (sensitivity study for use case 2).

SimBench assigns each LV load one of a few standard profiles (5-11 per network),
so many customers have identical time series. Real smart-meter data differ per
customer. This module turns each customer's standard profile into an individual
one, keeping its seasonal and daily shape and its mean consumption:

1. day shuffle: each day is taken from a random day of the same weekday within
   +-3 weeks, independently per customer;
2. behavioural time shift: a fixed offset per customer (sigma 45 min) plus a
   daily jitter (sigma 30 min);
3. 15-min variability: multiplicative log-normal AR(1) noise (phi 0.7, sigma 0.3);
4. appliance events (households only): about 3 per day, 1-2.5 kW for 15-60 min
   (scaled down for customers using under 1 kW on average),
   mostly in the morning and evening;
5. the result is scaled back to the original mean consumption;
6. electric vehicles: 10% of households charge at 3.7 kW, arriving 17:00-21:00,
   5-15 kWh per session, on 60% of days (added on top of the mean).

MV loads (aggregates of many customers) use noise sigma 0.1 and have no
appliance or EV events (those apply to household profiles only).
Reactive power keeps each load's original ratio Q/P. PV is left unchanged
(neighbouring PV systems see the same sun).
"""
import numpy as np

DAY = 96


def _arrival_weights():
    h = np.arange(DAY) / 4
    w = 0.2 + np.exp(-0.5 * ((h - 7.5) / 1.2) ** 2) + 1.5 * np.exp(-0.5 * ((h - 19) / 2.0) ** 2)
    return w / w.sum()


def individualise(lp, lq, profiles, start, steps, seed, noise=0.3):
    """lp, lq: full-year DataFrames (time x load) in MW / Mvar.
    profiles: profile name per load. Returns arrays (steps x load) for P and Q."""
    n_t, n_l = lp.shape
    P0, Q0 = lp.values, lq.values
    rng = np.random.default_rng(seed)
    aw = _arrival_weights()
    first_day, n_days = start // DAY, int(np.ceil(steps / DAY))
    P = np.zeros((steps, n_l)); Q = np.zeros((steps, n_l))
    for j in range(n_l):
        base = P0[:, j]
        offset = int(np.round(rng.normal(0, 3)))                    # 3 steps = 45 min
        out = np.empty(n_days * DAY)
        for d in range(n_days):
            src = first_day + d + 7 * rng.integers(-3, 4)
            src = int(np.clip(src, 1, n_t // DAY - 2))
            shift = offset + int(np.round(rng.normal(0, 2)))
            idx = np.clip(src * DAY + np.arange(DAY) - shift, 0, n_t - 1)
            out[d * DAY:(d + 1) * DAY] = base[idx]
        out = out[:steps]
        e = np.zeros(steps); z = rng.normal(0, 1, steps)
        for t in range(1, steps):
            e[t] = 0.7 * e[t - 1] + np.sqrt(1 - 0.49) * z[t]
        p = out * np.exp(noise * e - noise ** 2 / 2)
        household = str(profiles[j]).startswith("H")
        m0 = P0[start:start + steps, j].mean()
        size = min(1.0, m0 * 1000 / 1.0)                             # smaller events for small consumers
        if household:
            for d in range(n_days):
                for _ in range(rng.poisson(3)):
                    t0 = d * DAY + rng.choice(DAY, p=aw); dur = rng.integers(1, 5)
                    p[t0:t0 + dur] += size * rng.uniform(1.0, 2.5) / 1000
            p = p[:steps]
        if p.mean() > 0:
            p *= m0 / p.mean()
        ratio = (Q0[start:start + steps, j].sum() / P0[start:start + steps, j].sum()
                 if abs(P0[start:start + steps, j].sum()) > 1e-12 else 0.0)
        q = p * ratio
        if household and rng.random() < 0.10:                        # electric vehicle, active power only
            for d in range(n_days):
                if rng.random() < 0.6:
                    t0 = d * DAY + rng.integers(68, 84)              # 17:00-21:00
                    dur = int(np.round(rng.uniform(5, 15) / 3.7 * 4))
                    p[t0:t0 + dur] += 3.7 / 1000
            p = p[:steps]
        P[:, j], Q[:, j] = p, q
    return P, Q
