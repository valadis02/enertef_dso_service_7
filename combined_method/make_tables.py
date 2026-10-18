"""Print the result table of the combined method.

    python make_tables.py              # from a fresh run (work/)
    python make_tables.py --shipped    # from the results shipped in results/
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import SB_WORK, FW_WORK  # noqa: E402

COLS = ["documented", "validate_topology", "random_forest", "combination"]


def lv_uc1(sb):
    R = pd.concat([pd.read_csv(f) for f in glob.glob(f"{sb}/final_lv_month/res_*.csv")])
    return [R.documented.mean(), R.validate_topology.mean(), R.liu_only.mean(), R.combined.mean()], len(R)


def assign(sb, which):
    R = pd.read_csv(f"{sb}/final_{which}_month.csv")
    return [R.documented.mean(), R.validate_topology.mean(), R.liu_only.mean(), R.combined.mean()], len(R)


def mv(sb):
    R = pd.concat([pd.read_csv(f) for f in glob.glob(f"{sb}/final_mv85_month/res_*.csv")])
    out = []
    for m in ("documented", "validate_topology", "liu_only", "combined"):
        s = R[R.method == m]
        out.append((s.open_found.mean(), int(s.false_open.sum())))
    return out, len(R[R.method == "documented"])


def fw(fwd, pattern, open_metric):
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"{fwd}/{pattern}"))]
    out = []
    for m in ("doc", "validate_topology", "liu_only", "combined"):
        acc, of, fo = [], [], 0
        for r in rows:
            t = np.array(r["truth"]); s = np.array(r[m]); to = t == 0
            acc.append((s == t).mean()); of.append(((s == 0) & to).sum() / max(to.sum(), 1)); fo += int(((s == 0) & ~to).sum())
        out.append((np.mean(of), fo) if open_metric else np.mean(acc))
    return out, len(rows)


def pct(x):
    return f"{100 * x:.1f}%"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--shipped", action="store_true"); a = ap.parse_args()
    sb = os.path.join(HERE, "results", "simbench") if a.shipped else SB_WORK
    fwd = os.path.join(HERE, "results", "far_west") if a.shipped else FW_WORK
    lines = ["| Use case | Network | Cases | Documented | validate_topology | Random forest (Liu) | Combination |",
             "|---|---|---|---|---|---|---|"]
    def add(uc, net, fn):
        try:
            vals, n = fn()
        except Exception as e:                                  # missing outputs
            lines.append(f"| {uc} | {net} | - | not run ({type(e).__name__}) | | | |"); return
        cells = [f"{pct(v[0])} ({v[1]})" if isinstance(v, tuple) else pct(v) for v in vals]
        lines.append(f"| {uc} | {net} | {n} | " + " | ".join(cells) + " |")
    add("1. Topology", "SimBench LV", lambda: lv_uc1(sb))
    add("1. Topology", "Far West", lambda: fw(f"{fwd}/final_fw_month", "uc1_*.json", False))
    add("2. Customer-feeder", "SimBench LV", lambda: assign(sb, "feeder"))
    add("3. Phase", "SimBench LV", lambda: assign(sb, "phase"))
    add("4. Switches", "SimBench MV", lambda: mv(sb))
    add("4. Switches", "Far West", lambda: fw(f"{fwd}/final_fw7_month", "res_*.json", True))
    print("\n".join(lines))
    print("\nUse case 4: share of truly open elements found; in brackets, elements reported open that are closed.")
