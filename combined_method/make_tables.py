"""Print every result table of the combined method from a results folder.

    python make_tables.py                 # from the shipped results (results/)
    python make_tables.py --dir <folder>  # from a fresh run collected with run_all.py --stage collect

Folder layout (one sub-folder per scenario):
  simbench_lv_doc{70,80,90}/            use cases 1, 2, 3 on SimBench LV
  simbench_lv_realistic_doc{70,80,90}/  use cases 1, 3 with a realistic asset register
  simbench_mv_doc{70,80,90}/            use case 4 on SimBench MV
  simbench_mv_doc80_meter50/            use case 4, half of the MV/LV substations metered
  far_west_doc80/                       use cases 1, 4 on Far West
  far_west_fast_realistic_doc{70,80,90}/ use cases 1 (all levels) and 4 (80%), fast mode, wrong lengths
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M = ["documented", "variance_distance_spanning_tree", "liu_only", "combined"]
HEAD = ("| Use case · scenario | Documented | Variance-Distance Spanning Tree | Random forest (Liu) | Combination | Siamese (Liu) |\n"
        "|---|---|---|---|---|---|")


def pc(x):
    return f"{100 * x:.1f}"


def lv_uc1(d):
    R = pd.concat([pd.read_csv(f) for f in glob.glob(f"{d}/res_*.csv")])
    return [R[m].mean() for m in M], len(R)


def assign(f, liu="liu_cluster"):
    R = pd.read_csv(f)
    return [R.documented.mean(), R.variance_distance_spanning_tree.mean(), R[liu].mean(), R.combined.mean()], len(R)


def mv(d):
    R = pd.concat([pd.read_csv(f) for f in glob.glob(f"{d}/res_*.csv")])
    return [R[R.method == m].open_found.mean() for m in M], [int(R[R.method == m].false_open.sum()) for m in M]


def fw(d, pattern, open_metric):
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/{pattern}"))]
    out, fo = [], []
    for m in ["doc"] + M[1:]:
        acc, of, n = [], [], 0
        for r in rows:
            t = np.array(r["truth"]); s = np.array(r[m]); to = t == 0
            acc.append((s == t).mean()); of.append(((s == 0) & to).sum() / max(to.sum(), 1)); n += int(((s == 0) & ~to).sum())
        out.append(np.mean(of) if open_metric else np.mean(acc)); fo.append(n)
    return out, fo, len(rows)


def levels(fn, lv=(70, 80, 90)):
    vals = []
    for L in lv:
        try:
            vals.append(fn(L))
        except Exception:
            vals.append(None)
    cells = []
    for k in range(4):
        cells.append(" / ".join(pc(v[k]) if v else "–" for v in vals))
    return cells


def row(label, cells, siam="–"):
    return f"| {label} | " + " | ".join(cells) + f" | {siam} |"


def main(R):
    out = [HEAD]
    out.append(row("1 · LV", levels(lambda L: lv_uc1(f"{R}/simbench_lv_doc{L}/uc1_final_lv")[0])))
    out.append(row("realistic register (spare cables, wrong lengths)",
                   levels(lambda L: lv_uc1(f"{R}/simbench_lv_realistic_doc{L}/uc1_final_lv")[0])))
    out.append(row("same + training only from the register (80%)",
                   [pc(v) for v in lv_uc1(f"{R}/simbench_lv_realistic_doc80/uc1_selftrain")[0]]))
    v, _, _ = fw(f"{R}/far_west_doc80/uc1_final_fw", "uc1_*.json", False)
    out.append(row("1 · Far West (80% only)", [pc(x) for x in v]))
    out.append(row("fast mode + wrong lengths",
                   levels(lambda L: fw(f"{R}/far_west_fast_realistic_doc{L}/uc1_final_fw", "uc1_*.json", False)[0])))
    out.append(row("2 · LV", levels(lambda L: assign(f"{R}/simbench_lv_doc{L}/uc2_feeder.csv")[0])))
    out.append(row("no metering at the feeder heads", levels(lambda L: assign(f"{R}/simbench_lv_doc{L}/uc2_feeder_no_head_metering.csv")[0])))
    f = f"{R}/simbench_lv_doc70/uc2_feeder_30pct_unlabelled.csv"
    if os.path.exists(f):
        v, _ = assign(f)
        out.append(row("30% customers unlabelled, as in Liu", [pc(v[0]) + " (known)"] + [pc(x) for x in v[1:]]))
    sia = "–"
    sf = glob.glob(f"{HERE}/../uc3_siamese_method/results/phase_siamese_personalized/res_*.csv")
    if sf:
        sia = pc(pd.concat([pd.read_csv(x) for x in sf]).liu_siamese.mean()) + " at all levels (no register)"
    out.append(row("3 · LV", levels(lambda L: assign(f"{R}/simbench_lv_doc{L}/uc3_phase.csv")[0]), sia))
    out.append(row("wrong cable lengths in the register",
                   levels(lambda L: assign(f"{R}/simbench_lv_realistic_doc{L}/uc3_phase.csv")[0])))
    out.append(row("4 · MV", levels(lambda L: mv(f"{R}/simbench_mv_doc{L}/uc4_final_mv")[0])))
    v, _ = mv(f"{R}/simbench_mv_doc80_meter50/uc4_final_mv")
    out.append(row("meter at half of the MV/LV substations (80%)", [pc(x) for x in v]))
    v, _, _ = fw(f"{R}/far_west_doc80/uc4_final_fw", "res_*.json", True)
    out.append(row("4 · Far West (80% only)", [pc(x) for x in v]))
    v, _, _ = fw(f"{R}/far_west_fast_realistic_doc80/uc4_final_fw", "res_*.json", True)
    out.append(row("fast mode + wrong lengths", [pc(x) for x in v]))
    print("\n".join(out))
    print("\nCells: documentation 70 / 80 / 90% correct. Use case 4: share of truly open elements found.")
    print("Random forest (Liu): in use cases 2 and 3 his clustering.\n")
    # false openings for use case 4
    fo = ["| Use case 4 · false openings | Documented | Variance-Distance Spanning Tree | Random forest (Liu) | Combination |", "|---|---|---|---|---|"]
    for L in (70, 80, 90):
        fo.append(f"| MV {L}% | " + " | ".join(map(str, mv(f"{R}/simbench_mv_doc{L}/uc4_final_mv")[1])) + " |")
    fo.append("| MV 80%, half metered | " + " | ".join(map(str, mv(f"{R}/simbench_mv_doc80_meter50/uc4_final_mv")[1])) + " |")
    fo.append("| Far West 80% | " + " | ".join(map(str, fw(f"{R}/far_west_doc80/uc4_final_fw", "res_*.json", True)[1])) + " |")
    fo.append("| Far West 80%, fast + wrong lengths | " + " | ".join(map(str, fw(f"{R}/far_west_fast_realistic_doc80/uc4_final_fw", "res_*.json", True)[1])) + " |")
    print("\n".join(fo))
    print("\nCases: SimBench LV 72 per level, MV 64, Far West 93 (use case 1) and 62 (use case 4); fast mode: every other case.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=os.path.join(HERE, "results")); a = ap.parse_args()
    main(a.dir)
