"""Original method vs final method on the six main experiments (register 80% correct).

    python final_table.py --work work --clf xgb_tuned [--clf tabfm] [--out results/final_method]

Reads the per-case outputs that `run_all.py --stage original` and `--stage final --clf ...` write under
the work folder, prints one table and, with --out, writes summary.csv plus one per-case CSV per experiment.
Metrics (as in RESULTS.md):
  DSO 1 LV        share of lines in the correct state (mean over cases)
  DSO 2 / DSO 3   share of customers on the correct feeder / phase
  DSO 4 MV        share of the open points found, and false openings (sum)
  TSO 1           share of branches in the correct state, out-of-service branches found, false "out"
  TSO 4           share of the out-of-service branches found, false "out"
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

TSO = "far_west_doc80_angle01"


def lv(W, d):
    fs = glob.glob(os.path.join(W, "lv_doc80", "simbench", d, "res_*.csv"))
    if len(fs) < 6:
        return None
    x = pd.concat([pd.read_csv(f) for f in fs]).set_index("case")
    return pd.DataFrame({"value": 100 * x.combined})


def assign(W, f, col):
    p = os.path.join(W, "lv_doc80", "simbench", f)
    if not os.path.exists(p):
        return None
    x = pd.read_csv(p).set_index("case")
    return pd.DataFrame({"value": 100 * x[col]})


def mv(W, d):
    fs = glob.glob(os.path.join(W, "mv_doc80", "simbench", d, "res_*.csv"))
    if len(fs) < 7:
        return None
    a = pd.concat([pd.read_csv(f) for f in fs]); a = a[a.method == "combined"].set_index("case")
    return pd.DataFrame({"value": 100 * a.open_found, "false": a.false_open})


def fw(W, d, pat):
    fs = sorted(glob.glob(os.path.join(W, TSO, "far_west", d, pat)))
    if not fs:
        return None
    rows = []
    for f in fs:
        r = json.load(open(f)); t = np.array(r["truth"]); s = np.array(r["combined"]); out = t == 0
        rows.append(dict(case=os.path.basename(f), value=100 * (s == t).mean(), found=int(((s == 0) & out).sum()),
                         n_out=int(out.sum()), false=int(((s == 0) & ~out).sum())))
    return pd.DataFrame(rows).set_index("case")


def fmt(uc, x):
    if x is None:
        return "–"
    if uc in ("DSO 1 LV", "DSO 2 feeder", "DSO 3 phase"):
        return f"{x.value.mean():.1f}"
    if uc == "DSO 4 MV":
        return f"{x.value.mean():.1f} · {int(x['false'].sum())}"
    if uc == "TSO 1":
        return f"{x.value.mean():.2f} · {x.found.sum()}/{x.n_out.sum()} · {x['false'].sum()}"
    return f"{100 * x.found.sum() / x.n_out.sum():.1f} ({x.found.sum()}/{x.n_out.sum()}) · {x['false'].sum()}"


def experiments(W, clf):
    """(use case, original, final) per-case tables; None where the run is missing."""
    return [("DSO 1 LV", lv(W, "final_lv_month"), lv(W, f"final_lv_month_final_{clf}")),
            ("DSO 2 feeder", assign(W, "final_feeder_month.csv", "combined"),
             assign(W, f"final_feeder_month_final_{clf}.csv", "combined_p2")),
            ("DSO 3 phase", assign(W, "final_phase_month.csv", "combined"),
             assign(W, f"final_phase_month_final_{clf}.csv", "combined_p2")),
            ("DSO 4 MV", mv(W, "final_mv85_month"), mv(W, f"final_mv85_month_{clf}_p2")),
            ("TSO 1", fw(W, "final_fw_month", "uc1_*.json"), fw(W, f"final_fw_month_final_{clf}_pass2", "uc1_*.json")),
            ("TSO 4", fw(W, "final_fw7_month", "res_*.json"), fw(W, f"final_fw7_month_final_{clf}_pass2", "res_*.json"))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("CM_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")))
    ap.add_argument("--clf", action="append", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    clfs = a.clf or ["xgb_tuned"]
    rows = []
    for clf in clfs:
        for uc, o, f in experiments(a.work, clf):
            rows.append({"use case": uc, "classifier": clf, "original (RF, 1 pass)": fmt(uc, o), f"final": fmt(uc, f),
                         "cases": 0 if f is None else len(f)})
            if a.out and f is not None:
                d = os.path.join(a.out, clf); os.makedirs(d, exist_ok=True)
                per = f.add_prefix("final_")
                if o is not None:
                    per = o.add_prefix("original_").join(per, how="outer")
                per.to_csv(os.path.join(d, uc.lower().replace(" ", "_") + ".csv"))
    T = pd.DataFrame(rows)
    print("Metrics: DSO 1-3 % correct · DSO 4 % open points found · false · TSO 1 % correct · found · false · "
          "TSO 4 % found (found/all) · false")
    print(T.to_markdown(index=False) if hasattr(T, "to_markdown") else T.to_string(index=False))
    if a.out:
        os.makedirs(a.out, exist_ok=True); T.to_csv(os.path.join(a.out, "summary.csv"), index=False)
