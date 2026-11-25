"""Run the three extensions (UC1, LV, 80% correct documentation) and compare them with the baseline.

    python run_methods_ext.py --method gnn                              # 4: graph-aware classifier
    python run_methods_ext.py --method gnn --set GNN_GRAPH=0            # 4, control: same classifier, no graph features
    python run_methods_ext.py --method sa                               # 5: simulated annealing / MCMC
    python run_methods_ext.py --method milp --set OBJ=pwl --set TSEL=extreme --set KT=24
    python run_methods_ext.py --method gnn --scenario real              # same, realistic register (spare cables, wrong lengths)
    python run_methods_ext.py --method joint --scenario real            # 7: topology + line lengths jointly
    python run_methods_ext.py --method uc2                              # use case 2 (customer-feeder) with graph features
    python run_methods_ext.py --method mt --set TASKS=both              # FM steps 2-3: one GNN for UC1 + UC2 (also TASKS=uc1, uc2)
    python run_methods_ext.py --method mt --summary                     # table of the multi-task runs
    python run_methods_ext.py --compare                                 # table: every result folder vs combined, paired 95% CI
    python run_methods_ext.py --compare --scenario real

--scenario doc80 (default): ideal register, folder lv_doc80 (made by  run_all_ext.py --stage lv_uc1_80).
--scenario real           : realistic register (REG=real), folder lv_realistic_doc80. The first run there computes the
                            features and the baseline (random forest) itself, then the method.
--set KEY=VALUE           : settings of the methods (GNN_CLF, GNN_GRAPH, STEPS, LAM, KT, TLIM, OBJ, TSEL, ...).
                            They are passed explicitly, so a value left in os.environ from an earlier Spyder run is NOT used.
In Spyder:  runfile(PKG + r"\\run_methods_ext.py", args="--method gnn --set GNN_GRAPH=0", wdir=PKG)
            (set os.environ["CM_WORK"] = r"C:\\enertef_work" first)
"""
import argparse
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import run_all_ext as R

SCRIPT = {"sa": "exp_sa_lv_ext.py", "milp": "exp_milp_lv_ext.py", "gnn": "exp_gnn_lv_ext.py", "joint": "exp_joint_lv_ext.py",
          "uc2": "exp_final_feeder_ext.py", "gnnt": "exp_gnn_torch_lv_ext.py",
          "mt": "exp_multitask_ext.py"}
KNOBS = ["GNN_CLF", "GNN_GRAPH", "GNN_K", "STEPS", "SAMPLES", "T0FRAC", "LAM", "TSTRIDE", "SEED",
         "KT", "TLIM", "GAP", "OBJ", "TSEL", "MODEL", "CALIB", "TAU", "EPOCHS", "HIDDEN", "LAYERS", "LR", "SEEDS", "THREADS", "BATCH", "TASKS", "QUICK"]
SCEN = {"doc80": ("lv_doc80", dict(DOCERR="0.20", CANDS="all")),
        "real": ("lv_realistic_doc80", dict(DOCERR="0.20", CANDS="all", REG="real"))}


def make_env(scenario, sets):
    s, kw = SCEN[scenario]
    e = R.env(s, **kw)
    for k in KNOBS:                       # nothing inherited: only what is given with --set
        e.pop(k, None)
    e.update(sets)
    return s, e


def run_job(args, e, tag):
    """Like run_all_ext.run, but prints the error message of a failed job (Spyder does not show it otherwise)."""
    name = "_".join(a.replace("/", "-") for a in args)[:150] + "_" + tag
    done = os.path.join(e["CM_WORK"], "done", name)
    if os.path.exists(done):
        print("  (already done)", " ".join(args), tag, flush=True); return True
    print("  $", " ".join(args), tag, f"[{os.path.basename(e['CM_WORK'])}]", flush=True)
    r = subprocess.run([sys.executable, "-W", "ignore"] + args, cwd=R.SB_CODE, env=e, capture_output=True, text=True)
    if r.returncode:
        print(f"  ! exit {r.returncode}: {' '.join(args)}\n{r.stderr[-2500:]}", flush=True); return False
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "ok"
    print("    ", last, flush=True)
    os.makedirs(os.path.dirname(done), exist_ok=True); open(done, "w").close()
    return True


def ensure_baseline(scenario, w):
    """Features + baseline forest (MODEL=rf) of the scenario, as run_all_ext.lv_uc1 does. Skipped if already there."""
    s, kw = SCEN[scenario]
    e = R.env(s, **kw)
    for k in KNOBS:
        e.pop(k, None)
    have = glob.glob(os.path.join(e["CM_WORK"], "simbench", "final_lv_month", "res_*.csv"))
    if len(have) >= len(R.LV):
        return
    print(f"[{s}] features and baseline (random forest) first", flush=True)
    R.link_cache(R.lv_profiles(w), s)
    with ThreadPoolExecutor(w) as ex:
        list(ex.map(lambda i: run_job(["exp_lv_pipeline.py", "class_0.5", str(i)], e, "feat"), range(len(R.LV))))
    run_job(["exp_lv_pipeline.py", "class_0.5", "combine"], e, "feat")
    if run_job(["exp_final_lv_ext.py", "0"], e, "rf"):                  # trains the forest once
        with ThreadPoolExecutor(w) as ex:
            list(ex.map(lambda i: run_job(["exp_final_lv_ext.py", str(i)], e, "rf"), range(1, len(R.LV))))


def compare(scenario):
    """Every result folder of the scenario vs the baseline `combined`: mean and paired difference with 95% CI."""
    s, _ = SCEN[scenario]
    d = os.path.join(R.WORK, s, "simbench")
    load = lambda f: pd.concat([pd.read_csv(x) for x in sorted(glob.glob(os.path.join(d, f, "res_*.csv")))])
    base = load("final_lv_month")
    rows = [("baseline", col, base[["case", col]]) for col in
            ("documented", "liu_only", "variance_distance_spanning_tree", "ours_only", "combined_steps_1_3", "combined")]
    skip = {"case", "documented", "liu_only", "variance_distance_spanning_tree", "ours_only", "post_cert", "milp_gap", "scale_dev"}
    for f in sorted(glob.glob(os.path.join(d, "final_lv_month_*"))):
        if not glob.glob(os.path.join(f, "res_*.csv")):
            continue
        X = load(os.path.basename(f))
        for col in X.columns:
            if col not in skip:
                rows.append((os.path.basename(f).replace("final_lv_month_", ""), col, X[["case", col]]))
    out = []
    for folder, col, X in rows:
        M = base[["case", "combined"]].merge(X.rename(columns={col: "v"}), on="case")
        diff = 100 * (M.v - M.combined)
        n = int(diff.notna().sum())
        out.append(dict(folder=folder, column=col, n=n, mean=round(100 * M.v.mean(), 2), diff=round(diff.mean(), 2),
                        ci95=round(1.96 * diff.std(ddof=1) / np.sqrt(n), 2) if n > 1 else np.nan,
                        better=int((diff > 1e-9).sum()), worse=int((diff < -1e-9).sum())))
    T = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print(f"[{s}] percent correct lines; diff = paired difference to `combined` (95% CI treats the cases as independent)")
    print(T.to_string(index=False))
    path = os.path.join(R.WORK, s, f"compare_{scenario}.csv")
    T.to_csv(path, index=False); print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(SCRIPT))
    ap.add_argument("--scenario", choices=list(SCEN), default="doc80")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="", help="part of the 'done' marker; default: built from --set")
    ap.add_argument("--summary", action="store_true", help="the method's own summary")
    ap.add_argument("--compare", action="store_true", help="table of all methods vs combined")
    a = ap.parse_args()
    sets = dict(kv.split("=", 1) for kv in a.set)
    if a.compare:
        compare(a.scenario); return
    if not a.method:
        ap.error("--method is needed (or --compare)")
    s, e = make_env(a.scenario, sets)
    if a.summary:
        r = subprocess.run([sys.executable, "-W", "ignore", SCRIPT[a.method], "summary"], cwd=R.SB_CODE, env=e,
                           capture_output=True, text=True)
        print(r.stdout); print(r.stderr[-2000:]); return
    if a.method == "uc2":                 # use case 2: one job for all feeders (needs simbench), prints its own table
        e["ASSIGN_GATE"] = "1.01"         # as run_all.py: refinement on every customer
        print("  $", SCRIPT["uc2"], f"[{s}]  (several minutes)", flush=True)
        r = subprocess.run([sys.executable, "-W", "ignore", SCRIPT["uc2"]], cwd=R.SB_CODE, env=e, capture_output=True, text=True)
        print(r.stdout[-3000:]); print(r.stderr[-3000:] if r.returncode else "")
        return
    if a.scenario != "doc80":
        ensure_baseline(a.scenario, a.workers)
    tag = a.tag or ("_".join(f"{k}{v}" for k, v in sorted(sets.items())) or "default")
    script = SCRIPT[a.method]
    # feeder 0 alone first (it trains the classifier and writes probs.pkl), then the others in parallel
    if run_job([script, "0", tag], e, a.scenario):
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(lambda i: run_job([script, str(i), tag], e, a.scenario), range(1, len(R.LV))))


if __name__ == "__main__":
    main()                     # no sys.exit: in Spyder it would stop the rest of the cell
