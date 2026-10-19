"""Reproduce every result of the combined method.

    python run_all.py --stage all --workers 2
    python run_all.py --stage simbench          # LV + MV only (a few hours)
    python run_all.py --stage far_west          # needs the ACTIVSg2000 inputs, see README
    python run_all.py --stage personalized      # use case 2*, personalized customer profiles
    python run_all.py --stage tables            # print the result table from work/

Each stage skips work whose output already exists, so an interrupted run can
simply be started again. Settings of the reported results (one month of data,
voltage magnitude Class 0.5, documentation 85% correct) are fixed below.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import SB_CODE, FW_CODE, SB_WORK, FW_WORK, FW_DATA  # noqa: E402

LV = ["1-LV-rural1--0-sw", "1-LV-semiurb4--0-sw", "1-LV-urban6--0-sw",
      "1-LV-rural2--0-sw", "1-LV-rural3--0-sw", "1-LV-semiurb5--0-sw"]
MV = ["1-MV-rural--0-sw", "1-MV-semiurb--0-sw", "1-MV-urban--0-sw", "1-MV-comm--0-sw"]

# environment of the reported results
ENV = dict(os.environ, MONTH="1", TDM="1", MVSET="85", GLO="0.05", GHI="0.95", SUFFIX="_month",
           ONLYJ0="1", FWSTEP="1", FWHOURS="720", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           FWOUT_MULTI=os.path.join(FW_WORK, "scen_multi_q_month"),
           FWOUT7=os.path.join(FW_WORK, "scen_fw7_month"))


def run(args, cwd, env=None):
    print("  $", " ".join(args), flush=True)
    r = subprocess.run([sys.executable, "-W", "ignore"] + args, cwd=cwd, env=env or ENV)
    if r.returncode:
        print(f"  ! exit {r.returncode}: {' '.join(args)}", flush=True)


def parallel(jobs, workers):
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda j: run(*j), jobs))


def simbench(w):
    print("[SimBench LV] one month of power flow", flush=True)
    parallel([(["sb_sim.py", c], SB_CODE) for c in LV], w)
    print("[SimBench LV] features (use case 1)", flush=True)
    parallel([(["exp_lv_pipeline.py", "class_0.5", str(i)], SB_CODE) for i in range(len(LV))], w)
    run(["exp_lv_pipeline.py", "class_0.5", "combine"], SB_CODE)
    print("[SimBench LV] use case 1: variance_distance_spanning_tree / random forest / combination", flush=True)
    run(["exp_final_lv.py", "0"], SB_CODE)                    # trains the forests once
    parallel([(["exp_final_lv.py", str(i)], SB_CODE) for i in range(1, len(LV))], w)
    print("[SimBench LV] use case 2", flush=True)
    run(["exp_final_feeder.py"], SB_CODE)
    print("[SimBench MV] one month of power flow, 16 switching configurations per network", flush=True)
    parallel([(["mv_sim.py", "100000"], SB_CODE, dict(ENV, NETS=str(i))) for i in range(len(MV))], w)
    print("[SimBench MV] features", flush=True)
    parallel([(["exp_mv.py", "class_0.5", c, f"{lo}-{lo + 4}"], SB_CODE) for c in MV for lo in (0, 4, 8, 12)], w)
    print("[SimBench MV] use case 4", flush=True)
    run(["-c", "import sys; sys.argv=['x','summary']; import exp_final_mv as E; E.data()"], SB_CODE)
    parallel([(["exp_final_mv.py", str(i)], SB_CODE) for i in (0, 1, 3)] +
             [(["exp_final_mv.py", "2", f"{lo}-{lo + 4}"], SB_CODE) for lo in (0, 4, 8, 12)], w)


def far_west(w):
    need = ["case_ACTIVSg2000.m", "ts/load_MW.csv", "ts/load_MVAR.csv", "ts/ren_MW.csv"]
    miss = [f for f in need if not os.path.exists(os.path.join(FW_DATA, f))]
    if miss:
        sys.exit(f"Far West inputs missing in {FW_DATA}: {miss}. See README (Far West data).")
    if not os.path.exists(os.path.join(FW_DATA, "activsg2000_pp.json")):
        run(["build_net.py"], FW_CODE)
    cfg = json.load(open(os.path.join(FW_CODE, "configs.json")))
    q = os.path.join(FW_WORK, "scen_q_month"); os.makedirs(q, exist_ok=True)
    print("[Far West] one month of hourly AC power flow per configuration (several hours)", flush=True)
    jobs = [(["generate.py", "4000", "720", str(s), str(s + 1), q], FW_CODE) for s in cfg["singles"]
            if not os.path.exists(f"{q}/scen_{s:03d}.npz")]
    jobs += [(["gen_multi.py", str(i), str(i + 1)], FW_CODE) for i in cfg["multi"]]
    jobs += [(["gen_fw7.py", str(i), str(i + 1)], FW_CODE) for i in cfg["seven_out"]]
    parallel(jobs, w)
    print("[Far West] features", flush=True)
    n1 = 3 * (len(cfg["singles"]) + len(cfg["multi"]))         # case list incl. 3 draws; only draw 0 is built
    step = max(1, n1 // (4 * w))
    parallel([(["fw_ac.py", "build", "uc1", str(a), str(min(a + step, n1))], FW_CODE) for a in range(0, n1, step)], w)
    n4 = len(cfg["seven_out"]) + 5
    step = max(1, n4 // (4 * w))
    parallel([(["exp_final_fw7.py", "build", str(a), str(min(a + step, n4))], FW_CODE) for a in range(0, n4, step)], w)
    print("[Far West] use cases 1 and 4", flush=True)
    run(["exp_final_fw.py", "uc1", "run", "0", "0"], FW_CODE)  # trains the forests once
    run(["exp_final_fw7.py", "run", "0", "0"], FW_CODE)
    parallel([(["exp_final_fw.py", "uc1", "run", str(a), str(a + 6)], FW_CODE) for a in range(0, 96, 6)] +
             [(["exp_final_fw7.py", "run", str(a), str(a + 6)], FW_CODE) for a in range(0, 66, 6)], w)


def personalized(w):
    """SimBench LV with personalized customer profiles (PERSONALIZED_PROFILES.md):
    use case 2 variant. Use case 3 (siamese) is reproduced by ../uc3_siamese_method/run.py."""
    work = os.environ.get("CM_WORK_PERSONALIZED", os.path.join(HERE, "work_personalized"))
    env = dict(ENV, CM_WORK=work, INDIV="1")
    print("[SimBench LV, personalized profiles] one month of power flow", flush=True)
    parallel([(["sb_sim.py", c], SB_CODE, env) for c in LV], w)
    print("[SimBench LV, personalized profiles] use case 2, refinement on every customer", flush=True)
    run(["exp_final_feeder.py"], SB_CODE, dict(env, ASSIGN_GATE="1.01"))


def tables():
    run(["make_tables.py"], HERE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "simbench", "far_west", "personalized", "tables"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.stage in ("all", "simbench"):
        simbench(a.workers)
    if a.stage in ("all", "far_west"):
        far_west(a.workers)
    if a.stage in ("all", "personalized"):
        personalized(a.workers)
    tables()
