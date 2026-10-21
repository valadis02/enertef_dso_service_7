"""Reproduce every result of the combined method.

    python run_all.py --stage lv --workers 2            # SimBench LV, use cases 1-3, 70/80/90%
    python run_all.py --stage lv_realistic --workers 2  # realistic asset register, use cases 1 and 3
    python run_all.py --stage selftrain                 # use case 1, training only from the register (80%)
    python run_all.py --stage mv --workers 2            # SimBench MV, use case 4, 70/80/90% + half metered
    python run_all.py --stage far_west --workers 2      # Far West, use cases 1 and 4 (80%), full mode
    python run_all.py --stage far_west_fast --workers 2 # Far West, fast mode + wrong lengths, 70/80/90%
    python run_all.py --stage collect                   # copy the per-case results to work/results/
    python run_all.py --stage tables                    # print the tables from work/results/

Each scenario runs in its own folder under work/ (or CM_WORK). Steps whose output
exists are skipped, so an interrupted run can be started again. All random choices
are seeded: a rerun gives the shipped numbers. SimBench uses the personalized
customer profiles (PERSONALIZED_PROFILES.md).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SB_CODE, FW_CODE = os.path.join(HERE, "simbench"), os.path.join(HERE, "far_west")
WORK = os.environ.get("CM_WORK", os.path.join(HERE, "work"))
FW_DATA = os.environ.get("ACTIVSG_DATA", os.path.join(FW_CODE, "data"))
LV = ["1-LV-rural1--0-sw", "1-LV-semiurb4--0-sw", "1-LV-urban6--0-sw",
      "1-LV-rural2--0-sw", "1-LV-rural3--0-sw", "1-LV-semiurb5--0-sw"]
MV = ["1-MV-rural--0-sw", "1-MV-semiurb--0-sw", "1-MV-urban--0-sw", "1-MV-comm--0-sw"]
LEVELS = {70: "0.30", 80: "0.20", 90: "0.10"}           # documentation correct -> DOCERR
BASE = dict(os.environ, MONTH="1", TDM="1", SUFFIX="_month", ONLYJ0="1", GLO="0.05", GHI="0.95",
            OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", ACTIVSG_DATA=FW_DATA)


def env(scenario, **kw):
    e = dict(BASE, CM_WORK=os.path.join(WORK, scenario)); e.update({k: str(v) for k, v in kw.items()})
    return e


def run(args, cwd, e):
    out = os.path.join(e["CM_WORK"], "done", "_".join(a.replace("/", "-") for a in args)[:150])
    if os.path.exists(out):
        return
    print("  $", " ".join(args), f"[{os.path.basename(e['CM_WORK'])}]", flush=True)
    r = subprocess.run([sys.executable, "-W", "ignore"] + args, cwd=cwd, env=e)
    if r.returncode:
        print(f"  ! exit {r.returncode}: {' '.join(args)}", flush=True); return
    os.makedirs(os.path.dirname(out), exist_ok=True); open(out, "w").close()


def parallel(jobs, w):
    with ThreadPoolExecutor(w) as ex:
        list(ex.map(lambda j: run(*j), jobs))


def lv_profiles(w):
    """One month of LV power flow with personalized profiles, shared by every LV scenario."""
    e = env("lv_profiles", INDIV="1")
    parallel([(["sb_sim.py", c], SB_CODE, e) for c in LV], w)
    return os.path.join(e["CM_WORK"], "simbench", "sb_cache_month")


def link_cache(cache, scenario):
    d = os.path.join(WORK, scenario, "simbench"); os.makedirs(d, exist_ok=True)
    if not os.path.exists(os.path.join(d, "sb_cache_month")):
        os.symlink(cache, os.path.join(d, "sb_cache_month"))


def lv_uc1(e, w):
    parallel([(["exp_lv_pipeline.py", "class_0.5", str(i)], SB_CODE, e) for i in range(len(LV))], w)
    run(["exp_lv_pipeline.py", "class_0.5", "combine"], SB_CODE, e)
    run(["exp_final_lv.py", "0"], SB_CODE, e)                      # trains the forests once
    parallel([(["exp_final_lv.py", str(i)], SB_CODE, e) for i in range(1, len(LV))], w)


def lv(w):
    cache = lv_profiles(w)
    for L, err in LEVELS.items():
        s = f"lv_doc{L}"; link_cache(cache, s)
        print(f"[SimBench LV {L}%] use case 1 (no trusted line), 2, 3", flush=True)
        lv_uc1(env(s, DOCERR=err, CANDS="all"), w)
        run(["exp_final_feeder.py"], SB_CODE, env(s, DOCERR=err, ASSIGN_GATE="1.01"))
        run(["exp_final_phase.py"], SB_CODE, env(s, DOCERR=err))
        run(["exp_final_feeder.py"], SB_CODE, env(s, DOCERR=err, NOHEAD="1"))
    s = "lv_doc70_missing"; link_cache(cache, s)
    print("[SimBench LV 70%] use case 2 with 30% of the labels missing", flush=True)
    run(["exp_final_feeder.py"], SB_CODE, env(s, DOCERR="0.30", DOCMODE="missing", ASSIGN_GATE="1.01"))


def lv_realistic(w):
    cache = lv_profiles(w)
    for L, err in LEVELS.items():
        s = f"lv_realistic_doc{L}"; link_cache(cache, s)
        print(f"[SimBench LV {L}%] realistic register: use cases 1 and 3", flush=True)
        lv_uc1(env(s, DOCERR=err, CANDS="all", REG="real"), w)
        run(["exp_final_phase.py"], SB_CODE, env(s, DOCERR=err, REG="real"))


def selftrain(w):
    e = env("lv_realistic_doc80", DOCERR="0.20", CANDS="all", REG="real")
    print("[SimBench LV 80%] use case 1, training only from the register", flush=True)
    parallel([(["exp_lv_selftrain.py", str(i)], SB_CODE, e) for i in range(len(LV))], w)


def mv_level(L, err, w, meter=None):
    s = f"mv_doc{L}" + ("_meter50" if meter else "")
    e = env(s, DOCERR=err, MVSET="85", **({"MVMETER": meter} if meter else {}))
    cache = os.path.join(e["CM_WORK"], "simbench", "mv_cache85_month"); os.makedirs(cache, exist_ok=True)
    for i, code in enumerate(MV):
        # the simulated configurations depend only on how many open points are moved; share them
        m = subprocess.run([sys.executable, "-W", "ignore", "-c",
                            f"import simbench as sb, mv_sim as M, os; net=sb.get_simbench_net(M.MV_NETS[{i}]); "
                            f"n=int((M.mv_model(net)[3]==0).sum()); print(max(1, round(float(os.environ['DOCERR'])*n)))"],
                           cwd=SB_CODE, env=e, capture_output=True, text=True).stdout.strip().splitlines()[-1]
        se = env(f"mv_sim_m{m}", DOCERR=err, MVSET="85", INDIV="1", NETS=i)
        run(["mv_sim.py", "100000"], SB_CODE, se)
        src = os.path.join(se["CM_WORK"], "simbench", "mv_cache85_month")
        for f in os.listdir(src):
            if f.startswith(code) and not os.path.exists(os.path.join(cache, f)):
                os.symlink(os.path.join(src, f), os.path.join(cache, f))
    parallel([(["exp_mv.py", "class_0.5", c, f"{lo}-{lo + 4}"], SB_CODE, e) for c in MV for lo in (0, 4, 8, 12)], w)
    run(["-c", "import sys; sys.argv=['x','summary']; import exp_final_mv as E; E.data()"], SB_CODE, e)
    parallel([(["exp_final_mv.py", str(i)], SB_CODE, e) for i in (0, 1, 3)] +
             [(["exp_final_mv.py", "2", f"{lo}-{lo + 4}"], SB_CODE, e) for lo in (0, 4, 8, 12)], w)


def mv(w):
    for L, err in LEVELS.items():
        print(f"[SimBench MV {L}%] use case 4", flush=True)
        mv_level(L, err, w)
    print("[SimBench MV 80%] use case 4, half of the MV/LV substations metered", flush=True)
    mv_level(80, "0.20", w, meter="0.5")


def fw_inputs(w):
    need = ["case_ACTIVSg2000.m", "ts/load_MW.csv", "ts/load_MVAR.csv", "ts/ren_MW.csv"]
    miss = [f for f in need if not os.path.exists(os.path.join(FW_DATA, f))]
    if miss:
        sys.exit(f"Far West inputs missing in {FW_DATA}: {miss}. See README (Far West data).")
    e = env("fw_scenarios", FWSTEP=1, FWHOURS=720, FWK=5)
    q = os.path.join(e["CM_WORK"], "far_west")
    e.update(FWOUT_MULTI=os.path.join(q, "scen_multi_q_month"), FWOUT7=os.path.join(q, "scen_fw7_month"))
    if not os.path.exists(os.path.join(FW_DATA, "activsg2000_pp.json")):
        run(["build_net.py"], FW_CODE, e)
    cfg = json.load(open(os.path.join(FW_CODE, "configs.json")))
    os.makedirs(os.path.join(q, "scen_q_month"), exist_ok=True)
    print("[Far West] one month of hourly AC power flow per configuration (many hours)", flush=True)
    parallel([(["generate.py", "4000", "720", str(s), str(s + 1), os.path.join(q, "scen_q_month")], FW_CODE, e)
              for s in cfg["singles"]] +
             [(["gen_multi.py", str(i), str(i + 1)], FW_CODE, e) for i in cfg["multi"]] +
             [(["gen_fw7.py", str(i), str(i + 1)], FW_CODE, e) for i in cfg["seven_out"]], w)
    return q, cfg


def fw_scenario(s, q, cfg, w, uc4, **kw):
    e = env(s, FWSTEP=1, FWHOURS=720, FWK=5, **kw)
    d = os.path.join(e["CM_WORK"], "far_west"); os.makedirs(d, exist_ok=True)
    for name in ("scen_q_month", "scen_multi_q_month", "scen_fw7_month"):
        if not os.path.exists(os.path.join(d, name)):
            os.symlink(os.path.join(q, name), os.path.join(d, name))
    n1 = 3 * (len(cfg["singles"]) + len(cfg["multi"]))
    step = max(1, n1 // (4 * w))
    parallel([(["fw_ac.py", "build", "uc1", str(a), str(min(a + step, n1))], FW_CODE, e) for a in range(0, n1, step)], w)
    run(["exp_final_fw.py", "uc1", "run", "0", "0"], FW_CODE, e)  # trains the forests once
    parallel([(["exp_final_fw.py", "uc1", "run", str(a), str(a + 6)], FW_CODE, e) for a in range(0, 96, 6)], w)
    if uc4:
        n4 = len(cfg["seven_out"]) + 5; step = max(1, n4 // (4 * w))
        parallel([(["exp_final_fw7.py", "build", str(a), str(min(a + step, n4))], FW_CODE, e) for a in range(0, n4, step)], w)
        run(["exp_final_fw7.py", "run", "0", "0"], FW_CODE, e)
        parallel([(["exp_final_fw7.py", "run", str(a), str(a + 6)], FW_CODE, e) for a in range(0, 66, 6)], w)


def far_west(w):
    q, cfg = fw_inputs(w)
    print("[Far West 80%] use cases 1 and 4, full mode", flush=True)
    fw_scenario("far_west_doc80", q, cfg, w, True, DOCERR="0.20")


def far_west_fast(w):
    q, cfg = fw_inputs(w)
    for L, err in LEVELS.items():
        print(f"[Far West {L}%] fast mode (every 2nd hour, every other case), wrong lengths", flush=True)
        fw_scenario(f"far_west_fast_realistic_doc{L}", q, cfg, w, L == 80, DOCERR=err, REG="real", FWSUB=2, FWHALF=1)


def collect():
    R = os.path.join(WORK, "results")
    pairs = []
    for L in LEVELS:
        pairs += [(f"lv_doc{L}/simbench/final_lv_month", f"simbench_lv_doc{L}/uc1_final_lv"),
                  (f"lv_doc{L}/simbench/final_feeder_month.csv", f"simbench_lv_doc{L}/uc2_feeder.csv"),
                  (f"lv_doc{L}/simbench/final_phase_month.csv", f"simbench_lv_doc{L}/uc3_phase.csv"),
                  (f"lv_doc{L}/simbench/final_feeder_nohead_month.csv", f"simbench_lv_doc{L}/uc2_feeder_no_head_metering.csv"),
                  (f"lv_realistic_doc{L}/simbench/final_lv_month", f"simbench_lv_realistic_doc{L}/uc1_final_lv"),
                  (f"lv_realistic_doc{L}/simbench/final_phase_month.csv", f"simbench_lv_realistic_doc{L}/uc3_phase.csv"),
                  (f"mv_doc{L}/simbench/final_mv85_month", f"simbench_mv_doc{L}/uc4_final_mv"),
                  (f"far_west_fast_realistic_doc{L}/far_west/final_fw_month", f"far_west_fast_realistic_doc{L}/uc1_final_fw")]
    pairs += [("lv_doc70_missing/simbench/final_feeder_missing_month.csv", "simbench_lv_doc70/uc2_feeder_30pct_unlabelled.csv"),
              ("lv_realistic_doc80/simbench/final_lv_month_selftrain", "simbench_lv_realistic_doc80/uc1_selftrain"),
              ("mv_doc80_meter50/simbench/final_mv85_month", "simbench_mv_doc80_meter50/uc4_final_mv"),
              ("far_west_doc80/far_west/final_fw_month", "far_west_doc80/uc1_final_fw"),
              ("far_west_doc80/far_west/final_fw7_month", "far_west_doc80/uc4_final_fw"),
              ("far_west_fast_realistic_doc80/far_west/final_fw7_month", "far_west_fast_realistic_doc80/uc4_final_fw")]
    for a, b in pairs:
        src, dst = os.path.join(WORK, a), os.path.join(R, b)
        if not os.path.exists(src):
            print("  missing:", a); continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.pkl"))
        else:
            shutil.copy(src, dst)
    print("collected in", R)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["lv", "lv_realistic", "selftrain", "mv", "far_west",
                                                       "far_west_fast", "collect", "tables"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.stage == "collect":
        collect()
    elif a.stage == "tables":
        subprocess.run([sys.executable, os.path.join(HERE, "make_tables.py"), "--dir", os.path.join(WORK, "results")])
    else:
        globals()[a.stage](a.workers)
