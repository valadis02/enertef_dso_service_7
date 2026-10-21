"""Reproduce the use case 3 result: siamese method of Dong Liu on SimBench (personalized).

    python run.py --workers 2

1. one month of SimBench LV power flow with personalized customer profiles
   (combined_method/simbench/sb_sim.py with INDIV=1, profiles from indiv_profiles.py);
2. the siamese method on the 72 phase cases (6 feeders x 12 random phase draws);
3. the mean share of customers on the right phase.

Work files go to work/ (or CM_WORK). Takes about 2 h on 2 cores.
"""
import argparse
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SB_CODE = os.path.join(HERE, "..", "combined_method", "simbench")
LV = ["1-LV-rural1--0-sw", "1-LV-semiurb4--0-sw", "1-LV-urban6--0-sw",
      "1-LV-rural2--0-sw", "1-LV-rural3--0-sw", "1-LV-semiurb5--0-sw"]
WORK = os.environ.get("CM_WORK", os.path.join(HERE, "work"))
ENV = dict(os.environ, CM_WORK=WORK, MONTH="1", INDIV="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


def run(args, cwd):
    print("  $", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-W", "ignore"] + args, cwd=cwd, env=ENV, check=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=2); a = ap.parse_args()
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda c: run(["sb_sim.py", c], SB_CODE), LV))
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda i: run(["siamese.py", str(i), "5"], HERE), [2, 0, 1, 3, 4, 5]))
    import pandas as pd
    R = pd.concat([pd.read_csv(f) for f in glob.glob(f"{WORK}/simbench/phase_siamese_month/res_*.csv")])
    print(f"use case 3, SimBench (personalized), {len(R)} cases: documented {100 * R.documented.mean():.1f}%, "
          f"siamese {100 * R.liu_siamese.mean():.1f}%")
