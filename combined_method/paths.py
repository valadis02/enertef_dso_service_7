"""All locations used by the combined method. Override with environment variables.

CM_WORK       working directory for simulated data, features and results
              (default: combined_method/work)
ACTIVSG_DATA  folder with the ACTIVSg2000 inputs: case_ACTIVSg2000.m,
              activsg2000_pp.json, branches.csv, fw_outage_candidates.csv and ts/
              (default: combined_method/far_west/data)
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
TOPO = os.path.join(REPO, "variance_distance_spanning_tree")          # the original `topology` package
SB_CODE = os.path.join(ROOT, "simbench")
FW_CODE = os.path.join(ROOT, "far_west")
WORK = os.environ.get("CM_WORK", os.path.join(ROOT, "work"))
SB_WORK = os.path.join(WORK, "simbench")
FW_WORK = os.path.join(WORK, "far_west")
FW_DATA = os.environ.get("ACTIVSG_DATA", os.path.join(FW_CODE, "data"))
for _d in (SB_WORK, FW_WORK):
    os.makedirs(_d, exist_ok=True)
