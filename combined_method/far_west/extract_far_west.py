"""Keep only the Far West (area 1, buses 1001-1091) columns of the ACTIVSg2000 time series.

Usage:
    python extract_far_west.py "C:\\Users\\balt5\\Downloads\\ACTIVSg_Time_Series\\Time Series"

Writes three small CSVs to <ACTIVSG_DATA>/ts/: load_MW.csv, load_MVAR.csv, ren_MW.csv.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import TOPO, SB_CODE, SB_WORK, FW_CODE, FW_WORK, FW_DATA  # noqa: E402
import re
import sys
from pathlib import Path

import pandas as pd

FILES = {                                     # written to <ACTIVSG_DATA>/ts/
    "ACTIVISg2000_load_time_series_MW.csv": "load_MW.csv",
    "ACTIVISg2000_load_time_series_MVAR.csv": "load_MVAR.csv",
    "ACTIVISg2000_renewable_time_series_MW.csv": "ren_MW.csv",
}
FAR_WEST = range(1001, 1092)


def keep(col):
    if col in ("Date", "Time"):
        return True
    m = re.match(r"(Bus|Gen) (\d+) ", col)
    return bool(m) and int(m.group(2)) in FAR_WEST


def main(folder):
    folder = Path(folder)
    out_dir = Path(FW_DATA) / "ts"; out_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in FILES.items():
        # row 0 is a PowerWorld banner (and the Solar/Wind labels); row 1 is the header
        df = pd.read_csv(folder / src, header=1, low_memory=False)
        df = df[[c for c in df.columns if keep(c)]]
        df.to_csv(out_dir / dst, index=False)
        print(f"{dst}: {df.shape[0]} rows x {df.shape[1] - 2} columns")


if __name__ == "__main__":
    main(sys.argv[1])
