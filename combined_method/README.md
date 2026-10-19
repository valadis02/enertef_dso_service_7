# Combined method

Variance-Distance Spanning Tree + random forest (Liu) + physics refinement, for use cases 1, 2 and 4 on SimBench (LV, MV) and on the ACTIVSg2000 Far West transmission area. Use case 3 (phase identification) is in [`../uc3_siamese_method`](../uc3_siamese_method).

- [`RESULTS.md`](RESULTS.md) — results, minimum requirements, conclusions, limits
- [`METHOD.md`](METHOD.md) — full description of the method and of the evaluation protocol
- [`PERSONALIZED_PROFILES.md`](PERSONALIZED_PROFILES.md) — SimBench with personalized customer profiles (use case 2\* and use case 3)

## Layout

| Path | Content |
|---|---|
| `run_all.py` | reproduces every result, stage by stage |
| `make_tables.py` | prints the result table (`--shipped` for the results in `results/`) |
| `paths.py` | all locations; override with `CM_WORK` and `ACTIVSG_DATA` |
| `simbench/` | LV and MV code (use cases 1, 2, 4) |
| `far_west/` | ACTIVSg2000 Far West code (use cases 1 and 4) |
| `far_west/data/` | branch and outage-candidate tables of the Far West area |
| `far_west/configs.json` | the configurations used for the reported results |
| `results/` | per-case results behind the table |

Uses the `topology` package of [`../variance_distance_spanning_tree`](../variance_distance_spanning_tree).

| Use case | Network | Scripts (in order) |
|---|---|---|
| 1 Topology | LV | `sb_sim.py` → `exp_lv_pipeline.py` → `exp_liu_features.py` (features) → `exp_final_lv.py` |
| 2 Customer-feeder | LV | `sb_sim.py` → `exp_final_feeder.py` (uses `exp_feeder.py`) |
| 4 Switches | MV | `mv_sim.py` → `exp_mv.py` → `exp_final_mv.py` |
| 1 Topology | Far West | `build_net.py` → `generate.py`, `gen_multi.py` → `fw_ac.py build` → `exp_final_fw.py` |
| 4 Switches | Far West | `gen_fw7.py` → `exp_final_fw7.py build` → `exp_final_fw7.py run` |

## Setup

```bash
pip install -r requirements.txt
```

SimBench networks and profiles come with the `simbench` package.

**Far West data** (needed only for the Far West stage). From the Texas A&M Electric Grid Test Case Repository (ACTIVSg2000 synthetic grid), download:

1. the MATPOWER case `case_ACTIVSg2000.m` into `far_west/data/`;
2. the ACTIVSg2000 time series, then cut them to the Far West area:

```bash
python far_west/extract_far_west.py "<folder with ACTIVISg2000_load_time_series_MW.csv ...>"
```

This writes `far_west/data/ts/load_MW.csv`, `load_MVAR.csv`, `ren_MW.csv`. `run_all.py` builds `activsg2000_pp.json` from the case file.

## Reproduce

```bash
python make_tables.py --shipped                 # the table from the shipped results (seconds)
python run_all.py --stage simbench --workers 2  # LV + MV, about 3-4 h on 2 cores
python run_all.py --stage far_west --workers 2  # Far West, about 10-12 h on 2 cores
python run_all.py --stage personalized --workers 2 # use case 2*, personalized profiles, about 15 min
python run_all.py --stage tables                # the table from your run
```

Stages skip outputs that already exist, so an interrupted run can be restarted. Everything is written to `work/` (or `CM_WORK`). The settings of the reported results are fixed in `run_all.py`: one month of data (`MONTH=1`), residual on the fluctuations (`TDM=1`), MV switch protocol with 15% of the open points moved (`MVSET=85`), refinement gate 0.05–0.95. All random choices are seeded, so a rerun gives the same numbers.

## Notes

- Results are on simulated data; see the limits in [`RESULTS.md`](RESULTS.md).
- The Far West stage is the slow one: an hourly AC power flow of the full 2000-bus grid for a month, for 156 configurations.
