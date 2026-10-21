# Combined method

Builds on the random forest of Dong Liu (TU Delft) and adds the physics of the network: Variance-Distance Spanning Tree, the power-flow residual, radial constraints and a physics refinement. The same four steps for all four D2.2 use cases, on SimBench (LV, MV) and on the ACTIVSg2000 Far West transmission area. Use case 3 also has the siamese method of Dong Liu in [`../uc3_siamese_method`](../uc3_siamese_method).

- [`RESULTS.md`](RESULTS.md) — all results (70 / 80 / 90% correct documentation, realism checks), minimum requirements, limits
- [`METHOD.md`](METHOD.md) — full description of the method and of the evaluation protocol
- [`PERSONALIZED_PROFILES.md`](PERSONALIZED_PROFILES.md) — SimBench with personalized customer profiles, used everywhere

## Layout

| Path | Content |
|---|---|
| `run_all.py` | reproduces every result, scenario by scenario |
| `make_tables.py` | prints all result tables from `results/` (or `--dir work/results`) |
| `paths.py` | all locations; override with `CM_WORK` and `ACTIVSG_DATA` |
| `simbench/` | LV and MV code |
| `far_west/` | ACTIVSg2000 Far West code |
| `far_west/data/` | branch and outage-candidate tables of the Far West area |
| `far_west/configs.json` | the configurations used for the reported results |
| `results/` | per-case results behind every table, one folder per scenario |

Uses the `topology` package of [`../variance_distance_spanning_tree`](../variance_distance_spanning_tree).

| Use case | Network | Scripts (in order) |
|---|---|---|
| 1 Topology | LV | `sb_sim.py` → `exp_lv_pipeline.py` → `exp_final_lv.py` (`exp_lv_selftrain.py`: training only from the register) |
| 2 Customer-feeder | LV | `sb_sim.py` → `exp_final_feeder.py` (Liu's clustering in `liu_cluster.py`) |
| 3 Phase | LV | `sb_sim.py` → `exp_final_phase.py` (cases in `phase_cases.py`, three-phase model `phase_sim.py`) |
| 4 Switches | MV | `mv_sim.py` → `exp_mv.py` → `exp_final_mv.py` (`exp_mv_liu_switchbar.py`: Liu's original classifier) |
| 1 Topology | Far West | `build_net.py` → `generate.py`, `gen_multi.py` → `fw_ac.py build` → `exp_final_fw.py` |
| 4 Switches | Far West | `gen_fw7.py` → `exp_final_fw7.py build` → `exp_final_fw7.py run` |

Settings of a run are environment variables, set by `run_all.py`: `DOCERR` (0.30 / 0.20 / 0.10 = 70 / 80 / 90% correct), `CANDS=all` (no trusted line), `REG=real` (spare cables, wrong lengths), `MVMETER` (share of metered MV/LV substations), `NOHEAD=1`, `DOCMODE=missing`, `FWSUB`/`FWHALF` (Far West fast mode).

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
python make_tables.py                                  # all tables from the shipped results (seconds)
python run_all.py --stage lv --workers 2               # SimBench LV, use cases 1-3, 70/80/90% (a few hours)
python run_all.py --stage lv_realistic --workers 2     # realistic register
python run_all.py --stage selftrain --workers 2        # training only from the register (after lv_realistic)
python run_all.py --stage mv --workers 2               # SimBench MV, 70/80/90% + half metered (several hours)
python run_all.py --stage far_west --workers 2         # Far West full mode, 80% (a day on 2 cores)
python run_all.py --stage far_west_fast --workers 2    # Far West fast mode, 70/80/90%
python run_all.py --stage collect && python run_all.py --stage tables
```

Steps whose output exists are skipped, so an interrupted run can be restarted. Everything is written to `work/` (or `CM_WORK`). All random choices are seeded, so a rerun gives the shipped numbers.

## Notes

- Results are on simulated data; see the limits in [`RESULTS.md`](RESULTS.md).
- The Far West stage is the slow one: an hourly AC power flow of the full 2000-bus grid for a month, for 156 configurations.
