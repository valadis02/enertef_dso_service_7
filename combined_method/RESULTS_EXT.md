# Extensions of the combined method (Dong Liu, TU Delft)

All changes are in **new files** (`*_ext.py`); the original code is untouched. Same data, protocol and metric as the
original: SimBench LV with personalized profiles, 6 feeders × 12 cases = 72 cases, |V| class 0.5 without phase angle,
register 80% correct, training leave-one-feeder-out, share of correct lines (UC1) / customers (UC2).
Paired differences are against the original combination; the 95% CI treats the 72 cases as independent
(cases of one feeder are correlated, so the real uncertainty is larger).

## New files

| File | Content |
|---|---|
| `run_all_ext.py` | copy of `run_all.py`: classifier of step 2 selectable (`MODEL`, `CALIB`), stage `lv_uc1_80`, Windows-safe cache link |
| `run_methods_ext.py` | runs every extension below, `--scenario doc80/real`, `--set KEY=VALUE`, `--compare` (table with paired CIs) |
| `simbench/exp_final_lv_ext.py` | UC1 with other classifiers in step 2: gradient boosting, XGBoost, LightGBM, logistic regression, ± isotonic calibration |
| `simbench/exp_gnn_lv_ext.py` | UC1, graph features: 2 rounds of fixed message passing on the candidate graph + rank among competing lines, gradient boosting |
| `simbench/exp_gnn_torch_lv_ext.py` | UC1, **trainable** edge-aware graph neural network (torch only) |
| `simbench/exp_sa_lv_ext.py` | UC1, simulated annealing / MCMC over radial trees with J as likelihood (replaces step 4) |
| `simbench/exp_milp_lv_ext.py` | UC1, single-shot MILP: binary line variables, linearised power flow, radiality (scipy / HiGHS) |
| `simbench/exp_joint_lv_ext.py` | UC1, joint estimation of topology and line length factors |
| `simbench/exp_final_feeder_ext.py` | UC2 with graph features; writes the UC2 cache for the multi-task model |
| `simbench/exp_multitask_ext.py` | one graph format and one GNN for UC1 + UC2 (first step towards a foundation model) |

Run, e.g.: `python run_all_ext.py --stage lv_uc1_80`, then `python run_methods_ext.py --method gnnt` and
`python run_methods_ext.py --compare` (details in the head of each file).

## UC1, ideal register (80% correct)

| Method | Correct lines (%) | vs. combination |
|---|---|---|
| Documented register | 79.78 | |
| Random forest (Liu) alone | 80.22 | |
| Variance-Distance Spanning Tree | 94.79 | |
| Physics only (VDST + refinement) | 95.89 | |
| **Combination (original, random forest)** | **95.23** | |
| Other classifiers in step 2 (9 variants) | 94.81 – 95.27 | none clearly better |
| Annealing, pure physics / with forest prior | 94.60 / 95.61 | −0.63 / +0.38 ± 0.53 |
| MILP, pure physics / with forest gate | 89.73\* / 93.74\* | not solved to optimality on 5 of 6 feeders in 120 s |
| Joint topology + line lengths (physics only) | 95.67 | vs. 95.89 with fixed lengths |
| Graph features + gradient boosting | 97.00 | +1.77 ± 0.57 (47 better / 2 worse) |
| — control: same classifier, no graph features | 94.81 | −0.42 ± 0.28 |
| **Trainable GNN** | **98.22** | **+2.99 ± 0.56 (63 / 2)** |

\* MILP: mean over the cases with a solution (46 and 54 of 72).

## UC1, realistic register (20% spare cables, 20% of the lengths wrong by 10–30%)

| Method | Correct lines (%) | vs. combination |
|---|---|---|
| Documented register | 79.78 | |
| Random forest (Liu) alone | 69.54 | |
| Variance-Distance Spanning Tree | 83.46 | |
| Physics only | 84.54 | |
| **Combination (original)** | **89.67** | |
| Joint topology + line lengths (physics only) | 84.23 | −0.32 ± 0.36 vs. fixed lengths (length factors stay near 1) |
| Graph features + gradient boosting | 93.00 | +3.32 ± 0.69 (59 / 3) |
| — control: no graph features | 89.36 | −0.31 ± 0.54 |
| **Trainable GNN** | **94.31** | **+4.63 ± 1.07 (60 / 5)**; worse than the combination on the smallest feeder (rural1) |

## UC2 (customer-feeder), ideal register

| Method | Correct customers (%) |
|---|---|
| Documented | 79.12 |
| Combination (original; reproduces 94.2 of `RESULTS.md`) | 94.20 |
| Graph features + gradient boosting | 93.49 (before refinement 86.10, control without graph 81.65) |

The power-balance refinement dominates the final UC2 result; the graph features help the classifier (+4.5 points before refinement).

## One model for UC1 + UC2 (feasibility check: 40 epochs, 1 seed)

| | single-task GNN | multi-task GNN (UC1 + UC2) | multi- vs. single-task |
|---|---|---|---|
| UC1 (%) | 97.82 | 97.03 | −0.78 ± 0.94 (10 better / 34 worse) |
| UC2 (%) | 94.38 | 94.49 | +0.11 ± 1.38 |

One shared model learns both tasks; UC2 is unaffected, UC1 stays 1.8 points above the combination but slightly below
its single-task model. No positive transfer yet at this data size (60 training graphs per task).

## Limitations and next steps

Simulated data only, 6 feeders, 80% documentation, few seeds. UC3 (phase) and UC4 (MV switches) fit the same graph
format (customer → phase nodes; MV edges) but need their simulations (`phase_sim.py`, `mv_sim.py`) first.
Next: larger and more varied training data (more networks, error types, meter settings), loss weighting for the
multi-task model, UC3/UC4, then real measurements.
