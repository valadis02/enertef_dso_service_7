# Combined method — full description

One pipeline, the same four steps for all four use cases on every network (for use case 3 the siamese method of Dong Liu, in [`../uc3_siamese_method`](../uc3_siamese_method), is reported next to it):

1. physics features → 2. random forest → 3. decision under physical constraints → 4. physics refinement.

Origin of each part: Variance-Distance Spanning Tree, the power-flow residual, the constraints and the refinement are ours; the random forest classifier and the voltage-correlation features come from Liu et al., *A data-driven approach for topology correction in low voltage networks with DERs* (arXiv 2506.20238). The combination builds on his forest; its settings and his clustering settings follow his public code (github.com/distributionnetworksTUDelft/Data-driven-approach-for-topology-correction-in-LVDNs-).

## Notation and inputs

- Network: buses *i*, elements *e* (lines, cables, transformers, switches) with impedance *z_e = r_e + j x_e* from the asset register.
- Documentation: state *d_e* ∈ {0, 1} per element (or a feeder or phase label per customer), 70, 80 or 90% correct. Truth *s_e*.
- Candidates: elements whose state may be wrong. LV: every cable of the register (no line is trusted); MV: every line of the open rings (radial spurs cannot be open); Far West: every branch. Nothing about where the errors are is given.
- Measurements per time step *t*: |V_i(t)| with σ = 0.0025 p.u. (Class 0.5), P_i(t), Q_i(t) with 1% error, P and Q at the boundary (feeder heads or tie lines) with 1% error. No phase angle.

## Step 1 — features

**F1 Variance distance** (Variance-Distance Spanning Tree). With Ṽ the moving average of |V| over about four hours (15 samples at 15 min, 4 at 1 h):

    D_ij = Var_t( Ṽ_i(t) − Ṽ_j(t) )

Neighbouring buses move together, so D is small across a real line. Features: log10(D_e / median D of the candidates) and the rank of D_e among the candidates.

**F2 Variance-Distance Spanning Tree result.** Kruskal on D: candidates in increasing D while no cycle is formed. Feature: 1 if the element is kept. (On the meshed Far West network every element is a candidate, so this is a spanning tree on D.)

**F3 Power-flow residual on the fluctuations** (ours). For a topology hypothesis *s*, predict the voltages from the measured injections and compare:

- Radial LV and MV networks — linearised network equations with the substation bus as reference: Z_s = (Y_s reduced)⁻¹ and ΔV ≈ Re( Z_s · conj(S / |V|) ).
- Meshed Far West — full AC Newton–Raphson for all hours at once, measured injections at every bus, tie-line flows as injections at the boundary buses, one reference bus (measured |V|, angle 0) per electrical island. Single-element changes are evaluated by Newton steps with the reference Jacobian updated by the element's own low-rank contribution (Woodbury); elements documented out of service are evaluated by an exact warm-started solve. Hours in which the reference solution does not converge are excluded. If the documented topology has no power-flow solution in at least half of the hours, the reference topology is "all elements in service".

Residual r_i(t) = |V_i(t)| measured − predicted, with two means removed: the mean over buses at each time (unknown source voltage) and the time mean of each meter (constant model errors such as cable charging, transformer taps, meter bias). Then

    J(s) = Σ_t Σ_i ( r_i(t) / σ )²,      σ = max(meter σ, rms residual of the starting state)

Feature: dJ_e = ( J(d) − J(d with e changed) ) / J(d). Positive means the change explains the measurements better. On MV the change is a branch exchange (close the documented open point, open a line of the same ring) and dJ_e is the best exchange that involves e. On Far West two extra features: the reference state of the element, and, for use case 4, the change in the share of hours with a power-flow solution.

**F4 Voltage correlation** (Liu): log10(1 − ρ_ij), ρ the Pearson correlation of the smoothed voltages of the two ends.

**F5 Documented state** d_e.

Customer-feeder assignment (use case 2), per (customer *i*, feeder *k*): mean and top-3 correlation of ΔV_i (relative to the substation) with the customers documented on *k*; power-balance gain: relative drop of Σ_k ‖P_head,k − Σ_{i∈k} P_i‖² if *i* moves to *k*; documented label.

Phase (use case 3), per (customer *i*, phase *φ*): correlation with the substation voltage of *φ* and with the customers documented on *φ*; three-phase residual of the customer's voltage for phase *φ* (linearised, Z0 = 4 Z1); Liu's clustering label; documented label.

## Step 2 — random forest (Liu)

100 trees with the scikit-learn defaults, as in Liu's code; label = true state (or "customer is on this feeder / phase"), output probability p. Training never sees the case being evaluated:

- LV: leave one feeder out (each feeder has a single topology).
- MV: other switching configurations of the same network (4-fold, grouped by configuration).
- Far West: other configurations of the same network (5-fold, grouped by true configuration).

## Step 3 — decision under physical constraints

- LV, MV: Kruskal on p — candidates by decreasing p while no cycle is formed (radial result). The forest judges every element on its own and does not know that the network must be radial; this step turns its probabilities into a topology that can exist.
- Far West: Kruskal on p so that every bus is supplied, then every element with p > 0.5.
- Feeder, phase: highest p per customer.

## Step 4 — physics refinement

Local moves from the step-3 result, accepted while the residual J drops by more than 1%, restricted to decisions the forest is unsure about (0.05 < p < 0.95):

- LV: branch exchange (add a candidate, remove one on the loop it forms), pure additions and pure removals.
- MV: branch exchange between unsure lines.
- Far West: single element changes (first-order screening, exact confirmation), on every third hour of the month.
- Feeder: move a customer if the feeder-head power balance improves; applied to every customer.
- Phase: move a customer to the phase with a residual at least 5% lower.

The gate admits many elements, but few change: in use case 1 (LV, realistic register, 80%) 0–2 exchanges per case.

## Baselines in the tables

- **Variance-Distance Spanning Tree**: the original method of [`../variance_distance_spanning_tree`](../variance_distance_spanning_tree), unchanged. For feeder assignment, the same principle: each customer takes the majority documented label of its five closest customers by variance distance.
- **Random forest (Liu)**: use cases 1 and 4 — his forest per element on F4 + F5 (correlation and documentation), threshold 0.5, no physics, constraints or refinement. His original switch-bar classifier needs a switch-bar structure and a switching history that contains the true state; on SimBench MV it finds 58.9% of the open points (results/extra), because moved open points almost never appear in the training configurations. Use case 2 — his clustering of the transformed voltage correlation (α = 0.5, a = 1, complete linkage), clusters named by matching with the documentation; with missing labels his KNN (k = 5) on the voltage series. Use case 3 — his phase clustering (per-time standardisation, distance 1.1 − ρ, complete linkage, per feeder).

## Evaluation protocol

| | LV (use cases 1–3) | MV (use case 4) | Far West use case 1 | Far West use case 4 |
|---|---|---|---|---|
| Networks | 6 SimBench LV feeders, personalized customer profiles | 4 SimBench MV open-ring networks | ACTIVSg2000 area 1 (91 buses, 108 branches) | same |
| Truth | published topology; random feeder and phase errors | 16 configurations per network, open points moved along their ring | 93 configurations with 0–6 branches out | 62 configurations with 5 branches out |
| Documentation | 30 / 20 / 10% mixed errors; 30 / 20 / 10% wrong labels | 30 / 20 / 10% of the open points moved (rounded) | truth with 30 / 20 / 10% of the states flipped | 4 of the 5 outages right (80%) |
| Data | one month, 15 min | one month, 15 min | one month, 1 h | one month, 1 h |
| Cases | 72 per level | 64 per level | 93 | 62 |

Share correct = elements (or customers) in the right state; for use case 1 min(precision, recall) of the line set. For use case 4 only the elements that are truly open count, since the other lines never change.

## Realism checks

- **Realistic asset register (LV).** The cable list holds every real cable, every wrongly documented one and 20% spare cables between nodes 2–3 steps apart; 20% of the cables have a length 10–30% wrong in the register (measurements come from the true lengths). Phase: the same length errors.
- **Training only from the register (LV, 80%).** For each evaluated case the documented topology, completed to a radial tree, is taken as true; voltages are predicted on it from the measured P, Q with the register's line data; the register is corrupted again 8 times with the same error model; the forest is trained on these synthetic cases only and applied to the real measurements (`simbench/exp_lv_selftrain.py`).
- **Far West fast mode.** Every second hour of the month and every other case, with 20% of the branch lengths 10–30% wrong in the model. Used for 70 / 80 / 90%.
- **Fewer meters (MV).** Half of the MV/LV substations metered; unmetered buses get standard-profile pseudo-injections and the nearest meter's voltage.

## Use case 3

The combination above (features per customer and phase) and, next to it, the siamese method of Dong Liu, described in [`../uc3_siamese_method/README.md`](../uc3_siamese_method/README.md).
