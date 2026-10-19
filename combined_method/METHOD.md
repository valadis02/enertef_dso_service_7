# Combined method — full description

One pipeline, the same four steps for use cases 1, 2 and 4 on every network (use case 3 uses the siamese method of Dong Liu, in [`../uc3_siamese_method`](../uc3_siamese_method)):

1. physics features → 2. random forest → 3. decision under physical constraints → 4. physics refinement.

Origin of each part: Variance-Distance Spanning Tree, the power-flow residual, the constraints and the refinement are ours; the random forest classifier and the voltage-correlation features come from Liu et al., *A data-driven approach for topology correction in low voltage distribution networks with PVs* (arXiv 2506.20238).

## Notation and inputs

- Network: buses *i*, elements *e* (lines, cables, transformers, switches) with impedance *z_e = r_e + j x_e* from the asset database.
- Documentation: state *d_e* ∈ {0, 1} per element (or a feeder label per customer), about 85% correct. Truth *s_e*.
- Candidates: elements whose state is uncertain (switchable lines, the documented discrepancy candidates); all other elements are trusted and kept fixed.
- Measurements per time step *t*: |V_i(t)| with σ = 0.0025 p.u. (Class 0.5), P_i(t), Q_i(t) with 1% error, P and Q at the boundary (feeder heads or tie lines) with 1% error. No phase angle.

## Step 1 — features

**F1 Variance distance** (Variance-Distance Spanning Tree). With Ṽ the moving average of |V| over about four hours (15 samples at 15 min, 4 at 1 h):

    D_ij = Var_t( Ṽ_i(t) − Ṽ_j(t) )

Neighbouring buses move together, so D is small across a real line. Features: log10(D_e / median D of the trusted elements) and the rank of D_e among the candidates.

**F2 Variance-Distance Spanning Tree result.** Kruskal on D: trusted elements first, then candidates in increasing D while no cycle is formed. Feature: 1 if the element is kept. (On the meshed Far West network every element is a candidate, so this is a spanning tree on D.)

**F3 Power-flow residual on the fluctuations** (ours). For a topology hypothesis *s*, predict the voltages from the measured injections and compare:

- Radial LV and MV networks — linearised network equations with the substation bus as reference: Z_s = (Y_s reduced)⁻¹ and ΔV ≈ Re( Z_s · conj(S / |V|) ).
- Meshed Far West — full AC Newton–Raphson for all hours at once, measured injections at every bus, tie-line flows as injections at the boundary buses, one reference bus (measured |V|, angle 0) per electrical island. Single-element changes are evaluated by Newton steps with the reference Jacobian updated by the element's own low-rank contribution (Woodbury); elements documented out of service are evaluated by an exact warm-started solve. Hours in which the reference solution does not converge are excluded. If the documented topology has no power-flow solution in at least half of the hours, the reference topology is "all elements in service".

Residual r_i(t) = |V_i(t)| measured − predicted, with two means removed: the mean over buses at each time (unknown source voltage) and the time mean of each meter (constant model errors such as cable charging, transformer taps, meter bias). Then

    J(s) = Σ_t Σ_i ( r_i(t) / σ )²,      σ = max(meter σ, rms residual of the starting state)

Feature: dJ_e = ( J(d) − J(d with e changed) ) / J(d). Positive means the change explains the measurements better. On MV the change is a branch exchange (close the documented open point, open a line of the same ring) and dJ_e is the best exchange that involves e. On Far West two extra features: the reference state of the element, and, for use case 4, the change in the share of hours with a power-flow solution.

**F4 Voltage correlation** (Liu): log10(1 − ρ_ij), ρ the Pearson correlation of the smoothed voltages of the two ends.

**F5 Documented state** d_e.

Customer-feeder assignment (use case 2), per (customer *i*, feeder *k*): mean and top-3 correlation of ΔV_i (relative to the substation) with the customers documented on *k*; power-balance gain: relative drop of Σ_k ‖P_head,k − Σ_{i∈k} P_i‖² if *i* moves to *k*; documented label.

## Step 2 — random forest (Liu)

300 trees, minimum leaf 2, label = true state (or "customer is on this feeder"), output probability p. Training never sees the case being evaluated:

- LV: leave one feeder out (each feeder has a single topology).
- MV: other switching configurations of the same network (4-fold, grouped by configuration).
- Far West: other configurations of the same network (5-fold, grouped by true configuration).

## Step 3 — decision under physical constraints

- LV, MV: Kruskal on p — trusted elements first, candidates by decreasing p while no cycle is formed (radial result).
- Far West: Kruskal on p so that every bus is supplied, then every element with p > 0.5.
- Feeder: highest p per customer.

## Step 4 — physics refinement

Local moves from the step-3 result, accepted while the residual J drops by more than 1%, restricted to decisions the forest is unsure about (0.05 < p < 0.95):

- LV: branch exchange (add a candidate, remove one on the loop it forms), pure additions and pure removals.
- MV: branch exchange between unsure lines.
- Far West: single element changes (first-order screening, exact confirmation), on every third hour of the month.
- Feeder: move a customer if the feeder-head power balance improves. In the personalized-profile variant (95.2%\*) this is applied to every customer.

## Baselines in the tables

- **Variance-Distance Spanning Tree**: the original method of [`../variance_distance_spanning_tree`](../variance_distance_spanning_tree), unchanged. For feeder assignment, the same principle: each customer takes the majority documented label of its five closest customers by variance distance.
- **Random forest (Liu)**: the forest on F4 + F5 only (correlations and the 85% documentation), threshold 0.5 or highest probability, no physics, constraints or refinement.

## Evaluation protocol

| | LV (use cases 1–2) | MV (use case 4) | Far West use case 1 | Far West use case 4 |
|---|---|---|---|---|
| Networks | 6 SimBench LV feeders | 4 SimBench MV open-ring networks | ACTIVSg2000 area 1 (91 buses, 108 branches) | same |
| Truth | published topology; feeder of each customer | 16 configurations per network, 15% of the open points moved along their ring | 93 configurations with 0–6 branches out | 63 configurations with 7 branches out |
| Documentation | 15% mixed errors (§4.2 protocol) / 15% wrong feeder labels | the normal open points | truth with 15% of the 108 states flipped | 6 of the 7 outages right, the 7th on a wrong branch |
| Data | one month, 15 min | one month, 15 min | one month, 1 h | one month, 1 h |
| Cases | 72 (12 noise draws per feeder) | 64 | 93 | 63 |

Share correct = elements (or customers) in the right state. For use case 4 only the elements that are truly open count, since the other lines never change.

## Use case 3

Phase identification uses the siamese method of Dong Liu, described in [`../uc3_siamese_method/README.md`](../uc3_siamese_method/README.md).
