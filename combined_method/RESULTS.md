# Results

The best method per D2.2 use case, on SimBench and on Far West. For topology (use case 1), customer-feeder assignment (use case 2) and switches (use case 4) it is our combination of Variance-Distance Spanning Tree with the random forest of Dong Liu (TU Delft); for phase identification (use case 3) it is the siamese method of Dong Liu. It works well for topology on both networks, phase identification and MV switches; it gains little for switch validation on Far West.

## Minimum requirements

**Use cases 1, 2 and 4 (combination), identical on SimBench and Far West:**

- Documented topology, switch states or feeders about 85% correct.
- Line and transformer impedances from the asset database.
- Voltage magnitude at every metered bus, Class 0.5 (σ = 0.0025 p.u.). **No phase angle.**
- P and Q injections at every metered bus, 1% error.
- P and Q at the network boundary (feeder heads in the substation, or tie lines), 1% error.
- One month of data at the network's native resolution (SimBench 15 min, Far West 1 h).

**Use case 3 (siamese):**

- Active power of every customer (smart meter), 1% error.
- Active power of each phase at the LV side of the transformer, 1% error.
- One month of data at 15 min.
- No voltage, no documentation, no impedances.

These requirements are a choice of the team, based on meter accuracy standards (IEC 61557-12, ANSI C12.20) and on what DSOs and TSOs measure today. D2.2 does not specify them; they should be confirmed with the DSO and TSO nodes.

## Results

| Use case | Network | Documented | Variance-Distance Spanning Tree | Random forest (Liu) | Combination | Siamese (Liu) |
|---|---|---|---|---|---|---|
| 1. Topology | SimBench LV | 85.0% | 96.6% | 88.3% | **97.7%** | - |
| 1. Topology | Far West | 85.2% | 82.1% | 98.7% | **99.5%** | - |
| 2. Customer-feeder | SimBench LV | 84.7% | 69.7% | 85.4% | **87.7% / 95.2%\*** | - |
| 3. Phase | SimBench (personalized) | 84.7% | - | - | - | **98.5%** |
| 4. Switches | SimBench MV | 84.6% (80) | 78.5% (101) | 82.2% (53) | **96.5% (20)** | - |
| 4. Switches | Far West | 85.7% (63) | 72.8% (939) | 80.0% (65) | **86.8% (40)** | - |

**SimBench (personalized)**: the SimBench LV networks with a personalized load profile for every customer instead of SimBench's few shared standard profiles. See [`PERSONALIZED_PROFILES.md`](PERSONALIZED_PROFILES.md).

\* On SimBench (personalized), with the power-balance refinement applied to every customer.

- **Share correct**: elements (lines or customers) in the right state. For use case 4 only the elements that are truly open count (open points in MV, out-of-service branches in Far West); in brackets, elements reported open that are actually closed.
- **Variance-Distance Spanning Tree**: our original method in [`../variance_distance_spanning_tree`](../variance_distance_spanning_tree), unchanged. For use case 2 the same principle is applied to assignments (label of the electrically closest customers by variance distance).
- **Random forest (Liu)**: his classifier with his features (voltage correlations) and the 85% documentation, without physics, constraints or refinement.
- **Siamese (Liu)**: his phase identification method (IJEPES 2025), our implementation from the paper, in [`../uc3_siamese_method`](../uc3_siamese_method); uses only active power, no voltage and no documentation.
- Cases: LV 72, MV 64, Far West 93 (use case 1) and 63 (use case 4). On Far West the power-flow residual uses every third hour of the month (240 h) to keep the run time reasonable; variance distance and correlations use all 720 h.
- One month instead of one week raised the combination by 1.4 points (LV topology) and 1.9 (MV switches).

## Conclusions

- **Use case 1 works well on both networks.** On LV, Variance-Distance Spanning Tree alone is already strong and the combination adds about one point. On Far West, Variance-Distance Spanning Tree does not suit a meshed network (its spanning tree drops lines in service); the random forest reaches 98.7% and the combination 99.5%.
- **Use case 2 depends on the data.** 87.7% against 84.7% documented with SimBench's standard profiles, because many customers share an identical profile; 95.2%\* with personalized profiles.
- **Use case 3 works well with the siamese method.** 98.5% on SimBench (personalized), consistent with the 93–99% reported in the paper for 20 days of data.
- **Use case 4 works well on MV, little gain on Far West.** On MV 96.5% of the open points are found with 20 false openings. On Far West, with voltage magnitude only, 86.8% against 85.7% documented, with false outages cut from 63 to 40: in transmission the voltage magnitude changes little when one branch is out.

Use cases 2 and 3 do not exist on Far West: a transmission network has no LV customers, feeders or single-phase connections.

## Assumptions and limits

- Simulated data (SimBench network models, ACTIVSg2000 synthetic grid); no field validation. The personalized profiles are synthetic as well.
- Impedances in the asset database assumed exact; meters synchronised; every customer metered.
- MV: a meter at every MV/LV substation. Phase: single-phase customers.
- Far West: buses 1044 and 1091 (no in-area branch) are left out; hours where the power flow does not converge are excluded from the residual.
- The refinement thresholds (0.05–0.95 uncertainty gate, 1% residual drop) were chosen on the same simulated data, so the numbers are slightly optimistic; the same holds for the use case 2 variant.
