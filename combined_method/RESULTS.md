# Combined method — results

One method, with the same minimum requirements on SimBench and on Far West, covers the four D2.2 use cases. It combines validate_topology with the random forest of Liu (Dong) and is the best of the three in every use case and network. It works well for topology (use case 1, both networks), phase identification and MV switches; it gains little for customer-feeder assignment and for switch validation on Far West.

## Minimum requirements (identical everywhere)

- Documented topology, switch states, phases or feeders about 85% correct.
- Line and transformer impedances from the asset database.
- Voltage magnitude at every metered bus, Class 0.5 (σ = 0.0025 p.u.). **No phase angle.**
- P and Q injections at every metered bus, 1% error.
- P and Q at the network boundary (feeder heads in the substation, or tie lines), 1% error.
- One month of data at the network's native resolution (SimBench 15 min, Far West 1 h).

These requirements are a choice of the team, based on meter accuracy standards (IEC 61557-12, ANSI C12.20) and on what DSOs and TSOs measure today. D2.2 does not specify them and the Liu paper does not define them for other networks; they should be confirmed with the DSO and TSO nodes.

## Results

| Use case | Network | Documented | validate_topology | Random forest (Liu) | Combination |
|---|---|---|---|---|---|
| 1. Topology | SimBench LV | 85.0% | 96.6% | 88.3% | **97.7%** |
| 1. Topology | Far West | 85.2% | 82.1% | 98.7% | **99.5%** |
| 2. Customer-feeder | SimBench LV | 84.7% | 69.7% | 85.4% | **87.7%** |
| 3. Phase | SimBench LV | 84.7% | 89.9% | 87.3% | **97.3%** |
| 4. Switches | SimBench MV | 84.6% (80) | 78.5% (101) | 82.2% (53) | **96.5% (20)** |
| 4. Switches | Far West | 85.7% (63) | 72.8% (939) | 80.0% (65) | **86.8% (40)** |

- **Share correct**: elements (lines or customers) in the right state. For use case 4 only the elements that are truly open count (open points in MV, out-of-service branches in Far West); in brackets, elements reported open that are actually closed.
- **validate_topology**: the original method in [`../validate_topology`](../validate_topology), unchanged. For use cases 2 and 3 the same principle is applied to assignments (label of the electrically closest customers by variance distance).
- **Random forest (Liu)**: his classifier with his features (voltage correlations) and the 85% documentation, without physics, constraints or refinement.
- Cases: LV 72, MV 64, Far West 93 (use case 1) and 63 (use case 4). On Far West the power-flow residual uses every third hour of the month (240 h) to keep the run time reasonable; variance distance and correlations use all 720 h.
- One month instead of one week raised the combination by 1.4 points (LV topology), 2.1 (phase) and 1.9 (MV switches).

## Conclusions

- **Use case 1 works well on both networks.** On LV, validate_topology alone is already strong and the combination adds about one point. On Far West, validate_topology does not suit a meshed network (its spanning tree drops lines in service); the random forest reaches 98.7% and the combination 99.5%.
- **Use case 2: no good method yet.** 87.7% against 84.7% documented. SimBench uses a few standard load profiles, so many customers cannot be told apart; real smart meter data are needed.
- **Use case 3 works well.** The combination clearly beats both parts (97.3% against 89.9% and 87.3%).
- **Use case 4 works well on MV, little gain on Far West.** On MV 96.5% of the open points are found with 20 false openings. On Far West, with voltage magnitude only, 86.8% against 85.7% documented, with false outages cut from 63 to 40: in transmission the voltage magnitude changes little when one branch is out.

Use cases 2 and 3 do not exist on Far West: a transmission network has no LV customers, feeders or single-phase connections.

## Assumptions and limits

- Simulated data (SimBench network models, ACTIVSg2000 synthetic grid); no field validation.
- Impedances in the asset database assumed exact; meters synchronised; every customer metered.
- MV: a meter at every MV/LV substation. Phase: single-phase customers, Z0 = 4 Z1 for the cables (SimBench does not give it).
- Far West: buses 1044 and 1091 (no in-area branch) are left out; hours where the power flow does not converge are excluded from the residual.
- The refinement thresholds (0.05–0.95 uncertainty gate, 1% residual drop) were chosen on the same simulated data, so the numbers are slightly optimistic.
