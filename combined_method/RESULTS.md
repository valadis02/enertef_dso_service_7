# Results

Share of the network in the right state, for documentation 70 / 80 / 90% correct. The combination builds on the random forest of Dong Liu (TU Delft) and adds the physics of the network: variance distance (Variance-Distance Spanning Tree), the power-flow residual, radial constraints and a refinement. All numbers are reproduced by `python make_tables.py` from the per-case files in `results/`.

| Use case · scenario | Documented | Variance-Distance Spanning Tree | Random forest (Liu) | Combination | Siamese (Liu) |
|---|---|---|---|---|---|
| 1 · LV | 69.8 / 79.8 / 90.2 | 92.6 / 94.8 / 97.6 | 76.2 / 80.2 / 86.9 | 93.6 / 95.1 / 97.6 | – |
| realistic register (spare cables, wrong lengths) | 69.8 / 79.8 / 90.2 | 81.6 / 83.5 / 85.6 | 63.3 / 69.5 / 79.3 | 85.4 / 89.7 / 94.7 | – |
| same + training only from the register (80%) | 79.8 | 83.5 | 62.8 | 89.1 | – |
| 1 · Far West (80% only) | 79.6 | 82.1 | 98.2 | 99.4 | – |
| fast mode + wrong lengths | 70.4 / 79.6 / 89.8 | 81.9 / 81.9 / 81.9 | 98.0 / 98.1 / 98.3 | 99.0 / 99.2 / 99.3 | – |
| 2 · LV | 69.8 / 79.1 / 90.5 | 69.3 / 71.9 / 74.4 | 50.8 / 53.7 / 55.2 | 89.5 / 94.2 / 96.1 | – |
| no metering at the feeder heads | 69.8 / 79.1 / 90.5 | 69.3 / 71.9 / 74.4 | 50.8 / 53.7 / 55.2 | 70.5 / 80.7 / 91.3 | – |
| 30% customers unlabelled, as in Liu | 69.8 (known) | 91.9 | 90.5 | 96.8 | – |
| 3 · LV | 69.8 / 79.1 / 90.5 | 82.3 / 89.2 / 92.9 | 82.8 / 85.0 / 88.3 | 98.3 / 98.9 / 99.5 | 98.5 at all levels (no register) |
| wrong cable lengths in the register | 69.8 / 79.1 / 90.5 | 82.3 / 89.2 / 92.9 | 82.8 / 85.0 / 88.3 | 98.3 / 98.7 / 99.5 | – |
| 4 · MV | 71.5 / 81.5 / 86.9 | 78.9 / 78.3 / 76.7 | 69.3 / 77.7 / 83.7 | 90.6 / 95.0 / 95.9 | – |
| meter at half of the MV/LV substations (80%) | 81.5 | 29.2 | 69.8 | 88.3 | – |
| 4 · Far West (80% only) | 80.0 | 74.5 | 71.9 | 85.8 | – |
| fast mode + wrong lengths | 80.0 | 74.2 | 68.4 | 82.6 | – |


| Use case 4 · false openings | Documented | Variance-Distance Spanning Tree | Random forest (Liu) | Combination |
|---|---|---|---|---|
| MV 70% | 144 | 102 | 151 | 50 |
| MV 80% | 96 | 103 | 129 | 28 |
| MV 90% | 64 | 113 | 83 | 22 |
| MV 80%, half metered | 96 | 355 | 149 | 65 |
| Far West 80% | 62 | 1009 | 84 | 26 |

## Reading the table

- **Main rows** (bold use case numbers in the deck): SimBench with personalized customer profiles, one month, |V| Class 0.5, no phase angle. LV use case 1: no line of the register is trusted. Far West: full mode, 80% only.
- **Realistic register (LV).** The cable list also holds 20% spare cables, and 20% of the lengths are wrong by 10–30%. Variance-Distance Spanning Tree drops from about 95% to 82–86% (it relies on the cable list alone); the combination keeps 85–95% and leads by 4–9 points.
- **Training only from the register.** The forest never sees a verified topology: it is trained on synthetic errors added to the operator's own register, with the measured P, Q. 89.1% against 89.7% with verified training networks.
- **Far West fast mode** (every second hour, every other case, 20% of the branch lengths wrong): nearly identical to the full run for use case 1; for use case 4 the gain over the documentation shrinks from 5.8 to 2.6 points.
- **Use case 2 needs the feeder-head measurement.** Without it the combination stays at the level of the documentation (+0.7–1.6 points).
- **30% unlabelled, as in Liu's paper** (70% known labels, the rest correct): share of all customers right; of the unknown customers the combination finds 89.6%, Liu's KNN on the voltage series 68.7%.
- **Siamese (Liu)**, use case 3: uses only active power, no voltage and no register, so it is the same at every level.
- **Random forest (Liu)**: his forest per element on correlation and documentation (use cases 1, 4); his clustering (use cases 2, 3). Settings from his public code. His original switch-bar classifier on SimBench MV, 80%: 58.9% of the open points, 218 false openings (`results/extra/`): moved open points almost never appear in the switching history it is trained on.

## Minimum requirements

| Data | Source | Needed for |
|---|---|---|
| Old topology, with errors | Asset register | Use cases 1, 2, 4 |
| Cable types and lengths | Asset register | Residual (1, 4) |
| \|V\| Class 0.5, no phase angle | Smart meters, SCADA | Use cases 1, 2, 4 |
| P, Q per customer, 1% | Smart meters | All |
| P per LV feeder head | Substation metering | Use case 2 |
| P per phase at the transformer | Substation metering | Use case 3 (siamese) |
| One month, 15 min (TSO: 1 h) | — | All |
| Training examples | Simulation of the network from the register with synthetic errors, or a few verified feeders, or the SCADA switching history (MV) | Random forest (1–4) |

The register must list every cable that physically exists, even if its state is recorded wrongly. Geographic positions are not needed.

## Limits

- Simulated data only (SimBench, ACTIVSg2000 Far West); synthetic customer profiles. No field validation.
- LV use cases 1 and 2 are simulated as balanced; only use case 3 has a three-phase model.
- Every customer metered, synchronised, without gaps (partial metering tested on MV only).
- The refinement thresholds (gate 0.05–0.95, 1% residual drop) were chosen on the same simulations.
- Far West use case 4 at 70% and 90% would need new scenarios (5 outages allow 60, 80 or 100% only).
- Use cases 2 and 3 do not exist on a transmission network.
