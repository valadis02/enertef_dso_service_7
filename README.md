# DSO Service 7 — AI-based Grid Topology Identification

EnerTEF TEF DSO node · D2.2 §2.6.7

The repository has three parts.

| Folder | What it is |
|---|---|
| [`variance_distance_spanning_tree/`](variance_distance_spanning_tree/) | The original service: validates the documented topology of an LV network against smart meter voltages (variance distance + Kruskal), with its report and experiments. |
| [`combined_method/`](combined_method/) | All four use cases: the random forest of Dong Liu (TU Delft) combined with Variance-Distance Spanning Tree, the power-flow residual and a physics refinement, on SimBench (LV, MV) and on the ACTIVSg2000 Far West transmission area. Code to reproduce every result, the results and the summary ([`RESULTS.md`](combined_method/RESULTS.md)). |
| [`uc3_siamese_method/`](uc3_siamese_method/) | Use case 3 (phase identification): the siamese method of Dong Liu, our implementation from the paper, reported next to the combination. |

## Results at a glance (one month of data, no phase angle)

Share of the network in the right state for documentation 70 / 80 / 90% correct. The combination builds on the random forest of Dong Liu and adds the physics of the network.

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

Use case 4: share of truly open elements found. SimBench with personalized customer profiles ([`combined_method/PERSONALIZED_PROFILES.md`](combined_method/PERSONALIZED_PROFILES.md)).
Details, minimum requirements and limits: [`combined_method/RESULTS.md`](combined_method/RESULTS.md); method: [`combined_method/METHOD.md`](combined_method/METHOD.md).

All results are on simulated data (SimBench network models, ACTIVSg2000 synthetic grid); there is no field validation.
