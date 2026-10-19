# DSO Service 7 — AI-based Grid Topology Identification

EnerTEF TEF DSO node · D2.2 §2.6.7

The repository has three parts.

| Folder | What it is |
|---|---|
| [`variance_distance_spanning_tree/`](variance_distance_spanning_tree/) | The original service: validates the documented topology of an LV network against smart meter voltages (variance distance + Kruskal), with its report and experiments. |
| [`combined_method/`](combined_method/) | Use cases 1, 2 and 4: Variance-Distance Spanning Tree combined with the random forest of Dong Liu (TU Delft) and a physics refinement, on SimBench (LV, MV) and on the ACTIVSg2000 Far West transmission area. Code to reproduce every result, the results and the summary ([`RESULTS.md`](combined_method/RESULTS.md)). |
| [`uc3_siamese_method/`](uc3_siamese_method/) | Use case 3 (phase identification): the siamese method of Dong Liu, our implementation from the paper, on SimBench with personalized customer profiles. |

## Results at a glance (one month of data, no phase angle)

| Use case | Network | Documented | Variance-Distance Spanning Tree | Random forest (Liu) | Combination | Siamese (Liu) |
|---|---|---|---|---|---|---|
| 1. Topology | SimBench LV | 85.0% | 96.6% | 88.3% | **97.7%** | - |
| 1. Topology | Far West | 85.2% | 82.1% | 98.7% | **99.5%** | - |
| 2. Customer-feeder | SimBench LV | 84.7% | 69.7% | 85.4% | **87.7% / 95.2%\*** | - |
| 3. Phase | SimBench (personalized) | 84.7% | - | - | - | **98.5%** |
| 4. Switches | SimBench MV | 84.6% (80) | 78.5% (101) | 82.2% (53) | **96.5% (20)** | - |
| 4. Switches | Far West | 85.7% (63) | 72.8% (939) | 80.0% (65) | **86.8% (40)** | - |

SimBench (personalized): a personalized load profile for every customer instead of SimBench's few shared standard ones; \* on SimBench (personalized), with the refinement applied to every customer. See [`combined_method/PERSONALIZED_PROFILES.md`](combined_method/PERSONALIZED_PROFILES.md).

Use case 4: share of truly open elements found; in brackets, elements reported open that are actually closed.
Details: [`combined_method/RESULTS.md`](combined_method/RESULTS.md), method: [`combined_method/METHOD.md`](combined_method/METHOD.md).

All results are on simulated data (SimBench network models, ACTIVSg2000 synthetic grid); there is no field validation.
