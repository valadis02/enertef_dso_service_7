# DSO Service 7 — AI-based Grid Topology Identification

EnerTEF TEF DSO node · D2.2 §2.6.7

The repository has two parts.

| Folder | What it is |
|---|---|
| [`validate_topology/`](validate_topology/) | The original service: validates the documented topology of an LV network against smart meter voltages (variance distance + Kruskal), with its report and experiments. |
| [`combined_method/`](combined_method/) | The final method: validate_topology combined with the random forest of Liu (Dong) and a physics refinement, applied with the same minimum requirements to all four D2.2 use cases on SimBench (LV, MV) and on the ACTIVSg2000 Far West transmission area. All code to reproduce every result, the results and the summary. |

## Results at a glance (combined_method, voltage magnitude only, one month of data)

| Use case | Network | Documented | validate_topology | Random forest (Liu) | Combination |
|---|---|---|---|---|---|
| 1. Topology | SimBench LV | 85.0% | 96.6% | 88.3% | **97.7%** |
| 1. Topology | Far West | 85.2% | 82.1% | 98.7% | **99.5%** |
| 2. Customer-feeder | SimBench LV | 84.7% | 69.7% | 85.4% | **87.7%** |
| 3. Phase | SimBench LV | 84.7% | 89.9% | 87.3% | **97.3%** |
| 4. Switches | SimBench MV | 84.6% (80) | 78.5% (101) | 82.2% (53) | **96.5% (20)** |
| 4. Switches | Far West | 85.7% (63) | 72.8% (939) | 80.0% (65) | **86.8% (40)** |

Use case 4: share of truly open elements found; in brackets, elements reported open that are actually closed.
Details: [`combined_method/RESULTS.md`](combined_method/RESULTS.md), method: [`combined_method/METHOD.md`](combined_method/METHOD.md).

All results are on simulated data (SimBench network models, ACTIVSg2000 synthetic grid); there is no field validation.
