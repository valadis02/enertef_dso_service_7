# EnerTEF DSO Service 7 — AI-based Grid Topology Identification

Reconstructing real distribution-grid topology from voltage/power measurements,
instead of relying on possibly outdated asset-management records.

Part of the EnerTEF TEF DSO node (German node, partner Gridhound/RWTH).
Official service repo: https://github.com/ENERTEF/TEF-DSO-Service-7----AI-based-Grid-Topology-Identification-Service

## Status

- [x] Phase 1: synthetic data generation (pandapower, IEEE 33-bus feeder)
- [x] Phase 2: baseline — Pearson correlation + Maximum Spanning Tree (Kruskal)
      -> precision / recall / reconstruction accuracy = 1.0 on clean data
- [ ] Phase 3: robustness testing (noise, missing data, switching events)
- [ ] Phase 4: advanced methods (Graphical Lasso, MRF, GNN) — optional
- [ ] Phase 5: SimBench / real EnerTEF data

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/phase1_2_baseline.py
```

Outputs:
- `data/voltage_timeseries.csv` — simulated per-bus voltage magnitudes
- `results/phase1_2_baseline_results.csv` — evaluation metrics

## Method

1. Simulate `n_steps` power-flow snapshots on a known IEEE test feeder with
   randomized (correlated + per-load noise) load profiles -> per-bus voltage
   time-series with known ground-truth topology.
2. Compute the Pearson correlation matrix across all bus voltage time-series.
3. Extract the Maximum Spanning Tree (Kruskal) from the correlation graph as
   the predicted topology.
4. Evaluate edge-identification precision/recall and reconstruction accuracy
   against the known ground truth.

Reference: Bolognani et al., "Identification of power distribution network
topology via voltage correlation analysis" (IEEE CDC 2013); Frontiers in
Energy Research (2022), correlation + Kruskal method for LV networks.
