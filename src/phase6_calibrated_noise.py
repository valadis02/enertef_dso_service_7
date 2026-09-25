"""
DSO Service 7 - Calibrated noise test
Re-run the noise robustness test using REALISTIC noise levels derived
from standard smart meter accuracy classes, instead of arbitrary sigma
values.

Standard reference: accuracy class = % of full scale at 95% CI (~2 sigma).
  - ANSI C12.20 Class 0.5 (typical modern utility smart meter): 0.5% FS
  - Class 1 (older/cheaper meters): 1% FS
  - Class 0.1 (high-precision / PMU-grade): 0.1% FS
Since voltage is expressed in p.u. with nominal = 1.0, "% of full scale"
maps directly to p.u. Converting the 95% CI (2 sigma) bound to 1 sigma:
  sigma = (accuracy_class / 100) / 2
"""

import numpy as np
import pandas as pd

from phase1_2_baseline import load_feeder, ground_truth_graph, simulate_timeseries, correlation_mst, evaluate

METER_CLASSES = {
    "Class 0.1 (PMU-grade)": 0.001 / 2,    # 0.1% FS -> sigma = 0.0005 p.u.
    "Class 0.5 (typical smart meter)": 0.005 / 2,  # 0.5% FS -> sigma = 0.0025 p.u.
    "Class 1 (older/cheaper meter)": 0.01 / 2,     # 1% FS -> sigma = 0.005 p.u.
}


def main():
    net = load_feeder()
    true_graph = ground_truth_graph(net)
    voltage_df, _ = simulate_timeseries(net, n_steps=500)

    rng = np.random.default_rng(2026)
    rows = []
    print("Calibrated noise levels (from meter accuracy standards):\n")
    for label, sigma in METER_CLASSES.items():
        noisy = voltage_df + rng.normal(0, sigma, size=voltage_df.shape)
        mst, _ = correlation_mst(noisy)
        metrics = evaluate(mst, true_graph, voltage_df.columns)
        metrics["meter_class"] = label
        metrics["sigma_pu"] = sigma
        rows.append(metrics)
        print(f"{label:35s} (sigma={sigma:.4f} p.u.): "
              f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
              f"reconstruction_accuracy={metrics['reconstruction_accuracy']:.3f}")

    results = pd.DataFrame(rows)
    results.to_csv("results/phase6_calibrated_noise_results.csv", index=False)
    print("\nSaved results/phase6_calibrated_noise_results.csv")


if __name__ == "__main__":
    main()

    