"""
DSO Service 7 - Common-mode removal test
Quick test: does subtracting the network-wide average voltage signal
(the "common mode" driven by shared daily/weather patterns across all
buses) before computing correlation improve topology reconstruction on
real SimBench data?
"""

import numpy as np
import pandas as pd

from phase1_2_baseline import correlation_mst, evaluate
from phase5_simbench import load_simbench_feeder, ground_truth_graph, simulate_timeseries_simbench, add_realistic_meter_noise


def remove_common_mode(voltage_df):
    """Subtract the cross-sectional mean voltage at each timestep from
    every bus. This removes the shared/global fluctuation (e.g. everyone's
    voltage dips together when overall feeder load rises) and leaves the
    LOCAL deviation that should carry more structural (topology) signal."""
    common_mode = voltage_df.mean(axis=1)
    return voltage_df.sub(common_mode, axis=0)


def main():
    net = load_simbench_feeder()
    true_graph = ground_truth_graph(net)
    week_start = 96 * 7 * 26
    voltage_df = simulate_timeseries_simbench(net, n_steps=672, start=week_start)

    print("=== Baseline (raw correlation, no common-mode removal) ===")
    for label, df in [("clean", voltage_df), ("noisy (Class 0.5)", add_realistic_meter_noise(voltage_df))]:
        mst, _ = correlation_mst(df)
        m = evaluate(mst, true_graph, voltage_df.columns)
        print(f"  {label}: precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"recon_acc={m['reconstruction_accuracy']:.3f}")

    print("\n=== With common-mode removal ===")
    for label, df in [("clean", voltage_df), ("noisy (Class 0.5)", add_realistic_meter_noise(voltage_df))]:
        demeaned = remove_common_mode(df)
        mst, _ = correlation_mst(demeaned)
        m = evaluate(mst, true_graph, voltage_df.columns)
        print(f"  {label}: precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"recon_acc={m['reconstruction_accuracy']:.3f}")


if __name__ == "__main__":
    main()

    