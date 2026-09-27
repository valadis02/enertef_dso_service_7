"""Switching-event detection — the D2.2 metric that was still missing.

A load transfer (one line opened, a tie closed) happens partway through
the measurement period. The service monitors a sliding window and must
notice that the network no longer matches the documented topology.

Reported per event:
  detected      whether the change was flagged at all
  correct       whether the flagged change is the one that happened
  delay_h       hours from the event to the first window that flags it
  false_alarms  windows before the event that flagged a change
"""

import numpy as np
import networkx as nx

from _common import save, short
from topology import data, monitor
from topology.switching import plan_transfer, simulate_with_event

FEEDERS = ["1-LV-semiurb4--0-sw", "1-LV-urban6--0-sw", "1-LV-rural2--0-sw"]
EVENTS_PER_FEEDER = 3
EVENT_STEP = 240           # day 3 at 15-minute resolution
N_STEPS = 480              # five days
WINDOW = 96                # one day
STRIDE = 12                # every three hours
STEP_HOURS = 0.25


def candidate_set(before, open_edge, tie, n_decoy_lines=6, n_decoy_ties=6, seed=0):
    """What the operator would list as switchable: the lines and ties that
    physically exist, of which only one pair actually changes."""
    rng = np.random.default_rng(seed)
    cands = {frozenset(open_edge), frozenset(tie)}

    lines = [e for e in before.edges() if frozenset(e) not in cands]
    for i in rng.choice(len(lines), size=min(n_decoy_lines, len(lines)), replace=False):
        cands.add(frozenset(lines[i]))

    nodes = list(before.nodes())
    added = 0
    while added < n_decoy_ties:
        a, b = rng.choice(nodes, size=2, replace=False)
        e = frozenset((int(a), int(b)))
        if not before.has_edge(a, b) and e not in cands:
            cands.add(e)
            added += 1
    return cands


def main():
    rows = []
    for code in FEEDERS:
        base = data.load_simbench(code)
        for ev in range(EVENTS_PER_FEEDER):
            li, tie = plan_transfer(base, seed=ev)
            open_edge = (int(base.line.at[li, "from_bus"]), int(base.line.at[li, "to_bus"]))

            V, A, before, after = simulate_with_event(
                base, EVENT_STEP, li, tie, n_steps=N_STEPS)
            nodes = [c for c in V.columns if c in before]
            Vn, An = data.add_meter_noise(V, A, "class_0.5+angle", seed=100 + ev)
            cands = candidate_set(before, open_edge, tie, seed=ev)

            res = monitor(Vn, before, cands, nodes, An, window=WINDOW, stride=STRIDE)

            expected = {frozenset(open_edge), frozenset(tie)}
            first = res.first_detection(after=EVENT_STEP)
            correct = False
            if first is not None:
                changes = res.change_at(first)
                correct = {frozenset(e) for e in changes} == expected

            rows.append({
                "feeder": short(code), "event": ev,
                "opened": open_edge, "tie": tie,
                "detected": first is not None,
                "correct": correct,
                "delay_h": (first - EVENT_STEP) * STEP_HOURS if first is not None else None,
                "false_alarms": res.false_alarms(before=EVENT_STEP),
                "windows_before": sum(1 for e, _ in res.windows if e < EVENT_STEP),
                "alerts_total": len(res.alerts),
            })

    df = save(rows, "switching.csv")
    n = len(df)
    print(f"\nDetected:        {df.detected.sum()}/{n}")
    print(f"Correctly named: {df.correct.sum()}/{n}")
    print(f"Mean delay:      {df.delay_h.mean():.1f} h  (window = {WINDOW*STEP_HOURS:.0f} h)")
    print(f"False alarms:    {df.false_alarms.sum()} in {df.windows_before.sum()} pre-event windows")


if __name__ == "__main__":
    main()
