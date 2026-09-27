"""Near-real-time monitoring: detecting switching events as they happen.

D2.2 asks for the service to run continuously and to detect topology
changes following switching operations.

DESIGN NOTE. The obvious approach -- validate each window and alert when
the result differs from the documented topology -- does not work. The
validation step has a small baseline error (about one edge in forty on
realistic feeders), so almost every window differs from the documentation
by an edge or so and the monitor alarms constantly. Measured: 90 false
alarms in 108 pre-event windows.

What works is to compare each window against a *baseline built from the
monitor's own recent output*. The method's systematic mistakes appear
identically in every window and cancel; only a genuine change in the
network shows up as a difference. An alert needs the new state to persist
for `persistence` consecutive windows, which suppresses one-off noise.
"""

from collections import Counter
from dataclasses import dataclass, field

from .validation import validate_topology


@dataclass
class MonitoringResult:
    windows: list = field(default_factory=list)   # (end_step, change_vs_baseline)
    alerts: list = field(default_factory=list)    # (end_step, changed_edges)

    def first_detection(self, after=0):
        for end, changes in self.alerts:
            if end >= after:
                return end
        return None

    def false_alarms(self, before):
        return sum(1 for end, _ in self.alerts if end < before)

    def change_at(self, step):
        return dict(self.alerts).get(step)


def monitor(voltage, documented, possible_connections, nodes=None, angle=None,
            window=96, stride=12, baseline_windows=4, persistence=2,
            smooth_window=15):
    """Validate over a sliding window and alert on sustained changes.

    window:           samples per validation (96 = one day at 15 min)
    stride:           samples between windows (12 = every 3 hours)
    baseline_windows: how many recent windows define "normal"
    persistence:      consecutive windows a change must hold before alerting
    """
    nodes = list(voltage.columns) if nodes is None else list(nodes)
    res = MonitoringResult()
    history = []                  # validated topologies, as frozensets of edges
    streak_state, streak = None, 0

    for end in range(window, len(voltage) + 1, stride):
        sl = slice(end - window, end)
        V = voltage.iloc[sl].reset_index(drop=True)
        A = angle.iloc[sl].reset_index(drop=True) if angle is not None else None
        out = validate_topology(V, documented, nodes, A,
                                possible_connections=possible_connections,
                                smooth_window=smooth_window)
        current = frozenset(frozenset(e) for e in out.topology.edges())

        if len(history) < baseline_windows:
            history.append(current)
            res.windows.append((end, []))
            continue

        baseline = Counter(history[-baseline_windows:]).most_common(1)[0][0]
        diff = sorted(tuple(sorted(e)) for e in current ^ baseline)
        res.windows.append((end, diff))

        if diff:
            if streak_state == current:
                streak += 1
            else:
                streak_state, streak = current, 1
            if streak == persistence:
                res.alerts.append((end, diff))
                history = [current] * baseline_windows   # new normal
                streak_state, streak = None, 0
                continue
        else:
            streak_state, streak = None, 0

        history.append(current)
    return res
