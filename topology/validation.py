"""Validation of a documented topology — the operational mode.

D2.2 describes the service as comparing the inferred topology against the
DSO's asset database and raising alerts for operator verification. That is
what this module does, in one of two modes depending on what the DSO can
supply.

  With `possible_connections` — the set of physically existing lines,
  including those recorded as open (switches, tie lines). Edges outside
  that set are trusted; inside it the data decides. 94-99% accuracy.

  Without it — no correction is attempted. The documented edges are ranked
  by how poorly the measurements support them, and the worst are flagged
  for inspection. 25-93% precision depending on instrumentation.

The DSO is not asked where the errors are, only which connections are
physically possible. An entirely unrecorded new line falls outside that
set and cannot be recovered; see the report, section 4.5.
"""

from dataclasses import dataclass, field

import numpy as np
import networkx as nx

from .distance import electrical_distance


@dataclass
class ValidationResult:
    """What the service returns."""
    topology: nx.Graph                  # corrected, or the input unchanged
    alerts: list = field(default_factory=list)      # ranked discrepancies
    confidence: dict = field(default_factory=dict)  # per edge, 0-1
    corrected: bool = False

    def summary(self):
        return (f"{'corrected' if self.corrected else 'flagging only'}: "
                f"{self.topology.number_of_edges()} edges, "
                f"{len(self.alerts)} alerts")

    def to_dict(self):
        """Plain structure: topology, per-edge confidence, alerts."""
        def key(e):
            a, b = tuple(e)
            return [int(a), int(b)]
        return {
            "mode": "correction" if self.corrected else "flagging",
            "edges": [
                {"from": int(a), "to": int(b),
                 "confidence": round(self.confidence.get(frozenset((a, b)), 0.0), 4)}
                for a, b in self.topology.edges()
            ],
            "alerts": [
                {"edge": key(al["edge"]),
                 **{k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in al.items() if k != "edge"}}
                for al in self.alerts
            ],
        }

    def to_json(self, path=None, indent=2):
        """Serialise to JSON. Writes to `path` if given, else returns a string."""
        import json
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def to_pandapower(self, net):
        """Apply the validated topology to a pandapower network.

        Returns a copy of `net` with each line switched in or out of service
        to match the result. Lines are matched by their end buses. Connections
        in the result that have no corresponding line in `net` cannot be
        created without line parameters; they are returned in `missing` so
        the operator can add them.
        """
        import copy
        out = copy.deepcopy(net)
        wanted = {frozenset(e) for e in self.topology.edges()}
        seen = set()
        for i, r in out.line.iterrows():
            e = frozenset((int(r["from_bus"]), int(r["to_bus"])))
            out.line.at[i, "in_service"] = e in wanted
            seen.add(e)
        missing = [tuple(e) for e in wanted - seen]
        return out, missing


def _distance(voltage, angle, nodes, smooth):
    src = angle if angle is not None else voltage
    return electrical_distance(src[nodes], smooth)


def _confidence(D, idx, edges):
    """Map electrical distance to a 0-1 score.

    Rank-based rather than absolute: distances have no meaningful scale
    across feeders, but their ordering does.
    """
    if not edges:
        return {}
    d = np.array([D[idx[a], idx[b]] for a, b in edges])
    order = d.argsort().argsort()
    scores = 1.0 - order / max(len(d) - 1, 1)
    return {frozenset(e): float(s) for e, s in zip(edges, scores)}


def validate_topology(voltage, documented, nodes=None, angle=None,
                      possible_connections=None, smooth_window=15):
    """Validate, and where possible correct, a documented topology.

    voltage: DataFrame of per-bus voltage magnitudes.
    documented: nx.Graph from the DSO asset database.
    possible_connections: iterable of frozenset({u, v}) — the physically
        existing lines whose state is uncertain. Omit for flagging only.
    """
    nodes = list(voltage.columns) if nodes is None else list(nodes)
    nodes = [n for n in nodes if n in documented]
    D, dn = _distance(voltage, angle, nodes, smooth_window)
    idx = {c: i for i, c in enumerate(dn)}

    doc_edges = [(a, b) for a, b in documented.edges() if a in idx and b in idx]
    conf = _confidence(D, idx, doc_edges)

    if possible_connections is None:
        alerts = sorted(
            ({"edge": (a, b), "confidence": conf[frozenset((a, b))],
              "reason": "measurements do not support this documented connection"}
             for a, b in doc_edges),
            key=lambda x: x["confidence"],
        )
        return ValidationResult(topology=documented.copy(), alerts=alerts,
                                confidence=conf, corrected=False)

    candidates = {frozenset(e) for e in possible_connections}

    # trust everything outside the candidate set, then add candidates
    # shortest-first while the result stays a forest
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for a, b in doc_edges:
        if frozenset((a, b)) not in candidates:
            G.add_edge(a, b)

    def dist(e):
        a, b = tuple(e)
        return D[idx[a], idx[b]] if a in idx and b in idx else np.inf

    for e in sorted(candidates, key=dist):
        a, b = tuple(e)
        if a not in idx or b not in idx:
            continue
        if not (a in G and b in G and nx.has_path(G, a, b)):
            G.add_edge(a, b)

    before = {frozenset(e) for e in documented.edges()}
    after = {frozenset(e) for e in G.edges()}
    alerts = []
    for e in sorted(before - after, key=dist, reverse=True):
        alerts.append({"edge": tuple(e), "action": "documented but not supported",
                       "confidence": conf.get(e, 0.0)})
    for e in sorted(after - before, key=dist):
        alerts.append({"edge": tuple(e), "action": "supported but not documented",
                       "confidence": 1.0})

    return ValidationResult(topology=G, alerts=alerts,
                            confidence=_confidence(D, idx, list(G.edges())),
                            corrected=True)


def flag_suspicious_edges(voltage, documented, k, nodes=None, angle=None,
                          smooth_window=15):
    """Return the k documented edges the measurements support least."""
    res = validate_topology(voltage, documented, nodes, angle,
                            possible_connections=None, smooth_window=smooth_window)
    return [a["edge"] for a in res.alerts[:k]]
