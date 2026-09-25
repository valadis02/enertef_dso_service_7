# DSO Service 7 — AI-based Grid Topology Identification

EnerTEF TEF DSO node · D2.2 §2.6.7

Validates the topology recorded in a DSO's asset database against
operational measurements, and raises alerts for operator verification.

## Install

```bash
pip install -r requirements.txt
```

## Use

```python
from topology import data, validate_topology

net = data.load_simbench("1-LV-semiurb4--0-sw")
V, A = data.simulate_simbench(net)
V, A = data.add_meter_noise(V, A, "class_0.5+angle")

result = validate_topology(
    V, documented_topology, angle=A,
    possible_connections=switch_list,   # omit for flagging only
)

result.topology     # corrected topology
result.alerts       # ranked discrepancies
result.confidence   # score per edge
```

Supply `possible_connections` — the physically existing lines whose state
is uncertain, including those recorded as open — and the service corrects
the topology. Omit it and the service only ranks the documented edges by
how poorly the measurements support them.

The DSO is not asked where the errors are, only which connections are
physically possible.

## Accuracy

| Mode | Meters | Accuracy |
|---|---|---|
| Validation + correction | Class 0.5 | 94% |
| Validation + correction | Class 0.5 + 0.01° angle | 99% |
| Flagging only | Class 0.5 | 25–64% precision |
| Flagging only | + angle | 62–93% precision |
| Blind reconstruction | Class 0.5 | 12–31% |

Blind reconstruction is measurement-limited under realistic noise and is
kept as a capability, not as the operational mode.

## Layout

```
topology/     the service
experiments/  one script per table in the report
results/      CSV outputs
```

See `DSO_Service7_Summary.md` for the evaluation and its limits, and
`DSO_Service7_Delivery_Plan.md` for what remains.

Full experimental history, including the approaches that did not work, is
on the `archive` branch.

## Limits

No field validation: power flow and meter noise are simulated on SimBench
network models. Single-phase equivalent throughout, so phase
identification is not covered. LV only.
