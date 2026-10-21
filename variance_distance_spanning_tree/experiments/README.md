# Experiments

One script per table in `DSO_Service7_Summary.md`. Each writes a CSV to
`results/` and prints a summary.

| Script | Report section | Runtime |
|---|---|---|
| `exp_validation.py` | 4.2 — main result, six feeders | ~4 min |
| `exp_envelope.py` | 4.3 — error rate x candidate-set size | ~5 min |
| `exp_recall.py` | 4.5 — sensitivity to candidate recall | ~2 min |
| `exp_error_types.py` | 4.6 — which errors are observable | ~2 min |
| `exp_flagging.py` | 5 — flagging without a candidate list | ~3 min |
| `exp_blind.py` | 3 — the rejected use case | ~3 min |
| `exp_switching.py` | 4.8 — switching-event detection | ~5 min |

Run from this directory:

```bash
cd experiments
python exp_validation.py
```

Most of the runtime is SimBench power flow, not the method itself.

Seeds are fixed, so results are reproducible. If a number differs from the
report, that is a bug in the port, not a new finding — the original CSVs
are on the `archive` branch for comparison.

Not yet ported: the ablation (section 4.7) and the good-meter grid
(section 4.4).
