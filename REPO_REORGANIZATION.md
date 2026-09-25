# Αναδιοργάνωση repository

## Το πρόβλημα

Σήμερα το repo έχει 43+ phase scripts με αλυσίδα imports. Η τελική μέθοδος
(`phase30_final.py`) εξαρτάται από τουλάχιστον 7 άλλα phase αρχεία, και τα
πειράματα του κύριου use case (φάσεις 39–43) **δεν υπάρχουν ως αρχεία** —
έτρεξαν inline και υπάρχουν μόνο τα CSV αποτελεσμάτων.

## Στόχος

Ένα μικρό πακέτο που κάνει ένα πράγμα, και ένα script ανά πίνακα της
αναφοράς.

```
enertef_dso_service_7/
├── README.md
├── requirements.txt
├── SUMMARY.md                      ← DSO_Service7_Summary_v2.md
├── topology/
│   ├── __init__.py
│   ├── data.py          # φόρτωση δικτύων + προσομοίωση μετρήσεων
│   ├── distance.py      # ηλεκτρική απόσταση Var(Xi − Xj)
│   ├── blind.py         # τυφλή ανακατασκευή (ιεραρχία)
│   ├── validation.py    # επικύρωση γνωστής τοπολογίας + flagging
│   ├── errors.py        # προσομοίωση λαθών τεκμηρίωσης
│   └── metrics.py       # accuracy, precision, recall
├── experiments/
│   ├── exp_blind.py            # §3
│   ├── exp_validation.py       # §4.2
│   ├── exp_envelope.py         # §4.3
│   ├── exp_good_meters.py      # §4.4
│   ├── exp_recall.py           # §4.5
│   ├── exp_error_types.py      # §4.6
│   ├── exp_ablation.py         # §4.7
│   └── exp_flagging.py         # §5
└── results/
```

## Από πού έρχεται κάθε κομμάτι

### `topology/` — ο κώδικας

| Νέο αρχείο | Προέλευση | Συναρτήσεις |
|---|---|---|
| `data.py` | `phase1_2_baseline`, `phase5_simbench`, `phase10a_synthetic_generator`, `final_benchmark` | `load_ieee33`, `load_simbench`, `ground_truth_graph`, `simulate_measurements` (τάση + γωνία), `random_radial_network` |
| `distance.py` | `phase19_voltage_difference` | `electrical_distance` |
| `blind.py` | `phase22_hierarchy`, `phase25_hierarchy_v2`, `phase30_final` | `reconstruct` (με SNR gate, γωνία, clustering) |
| `validation.py` | `phase38_localised` + flagging από `phase15_use_cases` | `validate_topology`, `flag_suspicious_edges` |
| `errors.py` | `phase37_prior_informed`, inline φάση 39 | `corrupt_random`, `corrupt_open_switch`, `corrupt_reroute` |
| `metrics.py` | `phase1_2_baseline` | `evaluate` (επιστρέφει accuracy, precision, recall) |

### `experiments/` — ένα script ανά πίνακα

| Script | Πίνακας | Προέλευση | Υπάρχει ως αρχείο; |
|---|---|---|---|
| `exp_blind.py` | §3 | `final_benchmark` + `phase16_pmu_test` | ✅ |
| `exp_validation.py` | §4.2 | `phase38_localised` | ✅ |
| `exp_envelope.py` | §4.3 | φάση 40 | ❌ **inline, να γραφτεί** |
| `exp_good_meters.py` | §4.4 | φάση 41 | ❌ **inline, να γραφτεί** |
| `exp_recall.py` | §4.5 | φάση 42 | ❌ **inline, να γραφτεί** |
| `exp_error_types.py` | §4.6 | φάση 39 | ❌ **inline, να γραφτεί** |
| `exp_ablation.py` | §4.7 | φάση 43 | ❌ **inline, να γραφτεί** |
| `exp_flagging.py` | §5 | φάση 35 | ❌ **inline, να γραφτεί** |

Τα inline πειράματα υπάρχουν αυτούσια στο ιστορικό της συνομιλίας, οπότε
η μεταφορά τους σε αρχεία είναι μηχανική.

## Τι πάει στο archive

Όλα τα υπόλοιπα phase scripts πάνε σε **branch `archive`**, όχι διαγραφή.
Διατηρείται το ιστορικό και η αναφορά μπορεί να παραπέμπει εκεί για τα
αρνητικά αποτελέσματα.

```bash
git checkout -b archive        # κρατά την τωρινή κατάσταση όπως είναι
git push origin archive
git checkout main
# ... αναδιοργάνωση στο main
```

Φάσεις που πάνε μόνο στο archive: 3, 4, 5 (event-based), 6, 7, 8, 9,
10b, 11, 12, 13, 13b, 14, 17, 18, 18b, 20, 21, 23, 24, 25 (ως αυτόνομο),
και `visualize.py`.

## Επαλήθευση μετά την αναδιοργάνωση

Κάθε νέο experiment script πρέπει να αναπαράγει **τα ίδια νούμερα** με τον
αντίστοιχο πίνακα. Αν κάποιο διαφέρει, σημαίνει λάθος στη μεταφορά, όχι
νέο αποτέλεσμα.

Γνωστό σημείο προσοχής: μην φιλτράρεις το slack bus από τη λίστα κόμβων
(έχει ήδη προκαλέσει λάθος αποτέλεσμα μία φορά).

## Εκτίμηση

Αναδιοργάνωση κώδικα: μισή μέρα. Μεταφορά 6 inline πειραμάτων σε scripts:
μισή μέρα. Επαλήθευση ότι αναπαράγονται τα νούμερα: μισή μέρα.
