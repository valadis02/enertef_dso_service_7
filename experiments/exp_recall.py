"""Section 4.5 — the critical precondition.

What happens when the list of physically possible connections does not
contain all the real discrepancies. Break-even sits around 30% recall:
below that, correcting does net harm.
"""

from _common import SMALL, ERROR_RATE, load, save, short
from topology import data, errors, validate_topology, evaluate

RECALLS = [1.0, 0.75, 0.5, 0.25, 0.0]


def main():
    rows = []
    for code in SMALL[1:]:
        tg, V, A, nodes = load(code)
        n_err = max(1, round(tg.number_of_edges() * ERROR_RATE))
        documented = errors.mixed(tg, n_err)
        doc_acc = evaluate(documented, tg)["reconstruction_accuracy"]
        Vn, An = data.add_meter_noise(V, A, "good+angle")

        for r in RECALLS:
            candidates = errors.possible_connections(tg, documented, recall=r)
            res = validate_topology(Vn, documented, nodes, An,
                                    possible_connections=candidates)
            acc = evaluate(res.topology, tg)["reconstruction_accuracy"]
            rows.append({
                "feeder": short(code), "candidate_recall": f"{r:.0%}",
                "documented": round(doc_acc, 3), "validated": round(acc, 3),
                "gain": round(acc - doc_acc, 3),
            })

    df = save(rows, "recall.csv")
    print("\nMean by recall:")
    print(df.groupby("candidate_recall")[["documented", "validated", "gain"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
