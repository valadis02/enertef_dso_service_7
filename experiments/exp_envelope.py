"""Section 4.3 — operating envelope.

How accuracy holds up as the documentation gets worse and as the list of
possible connections gets more generous. Accuracy stays 95-100% throughout;
the gain over doing nothing grows with the error rate.
"""

from _common import SMALL, load, save, short
from topology import data, errors, validate_topology, evaluate

ERROR_RATES = [0.05, 0.10, 0.15, 0.25, 0.40]
DECOYS = [0.0, 1.0, 3.0, 8.0]


def main():
    rows = []
    for code in SMALL[1:] + ["1-LV-rural2--0-sw"]:
        tg, V, A, nodes = load(code)
        Vn, An = data.add_meter_noise(V, A, "class_0.5+angle")
        n_edges = tg.number_of_edges()

        for er in ERROR_RATES:
            n_err = max(1, round(n_edges * er))
            documented = errors.mixed(tg, n_err)
            doc_acc = evaluate(documented, tg)["reconstruction_accuracy"]

            for d in DECOYS:
                candidates = errors.possible_connections(tg, documented, decoy_factor=d)
                res = validate_topology(Vn, documented, nodes, An,
                                        possible_connections=candidates)
                rows.append({
                    "feeder": short(code), "error_rate": f"{er:.0%}", "decoys": f"x{d:g}",
                    "candidates_pct": round(100 * len(candidates) / n_edges),
                    "documented": round(doc_acc, 3),
                    "validated": round(evaluate(res.topology, tg)["reconstruction_accuracy"], 3),
                })

    df = save(rows, "envelope.csv")
    print("\nAccuracy by error rate and candidate-set size:")
    print(df.pivot_table(index="error_rate", columns="decoys",
                         values="validated").round(3).to_string())


if __name__ == "__main__":
    main()
