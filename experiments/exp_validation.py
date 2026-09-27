"""Section 4.2 — the main result.

Validation of a documented topology containing 15% errors, across six
SimBench feeders, with and without phase angle.
"""

from _common import ALL, ERROR_RATE, load, save, short
from topology import data, errors, validate_topology, evaluate


def main():
    rows = []
    for code in ALL:
        tg, V, A, nodes = load(code)
        n_err = max(1, round(tg.number_of_edges() * ERROR_RATE))
        documented = errors.mixed(tg, n_err)
        candidates = errors.possible_connections(tg, documented)
        doc_acc = evaluate(documented, tg)["reconstruction_accuracy"]

        for meters in ["class_0.5", "class_0.5+angle"]:
            Vn, An = data.add_meter_noise(V, A, meters)
            res = validate_topology(Vn, documented, nodes, An,
                                    possible_connections=candidates)
            m = evaluate(res.topology, tg)
            rows.append({
                "feeder": short(code), "buses": len(nodes), "meters": meters,
                "documented": round(doc_acc, 3),
                "validated": round(m["reconstruction_accuracy"], 3),
                "precision": round(m["precision"], 3),
                "recall": round(m["recall"], 3),
                "alerts": len(res.alerts),
            })

    df = save(rows, "validation.csv")
    print("\nMean by meter class:")
    print(df.groupby("meters")[["documented", "validated"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
