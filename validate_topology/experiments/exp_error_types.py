"""Section 4.6 — how observable each kind of documentation error is.

An open switch recorded as closed is the commonest case in practice and
the easiest to detect. A local reconnection is the hardest: the two
candidate parents are electrically almost identical.
"""

from _common import SMALL, ERROR_RATE, load, save, short
from topology import data, errors, validate_topology, evaluate

KINDS = {
    "open switch": errors.open_switch,
    "mixed": errors.mixed,
    "local reroute": errors.reroute,
}


def main():
    rows = []
    for code in SMALL[1:]:            # rural1 is too small for 15% to be meaningful
        tg, V, A, nodes = load(code)
        n_err = max(1, round(tg.number_of_edges() * ERROR_RATE))
        Vn, An = data.add_meter_noise(V, A, "class_0.5+angle")

        for name, make_errors in KINDS.items():
            documented = make_errors(tg, n_err)
            candidates = errors.possible_connections(tg, documented)
            res = validate_topology(Vn, documented, nodes, An,
                                    possible_connections=candidates)
            rows.append({
                "feeder": short(code), "error_type": name,
                "documented": round(evaluate(documented, tg)["reconstruction_accuracy"], 3),
                "validated": round(evaluate(res.topology, tg)["reconstruction_accuracy"], 3),
            })

    df = save(rows, "error_types.csv")
    print("\nMean by error type:")
    print(df.groupby("error_type")[["documented", "validated"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
